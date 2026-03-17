"""Train the surrogate MLP on collected PyElastica transition data.

Phase 1 (epochs 1-20):  Single-step MSE loss with noise injection.
Phase 2 (epochs 20+):   Add multi-step rollout loss (8-step BPTT).

Usage:
    python -m aprx_model_elastica.train_surrogate
    python -m aprx_model_elastica.train_surrogate --data-dir data/surrogate_rl_step
    python -m aprx_model_elastica.train_surrogate --no-wandb --epochs 50
"""

import argparse
import json
import math
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from aprx_model_elastica.train_config import SurrogateModelConfig, SurrogateTrainConfig
from aprx_model_elastica.dataset import FlatStepDataset, TrajectoryDataset
from aprx_model_elastica.model import SurrogateModel
from aprx_model_elastica.state import (
    StateNormalizer,
    action_to_omega_batch,
    encode_phase_batch,
    encode_n_cycles_batch,
    raw_to_relative,
    REL_COM_X, REL_COM_Y, REL_HEADING_SIN, REL_HEADING_COS,
    REL_COM_VEL_X, REL_COM_VEL_Y,
    REL_POS_X, REL_POS_Y, REL_VEL_X, REL_VEL_Y, REL_YAW, REL_OMEGA_Z,
    REL_STATE_DIM,
    # Legacy raw slices (kept for backward compat, not used in new code)
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
)
from src.configs.run_dir import setup_run_dir
from src.configs.console import ConsoleLogger
from src.trainers.logging_utils import collect_system_metrics
from src import wandb_utils


# STOP file for graceful stop (touch STOP to stop training)
STOP_FILE = "STOP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate model")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: from config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true", default=False,
                        help="Disable W&B logging (enabled by default)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--hidden-dims", type=str, default=None,
        help="Comma-separated hidden layer dims (e.g., '1024,1024,1024,1024')",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="W&B run name override",
    )
    # Architecture variant args
    parser.add_argument(
        "--rollout-weight", type=float, default=None,
        help="Rollout loss weight override (default: from config)",
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=None,
        help="Rollout steps override (default: from config)",
    )
    parser.add_argument(
        "--use-residual", action="store_true", default=False,
        help="Use ResidualSurrogateModel instead of base SurrogateModel",
    )
    parser.add_argument(
        "--history-k", type=int, default=None,
        help="History window size K for HistorySurrogateModel (0=disabled)",
    )
    parser.add_argument(
        "--arch", type=str, default=None,
        choices=["mlp", "residual", "transformer"],
        help="Architecture to use (overrides --use-residual)",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None,
        help="Number of transformer encoder layers (only with --arch transformer)",
    )
    parser.add_argument(
        "--n-heads", type=int, default=None,
        help="Number of attention heads (only with --arch transformer)",
    )
    parser.add_argument(
        "--d-model", type=int, default=None,
        help="Transformer embedding dimension (only with --arch transformer)",
    )
    return parser.parse_args()


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SurrogateTrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build LR scheduler with warmup.

    Supports config.lr_schedule: "cosine" (default) or "constant".
    """
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    if config.lr_schedule == "constant":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0
    else:  # "cosine"
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def probe_auto_batch_size(
    model: nn.Module,
    sample_input: dict,
    normalizer,
    device: str,
    use_amp: bool,
    start_bs: int = 4096,
    vram_target: float = 0.85,
) -> int:
    """Find the largest batch size that fits within vram_target of total GPU memory.

    Two-phase search:
      1. Coarse: double batch size until OOM (powers of 2 from start_bs)
      2. Fine: binary search between last passing and first failing

    Args:
        model: The surrogate model.
        sample_input: A single-sample dict from the dataset (used to build probe batches).
        normalizer: StateNormalizer for the model.
        device: CUDA device string.
        use_amp: Whether bf16 autocast is enabled.
        start_bs: Starting batch size for coarse search.
        vram_target: Fraction of total VRAM to target (default 0.85).

    Returns:
        Largest batch size that fits.
    """
    if not torch.cuda.is_available() or "cuda" not in device:
        print(f"  Auto-batch: CPU mode, using default {start_bs}")
        return start_bs

    total_vram = torch.cuda.get_device_properties(device).total_memory
    target_bytes = int(total_vram * vram_target)
    amp_ctx = _amp_context(use_amp, device)

    def _try_batch(bs: int) -> bool:
        """Run one forward+backward pass at batch size bs. Returns True if it fits."""
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            # Build probe batch by repeating the sample
            state = sample_input["state"].unsqueeze(0).expand(bs, -1).to(device)
            action = sample_input["action"].unsqueeze(0).expand(bs, -1).to(device)
            serp_time = sample_input["t_start"].unsqueeze(0).expand(bs).to(device)
            delta = sample_input["delta"].unsqueeze(0).expand(bs, -1).to(device)

            state_norm = normalizer.normalize_state(state) if normalizer else state
            delta_norm = normalizer.normalize_delta(delta) if normalizer else delta

            omega = action_to_omega_batch(action)
            phase_enc = encode_phase_batch(omega * serp_time)
            ncycles_enc = encode_n_cycles_batch(action)
            time_enc = torch.cat([phase_enc, ncycles_enc], dim=-1)

            ctx = amp_ctx if amp_ctx is not None else nullcontext()
            with ctx:
                pred = model(state_norm, action, time_enc)
                loss = nn.functional.mse_loss(pred, delta_norm)
            loss.backward()
            model.zero_grad()

            peak = torch.cuda.max_memory_allocated(device)
            return peak <= target_bytes
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model.zero_grad()
            return False

    # Phase 1: Coarse (powers of 2)
    last_pass = start_bs
    bs = start_bs
    while _try_batch(bs):
        last_pass = bs
        bs *= 2
    first_fail = bs

    # Phase 2: Fine (binary search)
    lo, hi = last_pass, first_fail
    while hi - lo > max(256, lo // 8):
        mid = (lo + hi) // 2
        # Round to multiple of 256 for alignment
        mid = (mid // 256) * 256
        if mid <= lo:
            break
        if _try_batch(mid):
            lo = mid
        else:
            hi = mid

    torch.cuda.empty_cache()
    model.zero_grad()
    return lo


def _amp_context(use_amp: bool, device: str):
    """Return bf16 autocast context if enabled, else nullcontext."""
    if use_amp and "cuda" in str(device):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def compute_single_step_loss(
    model: SurrogateModel,
    batch: dict,
    normalizer: StateNormalizer,
    noise_std: float,
    device: str,
    amp_ctx=None,
) -> tuple:
    """Compute single-step MSE loss on 130-dim relative state representation.

    Returns (total_loss, per_component_losses_dict).
    """
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    serp_time = batch["t_start"].to(device)
    delta = batch["delta"].to(device)

    # Add noise to input state for robustness
    if noise_std > 0 and model.training:
        state = state + noise_std * torch.randn_like(state)

    # Normalize (if enabled)
    state_norm = normalizer.normalize_state(state) if normalizer else state
    delta_norm = normalizer.normalize_delta(delta) if normalizer else delta

    # Encode oscillation phase omega*t (not raw t) + n_cycles
    omega = action_to_omega_batch(action)
    phase_enc = encode_phase_batch(omega * serp_time)    # (B, 2)
    ncycles_enc = encode_n_cycles_batch(action)           # (B, 2)
    time_enc = torch.cat([phase_enc, ncycles_enc], dim=-1)  # (B, 4)

    # Forward pass (bf16 autocast wraps forward + loss)
    ctx = amp_ctx if amp_ctx is not None else nullcontext()
    with ctx:
        pred_delta_norm = model(state_norm, action, time_enc)
        loss = nn.functional.mse_loss(pred_delta_norm, delta_norm)

    # Per-component losses (for logging, using 130-dim relative slices)
    with torch.no_grad():
        pred_delta = normalizer.denormalize_delta(pred_delta_norm) if normalizer else pred_delta_norm
        component_losses = {
            "com": nn.functional.mse_loss(
                pred_delta[:, REL_COM_X.start:REL_COM_Y.stop], delta[:, REL_COM_X.start:REL_COM_Y.stop]
            ).item(),
            "heading": nn.functional.mse_loss(
                pred_delta[:, REL_HEADING_SIN.start:REL_HEADING_COS.stop], delta[:, REL_HEADING_SIN.start:REL_HEADING_COS.stop]
            ).item(),
            "com_vel": nn.functional.mse_loss(
                pred_delta[:, REL_COM_VEL_X.start:REL_COM_VEL_Y.stop], delta[:, REL_COM_VEL_X.start:REL_COM_VEL_Y.stop]
            ).item(),
            "rel_pos_x": nn.functional.mse_loss(
                pred_delta[:, REL_POS_X], delta[:, REL_POS_X]
            ).item(),
            "rel_pos_y": nn.functional.mse_loss(
                pred_delta[:, REL_POS_Y], delta[:, REL_POS_Y]
            ).item(),
            "vel_x": nn.functional.mse_loss(
                pred_delta[:, REL_VEL_X], delta[:, REL_VEL_X]
            ).item(),
            "vel_y": nn.functional.mse_loss(
                pred_delta[:, REL_VEL_Y], delta[:, REL_VEL_Y]
            ).item(),
            "yaw": nn.functional.mse_loss(
                pred_delta[:, REL_YAW], delta[:, REL_YAW]
            ).item(),
            "omega_z": nn.functional.mse_loss(
                pred_delta[:, REL_OMEGA_Z], delta[:, REL_OMEGA_Z]
            ).item(),
        }

    return loss, component_losses


def compute_rollout_loss(
    model: SurrogateModel,
    batch: dict,
    normalizer: StateNormalizer,
    device: str,
    discount: float = 0.95,
) -> torch.Tensor:
    """Compute multi-step rollout loss via autoregressive unrolling.

    Args:
        batch: Dict with states (B, L+1, 124), actions (B, L, 5),
               serpenoid_times (B, L).
        discount: Per-step discount for the loss.

    Returns:
        Scalar loss.
    """
    states_seq = batch["states"].to(device)       # (B, L+1, 124)
    actions_seq = batch["actions"].to(device)      # (B, L, 5)
    times_seq = batch["serpenoid_times"].to(device)  # (B, L)

    rollout_len = actions_seq.shape[1]
    state = states_seq[:, 0]  # (B, 124) — start from ground truth

    total_loss = torch.tensor(0.0, device=device)
    for t in range(rollout_len):
        action = actions_seq[:, t]
        omega = action_to_omega_batch(action)
        phase_enc = encode_phase_batch(omega * times_seq[:, t])
        ncycles_enc = encode_n_cycles_batch(action)
        time_enc = torch.cat([phase_enc, ncycles_enc], dim=-1)

        # Predict next state
        state_norm = normalizer.normalize_state(state)
        delta_norm = model(state_norm, action, time_enc)
        delta = normalizer.denormalize_delta(delta_norm)
        state = state + delta  # autoregressive: use predicted state

        # Loss against ground truth
        target = states_seq[:, t + 1]
        step_loss = nn.functional.mse_loss(state, target)
        total_loss = total_loss + (discount ** t) * step_loss

    return total_loss / rollout_len


def evaluate(
    model: SurrogateModel,
    val_loader: DataLoader,
    normalizer: StateNormalizer,
    device: str,
    amp_ctx=None,
) -> dict:
    """Compute validation loss and R2 score.

    Returns dict with 'val_loss', 'r2', and per-component R2 scores.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_pred_deltas = []
    all_true_deltas = []
    with torch.no_grad():
        for batch in val_loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            serp_time = batch["t_start"].to(device)
            delta = batch["delta"].to(device)

            state_norm = normalizer.normalize_state(state) if normalizer else state
            delta_norm = normalizer.normalize_delta(delta) if normalizer else delta

            omega = action_to_omega_batch(action)
            phase_enc = encode_phase_batch(omega * serp_time)
            ncycles_enc = encode_n_cycles_batch(action)
            time_enc = torch.cat([phase_enc, ncycles_enc], dim=-1)

            ctx = amp_ctx if amp_ctx is not None else nullcontext()
            with ctx:
                pred_delta_norm = model(state_norm, action, time_enc)
                loss = nn.functional.mse_loss(pred_delta_norm, delta_norm)

            total_loss += loss.item()
            n_batches += 1

            # Denormalize for R2 computation
            pred_delta = normalizer.denormalize_delta(pred_delta_norm) if normalizer else pred_delta_norm
            all_pred_deltas.append(pred_delta)
            all_true_deltas.append(delta)

    val_loss = total_loss / max(1, n_batches)

    # R2 = 1 - SS_res / SS_tot
    all_pred = torch.cat(all_pred_deltas, dim=0)
    all_true = torch.cat(all_true_deltas, dim=0)
    ss_res = ((all_pred - all_true) ** 2).sum(dim=0)
    ss_tot = ((all_true - all_true.mean(dim=0)) ** 2).sum(dim=0)
    r2_per_dim = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
    r2_overall = r2_per_dim.mean().item()

    # Per-component R2 (using named slice constants from 130-dim relative state)
    r2_components = {
        "com": r2_per_dim[REL_COM_X.start:REL_COM_Y.stop].mean().item(),
        "heading": r2_per_dim[REL_HEADING_SIN.start:REL_HEADING_COS.stop].mean().item(),
        "com_vel": r2_per_dim[REL_COM_VEL_X.start:REL_COM_VEL_Y.stop].mean().item(),
        "rel_pos": r2_per_dim[REL_POS_X.start:REL_POS_Y.stop].mean().item(),
        "vel": r2_per_dim[REL_VEL_X.start:REL_VEL_Y.stop].mean().item(),
        "yaw": r2_per_dim[REL_YAW.start:REL_YAW.stop].mean().item(),
        "omega_z": r2_per_dim[REL_OMEGA_Z.start:REL_OMEGA_Z.stop].mean().item(),
    }

    return {
        "val_loss": val_loss,
        "r2": r2_overall,
        "r2_components": r2_components,
    }


def main():
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Config
    config = SurrogateTrainConfig()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.no_wandb:
        config.wandb_enabled = False
    if args.hidden_dims is not None:
        config.model.hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    # Architecture experiment overrides
    if args.rollout_weight is not None:
        config.rollout_loss_weight = args.rollout_weight
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    config.use_residual = args.use_residual
    if args.history_k is not None:
        config.history_k = args.history_k
    # --arch overrides --use-residual for cleaner sweep support
    if args.arch is not None:
        config.model.arch = args.arch
        if args.arch == "residual":
            config.use_residual = True
    if args.n_layers is not None:
        config.model.n_layers = args.n_layers
    if args.n_heads is not None:
        config.model.n_heads = args.n_heads
    if args.d_model is not None:
        config.model.d_model = args.d_model

    # Timestamped run directory
    run_dir = setup_run_dir(config)
    save_dir = run_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save full config snapshot
    config_snapshot = run_dir / "config.json"
    config_snapshot.write_text(json.dumps(asdict(config), indent=2, default=str))

    # Graceful shutdown handling
    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        print(f"\n{sig_name} received, will stop after current epoch...")
        shutdown_requested = True

    original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    # W&B setup — SurrogateTrainConfig uses flat fields instead of nested .wandb,
    # so we call wandb.init directly here rather than going through wandb_utils.setup_run.
    wandb_run = None
    if config.wandb_enabled:
        import wandb as _wandb
        wandb_run = _wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity or None,
            name=args.run_name or config.name,
            config=asdict(config),
            dir=str(run_dir),
        )
        _wandb.define_metric("train/*", step_metric="epoch")
        _wandb.define_metric("component/*", step_metric="epoch")
        _wandb.define_metric("tracking/*", step_metric="epoch")
        _wandb.define_metric("timing/*", step_metric="epoch")
        _wandb.define_metric("system/*", step_metric="epoch")

    # Metrics log file (JSON-lines, one object per epoch)
    metrics_log_path = run_dir / "metrics.jsonl"
    metrics_log_file = open(metrics_log_path, "a", encoding="utf-8")

    with ConsoleLogger(run_dir):
        # Load data (FlatStepDataset handles t_start key and episode-based splitting)
        print(f"Loading data from {config.data_dir}...")
        train_dataset = FlatStepDataset(
            config.data_dir, split="train", val_fraction=config.val_fraction
        )
        val_dataset = FlatStepDataset(
            config.data_dir, split="val", val_fraction=config.val_fraction
        )
        print(f"  Train: {len(train_dataset):,} transitions")
        print(f"  Val:   {len(val_dataset):,} transitions")
        print(f"  State dim: {train_dataset.states.shape[-1]}")

        # If data is still raw 124-dim (not pre-processed), convert on the fly
        if train_dataset.states.shape[-1] == 124:
            print("  Converting raw 124-dim states to 130-dim relative representation...")
            for ds in (train_dataset, val_dataset):
                ds.states = raw_to_relative(ds.states)
                ds.next_states = raw_to_relative(ds.next_states)
            print(f"  State dim after conversion: {train_dataset.states.shape[-1]}")

        # Compute normalization statistics (on 130-dim relative states)
        if config.normalize_inputs or config.normalize_targets:
            print("Computing normalization statistics...")
            normalizer = StateNormalizer(state_dim=REL_STATE_DIM, device=device)
            deltas = train_dataset.next_states - train_dataset.states
            normalizer.fit(train_dataset.states, deltas)
            if not config.normalize_inputs:
                normalizer.state_mean.zero_()
                normalizer.state_std.fill_(1.0)
            if not config.normalize_targets:
                normalizer.delta_mean.zero_()
                normalizer.delta_std.fill_(1.0)
            normalizer.save(str(save_dir / "normalizer.pt"))
        else:
            normalizer = None

        # Build model (--arch takes priority, then legacy --use-residual / --history-k)
        arch = config.model.arch
        if arch == "transformer":
            from aprx_model_elastica.model import TransformerSurrogateModel
            model = TransformerSurrogateModel(config.model).to(device)
        elif arch == "residual" or config.use_residual:
            from aprx_model_elastica.model import ResidualSurrogateModel
            model = ResidualSurrogateModel(config.model).to(device)
        elif config.history_k > 0:
            from aprx_model_elastica.model import HistorySurrogateModel
            model = HistorySurrogateModel(config.model, history_k=config.history_k).to(device)
        else:
            model = SurrogateModel(config.model).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {n_params:,} parameters ({trainable_params:,} trainable)")

        # Auto batch size probing
        if config.auto_batch_size and "cuda" in device:
            print("Probing auto batch size...")
            sample = train_dataset[0]
            probed_bs = probe_auto_batch_size(
                model, sample, normalizer, device, config.use_amp,
                start_bs=config.batch_size,
            )
            print(f"  Auto-batch: {config.batch_size} -> {probed_bs}")
            config.batch_size = probed_bs

        # Density-based sample weighting (upweight rare states)
        if config.use_density_weighting:
            print("Computing density-based sample weights...")
            sample_weights = train_dataset.get_sample_weights(
                n_bins=config.density_bins,
                clip_max=config.density_clip_max,
            )
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(train_dataset), replacement=True
            )
            print(f"  Weight range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, sampler=sampler,
                num_workers=2, pin_memory=True, drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True,
                num_workers=2, pin_memory=True, drop_last=True,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        # Trajectory dataset for multi-step loss (loaded lazily)
        traj_loader = None

        # Log one-time params (param_count and batch_size via wandb.config.update)
        extra_params = {
            "total_params": n_params,
            "trainable_params": trainable_params,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
            "batch_size": config.batch_size,
            "auto_batch_size": config.auto_batch_size,
        }
        extra_params.update(wandb_utils.collect_hardware_info(device))
        wandb_utils.log_extra_params(wandb_run, extra_params)
        if wandb_run is not None:
            import wandb as _wandb
            _wandb.config.update({
                "param_count": n_params,
                "batch_size": config.batch_size,
            }, allow_val_change=True)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = build_lr_scheduler(optimizer, config, len(train_loader))

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        stop_reason = "completed"

        # Mixed precision context
        amp_ctx = _amp_context(config.use_amp, device)
        grad_accum = config.gradient_accumulation_steps
        effective_batch = config.batch_size * grad_accum
        _system_log_counter = [0]

        print(f"\nTraining on {device}...")
        print(f"  Batch size: {config.batch_size} x {grad_accum} accum = {effective_batch} effective")
        print(f"  Mixed precision (bf16): {config.use_amp and 'cuda' in device}")
        print(f"  Early stopping patience: {config.patience}")
        print(f"  Run directory: {run_dir}")
        # Guard: FlatStepDataset has no multi-step trajectories
        if config.rollout_loss_weight > 0:
            print(f"  WARNING: rollout_loss_weight={config.rollout_loss_weight} but "
                  f"FlatStepDataset has no trajectory windows. Setting rollout_loss_weight=0.0.")
            config.rollout_loss_weight = 0.0
        print(f"  Single-step loss only (rollout_loss_weight=0)")

        epoch = 0
        for epoch in range(1, config.num_epochs + 1):
            # Check graceful shutdown (SIGINT/SIGTERM)
            if shutdown_requested:
                stop_reason = "signal"
                print("Shutdown requested, stopping training.")
                break

            # Check STOP file
            if Path(STOP_FILE).exists():
                stop_reason = "stop_file"
                print("STOP file detected, stopping training.")
                break

            t_epoch = time.monotonic()
            model.train()

            # Track losses
            epoch_loss = 0.0
            epoch_rollout_loss = 0.0
            n_batches = 0
            all_component_losses = {}

            optimizer.zero_grad()
            epoch_grad_norms = []

            # Single-step training
            for batch_idx, batch in enumerate(train_loader):
                loss, comp_losses = compute_single_step_loss(
                    model, batch, normalizer, config.state_noise_std, device,
                    amp_ctx=amp_ctx,
                )

                total_loss = loss

                # Multi-step rollout loss (after warmup)
                if epoch >= config.rollout_start_epoch and config.rollout_loss_weight > 0:
                    if traj_loader is None:
                        print(f"  Loading trajectory dataset (rollout_steps={config.rollout_steps})...")
                        traj_dataset = TrajectoryDataset(
                            config.data_dir,
                            rollout_length=config.rollout_steps,
                            split="train",
                            val_fraction=config.val_fraction,
                        )
                        if len(traj_dataset) == 0:
                            print(f"  WARNING: trajectory dataset has 0 valid windows "
                                  f"(need episodes with >{config.rollout_steps} contiguous steps). "
                                  f"Disabling rollout loss.")
                            config.rollout_loss_weight = 0
                            continue
                        traj_loader = DataLoader(
                            traj_dataset, batch_size=min(256, config.batch_size),
                            shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
                        )
                        traj_iter = iter(traj_loader)
                        print(f"  Trajectory dataset: {len(traj_dataset):,} windows")

                    try:
                        traj_batch = next(traj_iter)
                    except StopIteration:
                        traj_iter = iter(traj_loader)
                        traj_batch = next(traj_iter)

                    r_loss = compute_rollout_loss(model, traj_batch, normalizer, device)
                    total_loss = total_loss + config.rollout_loss_weight * r_loss
                    epoch_rollout_loss += r_loss.item()

                # Scale loss for gradient accumulation
                (total_loss / grad_accum).backward()

                # Step optimizer every grad_accum batches (or at end of epoch)
                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    epoch_grad_norms.append(gn.item())
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1
                for k, v in comp_losses.items():
                    all_component_losses[k] = all_component_losses.get(k, 0.0) + v

            epoch_loss /= max(1, n_batches)
            epoch_rollout_loss /= max(1, n_batches)
            for k in all_component_losses:
                all_component_losses[k] /= max(1, n_batches)

            # Validation
            val_result = evaluate(model, val_loader, normalizer, device, amp_ctx=amp_ctx)
            val_loss = val_result["val_loss"]
            r2_score = val_result["r2"]
            r2_components = val_result["r2_components"]

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), str(save_dir / "model.pt"))
            else:
                patience_counter += 1

            elapsed = time.monotonic() - t_epoch
            lr = optimizer.param_groups[0]["lr"]

            # Build epoch metrics dict
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "r2": r2_score,
                "rollout_loss": epoch_rollout_loss,
                "lr": lr,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "epoch_time_s": elapsed,
            }
            epoch_metrics.update({f"r2/{k}": v for k, v in r2_components.items()})
            epoch_metrics.update({f"component/{k}": v for k, v in all_component_losses.items()})

            # Write metrics.jsonl
            metrics_log_file.write(json.dumps(epoch_metrics, default=str) + "\n")
            metrics_log_file.flush()

            # Console log
            if epoch % 5 == 0 or epoch == 1:
                rollout_str = (
                    f" | rollout={epoch_rollout_loss:.6f}"
                    if epoch >= config.rollout_start_epoch
                    else ""
                )
                print(
                    f"Epoch {epoch:3d} | "
                    f"train={epoch_loss:.6f} | val={val_loss:.6f} | R2={r2_score:.4f}{rollout_str} | "
                    f"best={best_val_loss:.6f} | lr={lr:.6f} | "
                    f"patience={patience_counter}/{config.patience} | "
                    f"{elapsed:.1f}s"
                )

            # GPU memory tracking
            gpu_memory_mb = (
                torch.cuda.max_memory_allocated(device) / 1e6
                if torch.cuda.is_available() and "cuda" in device
                else 0.0
            )

            # W&B log (metric names match project dashboard panels)
            mean_grad_norm = sum(epoch_grad_norms) / max(1, len(epoch_grad_norms))
            wandb_log = {
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "r2": r2_score,
                "train_val_gap": epoch_loss - val_loss,
                "rollout_loss": epoch_rollout_loss,
                "lr": lr,
                "grad_norm": mean_grad_norm,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "epoch_time_s": elapsed,
                "gpu_memory_mb": gpu_memory_mb,
            }
            wandb_log.update({f"r2/{k}": v for k, v in r2_components.items()})
            wandb_log.update({f"component/{k}": v for k, v in all_component_losses.items()})

            # System metrics (GPU/CPU/RAM), throttled to every 5 epochs
            sys_metrics = collect_system_metrics(device, _system_log_counter, 5)
            wandb_log.update(sys_metrics)

            if wandb_run is not None:
                wandb_log["epoch"] = epoch
                wandb_run.log(wandb_log, step=epoch)

            # Early stopping
            if patience_counter >= config.patience:
                stop_reason = "early_stopping"
                print(f"Early stopping at epoch {epoch} (patience={config.patience})")
                break

        # Restore signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        print(f"\nTraining finished: {stop_reason}")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Checkpoint: {save_dir}/model.pt")
        print(f"  Normalizer: {save_dir}/normalizer.pt")
        print(f"  Run directory: {run_dir}")

        # Save final metrics
        final_metrics = {
            "best_val_loss": best_val_loss,
            "final_epoch": epoch,
            "stop_reason": stop_reason,
        }
        (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))

        # Upload best model as W&B artifact
        best_path = save_dir / "model.pt"
        if best_path.exists():
            wandb_utils.log_model_artifact(
                wandb_run,
                best_path,
                artifact_name="surrogate_model",
                metadata=final_metrics,
            )

    # Close metrics log file
    metrics_log_file.close()

    # Close W&B run
    wandb_utils.end_run(wandb_run)


def _watchdog_target():
    """Entry point for the watchdog child process (must be module-level for pickling)."""
    from src.utils.gpu_lock import GpuLock
    with GpuLock():
        main()


def _run_with_watchdog(timeout_minutes: int = 5):
    """Run main() with a watchdog that kills the process if it hangs after completion.

    PyTorch/CUDA can deadlock during post-training cleanup (gc.collect / empty_cache).
    The watchdog detects when main() has returned but the process is still alive,
    and force-kills it after timeout_minutes.
    """
    import multiprocessing
    import sys

    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=_watchdog_target)
    proc.start()
    proc.join()

    if proc.exitcode is None:
        # Process didn't exit — hung during cleanup
        print(
            f"\nWatchdog: process still alive after main() returned. "
            f"Waiting {timeout_minutes}min before force-kill...",
            file=sys.stderr,
        )
        proc.join(timeout=timeout_minutes * 60)
        if proc.is_alive():
            print("Watchdog: force-killing hung process.", file=sys.stderr)
            proc.kill()
            proc.join()

    # Propagate exit code (treat 137/143 from watchdog kill as success)
    if proc.exitcode and proc.exitcode not in (0, -9, 137, 143):
        sys.exit(proc.exitcode)


if __name__ == "__main__":
    _run_with_watchdog()
