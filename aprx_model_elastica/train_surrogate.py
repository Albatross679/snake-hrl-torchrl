"""Train the surrogate MLP on collected PyElastica transition data.

Phase 1 (epochs 1-20):  Single-step MSE loss with noise injection.
Phase 2 (epochs 20+):   Add multi-step rollout loss (8-step BPTT).

Usage:
    python -m aprx_model_elastica.train_surrogate
    python -m aprx_model_elastica.train_surrogate --data-dir data/surrogate --epochs 200
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from aprx_model_elastica.train_config import SurrogateModelConfig, SurrogateTrainConfig
from aprx_model_elastica.dataset import FlatStepDataset
from aprx_model_elastica.model import SurrogateModel
from aprx_model_elastica.state import (
    StateNormalizer,
    action_to_omega_batch,
    encode_phase_batch,
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
)


def find_max_batch_size(
    model: nn.Module,
    normalizer,
    device: str,
    state_dim: int = 124,
    action_dim: int = 5,
    min_batch: int = 1024,
    max_batch: int = 131072,
    headroom: float = 0.9,
) -> int:
    """Binary search for the largest batch size that fits in GPU VRAM.

    Runs a dummy forward+backward pass at each candidate size. Returns the
    largest power-of-2-aligned batch size that completes without OOM, scaled
    down by `headroom` to leave room for optimizer state and gradients during
    real training.

    Args:
        model: The surrogate model (already on device).
        normalizer: StateNormalizer (or None).
        device: CUDA device string.
        state_dim: State vector dimension.
        action_dim: Action vector dimension.
        min_batch: Smallest batch size to try.
        max_batch: Largest batch size to try.
        headroom: Fraction of max working batch to use (0.9 = keep 90%).

    Returns:
        The recommended batch size (always >= min_batch).
    """
    if not torch.cuda.is_available() or "cpu" in device:
        print("[auto-batch] No CUDA device — using default batch size")
        return min_batch

    from aprx_model_elastica.state import action_to_omega_batch, encode_phase_batch

    model.train()

    def _try_batch(bs: int) -> bool:
        """Try a forward+backward pass with batch size bs. Returns True if OK."""
        try:
            torch.cuda.empty_cache()
            state = torch.randn(bs, state_dim, device=device)
            action = torch.randn(bs, action_dim, device=device)
            omega = action_to_omega_batch(action)
            time_enc = encode_phase_batch(omega * torch.rand(bs, device=device))

            state_norm = normalizer.normalize_state(state) if normalizer else state
            pred = model(state_norm, action, time_enc)
            target = torch.randn_like(pred)
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()

            # Clean up
            model.zero_grad(set_to_none=True)
            del state, action, time_enc, state_norm, pred, target, loss, omega
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False

    # Binary search between min_batch and max_batch
    lo, hi = min_batch, max_batch
    best = min_batch

    # Quick check: can we even run min_batch?
    if not _try_batch(min_batch):
        print(f"[auto-batch] WARNING: even min_batch={min_batch} causes OOM!")
        return min_batch

    print(f"[auto-batch] Searching max batch size in [{min_batch}, {max_batch}]...")
    while lo <= hi:
        mid = (lo + hi) // 2
        # Align to 256 for GPU efficiency
        mid = max(min_batch, (mid // 256) * 256)
        if mid <= best and lo < hi:
            lo = mid + 256
            continue
        if _try_batch(mid):
            best = mid
            lo = mid + 256
        else:
            hi = mid - 256

    # Apply headroom factor and align to 256
    recommended = max(min_batch, int(best * headroom))
    recommended = (recommended // 256) * 256
    recommended = max(min_batch, recommended)

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[auto-batch] Max working batch: {best}")
    print(f"[auto-batch] Recommended (headroom={headroom}): {recommended}")
    print(f"[auto-batch] GPU memory: {total_mem:.1f} GB")

    return recommended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate model")
    parser.add_argument("--data-dir", type=str, default="data/surrogate")
    parser.add_argument("--save-dir", type=str, default="output/surrogate")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--hidden-dims", type=str, default=None,
        help="Comma-separated hidden layer dims (e.g., '256,256,256')",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="W&B run name (used when --wandb is enabled)",
    )
    parser.add_argument(
        "--save-best-val-loss", action="store_true", default=False,
        help="Save best_val_loss and final_epoch to metrics.json after training",
    )
    # Phase 3.1 architectural experiment args
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
        "--auto-batch", action="store_true", default=False,
        help="Auto-find max batch size that fits in GPU VRAM",
    )
    parser.add_argument(
        "--arch", type=str, default="mlp", choices=["mlp", "residual", "transformer"],
        help="Model architecture: mlp, residual, or transformer",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None,
        help="Transformer encoder layers (only used with --arch transformer)",
    )
    parser.add_argument(
        "--n-heads", type=int, default=None,
        help="Transformer attention heads (only used with --arch transformer)",
    )
    parser.add_argument(
        "--d-model", type=int, default=None,
        help="Transformer embedding dimension (only used with --arch transformer)",
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


def compute_single_step_loss(
    model: SurrogateModel,
    batch: dict,
    normalizer: StateNormalizer,
    noise_std: float,
    device: str,
) -> tuple:
    """Compute single-step MSE loss.

    Returns (total_loss, per_component_losses_dict).
    """
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    serp_time = batch.get("serpenoid_time", batch.get("t_start")).to(device)
    delta = batch["delta"].to(device)

    # Add noise to input state for robustness
    if noise_std > 0 and model.training:
        state = state + noise_std * torch.randn_like(state)

    # Normalize (if enabled)
    state_norm = normalizer.normalize_state(state) if normalizer else state
    delta_norm = normalizer.normalize_delta(delta) if normalizer else delta

    # Encode oscillation phase omega*t (not raw t)
    omega = action_to_omega_batch(action)
    time_enc = encode_phase_batch(omega * serp_time)

    # Forward pass
    pred_delta_norm = model(state_norm, action, time_enc)

    # MSE loss
    loss = nn.functional.mse_loss(pred_delta_norm, delta_norm)

    # Per-component losses (for logging)
    with torch.no_grad():
        pred_delta = normalizer.denormalize_delta(pred_delta_norm) if normalizer else pred_delta_norm
        component_losses = {
            "pos_x": nn.functional.mse_loss(pred_delta[:, POS_X], delta[:, POS_X]).item(),
            "pos_y": nn.functional.mse_loss(pred_delta[:, POS_Y], delta[:, POS_Y]).item(),
            "vel_x": nn.functional.mse_loss(pred_delta[:, VEL_X], delta[:, VEL_X]).item(),
            "vel_y": nn.functional.mse_loss(pred_delta[:, VEL_Y], delta[:, VEL_Y]).item(),
            "yaw": nn.functional.mse_loss(pred_delta[:, YAW], delta[:, YAW]).item(),
            "omega_z": nn.functional.mse_loss(pred_delta[:, OMEGA_Z], delta[:, OMEGA_Z]).item(),
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
        time_enc = encode_phase_batch(omega * times_seq[:, t])

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
) -> float:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss, _ = compute_single_step_loss(
                model, batch, normalizer, noise_std=0.0, device=device
            )
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


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
    config.data_dir = args.data_dir
    config.save_dir = args.save_dir
    config.wandb_enabled = args.wandb
    if args.hidden_dims is not None:
        config.model.hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    # Phase 3.1 architectural experiment overrides
    if args.rollout_weight is not None:
        config.rollout_loss_weight = args.rollout_weight
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    config.use_residual = args.use_residual
    if args.history_k is not None:
        config.history_k = args.history_k
    # Architecture selection (--arch overrides --use-residual)
    config.model.arch = args.arch
    if args.arch == "residual":
        config.use_residual = True
    if args.n_layers is not None:
        config.model.n_layers = args.n_layers
    if args.n_heads is not None:
        config.model.n_heads = args.n_heads
    if args.d_model is not None:
        config.model.d_model = args.d_model

    # Output directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    wandb_run = None
    if config.wandb_enabled:
        import wandb
        wandb_kwargs = dict(
            project=config.wandb_project,
            entity=config.wandb_entity or None,
            config={
                "model": vars(config.model),
                "training": {k: v for k, v in vars(config).items() if k != "model"},
            },
        )
        if args.run_name:
            wandb_kwargs["name"] = args.run_name
        wandb_run = wandb.init(**wandb_kwargs)

    # Load data
    print(f"Loading data from {config.data_dir}...")
    train_dataset = FlatStepDataset(
        config.data_dir, split="train", val_fraction=config.val_fraction
    )
    val_dataset = FlatStepDataset(
        config.data_dir, split="val", val_fraction=config.val_fraction
    )

    # Guard: rollout loss not supported with FlatStepDataset
    if config.rollout_loss_weight > 0:
        print("WARNING: rollout loss not supported with FlatStepDataset (no trajectories). Setting rollout_loss_weight=0.0")
        config.rollout_loss_weight = 0.0
    print(f"  Train: {len(train_dataset):,} transitions")
    print(f"  Val:   {len(val_dataset):,} transitions")

    # Density-based sample weighting (upweight rare states)
    sampler = None
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

    # Trajectory dataset for multi-step loss (loaded lazily)
    traj_loader = None

    # Compute normalization statistics
    if config.normalize_inputs or config.normalize_targets:
        print("Computing normalization statistics...")
        normalizer = StateNormalizer(device=device)
        train_deltas = train_dataset.next_states - train_dataset.states
        normalizer.fit(train_dataset.states, train_deltas)
        if not config.normalize_inputs:
            normalizer.state_mean.zero_()
            normalizer.state_std.fill_(1.0)
        if not config.normalize_targets:
            normalizer.delta_mean.zero_()
            normalizer.delta_std.fill_(1.0)
        normalizer.save(str(save_dir / "normalizer.pt"))
    else:
        normalizer = None

    # Build model based on --arch
    if config.model.arch == "transformer":
        from aprx_model_elastica.model import TransformerSurrogateModel
        model = TransformerSurrogateModel(config.model).to(device)
    elif config.model.arch == "residual" or config.use_residual:
        from aprx_model_elastica.model import ResidualSurrogateModel
        model = ResidualSurrogateModel(config.model).to(device)
    elif config.history_k > 0:
        from aprx_model_elastica.model import HistorySurrogateModel
        model = HistorySurrogateModel(config.model, history_k=config.history_k).to(device)
    else:
        model = SurrogateModel(config.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model.arch} — {n_params:,} parameters")

    # Log param_count to W&B at init
    if wandb_run is not None:
        import wandb
        wandb.config.update({"param_count": n_params})

    # Auto batch size: find max batch that fits in GPU VRAM
    if args.auto_batch and args.batch_size is None:
        config.batch_size = find_max_batch_size(
            model, normalizer, device,
            state_dim=config.model.state_dim,
            action_dim=config.model.action_dim,
        )
        print(f"  Using auto-detected batch size: {config.batch_size}")
    elif args.auto_batch and args.batch_size is not None:
        print("[auto-batch] --batch-size explicitly set, skipping auto-detection")

    # Log batch_size to W&B
    if wandb_run is not None:
        import wandb
        wandb.config.update({"batch_size": config.batch_size})

    # Create DataLoaders (after batch size is finalized)
    if sampler is not None:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=sampler,
            num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

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

    print(f"\nTraining for {config.num_epochs} epochs on {device}...")
    print(f"  Single-step loss: epochs 1-{config.rollout_start_epoch}")
    print(f"  + Rollout loss:   epochs {config.rollout_start_epoch + 1}+")

    for epoch in range(1, config.num_epochs + 1):
        t_epoch = time.monotonic()
        model.train()

        # Track losses
        epoch_loss = 0.0
        epoch_rollout_loss = 0.0
        n_batches = 0
        all_component_losses = {}

        # Single-step training
        for batch in train_loader:
            loss, comp_losses = compute_single_step_loss(
                model, batch, normalizer, config.state_noise_std, device
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
                    traj_loader = DataLoader(
                        traj_dataset, batch_size=min(256, config.batch_size),
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True,
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

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            for k, v in comp_losses.items():
                all_component_losses[k] = all_component_losses.get(k, 0.0) + v

        epoch_loss /= max(1, n_batches)
        epoch_rollout_loss /= max(1, n_batches)
        for k in all_component_losses:
            all_component_losses[k] /= max(1, n_batches)

        # Validation
        val_loss = evaluate(model, val_loader, normalizer, device)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(save_dir / "model.pt"))
            # Save config
            with open(save_dir / "config.json", "w") as f:
                json.dump(vars(config.model), f, indent=2)
        else:
            patience_counter += 1

        elapsed = time.monotonic() - t_epoch
        lr = optimizer.param_groups[0]["lr"]

        # Log
        if epoch % 5 == 0 or epoch == 1:
            rollout_str = (
                f" | rollout={epoch_rollout_loss:.6f}"
                if epoch >= config.rollout_start_epoch
                else ""
            )
            print(
                f"Epoch {epoch:3d}/{config.num_epochs} | "
                f"train={epoch_loss:.6f} | val={val_loss:.6f}{rollout_str} | "
                f"best={best_val_loss:.6f} | lr={lr:.6f} | "
                f"patience={patience_counter}/{config.patience} | "
                f"{elapsed:.1f}s"
            )

        if wandb_run is not None:
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
            log_dict = {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "rollout_loss": epoch_rollout_loss,
                "lr": lr,
                "best_val_loss": best_val_loss,
                "grad_norm": grad_norm,
                "epoch_time": elapsed,
                "train_val_gap": epoch_loss - val_loss,
                "gpu_memory_mb": gpu_mem_mb,
            }
            log_dict.update({f"component/{k}": v for k, v in all_component_losses.items()})
            wandb_run.log(log_dict)

        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch} (patience={config.patience})")
            break

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {save_dir}/model.pt")
    print(f"Normalizer saved to {save_dir}/normalizer.pt")

    # Save metrics.json for sweep runner
    if args.save_best_val_loss:
        metrics_path = save_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"best_val_loss": best_val_loss, "final_epoch": epoch}, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
