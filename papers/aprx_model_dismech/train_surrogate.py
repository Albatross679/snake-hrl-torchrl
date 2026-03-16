"""Train the surrogate MLP on collected DisMech transition data.

Phase 1 (epochs 1-20):  Single-step MSE loss with noise injection.
Phase 2 (epochs 20+):   Add multi-step rollout loss (8-step BPTT).

Usage:
    python -m aprx_model_dismech train
    python -m aprx_model_dismech train --data-dir data/surrogate_dismech_rl_step_rel128
    python -m aprx_model_dismech train --no-wandb --epochs 50
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

from aprx_model_dismech.train_config import SurrogateModelConfig, SurrogateTrainConfig
from aprx_model_dismech.dataset import FlatStepDataset
from aprx_model_dismech.model import SurrogateModel
from aprx_model_dismech.state import (
    StateNormalizer,
    action_to_omega_batch,
    encode_phase_batch,
    encode_n_cycles_batch,
    raw_to_relative,
    REL_COM_X, REL_COM_Y, REL_HEADING_SIN, REL_HEADING_COS,
    REL_POS_X, REL_POS_Y, REL_VEL_X, REL_VEL_Y, REL_YAW, REL_OMEGA_Z,
    REL_STATE_DIM,
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
)
from src.configs.run_dir import setup_run_dir
from src.configs.console import ConsoleLogger
from src.trainers.logging_utils import collect_system_metrics
from src import wandb_utils


# STOP file for graceful stop (touch STOP to stop training)
STOP_FILE = "STOP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate model (DisMech)")
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
    return parser.parse_args()


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SurrogateTrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build LR scheduler with warmup."""
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
    """Compute single-step MSE loss on 128-dim relative state representation."""
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    serp_time = batch["t_start"].to(device)
    delta = batch["delta"].to(device)

    if noise_std > 0 and model.training:
        state = state + noise_std * torch.randn_like(state)

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

    with torch.no_grad():
        pred_delta = normalizer.denormalize_delta(pred_delta_norm) if normalizer else pred_delta_norm
        component_losses = {
            "com": nn.functional.mse_loss(pred_delta[:, 0:2], delta[:, 0:2]).item(),
            "heading": nn.functional.mse_loss(pred_delta[:, 2:4], delta[:, 2:4]).item(),
            "rel_pos_x": nn.functional.mse_loss(pred_delta[:, REL_POS_X], delta[:, REL_POS_X]).item(),
            "rel_pos_y": nn.functional.mse_loss(pred_delta[:, REL_POS_Y], delta[:, REL_POS_Y]).item(),
            "vel_x": nn.functional.mse_loss(pred_delta[:, REL_VEL_X], delta[:, REL_VEL_X]).item(),
            "vel_y": nn.functional.mse_loss(pred_delta[:, REL_VEL_Y], delta[:, REL_VEL_Y]).item(),
            "yaw": nn.functional.mse_loss(pred_delta[:, REL_YAW], delta[:, REL_YAW]).item(),
            "omega_z": nn.functional.mse_loss(pred_delta[:, REL_OMEGA_Z], delta[:, REL_OMEGA_Z]).item(),
        }

    return loss, component_losses


def evaluate(
    model: SurrogateModel,
    val_loader: DataLoader,
    normalizer: StateNormalizer,
    device: str,
    amp_ctx=None,
) -> float:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss, _ = compute_single_step_loss(
                model, batch, normalizer, noise_std=0.0, device=device,
                amp_ctx=amp_ctx,
            )
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

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
    if args.rollout_weight is not None:
        config.rollout_loss_weight = args.rollout_weight
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    config.use_residual = args.use_residual
    if args.history_k is not None:
        config.history_k = args.history_k

    run_dir = setup_run_dir(config)
    save_dir = run_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = run_dir / "config.json"
    config_snapshot.write_text(json.dumps(asdict(config), indent=2, default=str))

    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        print(f"\n{sig_name} received, will stop after current epoch...")
        shutdown_requested = True

    original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

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

    metrics_log_path = run_dir / "metrics.jsonl"
    metrics_log_file = open(metrics_log_path, "a", encoding="utf-8")

    with ConsoleLogger(run_dir):
        print(f"Loading data from {config.data_dir}...")
        train_dataset = FlatStepDataset(
            config.data_dir, split="train", val_fraction=config.val_fraction
        )
        val_dataset = FlatStepDataset(
            config.data_dir, split="val", val_fraction=config.val_fraction
        )
        print(f"  Train: {len(train_dataset):,} transitions")
        print(f"  Val:   {len(val_dataset):,} transitions")

        if train_dataset.states.shape[-1] == 124:
            print("  Converting raw 124-dim states to 128-dim relative representation...")
            for ds in (train_dataset, val_dataset):
                ds.states = raw_to_relative(ds.states)
                ds.next_states = raw_to_relative(ds.next_states)
            print(f"  State dim after conversion: {train_dataset.states.shape[-1]}")

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

        if config.use_residual:
            from aprx_model_dismech.model import ResidualSurrogateModel
            model = ResidualSurrogateModel(config.model).to(device)
        else:
            model = SurrogateModel(config.model).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {n_params:,} parameters ({trainable_params:,} trainable)")

        if config.use_density_weighting:
            print("Computing density-based sample weights...")
            sample_weights = train_dataset.get_sample_weights(
                n_bins=config.density_bins, clip_max=config.density_clip_max,
            )
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(train_dataset), replacement=True
            )
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

        extra_params = {
            "total_params": n_params,
            "trainable_params": trainable_params,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
            "batch_size": config.batch_size,
        }
        extra_params.update(wandb_utils.collect_hardware_info(device))
        wandb_utils.log_extra_params(wandb_run, extra_params)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        scheduler = build_lr_scheduler(optimizer, config, len(train_loader))

        best_val_loss = float("inf")
        patience_counter = 0
        stop_reason = "completed"
        amp_ctx = _amp_context(config.use_amp, device)
        grad_accum = config.gradient_accumulation_steps
        effective_batch = config.batch_size * grad_accum
        _system_log_counter = [0]

        print(f"\nTraining on {device}...")
        print(f"  Batch size: {config.batch_size} x {grad_accum} accum = {effective_batch} effective")
        print(f"  Early stopping patience: {config.patience}")
        print(f"  Run directory: {run_dir}")

        epoch = 0
        for epoch in range(1, config.num_epochs + 1):
            if shutdown_requested:
                stop_reason = "signal"
                break
            if Path(STOP_FILE).exists():
                stop_reason = "stop_file"
                break

            t_epoch = time.monotonic()
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            all_component_losses = {}
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                loss, comp_losses = compute_single_step_loss(
                    model, batch, normalizer, config.state_noise_std, device, amp_ctx=amp_ctx,
                )
                (loss / grad_accum).backward()

                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1
                for k, v in comp_losses.items():
                    all_component_losses[k] = all_component_losses.get(k, 0.0) + v

            epoch_loss /= max(1, n_batches)
            for k in all_component_losses:
                all_component_losses[k] /= max(1, n_batches)

            val_loss = evaluate(model, val_loader, normalizer, device, amp_ctx=amp_ctx)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), str(save_dir / "model.pt"))
            else:
                patience_counter += 1

            elapsed = time.monotonic() - t_epoch
            lr = optimizer.param_groups[0]["lr"]

            epoch_metrics = {
                "epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss,
                "lr": lr, "best_val_loss": best_val_loss,
                "patience_counter": patience_counter, "epoch_time_s": elapsed,
            }
            epoch_metrics.update({f"component/{k}": v for k, v in all_component_losses.items()})
            metrics_log_file.write(json.dumps(epoch_metrics, default=str) + "\n")
            metrics_log_file.flush()

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d} | train={epoch_loss:.6f} | val={val_loss:.6f} | "
                    f"best={best_val_loss:.6f} | lr={lr:.6f} | "
                    f"patience={patience_counter}/{config.patience} | {elapsed:.1f}s"
                )

            wandb_log = {
                "train/loss": epoch_loss, "train/val_loss": val_loss,
                "train/lr": lr, "tracking/best_val_loss": best_val_loss,
                "tracking/patience_counter": patience_counter,
                "timing/epoch_time_s": elapsed,
            }
            wandb_log.update({f"component/{k}": v for k, v in all_component_losses.items()})
            sys_metrics = collect_system_metrics(device, _system_log_counter, 5)
            wandb_log.update(sys_metrics)

            if wandb_run is not None:
                wandb_log["epoch"] = epoch
                wandb_run.log(wandb_log, step=epoch)

            if patience_counter >= config.patience:
                stop_reason = "early_stopping"
                print(f"Early stopping at epoch {epoch}")
                break

        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        print(f"\nTraining finished: {stop_reason}")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Checkpoint: {save_dir}/model.pt")

        final_metrics = {"best_val_loss": best_val_loss, "final_epoch": epoch, "stop_reason": stop_reason}
        (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))

        best_path = save_dir / "model.pt"
        if best_path.exists():
            wandb_utils.log_model_artifact(
                wandb_run, best_path, artifact_name="surrogate_model_dismech", metadata=final_metrics,
            )

    metrics_log_file.close()
    wandb_utils.end_run(wandb_run)


if __name__ == "__main__":
    main()
