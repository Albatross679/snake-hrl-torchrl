"""Train surrogate model with physics regularizer and optional lambda sweep.

Extends the base surrogate training loop with:
- PhysicsRegularizer for soft physics constraint losses
- ReLoBRaLo adaptive loss balancing
- Curriculum warmup (data-only for first N% of epochs)
- Lambda sweep mode for comparing regularization strengths
- Per-component RMSE evaluation in physical units

Usage:
    python -m src.pinn.train_regularized
    python -m src.pinn.train_regularized --lambda-phys 0.1
    python -m src.pinn.train_regularized --sweep
    python -m src.pinn.train_regularized --no-wandb --epochs 3 --lambda-phys 0.01
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.configs.base import WandB, resolve_device, save_config
from src.pinn.regularizer import PhysicsRegularizer
from src.pinn.loss_balancing import ReLoBRaLo
from src.pinn._state_slices import (
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
    RAW_STATE_DIM, NUM_NODES, NUM_ELEMENTS,
)
from src.utils.cleanup import cleanup_vram
from src.wandb_utils import (
    setup_run, log_metrics, log_extra_params,
    log_model_artifact, end_run, _count_parameters, collect_hardware_info,
)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False


def _sigterm_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\nSIGTERM received — finishing current epoch and saving checkpoint.")


def _check_stop(run_dir: Path) -> bool:
    """Return True if a STOP file exists in the run dir or cwd, or SIGTERM was received."""
    if _STOP_REQUESTED:
        return True
    for p in [run_dir / "STOP", Path("STOP")]:
        if p.exists():
            return True
    return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RegularizerTrainConfig:
    """Physics-regularized surrogate training configuration."""

    # Identity
    name: str = "pinn_regularized"
    seed: int = 42
    device: str = "auto"

    # Data
    data_dir: str = "data/surrogate_rl_step"
    val_fraction: float = 0.1

    # Model
    hidden_dims: str = "512,512,512"
    use_residual: bool = False

    # Physics regularizer
    lambda_phys: float = 0.01
    use_relobralo: bool = True
    curriculum_warmup: float = 0.2

    # Training
    num_epochs: int = 9999
    lr: float = 1e-3
    batch_size: int = 4096
    patience: int = 30
    grad_clip: float = 1.0

    # Sweep
    sweep: bool = False

    # Output
    save_dir: str = "output/surrogate/pinn_regularized"

    # W&B
    wandb: WandB = field(default_factory=lambda: WandB(
        enabled=True, project="snake-hrl-pinn",
    ))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_cli(cls) -> "RegularizerTrainConfig":
        """Parse CLI args into config, overriding defaults."""
        import argparse
        p = argparse.ArgumentParser(description="Train physics-regularized surrogate")
        p.add_argument("--data-dir", type=str)
        p.add_argument("--lambda-phys", type=float)
        p.add_argument("--epochs", type=int, dest="num_epochs")
        p.add_argument("--lr", type=float)
        p.add_argument("--hidden-dims", type=str)
        p.add_argument("--use-residual", action="store_true", default=None)
        p.add_argument("--no-wandb", action="store_true", default=False)
        p.add_argument("--run-name", type=str, default=None)
        p.add_argument("--save-dir", type=str)
        p.add_argument("--device", type=str)
        p.add_argument("--no-relobralo", action="store_true", default=False)
        p.add_argument("--curriculum-warmup", type=float)
        p.add_argument("--patience", type=int)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--sweep", action="store_true", default=False)
        p.add_argument("--val-fraction", type=float)
        p.add_argument("--seed", type=int)
        args = p.parse_args()

        cfg = cls()
        # Override non-None CLI args
        if args.data_dir is not None:
            cfg.data_dir = args.data_dir
        if args.lambda_phys is not None:
            cfg.lambda_phys = args.lambda_phys
        if args.num_epochs is not None:
            cfg.num_epochs = args.num_epochs
        if args.lr is not None:
            cfg.lr = args.lr
        if args.hidden_dims is not None:
            cfg.hidden_dims = args.hidden_dims
        if args.use_residual is not None:
            cfg.use_residual = args.use_residual
        if args.save_dir is not None:
            cfg.save_dir = args.save_dir
        if args.device is not None:
            cfg.device = args.device
        if args.no_relobralo:
            cfg.use_relobralo = False
        if args.curriculum_warmup is not None:
            cfg.curriculum_warmup = args.curriculum_warmup
        if args.patience is not None:
            cfg.patience = args.patience
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.val_fraction is not None:
            cfg.val_fraction = args.val_fraction
        if args.seed is not None:
            cfg.seed = args.seed
        if args.sweep:
            cfg.sweep = True
        if args.no_wandb:
            cfg.wandb.enabled = False
        if args.run_name is not None:
            cfg.name = args.run_name

        return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_context(device: str):
    """Return bf16 autocast context or nullcontext for CPU."""
    if device.startswith("cuda") and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:  # Ampere+
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_curriculum_weight(
    epoch: int,
    total_epochs: int,
    max_weight: float,
    warmup_frac: float = 0.2,
) -> float:
    """Compute curriculum-ramped physics loss weight.

    Returns 0 during warmup, linearly ramps to max_weight over 30% of
    remaining epochs, then holds at max_weight.
    """
    warmup_end = int(warmup_frac * total_epochs)
    ramp_end = warmup_end + int(0.3 * total_epochs)

    if epoch < warmup_end:
        return 0.0
    elif epoch < ramp_end:
        progress = (epoch - warmup_end) / max(1, ramp_end - warmup_end)
        return max_weight * progress
    else:
        return max_weight


def per_component_rmse(
    pred_deltas: torch.Tensor,
    true_deltas: torch.Tensor,
) -> dict:
    """Compute per-component RMSE in physical units."""
    def _rmse(pred, true):
        return torch.sqrt(F.mse_loss(pred, true)).item()

    return {
        "pos_x_mm": _rmse(pred_deltas[:, POS_X], true_deltas[:, POS_X]) * 1000,
        "pos_y_mm": _rmse(pred_deltas[:, POS_Y], true_deltas[:, POS_Y]) * 1000,
        "vel_x_mm_s": _rmse(pred_deltas[:, VEL_X], true_deltas[:, VEL_X]) * 1000,
        "vel_y_mm_s": _rmse(pred_deltas[:, VEL_Y], true_deltas[:, VEL_Y]) * 1000,
        "yaw_rad": _rmse(pred_deltas[:, YAW], true_deltas[:, YAW]),
        "omega_z_rad_s": _rmse(pred_deltas[:, OMEGA_Z], true_deltas[:, OMEGA_Z]),
    }


def _ensure_papers_path():
    """Add papers/ to sys.path and pre-register packages to avoid __init__ chains."""
    import sys
    import types

    if "papers" not in sys.path:
        sys.path.insert(0, "papers")

    for pkg in ["aprx_model_elastica", "locomotion_elastica"]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
            sys.modules[pkg].__path__ = [f"papers/{pkg}"]
            sys.modules[pkg].__package__ = pkg


def _load_data_and_normalizer(data_dir: str, val_fraction: float, device: str):
    """Load FlatStepDataset + StateNormalizer, keep both raw and relative states."""
    _ensure_papers_path()

    from aprx_model_elastica.dataset import FlatStepDataset
    from aprx_model_elastica.state import (
        StateNormalizer, raw_to_relative, relative_to_raw,
        REL_STATE_DIM,
    )

    train_ds = FlatStepDataset(data_dir, split="train", val_fraction=val_fraction)
    val_ds = FlatStepDataset(data_dir, split="val", val_fraction=val_fraction)

    # Keep raw 124-dim states for physics regularizer
    raw_train_states = train_ds.states.clone()
    raw_train_next = train_ds.next_states.clone()
    raw_val_states = val_ds.states.clone()
    raw_val_next = val_ds.next_states.clone()

    # Convert to 130-dim relative for model training
    if train_ds.states.shape[-1] == 124:
        for ds in (train_ds, val_ds):
            ds.states = raw_to_relative(ds.states)
            ds.next_states = raw_to_relative(ds.next_states)

    # Compute normalizer on relative states
    normalizer = StateNormalizer(state_dim=REL_STATE_DIM, device=device)
    deltas = train_ds.next_states - train_ds.states
    normalizer.fit(train_ds.states, deltas)

    return (train_ds, val_ds, normalizer, raw_train_states, raw_train_next,
            raw_val_states, raw_val_next)


def _build_model(hidden_dims_str: str, use_residual: bool, device: str):
    """Build surrogate model."""
    _ensure_papers_path()

    from aprx_model_elastica.train_config import SurrogateModelConfig
    from aprx_model_elastica.model import SurrogateModel

    hidden_dims = [int(x) for x in hidden_dims_str.split(",")]
    config = SurrogateModelConfig(hidden_dims=hidden_dims)

    if use_residual:
        from aprx_model_elastica.model import ResidualSurrogateModel
        model = ResidualSurrogateModel(config).to(device)
    else:
        model = SurrogateModel(config).to(device)

    return model


def _encode_time(action, serp_time):
    """Build time encoding from action and serpenoid time."""
    _ensure_papers_path()

    from aprx_model_elastica.state import (
        action_to_omega_batch, encode_phase_batch, encode_n_cycles_batch,
    )

    omega = action_to_omega_batch(action)
    phase_enc = encode_phase_batch(omega * serp_time)
    ncycles_enc = encode_n_cycles_batch(action)
    return torch.cat([phase_enc, ncycles_enc], dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_single_lambda(
    cfg: RegularizerTrainConfig,
    lambda_phys: float,
    run_dir: Path,
) -> dict:
    """Train a single model with given lambda_phys. Returns eval metrics."""
    device = resolve_device(cfg.device)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    run_cfg = RegularizerTrainConfig(**{**asdict(cfg), "lambda_phys": lambda_phys})
    save_config(run_cfg, run_dir / "config.json")

    # Metrics file (incremental)
    metrics_file = open(run_dir / "metrics.jsonl", "w")

    print(f"\n{'='*60}")
    print(f"Training with lambda_phys={lambda_phys}")
    print(f"Run dir: {run_dir}")
    print(f"{'='*60}")

    # Load data
    (train_ds, val_ds, normalizer, raw_train_states, raw_train_next,
     raw_val_states, raw_val_next) = _load_data_and_normalizer(
        cfg.data_dir, cfg.val_fraction, device
    )
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # Save normalizer
    normalizer.save(str(run_dir / "normalizer.pt"))

    # Build model
    model = _build_model(cfg.hidden_dims, cfg.use_residual, device)
    param_info = _count_parameters(model)
    print(f"Model: {param_info['total_params']:,} parameters")

    # Physics regularizer
    regularizer = PhysicsRegularizer(dt=0.5).to(device)

    # Loss balancer
    balancer = ReLoBRaLo(n_losses=2) if cfg.use_relobralo else None

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=1e-4
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    total_steps = cfg.num_epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # W&B via wandb_utils — use a shallow copy with overrides
    import copy
    wandb_cfg = copy.copy(cfg)
    wandb_cfg.lambda_phys = lambda_phys
    wandb_cfg.name = cfg.name or f"reg_lambda_{lambda_phys}"
    wandb_run = setup_run(wandb_cfg, run_dir)

    # Log one-time params
    hw_info = collect_hardware_info(device)
    log_extra_params(wandb_run, {
        **param_info,
        "num_train_samples": len(train_ds),
        "num_dev_samples": len(val_ds),
        **hw_info,
    })

    # Import relative_to_raw for physics loss computation
    _ensure_papers_path()
    from aprx_model_elastica.state import relative_to_raw

    # AMP context
    amp_ctx = get_amp_context(device)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.num_epochs + 1):
        # Check graceful stop
        if _check_stop(run_dir):
            print(f"  Stop requested at epoch {epoch} — saving and exiting.")
            break

        epoch_start = time.time()
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        n_batches = 0

        phys_weight = get_curriculum_weight(
            epoch, cfg.num_epochs, lambda_phys, cfg.curriculum_warmup
        )

        for batch in train_loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            serp_time = batch["t_start"].to(device)
            delta = batch["delta"].to(device)

            state_norm = normalizer.normalize_state(state)
            delta_norm = normalizer.normalize_delta(delta)
            time_enc = _encode_time(action, serp_time)

            # Forward + loss inside AMP context
            with amp_ctx:
                pred_delta_norm = model(state_norm, action, time_enc)
                loss_data = F.mse_loss(pred_delta_norm, delta_norm)

                # Physics loss (on raw 124-dim states)
                if phys_weight > 0:
                    pred_delta_rel_grad = normalizer.denormalize_delta(pred_delta_norm)
                    pred_next_rel_grad = state + pred_delta_rel_grad
                    pred_next_raw_grad = relative_to_raw(pred_next_rel_grad)
                    raw_state_batch_nograd = relative_to_raw(state.detach())
                    delta_pred_raw_grad = pred_next_raw_grad - raw_state_batch_nograd

                    loss_phys = regularizer(raw_state_batch_nograd, delta_pred_raw_grad)
                else:
                    loss_phys = torch.tensor(0.0, device=device)

                # Combine losses
                if balancer is not None and phys_weight > 0:
                    weights = balancer.update([loss_data, loss_phys])
                    loss = weights[0] * loss_data + weights[1] * phys_weight * loss_phys
                else:
                    loss = loss_data + phys_weight * loss_phys

            # Backward OUTSIDE AMP context
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_data_loss += loss_data.item()
            epoch_phys_loss += loss_phys.item()
            n_batches += 1

        epoch_data_loss /= max(1, n_batches)
        epoch_phys_loss /= max(1, n_batches)
        epoch_time = time.time() - epoch_start

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.inference_mode():
            for batch in val_loader:
                state = batch["state"].to(device)
                action = batch["action"].to(device)
                serp_time = batch["t_start"].to(device)
                delta = batch["delta"].to(device)

                state_norm = normalizer.normalize_state(state)
                delta_norm = normalizer.normalize_delta(delta)
                time_enc = _encode_time(action, serp_time)

                with amp_ctx:
                    pred = model(state_norm, action, time_enc)
                    val_loss_sum += F.mse_loss(pred, delta_norm).item()
                val_n += 1

        val_loss = val_loss_sum / max(1, val_n)

        # Checkpoint: save best AND last
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(run_dir / "model_best.pt"))
        else:
            patience_counter += 1
        torch.save(model.state_dict(), str(run_dir / "model_last.pt"))

        lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "epoch": epoch,
            "loss_data": epoch_data_loss,
            "loss_phys": epoch_phys_loss,
            "phys_weight": phys_weight,
            "val_loss": val_loss,
            "tracking/best_val_loss": best_val_loss,
            "lr": lr,
            "tracking/patience": patience_counter,
            "timing/epoch_s": epoch_time,
        }

        # Write incrementally to metrics.jsonl
        metrics_file.write(json.dumps(epoch_metrics) + "\n")
        metrics_file.flush()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | data={epoch_data_loss:.6f} | "
                f"phys={epoch_phys_loss:.6f} (w={phys_weight:.4f}) | "
                f"val={val_loss:.6f} | best={best_val_loss:.6f} | "
                f"lr={lr:.6f} | patience={patience_counter}/{cfg.patience}"
            )

        # W&B logging
        log_metrics(wandb_run, epoch_metrics, step=epoch)

        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    metrics_file.close()

    # Final evaluation with per-component RMSE
    print("Computing per-component RMSE on validation set...")
    best_path = run_dir / "model_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(str(best_path), weights_only=True))
    model.eval()

    all_pred_raw = []
    all_true_raw = []

    with torch.inference_mode():
        for batch in val_loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            serp_time = batch["t_start"].to(device)
            delta = batch["delta"].to(device)

            state_norm = normalizer.normalize_state(state)
            time_enc = _encode_time(action, serp_time)

            with amp_ctx:
                pred_delta_norm = model(state_norm, action, time_enc)

            pred_delta_rel = normalizer.denormalize_delta(pred_delta_norm)
            true_delta_rel = delta

            pred_next_rel = state + pred_delta_rel
            true_next_rel = state + true_delta_rel

            pred_next_raw = relative_to_raw(pred_next_rel)
            true_next_raw = relative_to_raw(true_next_rel)

            raw_state_batch = relative_to_raw(state)
            all_pred_raw.append(pred_next_raw - raw_state_batch)
            all_true_raw.append(true_next_raw - raw_state_batch)

    all_pred_raw = torch.cat(all_pred_raw, dim=0)
    all_true_raw = torch.cat(all_true_raw, dim=0)

    rmse = per_component_rmse(all_pred_raw, all_true_raw)

    # R2 scores
    ss_res = ((all_pred_raw - all_true_raw) ** 2).sum(dim=0)
    ss_tot = ((all_true_raw - all_true_raw.mean(dim=0)) ** 2).sum(dim=0)
    r2_per_dim = 1.0 - ss_res / ss_tot.clamp(min=1e-8)

    r2_components = {
        "pos_x": r2_per_dim[POS_X].mean().item(),
        "pos_y": r2_per_dim[POS_Y].mean().item(),
        "vel_x": r2_per_dim[VEL_X].mean().item(),
        "vel_y": r2_per_dim[VEL_Y].mean().item(),
        "yaw": r2_per_dim[YAW].mean().item(),
        "omega_z": r2_per_dim[OMEGA_Z].mean().item(),
    }

    eval_result = {
        "lambda_phys": lambda_phys,
        "best_val_loss": best_val_loss,
        "rmse": rmse,
        "r2_components": r2_components,
        "r2_overall": r2_per_dim.mean().item(),
    }

    (run_dir / "eval_metrics.json").write_text(json.dumps(eval_result, indent=2))

    print(f"\nResults for lambda={lambda_phys}:")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  R2 overall: {eval_result['r2_overall']:.4f}")
    print(f"  Per-component RMSE:")
    for k, v in rmse.items():
        print(f"    {k}: {v:.4f}")
    print(f"  Per-component R2:")
    for k, v in r2_components.items():
        print(f"    {k}: {v:.4f}")

    # Log final eval to W&B
    if wandb_run is not None:
        eval_wandb = {"eval/" + k: v for k, v in rmse.items()}
        eval_wandb.update({"eval_r2/" + k: v for k, v in r2_components.items()})
        log_metrics(wandb_run, eval_wandb, step=epoch)

    # Upload best model artifact
    log_model_artifact(wandb_run, best_path, f"pinn-reg-lambda-{lambda_phys}")

    end_run(wandb_run)

    return eval_result


def generate_comparison_plots(results: list[dict], save_dir: Path):
    """Generate comparison bar charts for lambda sweep results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    figures_dir = Path("figures/pinn")
    figures_dir.mkdir(parents=True, exist_ok=True)

    lambdas = [r["lambda_phys"] for r in results]
    labels = [f"\u03bb={l}" for l in lambdas]

    # Sweep plot: overall R2 and val_loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    r2_vals = [r["r2_overall"] for r in results]
    axes[0].bar(labels, r2_vals, color="steelblue")
    axes[0].set_ylabel("R\u00b2 (overall)")
    axes[0].set_title("Overall R\u00b2 vs Lambda")
    axes[0].set_ylim(0, 1)

    val_losses = [r["best_val_loss"] for r in results]
    axes[1].bar(labels, val_losses, color="coral")
    axes[1].set_ylabel("Best Val Loss")
    axes[1].set_title("Validation Loss vs Lambda")

    plt.tight_layout()
    plt.savefig(figures_dir / "regularizer_sweep.png", dpi=150)
    plt.close()
    print(f"Saved {figures_dir / 'regularizer_sweep.png'}")

    # Per-component RMSE comparison
    components = list(results[0]["rmse"].keys())
    n_components = len(components)
    x = range(n_components)
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, r in enumerate(results):
        vals = [r["rmse"][c] for c in components]
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width, label=f"\u03bb={r['lambda_phys']}")

    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_ylabel("RMSE (physical units)")
    ax.set_title("Per-Component RMSE: Physics Regularizer Lambda Sweep")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "regularizer_per_component.png", dpi=150)
    plt.close()
    print(f"Saved {figures_dir / 'regularizer_per_component.png'}")


def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)

    cfg = RegularizerTrainConfig.from_cli()
    set_seed(cfg.seed)

    if cfg.sweep:
        lambdas = [0.001, 0.01, 0.1, 1.0]
        results = []
        for lam in lambdas:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(cfg.save_dir) / f"lambda_{lam}_{ts}"
            result = train_single_lambda(cfg, lam, run_dir)
            results.append(result)
            # VRAM cleanup between sequential runs
            cleanup_vram()

        generate_comparison_plots(results, Path(cfg.save_dir))

        combined_path = Path(cfg.save_dir) / "sweep_results.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(results, indent=2))
        print(f"\nSweep complete! Results saved to {combined_path}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.save_dir) / f"{cfg.name}_{ts}"
        train_single_lambda(cfg, cfg.lambda_phys, run_dir)


if __name__ == "__main__":
    from src.utils.gpu_lock import GpuLock
    with GpuLock():
        main()
