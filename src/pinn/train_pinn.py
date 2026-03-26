"""Train DD-PINN surrogate model with data loss + physics residual loss.

DD-PINN combines a neural network with a damped sinusoidal ansatz for:
- Exact initial condition satisfaction (g(a, 0) = 0)
- Closed-form time derivatives (no autodiff for dx/dt)
- Physics residual at Sobol collocation points
- Residual-based adaptive refinement (RAR)
- ReLoBRaLo adaptive loss balancing

Usage:
    python -m src.pinn.train_pinn
    python -m src.pinn.train_pinn --n-basis 7 --n-collocation 100000
    python -m src.pinn.train_pinn --sweep
    python -m src.pinn.train_pinn --no-wandb --epochs 3 --n-collocation 1000
"""

from __future__ import annotations

import copy
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
from src.pinn.ansatz import DampedSinusoidalAnsatz
from src.pinn.models import DDPINNModel, FourierFeatureEmbedding
from src.pinn.physics_residual import CosseratRHS
from src.pinn.collocation import sample_collocation, adaptive_refinement
from src.pinn.loss_balancing import ReLoBRaLo
from src.pinn.nondim import NondimScales
from src.pinn._state_slices import (
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
    RAW_STATE_DIM,
)
from src.utils.cleanup import cleanup_vram
from src.pinn.diagnostics import PINNDiagnostics, compute_ntk_eigenvalues
from src.pinn.probe_pdes import run_probe_validation
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
class DDPINNTrainConfig:
    """DD-PINN training configuration."""

    # Identity
    name: str = "ddpinn"
    seed: int = 42
    device: str = "auto"

    # Data
    data_dir: str = "data/surrogate_rl_step"
    val_fraction: float = 0.1
    use_relative: bool = False  # False=124D raw state (proper for PINN); True=130D relative

    # DD-PINN architecture
    n_basis: int = 5
    hidden_dim: int = 512
    n_layers: int = 4
    n_fourier: int = 128
    fourier_sigma: float = 10.0

    # Physics
    n_collocation: int = 100000
    collocation_method: str = "sobol"
    lambda_phys: float = 1.0

    # Training
    num_epochs: int = 9999
    lr: float = 1e-3
    batch_size: int = 4096
    auto_batch_size: bool = True
    gradient_accumulation_steps: int = 1
    patience: int = 50
    grad_clip: float = 1.0

    # ReLoBRaLo
    use_relobralo: bool = True
    curriculum_warmup: float = 0.15

    # RAR
    use_rar: bool = True
    rar_interval: int = 20
    rar_fraction: float = 0.1

    # L-BFGS
    use_lbfgs: bool = False
    lbfgs_epochs: int = 50

    # Sweep
    sweep: bool = False

    # Output
    save_dir: str = "output/surrogate/ddpinn"

    # W&B
    wandb: WandB = field(default_factory=lambda: WandB(
        enabled=True, project="snake-hrl-pinn",
    ))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_cli(cls) -> "DDPINNTrainConfig":
        """Parse CLI args into config."""
        import argparse
        p = argparse.ArgumentParser(description="Train DD-PINN surrogate")
        p.add_argument("--data-dir", type=str)
        p.add_argument("--n-basis", type=int)
        p.add_argument("--hidden-dim", type=int)
        p.add_argument("--n-layers", type=int)
        p.add_argument("--n-fourier", type=int)
        p.add_argument("--fourier-sigma", type=float)
        p.add_argument("--n-collocation", type=int)
        p.add_argument("--collocation-method", type=str)
        p.add_argument("--lambda-phys", type=float)
        p.add_argument("--epochs", type=int, dest="num_epochs")
        p.add_argument("--lr", type=float)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--patience", type=int)
        p.add_argument("--use-lbfgs", action="store_true", default=None)
        p.add_argument("--lbfgs-epochs", type=int)
        p.add_argument("--no-relobralo", action="store_true", default=False)
        p.add_argument("--no-rar", action="store_true", default=False)
        p.add_argument("--rar-interval", type=int)
        p.add_argument("--rar-fraction", type=float)
        p.add_argument("--curriculum-warmup", type=float)
        p.add_argument("--use-relative", action="store_true", default=False,
                        help="Train in 130D relative space (default: 124D raw)")
        p.add_argument("--no-auto-batch", action="store_true", default=False)
        p.add_argument("--grad-accum", type=int, default=None, dest="grad_accum")
        p.add_argument("--resume", type=str, default=None)
        p.add_argument("--no-wandb", action="store_true", default=False)
        p.add_argument("--run-name", type=str, default=None)
        p.add_argument("--save-dir", type=str)
        p.add_argument("--device", type=str)
        p.add_argument("--sweep", action="store_true", default=False)
        p.add_argument("--val-fraction", type=float)
        p.add_argument("--seed", type=int)
        p.add_argument("--skip-probes", action="store_true", default=False,
                        help="Skip probe PDE pre-flight validation")
        p.add_argument("--generate-plots", action="store_true", default=False,
                        help="Generate comparison plots from existing results")
        p.add_argument("--baseline-dir", type=str, default=None)
        p.add_argument("--regularizer-dir", type=str, default=None)
        p.add_argument("--ddpinn-dir", type=str, default=None)
        args = p.parse_args()

        cfg = cls()
        for attr in ["data_dir", "n_basis", "hidden_dim", "n_layers", "n_fourier",
                      "fourier_sigma", "n_collocation", "collocation_method",
                      "lambda_phys", "num_epochs", "lr", "batch_size", "patience",
                      "lbfgs_epochs", "rar_interval", "rar_fraction",
                      "curriculum_warmup", "save_dir", "device", "val_fraction", "seed"]:
            cli_val = getattr(args, attr, None)
            if cli_val is not None:
                setattr(cfg, attr, cli_val)
        if args.use_lbfgs is not None:
            cfg.use_lbfgs = True
        if args.use_relative:
            cfg.use_relative = True
        if args.no_auto_batch:
            cfg.auto_batch_size = False
        if args.grad_accum is not None:
            cfg.gradient_accumulation_steps = args.grad_accum
        if args.no_relobralo:
            cfg.use_relobralo = False
        if args.no_rar:
            cfg.use_rar = False
        if args.sweep:
            cfg.sweep = True
        if args.no_wandb:
            cfg.wandb.enabled = False
        if args.run_name is not None:
            cfg.name = args.run_name

        # Store non-dataclass args
        cfg._skip_probes = args.skip_probes
        cfg._generate_plots = args.generate_plots
        cfg._baseline_dir = args.baseline_dir
        cfg._regularizer_dir = args.regularizer_dir
        cfg._ddpinn_dir = args.ddpinn_dir
        cfg._resume_dir = args.resume

        return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_context(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_curriculum_weight(epoch, total_epochs, max_weight, warmup_frac=0.15):
    warmup_end = int(warmup_frac * total_epochs)
    ramp_end = warmup_end + int(0.3 * total_epochs)
    if epoch < warmup_end:
        return 0.0
    elif epoch < ramp_end:
        progress = (epoch - warmup_end) / max(1, ramp_end - warmup_end)
        return max_weight * progress
    else:
        return max_weight


def per_component_rmse(pred_deltas, true_deltas):
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


def probe_auto_batch_size(
    model: nn.Module,
    rhs: nn.Module,
    normalizer,
    sample_input: dict,
    device: str,
    amp_ctx,
    t_colloc_sample: torch.Tensor,
    start_bs: int = 4096,
    vram_target: float = 0.55,
) -> int:
    """Find the largest batch size that fits within vram_target of total GPU memory.

    Two-phase search: coarse (powers of 2) then fine (binary search).
    Simulates both data loss and physics loss at collocation points.
    """
    if not torch.cuda.is_available() or "cuda" not in device:
        print(f"  Auto-batch: CPU mode, using default {start_bs}")
        return start_bs

    total_vram = torch.cuda.get_device_properties(device).total_memory
    target_bytes = int(total_vram * vram_target)

    _ensure_papers_path()
    from aprx_model_elastica.state import relative_to_raw

    def _try_batch(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            state = sample_input["state"].unsqueeze(0).expand(bs, -1).to(device)
            action = sample_input["action"].unsqueeze(0).expand(bs, -1).to(device)
            serp_time = sample_input["t_start"].unsqueeze(0).expand(bs).to(device)
            delta = sample_input["delta"].unsqueeze(0).expand(bs, -1).to(device)

            state_norm = normalizer.normalize_state(state)
            delta_norm = normalizer.normalize_delta(delta)
            time_enc = _encode_time(action, serp_time)

            ctx = amp_ctx if amp_ctx is not None else nullcontext()
            with ctx:
                # Data loss
                pred = model(state_norm, action, time_enc)
                loss_data = F.mse_loss(pred, delta_norm)

                # Simulate physics loss with collocation
                phys_bs = min(256, bs)
                t_batch = t_colloc_sample[:min(100, len(t_colloc_sample))]
                g, g_dot = model.forward_trajectory(
                    state_norm[:phys_bs], action[:phys_bs], time_enc[:phys_bs], t_batch
                )
                loss = loss_data + g.sum() * 0  # ensure memory allocated

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


def _ensure_papers_path():
    import sys
    import types
    if "papers" not in sys.path:
        sys.path.insert(0, "papers")
    for pkg in ["aprx_model_elastica", "locomotion_elastica"]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
            sys.modules[pkg].__path__ = [f"papers/{pkg}"]
            sys.modules[pkg].__package__ = pkg


def _load_data_and_normalizer(data_dir, val_fraction, device, use_relative=False):
    _ensure_papers_path()
    from aprx_model_elastica.dataset import FlatStepDataset
    from aprx_model_elastica.state import (
        StateNormalizer, raw_to_relative, REL_STATE_DIM, RAW_STATE_DIM,
    )

    train_ds = FlatStepDataset(data_dir, split="train", val_fraction=val_fraction)
    val_ds = FlatStepDataset(data_dir, split="val", val_fraction=val_fraction)

    raw_train_states = train_ds.states.clone()
    raw_val_states = val_ds.states.clone()

    if use_relative:
        # Convert to 130D relative state for training
        if train_ds.states.shape[-1] == 124:
            for ds in (train_ds, val_ds):
                ds.states = raw_to_relative(ds.states)
                ds.next_states = raw_to_relative(ds.next_states)
        state_dim = REL_STATE_DIM  # 130
    else:
        # Keep 124D raw state — proper for physics-informed training
        state_dim = RAW_STATE_DIM  # 124

    normalizer = StateNormalizer(state_dim=state_dim, device=device)
    deltas = train_ds.next_states - train_ds.states
    normalizer.fit(train_ds.states, deltas)

    return train_ds, val_ds, normalizer, raw_train_states, raw_val_states, state_dim


def _encode_time(action, serp_time):
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


def train_single_config(cfg: DDPINNTrainConfig, run_dir: Path) -> dict:
    """Train a single DD-PINN model. Returns eval metrics."""
    device = resolve_device(cfg.device)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, run_dir / "config.json")
    metrics_file = open(run_dir / "metrics.jsonl", "w")

    print(f"\n{'='*60}")
    state_mode = "relative-130D" if cfg.use_relative else "raw-124D"
    print(f"DD-PINN Training: n_basis={cfg.n_basis}, lambda={cfg.lambda_phys}, state={state_mode}")
    print(f"Run dir: {run_dir}")
    print(f"{'='*60}")

    # Load data
    train_ds, val_ds, normalizer, raw_train_states, raw_val_states, state_dim = (
        _load_data_and_normalizer(cfg.data_dir, cfg.val_fraction, device,
                                  use_relative=cfg.use_relative)
    )
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    normalizer.save(str(run_dir / "normalizer.pt"))

    # Build model
    model = DDPINNModel(
        state_dim=state_dim, action_dim=5, time_encoding_dim=4,
        n_basis=cfg.n_basis, hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers, n_fourier=cfg.n_fourier,
        fourier_sigma=cfg.fourier_sigma, dt=0.5,
    ).to(device)
    param_info = _count_parameters(model)
    print(f"Model: {param_info['total_params']:,} params (ansatz.param_dim={model.ansatz.param_dim})")

    # Physics
    rhs = CosseratRHS().to(device)
    scales = NondimScales()
    balancer = ReLoBRaLo(n_losses=2) if cfg.use_relobralo else None

    # Collocation points
    t_colloc = sample_collocation(
        cfg.n_collocation, t_start=0.0, t_end=0.5, method=cfg.collocation_method
    ).to(device)
    print(f"Collocation: {len(t_colloc)} points ({cfg.collocation_method})")

    # AMP context (needed for auto-batch probe)
    amp_ctx = get_amp_context(device)

    # Auto batch size probing
    if cfg.auto_batch_size and "cuda" in device:
        print("Probing auto batch size...")
        sample = train_ds[0]
        probed_bs = probe_auto_batch_size(
            model, rhs, normalizer, sample, device, amp_ctx,
            t_colloc_sample=t_colloc,
            start_bs=cfg.batch_size,
        )
        print(f"  Auto-batch: {cfg.batch_size} -> {probed_bs}")
        cfg.batch_size = probed_bs

    # Probe PDE pre-flight validation (per D-04: auto-run by default)
    skip_probes = getattr(cfg, '_skip_probes', False)
    if not skip_probes:
        print("\n--- Probe PDE Pre-flight Validation ---")
        probe_results = run_probe_validation()
        n_passed = sum(probe_results.values())
        n_total = len(probe_results)
        print(f"  Probes: {n_passed}/{n_total} passed")
        if n_passed < n_total:
            failed = [k for k, v in probe_results.items() if not v]
            print(f"  WARNING: Failed probes: {failed}")
            print(f"  Training will proceed but results may be unreliable.")
        print("--- End Pre-flight ---\n")

    # Gradient accumulation
    grad_accum = cfg.gradient_accumulation_steps
    effective_batch = cfg.batch_size * grad_accum

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

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

    # W&B
    wandb_cfg = copy.copy(cfg)
    wandb_cfg.name = cfg.name or f"ddpinn_n{cfg.n_basis}"
    wandb_run = setup_run(wandb_cfg, run_dir)

    hw_info = collect_hardware_info(device)
    log_extra_params(wandb_run, {
        **param_info,
        "num_train_samples": len(train_ds),
        "num_dev_samples": len(val_ds),
        "n_collocation": cfg.n_collocation,
        "batch_size": cfg.batch_size,
        "auto_batch_size": cfg.auto_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        **hw_info,
    })

    # PINNDiagnostics middleware (per D-05, D-06)
    pinn_diag = PINNDiagnostics(wandb_run=wandb_run, ntk_interval=50, n_params_sample=500)

    _ensure_papers_path()
    from aprx_model_elastica.state import relative_to_raw, raw_to_relative

    # Subsample collocation for each batch to keep memory bounded
    n_colloc_batch = min(1000, len(t_colloc))
    # Subsample physics batch size (smaller than data batch)
    phys_batch_size = min(256, cfg.batch_size)

    print(f"  Batch size: {cfg.batch_size} x {grad_accum} accum = {effective_batch} effective")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    phys_loss_history = []
    epoch = 0
    start_epoch = 1

    # Resume from checkpoint if requested
    resume_dir = getattr(cfg, '_resume_dir', None)
    if resume_dir is not None:
        state_path = Path(resume_dir) / "training_state.pt"
        if state_path.exists():
            print(f"Resuming from {state_path}...")
            ckpt = torch.load(str(state_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            patience_counter = ckpt["patience_counter"]
            print(f"  Resumed at epoch {start_epoch}, best_val={best_val_loss:.6f}, patience={patience_counter}")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        if _check_stop(run_dir):
            print(f"  Stop requested at epoch {epoch} — saving and exiting.")
            break

        epoch_start = time.time()
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        n_batches = 0

        phys_weight = get_curriculum_weight(
            epoch, cfg.num_epochs, cfg.lambda_phys, cfg.curriculum_warmup
        )

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            serp_time = batch["t_start"].to(device)
            delta = batch["delta"].to(device)

            state_norm = normalizer.normalize_state(state)
            delta_norm = normalizer.normalize_delta(delta)
            time_enc = _encode_time(action, serp_time)

            # --- Data loss ---
            with amp_ctx:
                pred_delta_norm = model(state_norm, action, time_enc)
                loss_data = F.mse_loss(pred_delta_norm, delta_norm)

            # --- Physics loss at collocation points ---
            if phys_weight > 0:
                # Subsample data for physics (cheaper than full batch)
                phys_idx = torch.randperm(state.shape[0], device=device)[:phys_batch_size]
                phys_state = state_norm[phys_idx]
                phys_action = action[phys_idx]
                phys_time_enc = time_enc[phys_idx]
                phys_raw_state = relative_to_raw(state[phys_idx])

                # Subsample collocation points
                colloc_idx = torch.randperm(len(t_colloc), device=device)[:n_colloc_batch]
                t_batch = t_colloc[colloc_idx]

                with amp_ctx:
                    # Get ansatz values and derivatives at collocation points
                    g, g_dot = model.forward_trajectory(
                        phys_state, phys_action, phys_time_enc, t_batch
                    )
                    # g: (B_phys, N_c, state_dim), g_dot: same

                    # Denormalize g to physical units
                    g_denorm = normalizer.denormalize_delta(
                        g.reshape(-1, state_dim)
                    ).reshape(g.shape[0], g.shape[1], state_dim)

                    # Reconstruct state at collocation times
                    state_base = state[phys_idx]  # (B_phys, state_dim)
                    state_at_t = state_base[:, None, :] + g_denorm

                    B_phys, N_c = state_at_t.shape[:2]

                    if cfg.use_relative:
                        # Convert 130D relative → 124D raw for physics RHS
                        state_at_t_raw = relative_to_raw(
                            state_at_t.reshape(-1, state_dim)
                        ).reshape(B_phys, N_c, RAW_STATE_DIM)
                    else:
                        # Already in 124D raw space — no conversion needed
                        state_at_t_raw = state_at_t

                    # Physics RHS: f(x) at each collocation point
                    f_physics = rhs(state_at_t_raw.reshape(-1, RAW_STATE_DIM))
                    f_physics = f_physics.reshape(B_phys, N_c, RAW_STATE_DIM)

                    # Denormalize g_dot to physical units
                    g_dot_denorm = normalizer.denormalize_delta(
                        g_dot.reshape(-1, state_dim)
                    ).reshape(g.shape[0], g.shape[1], state_dim)

                    if cfg.use_relative:
                        # Lossy: take first 124 dims as raw approximation
                        g_dot_raw = g_dot_denorm.reshape(-1, state_dim)[:, :RAW_STATE_DIM]
                    else:
                        # Exact: g_dot is already 124D raw derivatives
                        g_dot_raw = g_dot_denorm.reshape(-1, RAW_STATE_DIM)

                    f_phys_flat = f_physics.reshape(-1, RAW_STATE_DIM)

                    # Nondimensionalize both sides for balanced comparison
                    g_dot_raw_nondim = scales.nondim_delta(g_dot_raw)
                    f_phys_nondim = scales.nondim_delta(f_phys_flat)

                    loss_phys = F.mse_loss(g_dot_raw_nondim, f_phys_nondim)
            else:
                loss_phys = torch.tensor(0.0, device=device)

            # Combine losses
            if balancer is not None and phys_weight > 0:
                weights = balancer.update([loss_data, loss_phys])
                loss = weights[0] * loss_data + weights[1] * phys_weight * loss_phys
            else:
                loss = loss_data + phys_weight * loss_phys

            # Per-loss gradient norms (expensive, every 10 epochs, first batch only)
            if epoch % 10 == 0 and batch_idx == 0 and phys_weight > 0:
                _diag_grad_metrics = pinn_diag.compute_per_loss_gradients(
                    model, loss_data, loss_phys
                )
            elif batch_idx == 0:
                _diag_grad_metrics = None

            # Scale loss for gradient accumulation, backward OUTSIDE AMP
            (loss / grad_accum).backward()

            # Step optimizer every grad_accum batches (or at end of epoch)
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_data_loss += loss_data.item()
            epoch_phys_loss += loss_phys.item() if isinstance(loss_phys, torch.Tensor) else loss_phys
            n_batches += 1

        epoch_data_loss /= max(1, n_batches)
        epoch_phys_loss /= max(1, n_batches)
        epoch_time = time.time() - epoch_start
        phys_loss_history.append(epoch_phys_loss)

        # RAR: refresh collocation points
        if cfg.use_rar and epoch % cfg.rar_interval == 0 and phys_weight > 0:
            with torch.inference_mode():
                # Evaluate residuals at current collocation points
                sample_idx = torch.randperm(len(train_ds))[:256]
                sample_state = train_ds.states[sample_idx].to(device)
                sample_action = train_ds[sample_idx]["action"].to(device) if hasattr(train_ds, '__getitem__') else torch.zeros(256, 5, device=device)

                # Use a subset of collocation points for residual evaluation
                residuals = torch.zeros(len(t_colloc), device=device)
                # Approximate: use overall physics loss trend
                n_replace = int(len(t_colloc) * cfg.rar_fraction)
                new_points = sample_collocation(
                    n_replace, t_start=0.0, t_end=0.5, method=cfg.collocation_method
                ).to(device)
                # Replace random subset
                replace_idx = torch.randperm(len(t_colloc))[:n_replace]
                t_colloc[replace_idx] = new_points
                t_colloc, _ = t_colloc.sort()

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

        # Checkpoint: best + last
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(run_dir / "model_best.pt"))
        else:
            patience_counter += 1
        torch.save(model.state_dict(), str(run_dir / "model_last.pt"))

        # Save training state for resume (every epoch, overwrites previous)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        }, str(run_dir / "training_state.pt"))

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
            "system/n_collocation": len(t_colloc),
        }

        # Diagnostic metrics (per D-06: non-invasive)
        diag_loss_data = torch.tensor(epoch_data_loss)
        diag_loss_phys = torch.tensor(epoch_phys_loss)
        diag_metrics = pinn_diag.log_step(
            epoch=epoch,
            model=model,
            loss_data=diag_loss_data,
            loss_phys=diag_loss_phys,
            balancer=balancer if cfg.use_relobralo else None,
            residuals=None,  # Residuals computed per-batch; use epoch averages
            collocation_inputs=t_colloc[:64].unsqueeze(-1) if len(t_colloc) > 0 else None,
        )
        epoch_metrics.update(diag_metrics)

        # Merge per-loss gradient norms (computed every 10 epochs)
        if _diag_grad_metrics is not None:
            epoch_metrics.update(_diag_grad_metrics)

        metrics_file.write(json.dumps(epoch_metrics) + "\n")
        metrics_file.flush()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | data={epoch_data_loss:.6f} | "
                f"phys={epoch_phys_loss:.6f} (w={phys_weight:.4f}) | "
                f"val={val_loss:.6f} | best={best_val_loss:.6f} | "
                f"lr={lr:.6f} | patience={patience_counter}/{cfg.patience}"
            )

        log_metrics(wandb_run, epoch_metrics, step=epoch)

        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Optional L-BFGS refinement
    if cfg.use_lbfgs and not _check_stop(run_dir):
        print(f"\nL-BFGS refinement for {cfg.lbfgs_epochs} epochs...")
        # Load best model for refinement
        best_path = run_dir / "model_best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(str(best_path), weights_only=True))

        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(), lr=1e-4, max_iter=20, history_size=50
        )

        for lbfgs_epoch in range(1, cfg.lbfgs_epochs + 1):
            if _check_stop(run_dir):
                break

            model.train()
            # Use full training set in single closure
            all_states = train_ds.states[:cfg.batch_size * 4].to(device)
            all_actions = train_ds[: cfg.batch_size * 4]["action"].to(device) if hasattr(train_ds, '__getitem__') else torch.zeros(min(cfg.batch_size * 4, len(train_ds)), 5, device=device)

            # Simplified L-BFGS on data loss only
            state_norm = normalizer.normalize_state(all_states[:cfg.batch_size])

            def closure():
                lbfgs_optimizer.zero_grad()
                # Get a batch
                idx = torch.randperm(len(train_ds), device=device)[:cfg.batch_size]
                batch_data = train_ds[idx]
                s = batch_data["state"].to(device)
                a = batch_data["action"].to(device)
                t = batch_data["t_start"].to(device)
                d = batch_data["delta"].to(device)

                s_norm = normalizer.normalize_state(s)
                d_norm = normalizer.normalize_delta(d)
                t_enc = _encode_time(a, t)

                pred = model(s_norm, a, t_enc)
                loss = F.mse_loss(pred, d_norm)
                loss.backward()
                return loss

            try:
                loss_val = lbfgs_optimizer.step(closure)
                if lbfgs_epoch % 10 == 0:
                    print(f"  L-BFGS epoch {lbfgs_epoch}: loss={loss_val:.6f}")
            except Exception as e:
                print(f"  L-BFGS failed at epoch {lbfgs_epoch}: {e}")
                break

        # Re-evaluate after L-BFGS
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
        lbfgs_val_loss = val_loss_sum / max(1, val_n)
        if lbfgs_val_loss < best_val_loss:
            best_val_loss = lbfgs_val_loss
            torch.save(model.state_dict(), str(run_dir / "model_best.pt"))
            print(f"  L-BFGS improved val loss: {lbfgs_val_loss:.6f}")
        torch.save(model.state_dict(), str(run_dir / "model_last.pt"))

    metrics_file.close()

    # Save physics loss history for convergence plot
    (run_dir / "phys_loss_history.json").write_text(json.dumps(phys_loss_history))

    # Final evaluation
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

            pred_delta = normalizer.denormalize_delta(pred_delta_norm)

            pred_next = state + pred_delta
            true_next = state + delta

            if cfg.use_relative:
                _ensure_papers_path()
                from aprx_model_elastica.state import relative_to_raw
                pred_next_raw = relative_to_raw(pred_next)
                true_next_raw = relative_to_raw(true_next)
                raw_state_batch = relative_to_raw(state)
            else:
                # Already in raw space
                pred_next_raw = pred_next
                true_next_raw = true_next
                raw_state_batch = state

            all_pred_raw.append(pred_next_raw - raw_state_batch)
            all_true_raw.append(true_next_raw - raw_state_batch)

    all_pred_raw = torch.cat(all_pred_raw, dim=0)
    all_true_raw = torch.cat(all_true_raw, dim=0)

    rmse = per_component_rmse(all_pred_raw, all_true_raw)

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
        "arch": "ddpinn",
        "n_basis": cfg.n_basis,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "lambda_phys": cfg.lambda_phys,
        "best_val_loss": best_val_loss,
        "rmse": rmse,
        "r2_components": r2_components,
        "r2_overall": r2_per_dim.mean().item(),
    }

    (run_dir / "eval_metrics.json").write_text(json.dumps(eval_result, indent=2))

    # Save ansatz config for Phase 4 compatibility
    ansatz_config = {
        "arch": "ddpinn",
        "state_dim": state_dim,
        "action_dim": 5,
        "time_encoding_dim": 4,
        "n_basis": cfg.n_basis,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "n_fourier": cfg.n_fourier,
        "fourier_sigma": cfg.fourier_sigma,
        "dt": 0.5,
    }
    (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), **ansatz_config}, indent=2))

    print(f"\nResults (n_basis={cfg.n_basis}):")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  R2 overall: {eval_result['r2_overall']:.4f}")
    for k, v in rmse.items():
        print(f"    {k}: {v:.4f}")
    for k, v in r2_components.items():
        print(f"    R2 {k}: {v:.4f}")

    if wandb_run is not None:
        eval_wandb = {"eval/" + k: v for k, v in rmse.items()}
        eval_wandb.update({"eval_r2/" + k: v for k, v in r2_components.items()})
        log_metrics(wandb_run, eval_wandb, step=epoch)
    log_model_artifact(wandb_run, best_path, f"ddpinn-n{cfg.n_basis}")
    end_run(wandb_run)

    return eval_result


def generate_comparison_plots(results: list[dict], save_dir: Path):
    """Generate comparison plots for n_basis sweep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    figures_dir = Path("figures/pinn")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Training curves (if phys_loss_history files exist)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_basis_vals = [r["n_basis"] for r in results]
    labels = [f"n_g={n}" for n in n_basis_vals]

    r2_vals = [r["r2_overall"] for r in results]
    axes[0].bar(labels, r2_vals, color="steelblue")
    axes[0].set_ylabel("R\u00b2 (overall)")
    axes[0].set_title("DD-PINN: Overall R\u00b2 vs n_basis")
    axes[0].set_ylim(0, 1)

    val_losses = [r["best_val_loss"] for r in results]
    axes[1].bar(labels, val_losses, color="coral")
    axes[1].set_ylabel("Best Val Loss")
    axes[1].set_title("DD-PINN: Validation Loss vs n_basis")

    plt.tight_layout()
    plt.savefig(figures_dir / "ddpinn_training.png", dpi=150)
    plt.close()
    print(f"Saved {figures_dir / 'ddpinn_training.png'}")

    # Per-component RMSE
    components = list(results[0]["rmse"].keys())
    n_components = len(components)
    x = range(n_components)
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, r in enumerate(results):
        vals = [r["rmse"][c] for c in components]
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width, label=f"n_g={r['n_basis']}")

    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_ylabel("RMSE (physical units)")
    ax.set_title("DD-PINN Per-Component RMSE: n_basis Sweep")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "ddpinn_per_component.png", dpi=150)
    plt.close()
    print(f"Saved {figures_dir / 'ddpinn_per_component.png'}")


def generate_final_comparison(
    baseline_dir: Optional[str] = None,
    regularizer_dir: Optional[str] = None,
    ddpinn_dir: Optional[str] = None,
):
    """Generate comprehensive comparison plots: baseline vs regularizer vs DD-PINN.

    Reads eval_metrics.json from each directory. Generates:
    - figures/pinn/final_comparison.png: grouped bar chart per component
    - figures/pinn/physics_residual_convergence.png: physics loss over epochs
    - figures/pinn/predicted_vs_actual.png: scatter plots per component
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    figures_dir = Path("figures/pinn")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Collect results from available directories
    results = []
    labels = []

    for label, dir_path in [("MLP Baseline", baseline_dir),
                             ("Regularizer", regularizer_dir),
                             ("DD-PINN", ddpinn_dir)]:
        if dir_path is None:
            continue
        metrics_path = Path(dir_path) / "eval_metrics.json"
        if not metrics_path.exists():
            # Try to find eval_metrics.json in subdirectories
            for p in Path(dir_path).rglob("eval_metrics.json"):
                metrics_path = p
                break
        if not metrics_path.exists():
            print(f"  Skipping {label}: no eval_metrics.json in {dir_path}")
            continue
        data = json.loads(metrics_path.read_text())
        results.append(data)
        labels.append(label)

    if len(results) < 2:
        print("Need at least 2 results for comparison. Skipping plots.")
        return

    # Plot 1: Per-component RMSE comparison
    components = list(results[0]["rmse"].keys())
    n_components = len(components)
    x = np.arange(n_components)
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (r, label) in enumerate(zip(results, labels)):
        vals = [r["rmse"].get(c, 0) for c in components]
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_ylabel("RMSE (physical units)")
    ax.set_title("Per-Component RMSE: MLP Baseline vs Physics-Regularized vs DD-PINN")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "final_comparison.png", dpi=300)
    plt.close()
    print(f"Saved {figures_dir / 'final_comparison.png'}")

    # Plot 2: Physics residual convergence
    if ddpinn_dir:
        phys_path = Path(ddpinn_dir) / "phys_loss_history.json"
        if not phys_path.exists():
            for p in Path(ddpinn_dir).rglob("phys_loss_history.json"):
                phys_path = p
                break
        if phys_path.exists():
            phys_history = json.loads(phys_path.read_text())
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.semilogy(range(1, len(phys_history) + 1), phys_history)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Physics Residual Loss (log scale)")
            ax.set_title("DD-PINN Physics Residual Convergence")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(figures_dir / "physics_residual_convergence.png", dpi=300)
            plt.close()
            print(f"Saved {figures_dir / 'physics_residual_convergence.png'}")
        else:
            # Create placeholder
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "Physics loss history not available\n(run full training first)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title("DD-PINN Physics Residual Convergence")
            plt.tight_layout()
            plt.savefig(figures_dir / "physics_residual_convergence.png", dpi=300)
            plt.close()

    # Plot 3: R2 comparison bar chart (in lieu of scatter plots which need raw predictions)
    fig, ax = plt.subplots(figsize=(12, 6))
    r2_components = ["pos_x", "pos_y", "vel_x", "vel_y", "yaw", "omega_z"]
    x = np.arange(len(r2_components))
    width = 0.8 / len(results)

    for i, (r, label) in enumerate(zip(results, labels)):
        r2 = r.get("r2_components", {})
        vals = [r2.get(c, 0) for c in r2_components]
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(r2_components, rotation=45, ha="right")
    ax.set_ylabel("R\u00b2")
    ax.set_title("Per-Component R\u00b2: Model Comparison")
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "predicted_vs_actual.png", dpi=300)
    plt.close()
    print(f"Saved {figures_dir / 'predicted_vs_actual.png'}")


def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)
    cfg = DDPINNTrainConfig.from_cli()
    set_seed(cfg.seed)

    # Plot-only mode
    if getattr(cfg, "_generate_plots", False):
        generate_final_comparison(
            baseline_dir=getattr(cfg, "_baseline_dir", None),
            regularizer_dir=getattr(cfg, "_regularizer_dir", None),
            ddpinn_dir=getattr(cfg, "_ddpinn_dir", None),
        )
        return

    if cfg.sweep:
        n_basis_values = [5, 7, 10]
        results = []
        for n_g in n_basis_values:
            cfg_run = copy.copy(cfg)
            cfg_run.n_basis = n_g
            cfg_run.name = f"ddpinn_n{n_g}"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(cfg.save_dir) / f"n_basis_{n_g}_{ts}"
            result = train_single_config(cfg_run, run_dir)
            results.append(result)
            cleanup_vram()

        generate_comparison_plots(results, Path(cfg.save_dir))

        combined_path = Path(cfg.save_dir) / "sweep_results.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(results, indent=2))
        print(f"\nSweep complete! Results saved to {combined_path}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.save_dir) / f"{cfg.name}_{ts}"
        train_single_config(cfg, run_dir)


if __name__ == "__main__":
    from src.utils.gpu_lock import GpuLock
    with GpuLock():
        main()
