"""PINN diagnostic middleware for training failure detection.

Non-invasive diagnostic layer that computes and logs PINN-specific metrics
to W&B each epoch. Mirrors src/trainers/diagnostics.py pattern for RL.

Provides:
- PINNDiagnostics: middleware class with deque-based history, called via log_step()
- compute_ntk_eigenvalues: standalone NTK eigenvalue computation with parameter subsampling

Metrics are logged under the ``diagnostics/`` W&B prefix. Per decision D-07,
this module is log-only -- no wandb.alert(), no auto-stopping.

References:
    Wang et al. "When and Why PINNs Fail to Train" (NeurIPS 2021) -- NTK analysis
    Bischof & Kraus (arXiv:2110.09813) -- ReLoBRaLo health monitoring
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.pinn._state_slices import POS_X, VEL_X, YAW, OMEGA_Z


# ---------------------------------------------------------------------------
# Standalone NTK computation (per D-09, D-10, D-11)
# ---------------------------------------------------------------------------

def compute_ntk_eigenvalues(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    n_params_sample: int = 500,
) -> Dict[str, float]:
    """Approximate NTK eigenvalue spectrum for convergence analysis.

    Computes Jacobian of model output w.r.t. a subsample of parameters,
    then derives the NTK approximation K = J @ J.T and its eigenvalues.

    Args:
        model: Neural network model.
        sample_inputs: (N, input_dim) tensor of collocation inputs.
        n_params_sample: Max number of parameters to subsample for tractability.

    Returns:
        Dict with ntk/eigenvalue_max, ntk/eigenvalue_min,
        ntk/condition_number, ntk/spectral_decay_rate.
    """
    was_training = model.training
    model.eval()

    try:
        # Flatten all parameters and record which are trainable
        all_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in all_params)
        actual_n = min(n_params_sample, total_params)

        # Build parameter index map: which flat indices to subsample
        param_indices = torch.randperm(total_params)[:actual_n]

        # Compute per-sample Jacobian via backward passes
        N = sample_inputs.shape[0]
        J = torch.zeros(N, actual_n)

        for i in range(N):
            model.zero_grad()
            inp = sample_inputs[i : i + 1].detach().requires_grad_(False)
            out = model(inp)
            # Sum over output dims if multi-output
            scalar_out = out.sum()
            scalar_out.backward(retain_graph=False)

            # Collect gradients into flat vector, subsample
            grad_flat = torch.cat(
                [p.grad.flatten() for p in all_params if p.grad is not None]
            )
            J[i] = grad_flat[param_indices]

        # NTK approximation: K = J @ J.T  (N x N)
        K = J @ J.T

        # Eigenvalues (symmetric positive semi-definite)
        eigenvalues = torch.linalg.eigvalsh(K)
        eigenvalues = eigenvalues.clamp(min=0)  # numerical stability

        eig_max = eigenvalues[-1].item()
        eig_min = eigenvalues[0].item()
        mid_idx = max(1, len(eigenvalues) // 2)
        eig_mid = eigenvalues[-mid_idx].item()

        return {
            "ntk/eigenvalue_max": eig_max,
            "ntk/eigenvalue_min": eig_min,
            "ntk/condition_number": eig_max / max(eig_min, 1e-10),
            "ntk/spectral_decay_rate": eig_max / max(eig_mid, 1e-10),
        }

    finally:
        if was_training:
            model.train()
        model.zero_grad()


# ---------------------------------------------------------------------------
# PINNDiagnostics middleware (per D-05, D-06, D-07)
# ---------------------------------------------------------------------------

class PINNDiagnostics:
    """Non-invasive diagnostic middleware for PINN training.

    Mirrors src/trainers/diagnostics.py pattern for RL.
    Computes and logs PINN-specific metrics each epoch.
    Per D-07: log-only to W&B (no alerts, no auto-stopping).

    Args:
        wandb_run: Active W&B run or None.
        ntk_interval: Compute NTK every N epochs (default 50, per D-09).
        n_params_sample: Parameters to subsample for NTK (default 500, per D-10).
    """

    def __init__(
        self,
        wandb_run=None,
        ntk_interval: int = 50,
        n_params_sample: int = 500,
    ):
        self.wandb_run = wandb_run
        self.ntk_interval = ntk_interval
        self.n_params_sample = n_params_sample
        self._history = {
            "loss_ratio": deque(maxlen=100),
            "grad_norm_data": deque(maxlen=100),
            "grad_norm_phys": deque(maxlen=100),
            "residual_mean": deque(maxlen=100),
        }

    # ---- Compute methods (per D-08) ----

    def compute_loss_ratio(
        self, loss_data: torch.Tensor, loss_phys: torch.Tensor
    ) -> float:
        """Ratio of physics loss to data loss.

        Healthy range: [0.1, 10]. Ratio > 100 indicates convergence mismatch.
        """
        ratio = loss_phys.item() / max(loss_data.item(), 1e-10)
        self._history["loss_ratio"].append(ratio)
        return ratio

    def compute_per_loss_gradients(
        self, model: nn.Module, loss_data: torch.Tensor, loss_phys: torch.Tensor
    ) -> Dict[str, float]:
        """Separate gradient norms for each loss term.

        Computes backward on each loss individually with retain_graph=True,
        records norms, then restores model.zero_grad().
        """
        # Save any existing gradients
        saved_grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                saved_grads[name] = p.grad.clone()

        # Data loss gradients
        model.zero_grad()
        loss_data.backward(retain_graph=True)
        grad_data_parts = [
            p.grad.flatten() for p in model.parameters() if p.grad is not None
        ]
        norm_data = (
            torch.cat(grad_data_parts).norm().item() if grad_data_parts else 0.0
        )

        # Physics loss gradients
        model.zero_grad()
        loss_phys.backward(retain_graph=True)
        grad_phys_parts = [
            p.grad.flatten() for p in model.parameters() if p.grad is not None
        ]
        norm_phys = (
            torch.cat(grad_phys_parts).norm().item() if grad_phys_parts else 0.0
        )

        # Restore original gradients
        model.zero_grad()
        for name, p in model.named_parameters():
            if name in saved_grads:
                p.grad = saved_grads[name]

        self._history["grad_norm_data"].append(norm_data)
        self._history["grad_norm_phys"].append(norm_phys)

        return {
            "diagnostics/grad_norm_data": norm_data,
            "diagnostics/grad_norm_phys": norm_phys,
            "diagnostics/grad_norm_ratio": norm_phys / max(norm_data, 1e-10),
        }

    def compute_residual_statistics(self, residuals: torch.Tensor) -> Dict[str, float]:
        """Per-component residual statistics for spatial analysis.

        Args:
            residuals: (N, state_dim) or (N,) tensor of physics residuals.
        """
        r_abs = residuals.abs()
        mean_val = r_abs.mean().item()
        self._history["residual_mean"].append(mean_val)

        return {
            "diagnostics/residual_mean": mean_val,
            "diagnostics/residual_max": r_abs.max().item(),
            "diagnostics/residual_std": residuals.std().item(),
            "diagnostics/residual_p95": r_abs.quantile(0.95).item(),
        }

    def compute_relobralo_health(self, weights: torch.Tensor) -> Dict[str, float]:
        """Track ReLoBRaLo weight evolution.

        Args:
            weights: (n_losses,) tensor from ReLoBRaLo.weights.
        """
        return {
            "diagnostics/relobralo_w_data": weights[0].item(),
            "diagnostics/relobralo_w_phys": weights[1].item(),
            "diagnostics/relobralo_ratio": weights[1].item()
            / max(weights[0].item(), 1e-10),
        }

    def compute_per_component_violations(
        self, residuals: torch.Tensor
    ) -> Dict[str, float]:
        """Per-component group violation magnitudes.

        Args:
            residuals: (N, 124) tensor of per-state-component physics violations.

        Returns:
            Mean absolute violation for position, velocity, yaw, omega groups.
        """
        return {
            "diagnostics/violation_pos_x": residuals[:, POS_X].abs().mean().item(),
            "diagnostics/violation_vel_x": residuals[:, VEL_X].abs().mean().item(),
            "diagnostics/violation_yaw": residuals[:, YAW].abs().mean().item(),
            "diagnostics/violation_omega_z": residuals[:, OMEGA_Z].abs().mean().item(),
        }

    # ---- Main entry point ----

    def log_step(
        self,
        epoch: int,
        model: nn.Module,
        loss_data: torch.Tensor,
        loss_phys: torch.Tensor,
        balancer=None,
        residuals: Optional[torch.Tensor] = None,
        collocation_inputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute all diagnostics for one epoch and return merged dict.

        Called once per epoch from train_pinn.py. Per D-06: non-invasive.
        Per D-07: logs to W&B only (no alerts, no auto-stopping).

        Args:
            epoch: Current training epoch.
            model: The DD-PINN model.
            loss_data: Scalar data loss tensor.
            loss_phys: Scalar physics loss tensor.
            balancer: ReLoBRaLo instance (optional).
            residuals: Physics residual tensor (optional).
            collocation_inputs: Collocation point tensor for NTK (optional).

        Returns:
            Dict of all diagnostic metrics computed this epoch.
        """
        metrics: Dict[str, float] = {}

        # Always compute loss ratio
        metrics["diagnostics/loss_ratio"] = self.compute_loss_ratio(
            loss_data, loss_phys
        )

        # ReLoBRaLo health (if balancer provided)
        if balancer is not None:
            metrics.update(self.compute_relobralo_health(balancer.weights))

        # Residual stats (if residuals provided)
        if residuals is not None:
            metrics.update(self.compute_residual_statistics(residuals))
            if residuals.dim() == 2 and residuals.shape[1] >= 124:
                metrics.update(self.compute_per_component_violations(residuals))

        # NTK (expensive, per D-09: every ntk_interval epochs)
        if collocation_inputs is not None and epoch % self.ntk_interval == 0:
            ntk_metrics = compute_ntk_eigenvalues(
                model, collocation_inputs, self.n_params_sample
            )
            metrics.update(ntk_metrics)

        # Log to W&B (per D-07: log-only, no alerts)
        if self.wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=epoch)
            except Exception:
                pass

        return metrics
