"""Collocation point sampling for PINN/DD-PINN training.

Provides Sobol quasi-random and uniform sampling for temporal collocation,
plus Residual-based Adaptive Refinement (RAR) for concentrating points
in high-residual regions.
"""

from __future__ import annotations

import torch
from scipy.stats.qmc import Sobol


def sample_collocation(
    n_points: int,
    t_start: float = 0.0,
    t_end: float = 0.5,
    method: str = "sobol",
) -> torch.Tensor:
    """Sample collocation points in a time interval.

    Args:
        n_points: Number of collocation points to generate.
        t_start: Start of time interval.
        t_end: End of time interval.
        method: Sampling method, "sobol" or "uniform".

    Returns:
        Sorted (n_points,) float32 tensor of collocation times in [t_start, t_end].
    """
    if method == "sobol":
        sampler = Sobol(d=1, scramble=True)
        # Sobol requires power-of-2 samples; draw enough and truncate
        m = max(1, (n_points - 1).bit_length())
        n_pow2 = 2 ** m
        samples = sampler.random(n_pow2)[:n_points, 0]  # (n_points,)
        t = torch.tensor(samples, dtype=torch.float32)
    elif method == "uniform":
        t = torch.rand(n_points)
    else:
        raise ValueError(f"Unknown collocation method: {method}")

    # Scale to [t_start, t_end] and sort
    t = t_start + t * (t_end - t_start)
    t, _ = torch.sort(t)
    return t


def adaptive_refinement(
    residuals: torch.Tensor,
    t_colloc: torch.Tensor,
    n_new: int = 100,
    beta: float = 2.0,
) -> torch.Tensor:
    """Residual-based Adaptive Refinement (RAR) for collocation points.

    Samples new collocation points concentrated near regions where the
    physics residual is high. Each existing collocation point has sampling
    probability proportional to |residual|^beta.

    Args:
        residuals: (N,) tensor of residual magnitudes at existing collocation points.
        t_colloc: (N,) tensor of existing collocation point times.
        n_new: Number of new points to generate.
        beta: Exponent controlling concentration (higher = more focused).

    Returns:
        (n_new,) sorted tensor of new collocation times.
    """
    # Probability weights: |residual|^beta
    weights = residuals.abs().pow(beta)
    weights = weights / (weights.sum() + 1e-10)

    # Sample indices with replacement according to weights
    indices = torch.multinomial(weights, n_new, replacement=True)

    # Get base times from sampled indices
    base_times = t_colloc[indices]

    # Add small noise for diversity (uniform in [-dt/2, dt/2] where dt is
    # average spacing between collocation points)
    dt_avg = (t_colloc.max() - t_colloc.min()) / max(len(t_colloc) - 1, 1)
    noise = (torch.rand(n_new) - 0.5) * dt_avg

    new_times = base_times + noise

    # Clamp to valid range
    new_times = new_times.clamp(t_colloc.min().item(), t_colloc.max().item())

    # Sort
    new_times, _ = torch.sort(new_times)
    return new_times
