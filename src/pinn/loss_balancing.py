"""ReLoBRaLo: Relative Loss Balancing with Random Lookback.

Adaptive loss weighting for multi-term PINN losses. Adjusts weights based
on relative rate of change of each loss term, with exponential moving average
smoothing and random lookback for stability.

Reference: Bischof & Kraus, "Multi-Objective Loss Balancing for Physics-Informed
Deep Learning" (arXiv:2110.09813).
"""

from __future__ import annotations

from typing import List, Optional

import torch


class ReLoBRaLo:
    """Adaptive multi-term loss balancer using ReLoBRaLo algorithm.

    Args:
        n_losses: Number of loss terms to balance.
        alpha: Exponential moving average decay for weight smoothing.
        temperature: Softmax temperature (higher = more uniform weights).
        rho: Probability of using previous step (vs random lookback).
    """

    def __init__(
        self,
        n_losses: int,
        alpha: float = 0.999,
        temperature: float = 1.0,
        rho: float = 0.999,
    ):
        self.n_losses = n_losses
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho

        self.weights = torch.ones(n_losses)
        self.prev_losses: Optional[torch.Tensor] = None
        self.lookback_losses: Optional[torch.Tensor] = None

    def update(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """Update weights given current loss values.

        Args:
            losses: List of scalar loss tensors (one per term).

        Returns:
            Tensor of shape (n_losses,) with updated weights summing to n_losses.
        """
        current = torch.tensor([l.item() for l in losses])

        if self.prev_losses is None:
            self.prev_losses = current.clone()
            self.lookback_losses = current.clone()
            return self.weights.clone()

        # Random lookback: with probability (1-rho), use lookback instead of prev
        if torch.rand(1).item() < self.rho:
            ref_losses = self.prev_losses
        else:
            ref_losses = self.lookback_losses

        # Compute relative change ratios
        ratios = current / (ref_losses + 1e-8)

        # Softmax to get new weights, scaled to sum to n_losses
        new_weights = torch.softmax(ratios / self.temperature, dim=0) * self.n_losses

        # Exponential moving average
        self.weights = self.alpha * self.weights + (1.0 - self.alpha) * new_weights

        # Update references
        self.lookback_losses = self.prev_losses.clone()
        self.prev_losses = current.clone()

        return self.weights.clone()
