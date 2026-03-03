"""Reward functions for DQN gait learning (Shi et al., 2020).

Forward locomotion:
    R(s,a) = c₁ · Δx - c₂ · P₀ + c₃ · R_θ

Reorientation:
    R(s,a) = c₁ · Δθ - c₂ · P₀

where P₀ is a penalty for zero displacement and R_θ encourages
maintaining forward-facing orientation.
"""

import math

import numpy as np


def orientation_reward(theta_new: float) -> float:
    """Orientation bonus R_θ from Eq. (9).

    R_θ = 1              if -π/4 ≤ θ_new ≤ π/4
    R_θ = π/4 - |θ_new|  otherwise

    Args:
        theta_new: Orientation after action (radians).

    Returns:
        Scalar orientation reward.
    """
    if -math.pi / 4 <= theta_new <= math.pi / 4:
        return 1.0
    return math.pi / 4 - abs(theta_new)


def compute_forward_reward(
    delta_x: float,
    theta_new: float,
    c1: float = 10.0,
    c2: float = 10.0,
    c3: float = 1.0,
) -> float:
    """Forward locomotion reward from Eq. (8).

    R(s,a) = c₁ · Δx - c₂ · P₀ + c₃ · R_θ

    Positive Δx = forward, negative = backward.
    P₀ = 1 if Δx ≈ 0, else 0.

    Args:
        delta_x: Body-frame x-displacement.
        theta_new: New orientation angle.
        c1, c2, c3: Reward weights.

    Returns:
        Scalar reward.
    """
    P0 = 1.0 if abs(delta_x) < 1e-6 else 0.0
    R_theta = orientation_reward(theta_new)
    return c1 * delta_x - c2 * P0 + c3 * R_theta


def compute_reorientation_reward(
    delta_theta: float,
    c1: float = 10.0,
    c2: float = 10.0,
) -> float:
    """Reorientation reward from Eq. (10).

    R(s,a) = c₁ · Δθ - c₂ · P₀

    Positive Δθ = desired rotation direction.
    P₀ = 1 if Δθ ≈ 0, else 0.

    Args:
        delta_theta: Change in orientation (radians).
        c1, c2: Reward weights.

    Returns:
        Scalar reward.
    """
    P0 = 1.0 if abs(delta_theta) < 1e-6 else 0.0
    return c1 * delta_theta - c2 * P0
