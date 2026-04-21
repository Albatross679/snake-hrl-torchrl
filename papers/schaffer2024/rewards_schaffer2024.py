"""Reward functions for biohybrid lattice worm (Schaffer et al., 2024).

Following Naughton et al. (2021) style:
    R = -n^2 + phi(n)
where n = distance to target, phi(n) gives bonuses for proximity.

NaN penalty: -2.0 if simulation produces NaN (instability).
"""

import numpy as np


def compute_navigation_reward(
    worm_pos: np.ndarray,
    target_pos: np.ndarray,
    threshold: float = 0.001,
) -> float:
    """Distance-based reward with proximity bonuses.

    Args:
        worm_pos: Current worm centroid position, shape (3,).
        target_pos: Target position, shape (3,).
        threshold: Success radius d (1 mm).

    Returns:
        Scalar reward.
    """
    dist = np.linalg.norm(worm_pos - target_pos)

    # Base: quadratic distance penalty
    reward = -(dist ** 2)

    # Proximity bonuses
    if dist < 2 * threshold:
        reward += 0.5  # Close
    if dist < threshold:
        reward += 2.0  # Success

    return float(reward)


def compute_nan_penalty() -> float:
    """Penalty for simulation instability (NaN detected)."""
    return -2.0
