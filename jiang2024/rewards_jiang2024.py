"""Reward functions for COBRA navigation (Jiang et al., 2024).

Implements the reward function from Eq. 12 of the paper:
    r = w1 * r1 + w2 * r2 + w3 * r3
where:
    r1 = 1 / (0.1 + d)            -- proximity reward
    r2 = d_prev - d               -- velocity reward (approaching target)
    r3 = -||a - a_prev||          -- smoothness penalty
"""

import numpy as np


def compute_proximity_reward(distance: float) -> float:
    """Proximity reward: higher when closer to waypoint.

    r1 = 1 / (0.1 + d)

    Args:
        distance: Current distance to waypoint (meters).

    Returns:
        Proximity reward (always positive).
    """
    return 1.0 / (0.1 + distance)


def compute_velocity_reward(dist_prev: float, dist_curr: float) -> float:
    """Velocity reward: positive when approaching waypoint.

    r2 = d_prev - d_curr

    Args:
        dist_prev: Distance to waypoint at previous step.
        dist_curr: Distance to waypoint at current step.

    Returns:
        Velocity reward (positive = approaching).
    """
    return dist_prev - dist_curr


def compute_smoothness_penalty(
    action: np.ndarray,
    prev_action: np.ndarray,
) -> float:
    """Smoothness penalty: penalizes abrupt action changes.

    r3 = -||a - a_prev||

    Args:
        action: Current action vector.
        prev_action: Previous action vector.

    Returns:
        Smoothness penalty (always <= 0).
    """
    return -float(np.linalg.norm(action - prev_action))


def compute_navigation_reward(
    distance: float,
    dist_prev: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    w1: float = 1.0,
    w2: float = 5.0,
    w3: float = 0.1,
) -> float:
    """Combined navigation reward (Eq. 12).

    r = w1 * r1 + w2 * r2 + w3 * r3

    Args:
        distance: Current distance to waypoint.
        dist_prev: Distance to waypoint at previous step.
        action: Current action vector.
        prev_action: Previous action vector.
        w1: Weight for proximity reward.
        w2: Weight for velocity reward.
        w3: Weight for smoothness penalty.

    Returns:
        Scalar reward.
    """
    r1 = compute_proximity_reward(distance)
    r2 = compute_velocity_reward(dist_prev, distance)
    r3 = compute_smoothness_penalty(action, prev_action)
    return w1 * r1 + w2 * r2 + w3 * r3
