"""Reward functions for Elastica-RL benchmark (Naughton et al., 2021).

Core reward:
    R = -n^2 + phi(n)

where n = ||x_tip - x_target|| and phi(n) provides proximity bonuses
with increasing reward as the tip approaches the target.
"""

import numpy as np


def compute_tracking_reward(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    thresholds: tuple = (0.1, 0.05, 0.02, 0.01),
    bonuses: tuple = (0.25, 0.5, 1.0, 2.0),
) -> float:
    """Distance-based reward with proximity bonuses (Cases 1, 3, 4).

    R = -n^2 + phi(n)

    phi(n) gives incremental bonuses as the tip enters progressively
    tighter proximity thresholds around the target.

    Args:
        tip_pos: Current tip position, shape (3,).
        target_pos: Target position, shape (3,).
        thresholds: Distance thresholds for bonuses (descending).
        bonuses: Reward bonus at each threshold.

    Returns:
        Scalar reward.
    """
    dist = np.linalg.norm(tip_pos - target_pos)

    # Quadratic distance penalty
    reward = -(dist ** 2)

    # Proximity bonuses
    for thresh, bonus in zip(thresholds, bonuses):
        if dist < thresh:
            reward += bonus

    return float(reward)


def compute_reaching_reward(
    tip_pos: np.ndarray,
    tip_tangent: np.ndarray,
    target_pos: np.ndarray,
    target_orient: np.ndarray,
    position_weight: float = 0.7,
    orientation_weight: float = 0.3,
) -> float:
    """Reaching reward with position and orientation matching (Case 2).

    Combines the standard tracking reward with an orientation alignment
    bonus based on the cosine similarity between tip tangent and target
    orientation.

    Args:
        tip_pos: Tip position, shape (3,).
        tip_tangent: Tip tangent direction (unit), shape (3,).
        target_pos: Target position, shape (3,).
        target_orient: Target orientation (unit), shape (3,).
        position_weight: Weight for position component.
        orientation_weight: Weight for orientation component.

    Returns:
        Scalar reward.
    """
    # Position reward (same as tracking)
    pos_reward = compute_tracking_reward(tip_pos, target_pos)

    # Orientation reward (cosine similarity → [0, 1])
    cos_sim = np.clip(np.dot(tip_tangent, target_orient), -1.0, 1.0)
    orient_reward = (1.0 + cos_sim) / 2.0

    return float(position_weight * pos_reward + orientation_weight * orient_reward)


def compute_obstacle_reward(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    total_penetration: float,
    collision_penalty: float = 5.0,
) -> float:
    """Reward for reaching through obstacles (Cases 3, 4).

    Standard tracking reward plus a penalty proportional to total
    penetration depth into obstacles.

    Args:
        tip_pos: Tip position, shape (3,).
        target_pos: Target position, shape (3,).
        total_penetration: Sum of penetration depths, >= 0.
        collision_penalty: Penalty scale factor.

    Returns:
        Scalar reward.
    """
    reward = compute_tracking_reward(tip_pos, target_pos)
    reward -= collision_penalty * total_penetration

    return float(reward)
