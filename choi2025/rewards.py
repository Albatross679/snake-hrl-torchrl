"""Reward functions for soft manipulator tasks (Choi & Tong, 2025).

Following the Naughton et al. (2021) Elastica benchmark style:
distance-based rewards with optional improvement bonuses and penalties.
"""

import numpy as np


def compute_follow_target_reward(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    prev_tip_pos: np.ndarray,
) -> float:
    """Reward for following a moving target.

    Combines a distance penalty with an improvement bonus for moving
    closer to the target.

    Args:
        tip_pos: Current tip position, shape (3,).
        target_pos: Target position, shape (3,).
        prev_tip_pos: Previous tip position, shape (3,).

    Returns:
        Scalar reward.
    """
    dist = np.linalg.norm(tip_pos - target_pos)
    prev_dist = np.linalg.norm(prev_tip_pos - target_pos)

    # Distance penalty (exponential decay)
    distance_reward = np.exp(-5.0 * dist)

    # Improvement bonus
    improvement = prev_dist - dist
    improvement_bonus = 10.0 * improvement

    return float(distance_reward + improvement_bonus)


def compute_ik_reward(
    tip_pos: np.ndarray,
    tip_tangent: np.ndarray,
    target_pos: np.ndarray,
    target_orient: np.ndarray,
) -> float:
    """Reward for inverse kinematics (position + orientation).

    Args:
        tip_pos: Current tip position, shape (3,).
        tip_tangent: Current tip tangent vector (unit), shape (3,).
        target_pos: Target position, shape (3,).
        target_orient: Target orientation (unit vector), shape (3,).

    Returns:
        Scalar reward.
    """
    # Position error
    pos_dist = np.linalg.norm(tip_pos - target_pos)
    pos_reward = np.exp(-5.0 * pos_dist)

    # Orientation error (cosine similarity)
    cos_sim = np.clip(np.dot(tip_tangent, target_orient), -1.0, 1.0)
    orient_reward = (1.0 + cos_sim) / 2.0  # Maps [-1, 1] → [0, 1]

    # Combined: position weighted more than orientation
    return float(0.7 * pos_reward + 0.3 * orient_reward)


def compute_obstacle_reward(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    prev_tip_pos: np.ndarray,
    total_penetration: float,
    contact_penalty: float = 10.0,
) -> float:
    """Reward for reaching target while avoiding obstacles.

    Args:
        tip_pos: Current tip position, shape (3,).
        target_pos: Target position, shape (3,).
        prev_tip_pos: Previous tip position, shape (3,).
        total_penetration: Sum of penetration depths across all nodes.
        contact_penalty: Penalty scaling for obstacle penetration.

    Returns:
        Scalar reward.
    """
    dist = np.linalg.norm(tip_pos - target_pos)
    prev_dist = np.linalg.norm(prev_tip_pos - target_pos)

    # Distance reward
    distance_reward = np.exp(-5.0 * dist)

    # Improvement bonus
    improvement = prev_dist - dist
    improvement_bonus = 10.0 * improvement

    # Obstacle penetration penalty
    penalty = -contact_penalty * total_penetration

    return float(distance_reward + improvement_bonus + penalty)
