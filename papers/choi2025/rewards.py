"""Reward functions for soft manipulator tasks (Choi & Tong, 2025).

Following the Naughton et al. (2021) Elastica benchmark style:
distance-based rewards with optional improvement bonuses and penalties.
"""

import numpy as np


def compute_follow_target_reward(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    prev_tip_pos: np.ndarray,
    tip_tangent: np.ndarray | None = None,
    heading_weight: float = 0.0,
    prev_dist: float | None = None,
    pbrs_gamma: float = 0.0,
    pbrs_only: bool = False,
    improvement_weight: float = 0.0,
    return_components: bool = False,
) -> float | tuple[float, dict]:
    """Reward for following a moving target.

    Distance-based exponential reward with optional PBRS shaping.

    PBRS (Potential-Based Reward Shaping, Ng et al. 1999):
        Φ(s) = -dist(tip, target)
        F(s, s') = γ_pbrs · Φ(s') - Φ(s) = prev_dist - γ_pbrs · dist
        Guaranteed not to change the optimal policy.

    Args:
        tip_pos: Current tip position, shape (3,).
        target_pos: Target position, shape (3,).
        prev_tip_pos: Previous tip position, shape (3,).
        tip_tangent: Tip tangent (unit vector), shape (3,). Required if heading_weight > 0.
        heading_weight: Weight for heading component in [0, 1].
            Total reward = (1 - w) * dist_reward + w * heading_reward.
            Both components are bounded [0, 1], so total is [0, 1].
        prev_dist: Distance at previous step (for PBRS). None on first step.
        pbrs_gamma: PBRS discount factor. 0.0 = disabled. Typical: 0.99.
        pbrs_only: If True, use ONLY the PBRS shaping signal (no base distance reward).
        improvement_weight: Multiplier for step-wise improvement bonus: w*(prev_dist - dist).
            Paper uses 10.0. 0.0 = disabled.
        return_components: If True, return (reward, components_dict) for per-component logging.

    Returns:
        Scalar reward, or (reward, components_dict) if return_components=True.
    """
    dist = np.linalg.norm(tip_pos - target_pos)
    dist_reward = np.exp(-5.0 * dist)
    heading_reward = 0.0
    pbrs_reward = 0.0
    improvement_reward = 0.0

    if pbrs_only:
        # PBRS-only mode: no base reward, only shaping signal
        total = 0.0
    elif heading_weight > 0.0 and tip_tangent is not None:
        # Direction from tip to target
        to_target = target_pos - tip_pos
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm > 1e-8:
            to_target_dir = to_target / to_target_norm
            cos_sim = np.dot(tip_tangent, to_target_dir)
            heading_reward = (1.0 + cos_sim) / 2.0  # Maps [-1, 1] → [0, 1]
        else:
            heading_reward = 1.0
        total = float((1.0 - heading_weight) * dist_reward + heading_weight * heading_reward)
    else:
        total = float(dist_reward)

    # Improvement bonus: w * (prev_dist - dist)  (paper uses w=10)
    if improvement_weight > 0.0 and prev_dist is not None:
        improvement_reward = float(improvement_weight * (prev_dist - dist))
        total += improvement_reward

    # PBRS: F(s, s') = prev_dist - γ * dist  (Φ(s) = -dist)
    if pbrs_gamma > 0.0 and prev_dist is not None:
        pbrs_reward = float(prev_dist - pbrs_gamma * dist)
        total += pbrs_reward

    if return_components:
        return total, {
            "dist_to_goal": float(dist),
            "reward_dist": float(dist_reward),
            "reward_align": float(heading_reward),
            "reward_pbrs": float(pbrs_reward),
            "reward_improve": float(improvement_reward),
        }
    return total


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
    improvement_bonus = 2.0 * improvement  # Reduced from 10.0 to improve reward SNR for PPO

    # Obstacle penetration penalty
    penalty = -contact_penalty * total_penetration

    return float(distance_reward + improvement_bonus + penalty)
