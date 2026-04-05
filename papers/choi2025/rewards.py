"""Reward functions for soft manipulator tasks (Choi & Tong, 2025).

Normalized multi-component architecture:
- Base components normalized to known ranges, then weighted
- PBRS decomposed per-objective (distance, heading), raw/unweighted
- Weights are pure importance coefficients, not scale-entangled knobs
"""

import numpy as np


def compute_follow_target_reward(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    prev_tip_pos: np.ndarray,
    tip_tangent: np.ndarray | None = None,
    prev_tip_tangent: np.ndarray | None = None,
    dist_weight: float = 1.0,
    heading_weight: float = 0.0,
    smooth_weight: float = 0.0,
    prev_dist: float | None = None,
    pbrs_gamma: float = 0.0,
    action: np.ndarray | None = None,
    prev_action: np.ndarray | None = None,
    action_dim: int = 10,
    workspace_radius: float = 1.0,
    reward_steepness: float = 5.0,
    return_components: bool = False,
) -> float | tuple[float, dict]:
    """Reward for following a moving target.

    Architecture:
        total = Σ wᵢ · normalize(rᵢ)   (base components, normalized then weighted)
              + Σ (γ·Φⱼ(s') - Φⱼ(s))   (PBRS components, raw/unweighted)

    Base components (all normalized to known ranges):
        - Distance:  exp(-k·dist)             → [0, 1], weight=1.0 (always on)
        - Heading:   (1+cos_sim)/2          → [0, 1], weight=heading_weight
        - Smoothness: -||Δa||²/(2·action_dim) → [-1, 0], weight=smooth_weight

    PBRS components (policy-invariant, Ng et al. 1999):
        - Distance PBRS: Φ(s) = -dist/workspace_radius
        - Heading PBRS:  Φ(s) = cos_sim(tip_tangent, to_target_dir)

    Args:
        tip_pos: Current tip position, shape (3,).
        target_pos: Target position, shape (3,).
        prev_tip_pos: Previous tip position, shape (3,).
        tip_tangent: Tip tangent (unit vector), shape (3,).
        prev_tip_tangent: Previous tip tangent, shape (3,). For heading PBRS.
        heading_weight: Weight for heading base component. 0.0 = disabled.
        smooth_weight: Weight for action smoothness penalty. 0.0 = disabled.
        prev_dist: Distance at previous step (for PBRS). None on first step.
        pbrs_gamma: PBRS discount factor. 0.0 = disabled. Typical: 0.99.
        action: Current action, shape (action_dim,).
        prev_action: Previous action, shape (action_dim,). None on first step.
        action_dim: Action dimensionality (for smoothness normalization).
        workspace_radius: Workspace radius for distance PBRS normalization.
        return_components: If True, return (reward, components_dict).

    Returns:
        Scalar reward, or (reward, components_dict) if return_components=True.
    """
    dist = np.linalg.norm(tip_pos - target_pos)

    # === Base components (normalized, weighted) ===
    dist_reward = np.exp(-reward_steepness * dist)  # [0, 1]

    heading_reward = 0.0
    cos_sim = 0.0
    if heading_weight > 0.0 and tip_tangent is not None:
        to_target = target_pos - tip_pos
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm > 1e-8:
            to_target_dir = to_target / to_target_norm
            cos_sim = float(np.dot(tip_tangent, to_target_dir))
            heading_reward = (1.0 + cos_sim) / 2.0  # [0, 1]
        else:
            cos_sim = 1.0
            heading_reward = 1.0

    smooth_penalty = 0.0
    if smooth_weight > 0.0 and action is not None and prev_action is not None:
        delta = action - prev_action
        smooth_penalty = float(-np.sum(delta ** 2) / (2 * action_dim))  # [-1, 0]

    total = float(dist_weight * dist_reward
                  + heading_weight * heading_reward
                  + smooth_weight * smooth_penalty)

    # === PBRS (raw, unweighted, naturally centered) ===
    pbrs_dist = 0.0
    pbrs_head = 0.0

    if pbrs_gamma > 0.0 and prev_dist is not None:
        phi_dist = -dist / workspace_radius
        phi_dist_prev = -prev_dist / workspace_radius
        pbrs_dist = float(pbrs_gamma * phi_dist - phi_dist_prev)
        total += pbrs_dist

        if (tip_tangent is not None and prev_tip_tangent is not None):
            # Current heading potential
            to_target = target_pos - tip_pos
            to_target_norm = np.linalg.norm(to_target)
            if to_target_norm > 1e-8:
                to_target_dir = to_target / to_target_norm
                phi_head = float(np.dot(tip_tangent, to_target_dir))
            else:
                phi_head = 1.0

            # Previous heading potential
            prev_to_target = target_pos - prev_tip_pos
            prev_to_target_norm = np.linalg.norm(prev_to_target)
            if prev_to_target_norm > 1e-8:
                prev_to_target_dir = prev_to_target / prev_to_target_norm
                phi_head_prev = float(np.dot(prev_tip_tangent, prev_to_target_dir))
            else:
                phi_head_prev = 1.0

            pbrs_head = float(pbrs_gamma * phi_head - phi_head_prev)
            total += pbrs_head

    if return_components:
        return total, {
            "dist_to_goal": float(dist),
            "reward_dist": float(dist_reward),
            "reward_align": float(heading_reward),
            "reward_pbrs": float(pbrs_dist),
            "reward_pbrs_head": float(pbrs_head),
            "reward_smooth": float(smooth_penalty),
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
    max_penetration: float = 0.1,
) -> float:
    """Reward for reaching target while avoiding obstacles.

    Components (normalized ranges):
        - Distance:    exp(-5·dist)                          → [0, 1]
        - Improvement: clip(prev_dist - dist, -0.1, 0.1)/0.1 → [-1, 1]
        - Contact:     -total_penetration/max_penetration     → [-1, 0] (clamped)

    Args:
        tip_pos: Current tip position, shape (3,).
        target_pos: Target position, shape (3,).
        prev_tip_pos: Previous tip position, shape (3,).
        total_penetration: Sum of penetration depths across all nodes.
        contact_penalty: Weight for contact penalty component.
        max_penetration: Maximum expected penetration for normalization.

    Returns:
        Scalar reward.
    """
    dist = np.linalg.norm(tip_pos - target_pos)
    prev_dist = np.linalg.norm(prev_tip_pos - target_pos)

    # Distance reward: [0, 1]
    distance_reward = np.exp(-5.0 * dist)

    # Improvement bonus (normalized): [-1, 1]
    improvement = np.clip(prev_dist - dist, -0.1, 0.1) / 0.1
    improvement_bonus = 2.0 * improvement

    # Contact penalty (normalized): [-1, 0]
    normalized_pen = min(total_penetration / max(max_penetration, 1e-8), 1.0)
    penalty = -contact_penalty * normalized_pen

    return float(distance_reward + improvement_bonus + penalty)
