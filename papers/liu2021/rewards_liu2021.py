"""Artificial Potential Field reward for contact-aware locomotion (Liu et al., 2021).

R = ω₁·R_goal + ω₂·R_att + ω₃·R_rep

R_goal: Multi-level termination reward based on heading alignment
R_att: Alignment of velocity with attractive force (toward goal)
R_rep: Alignment of velocity with repulsive force (FIRAS potential)

The repulsive term can encourage beneficial obstacle contact,
enabling obstacle-aided locomotion where contacts provide propulsion.
"""

import numpy as np

from liu2021.configs_liu2021 import APFRewardConfig


def attractive_potential(pos: np.ndarray, goal: np.ndarray,
                         k_att: float = 1.0) -> float:
    """Attractive potential field: U_att = 0.5 * k_att * ||p - p_g||²."""
    return 0.5 * k_att * np.linalg.norm(pos - goal) ** 2


def attractive_force(pos: np.ndarray, goal: np.ndarray,
                     k_att: float = 1.0) -> np.ndarray:
    """Attractive force: F_att = -∇U_att = -k_att * (p - p_g)."""
    return -k_att * (pos - goal)


def repulsive_potential(pos: np.ndarray, obstacle_pos: np.ndarray,
                        k_rep: float = 0.5, rho_0: float = 0.05) -> float:
    """FIRAS repulsive potential for a single obstacle.

    U_rep = 0.5 * k_rep * (1/ρ - 1/ρ₀)²  if ρ ≤ ρ₀
            0                               if ρ > ρ₀

    where ρ = ||p - p_o|| is the distance to the obstacle.
    """
    rho = np.linalg.norm(pos - obstacle_pos)
    if rho <= 0:
        rho = 1e-6
    if rho > rho_0:
        return 0.0
    return 0.5 * k_rep * (1.0 / rho - 1.0 / rho_0) ** 2


def repulsive_force(pos: np.ndarray, obstacle_pos: np.ndarray,
                    k_rep: float = 0.5, rho_0: float = 0.05) -> np.ndarray:
    """Repulsive force from a single obstacle: F_rep = -∇U_rep."""
    diff = pos - obstacle_pos
    rho = np.linalg.norm(diff)
    if rho <= 0:
        rho = 1e-6
    if rho > rho_0:
        return np.zeros_like(pos)

    direction = diff / rho
    magnitude = k_rep * (1.0 / rho - 1.0 / rho_0) / (rho ** 2)
    return magnitude * direction


def total_repulsive_force(pos: np.ndarray, obstacles: np.ndarray,
                          k_rep: float = 0.5, rho_0: float = 0.05) -> np.ndarray:
    """Total repulsive force from all obstacles.

    Args:
        pos: Agent position, shape (2,).
        obstacles: Obstacle positions, shape (n, 2).
        k_rep: Repulsive force constant.
        rho_0: Activation radius.

    Returns:
        Total repulsive force, shape (2,).
    """
    F = np.zeros_like(pos)
    for obs in obstacles:
        F += repulsive_force(pos, obs, k_rep, rho_0)
    return F


def compute_goal_reward(
    head_pos: np.ndarray,
    goal_pos: np.ndarray,
    heading_angle: float,
    goal_levels: list = None,
) -> float:
    """Multi-level goal reward R_goal.

    R_goal = cos(θ_g) · Σ_{k=0}^{i} (1/l_k) · 1(ρ_g < l_k)

    where θ_g is the angle between locomotion direction and goal direction,
    l_k are the level radii, and ρ_g is the distance to goal.

    Args:
        head_pos: Head position, shape (2,).
        goal_pos: Goal position, shape (2,).
        heading_angle: Deviation angle between heading and goal direction.
        goal_levels: Accepting radii [l_0, l_1, ...].

    Returns:
        Scalar goal reward.
    """
    if goal_levels is None:
        goal_levels = [0.15, 0.10, 0.05]

    dist = np.linalg.norm(head_pos - goal_pos)
    cos_theta = np.cos(heading_angle)

    level_sum = 0.0
    for lk in goal_levels:
        if dist < lk:
            level_sum += 1.0 / lk

    return cos_theta * level_sum


def compute_apf_reward(
    pos: np.ndarray,
    vel: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    heading_angle: float,
    config: APFRewardConfig = None,
) -> float:
    """Full APF reward (Eq. 5 in paper).

    R = ω₁·R_goal + ω₂·R_att + ω₃·R_rep

    R_att = v · F_att(p)  (velocity aligned with attraction)
    R_rep = v · F_rep(p)  (velocity aligned with repulsion — can be positive
                           when contact pushes robot toward goal)

    Args:
        pos: Agent position, shape (2,).
        vel: Agent velocity, shape (2,).
        goal: Goal position, shape (2,).
        obstacles: Obstacle positions, shape (n, 2).
        heading_angle: Heading deviation from goal direction.
        config: Reward configuration.

    Returns:
        Scalar reward.
    """
    if config is None:
        config = APFRewardConfig()

    # R_goal
    R_goal = compute_goal_reward(pos, goal, heading_angle, config.goal_levels)

    # R_att = v · F_att
    F_att = attractive_force(pos, goal, config.k_att)
    R_att = np.dot(vel, F_att)

    # R_rep = v · F_rep
    F_rep = total_repulsive_force(pos, obstacles, config.k_rep, config.rho_0)
    R_rep = np.dot(vel, F_rep)

    return config.omega_1 * R_goal + config.omega_2 * R_att + config.omega_3 * R_rep
