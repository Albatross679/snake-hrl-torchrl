"""Potential-field reward for snake locomotion (following Liu et al. 2023, eq. 14).

R = c_v * v_g  +  c_g * v_g * cos(theta_g) / dist

where:
    v_g:       velocity toward goal (projected onto CoM-to-goal direction)
    theta_g:   heading angle error to goal (radians)
    dist:      distance from CoM to goal
"""

import math

from locomotion_elastica.config import LocomotionRewardConfig


def compute_goal_reward(
    config: LocomotionRewardConfig,
    v_g: float,
    dist_to_goal: float,
    theta_g: float,
) -> float:
    """Compute potential-field reward.

    Args:
        config: Reward coefficients (c_v, c_g).
        v_g: Velocity toward goal (positive = approaching).
        dist_to_goal: Current distance to goal.
        theta_g: Heading angle error to goal (radians).

    Returns:
        Scalar reward.
    """
    # Velocity reward: encourage moving toward goal
    reward = config.c_v * v_g

    # Potential-field term: velocity aligned with attractive force / distance
    epsilon = 1e-6
    if dist_to_goal > epsilon:
        reward += config.c_g * v_g * math.cos(theta_g) / dist_to_goal

    return reward
