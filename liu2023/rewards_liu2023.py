"""Reward functions for Liu et al. (2023) goal-tracking locomotion.

Implements equation (14) from the paper:
    R = c_v * v_g + c_g * U + curriculum_bonus

where:
    v_g: velocity toward goal (projected)
    U: potential field term (velocity dot attractive force / distance)
    curriculum_bonus: proximity bonus across curriculum levels
"""

import math
from typing import List

import numpy as np


def compute_goal_tracking_reward(
    v_g: float,
    dist_to_goal: float,
    theta_g: float,
    goal_radii: List[float],
    c_v: float = 1.0,
    c_g: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """Compute the potential field reward (eq. 14).

    Args:
        v_g: Velocity toward goal (positive = approaching).
        dist_to_goal: Current distance to goal.
        theta_g: Heading angle error to goal (radians).
        goal_radii: List of goal radii from all curriculum levels
            reached so far (largest to smallest).
        c_v: Velocity reward coefficient.
        c_g: Goal proximity reward coefficient.
        epsilon: Small constant to avoid division by zero.

    Returns:
        Scalar reward.
    """
    # Velocity reward: encourage moving toward goal
    reward_velocity = c_v * v_g

    # Potential field: velocity aligned with attractive force / distance
    cos_theta = math.cos(theta_g)
    if dist_to_goal > epsilon:
        reward_potential = c_g * v_g * cos_theta / dist_to_goal
    else:
        reward_potential = 0.0

    # Curriculum bonus: sum of 1/r_k for each radius r_k where dist < r_k
    reward_bonus = 0.0
    for r_k in goal_radii:
        if dist_to_goal < r_k and r_k > epsilon:
            reward_bonus += c_g * cos_theta * (1.0 / r_k)

    return reward_velocity + reward_potential + reward_bonus


def check_goal_reached(dist_to_goal: float, goal_radius: float) -> bool:
    """Check if the snake has reached the goal."""
    return dist_to_goal <= goal_radius


def check_starvation(v_g_history: List[float], timeout: int = 60) -> bool:
    """Check starvation condition: v_g negative for timeout consecutive steps."""
    if len(v_g_history) < timeout:
        return False
    return all(v < 0 for v in v_g_history[-timeout:])
