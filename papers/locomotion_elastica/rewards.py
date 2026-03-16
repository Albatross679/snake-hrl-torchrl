"""Distance-based potential reward for snake locomotion.

R = c_dist * (prev_dist - curr_dist) + c_align * cos(theta_g) + goal_bonus

where:
    prev_dist: previous distance to goal
    curr_dist: current distance to goal
    theta_g:   heading angle error to goal (radians)
"""

import math

from locomotion_elastica.config import LocomotionRewardConfig


def compute_goal_reward(
    config: LocomotionRewardConfig,
    prev_dist: float,
    curr_dist: float,
    theta_g: float,
    goal_reached: bool,
) -> float:
    """Compute distance-based potential reward.

    Args:
        config: Reward coefficients.
        prev_dist: Previous distance to goal.
        curr_dist: Current distance to goal.
        theta_g: Heading angle error to goal (radians).
        goal_reached: Whether the goal was reached this step.

    Returns:
        Scalar reward.
    """
    # Distance reduction: positive when getting closer
    reward = config.c_dist * (prev_dist - curr_dist)

    # Heading alignment bonus: encourage facing the goal
    reward += config.c_align * math.cos(theta_g)

    # Goal reached bonus
    if goal_reached:
        reward += config.goal_bonus

    return reward
