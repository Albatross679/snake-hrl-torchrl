"""Reward functions for locomotion tasks (Bing et al., IJCAI 2019).

Faithfully ported from the original implementation.
"""

import numpy as np


def compute_power_velocity_reward(
    target_v: float,
    velocity: float,
    power_normalized: float,
) -> float:
    """Power-velocity reward from the original paper.

    Maximizes velocity tracking while minimizing energy usage.

    Args:
        target_v: Target velocity (m/s).
        velocity: Measured velocity toward target (m/s).
        power_normalized: Normalized power consumption in [0, 1].

    Returns:
        Scalar reward (product of velocity and power terms).
    """
    a1 = 0.2
    a2 = 0.2
    rew_v = (1.0 - np.abs(target_v - velocity) / a1) ** (1.0 / a2)

    b1 = 0.6
    rew_p = np.abs(1.0 - power_normalized) ** (b1 ** (-2.0))

    reward = float(rew_v * rew_p)
    return reward


def compute_target_tracking_reward(
    target_distance: float,
    dist_before: float,
    dist_after: float,
    target_dist_min: float = 2.0,
    target_dist_max: float = 6.0,
) -> float:
    """Target-tracking reward from the original paper.

    Reward is proportional to improvement in maintaining target distance.

    Args:
        target_distance: Desired distance from head to target (m).
        dist_before: Distance to target before step.
        dist_after: Distance to target after step.
        target_dist_min: Minimum distance bound for normalization.
        target_dist_max: Maximum distance bound for normalization.

    Returns:
        Scalar reward.
    """
    distance_range = (target_dist_max - target_dist_min) / 2.0
    diff_before = np.abs(target_distance - dist_before)
    diff_after = np.abs(target_distance - dist_after)
    reward = float((diff_before - diff_after) / distance_range)
    return reward


def compute_energy_normalized(
    sensor_actuatorfrcs: np.ndarray,
    joint_velocities: np.ndarray,
    actuator_gear: float = 1.0,
    force_max: float = 20.0,
    max_joint_vel: float = 25.0,
) -> tuple:
    """Compute normalized energy usage matching original implementation.

    Args:
        sensor_actuatorfrcs: Raw actuator force sensor readings, shape (8,).
        joint_velocities: Joint angular velocities, shape (8,).
        actuator_gear: Actuator gear ratio (1.0 in original).
        force_max: Maximum actuator force (20.0 N in original).
        max_joint_vel: Maximum joint velocity for normalization.

    Returns:
        Tuple of (power_normalized, power_absolute).
    """
    actuator_torques = np.array(sensor_actuatorfrcs) * actuator_gear
    actuator_energy = np.mean(np.abs(actuator_torques * joint_velocities))
    actuator_energies_max = force_max * actuator_gear * max_joint_vel
    power_normalized = actuator_energy / actuator_energies_max
    return float(power_normalized), float(actuator_energy)
