"""Reward functions for hierarchical path following (Liu, Guo & Fang, 2022).

r_t = r_p + r_e - p_h

where:
    r_p: Path proximity reward (approach desired path)
    r_e: Endpoint progress reward (move toward target on path)
    p_h: Visual localization stabilization penalty (excessive head swings)
"""

import numpy as np


def compute_path_reward(
    dist_to_path: float,
    d1: float = 0.1,
    d2: float = 0.5,
    c_p: float = 1.0,
) -> float:
    """Path proximity reward r_p (Eq. 10).

    r_p = c_p                           if |d^p| < d₁
          c_p · exp(d₁ - |d^p|)         if d₁ ≤ |d^p| ≤ d₂
          0                              if |d^p| > d₂

    Args:
        dist_to_path: Perpendicular distance to desired path |d^p|.
        d1: Inner threshold (full reward).
        d2: Outer threshold (zero reward).
        c_p: Reward weight.

    Returns:
        Scalar reward.
    """
    d = abs(dist_to_path)
    if d < d1:
        return c_p
    elif d <= d2:
        return c_p * np.exp(d1 - d)
    else:
        return 0.0


def compute_endpoint_reward(
    dist_to_target_prev: float,
    dist_to_target_curr: float,
    c_e: float = 0.5,
) -> float:
    """Endpoint progress reward r_e (Eq. 11).

    r_e = c_e · (d^e_t - d^e_{t+1})

    Positive when robot moves closer to the target point on the path.

    Args:
        dist_to_target_prev: Previous distance to target.
        dist_to_target_curr: Current distance to target.
        c_e: Reward weight.

    Returns:
        Scalar reward.
    """
    return c_e * (dist_to_target_prev - dist_to_target_curr)


def compute_head_swing_penalty(
    head_angle_current: float,
    head_angle_previous: float,
    angle_threshold: float = 0.3,
    c_h: float = -0.1,
) -> float:
    """Visual localization stabilization penalty p_h (Eq. 12).

    p_h = c_h      if |φ¹_{t+1} - φ¹_t| ≥ φ_*
          0         if |φ¹_{t+1} - φ¹_t| < φ_*

    Penalizes excessive head swings that would cause the pan-tilt
    camera to lose the visual marker.

    Args:
        head_angle_current: Current head angle φ¹_{t+1}.
        head_angle_previous: Previous head angle φ¹_t.
        angle_threshold: Maximum allowed head swing φ_*.
        c_h: Penalty value (negative).

    Returns:
        Scalar penalty (negative or zero).
    """
    swing = abs(head_angle_current - head_angle_previous)
    if swing >= angle_threshold:
        return c_h
    return 0.0


def compute_total_reward(
    dist_to_path: float,
    dist_to_target_prev: float,
    dist_to_target_curr: float,
    head_angle_current: float,
    head_angle_previous: float,
    c_p: float = 1.0,
    c_e: float = 0.5,
    d1: float = 0.1,
    d2: float = 0.5,
    angle_threshold: float = 0.3,
    c_h: float = -0.1,
) -> float:
    """Total reward r_t = r_p + r_e - p_h (Eq. 9).

    Returns:
        Scalar reward.
    """
    r_p = compute_path_reward(dist_to_path, d1, d2, c_p)
    r_e = compute_endpoint_reward(dist_to_target_prev, dist_to_target_curr, c_e)
    p_h = compute_head_swing_penalty(
        head_angle_current, head_angle_previous, angle_threshold, c_h
    )

    return r_p + r_e - p_h
