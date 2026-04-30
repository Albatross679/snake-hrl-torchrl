"""Curriculum manager for Liu et al. (2023) goal-tracking training.

Implements the 12-level curriculum from Table III of the paper.
Each level specifies a goal distance range, heading angle range, and goal radius.
The agent advances when its success rate exceeds a threshold over a sliding window.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class CurriculumLevel:
    """Single curriculum level specification."""
    distance_min: float
    distance_max: float
    angle_min: float  # degrees
    angle_max: float  # degrees
    goal_radius: float


# Table III from Liu et al. (2023)
DEFAULT_LEVELS: List[CurriculumLevel] = [
    CurriculumLevel(0.5, 1.0,   0,  15, 0.50),   # Level 1
    CurriculumLevel(0.5, 1.5,   0,  30, 0.50),   # Level 2
    CurriculumLevel(0.5, 2.0,   0,  45, 0.45),   # Level 3
    CurriculumLevel(0.5, 2.5,   0,  60, 0.40),   # Level 4
    CurriculumLevel(0.5, 3.0,   0,  90, 0.35),   # Level 5
    CurriculumLevel(0.5, 3.5,   0, 120, 0.30),   # Level 6
    CurriculumLevel(0.5, 4.0,   0, 150, 0.25),   # Level 7
    CurriculumLevel(0.5, 4.5,   0, 180, 0.20),   # Level 8
    CurriculumLevel(0.5, 5.0,   0, 180, 0.20),   # Level 9
    CurriculumLevel(0.5, 5.5,   0, 180, 0.15),   # Level 10
    CurriculumLevel(0.5, 6.0,   0, 180, 0.15),   # Level 11
    CurriculumLevel(0.5, 6.5,   0, 180, 0.10),   # Level 12
]


class CurriculumManager:
    """Manages curriculum progression during training.

    Tracks success rate over a sliding window and advances the
    curriculum level when the threshold is exceeded.
    """

    def __init__(
        self,
        levels: List[CurriculumLevel] = None,
        success_threshold: float = 0.9,
        eval_window: int = 100,
    ):
        self.levels = levels or DEFAULT_LEVELS
        self.success_threshold = success_threshold
        self.eval_window = eval_window

        self._current_level = 0
        self._episode_results: List[bool] = []

    @property
    def current_level(self) -> int:
        return self._current_level

    @property
    def max_level(self) -> int:
        return len(self.levels) - 1

    @property
    def level_spec(self) -> CurriculumLevel:
        return self.levels[self._current_level]

    @property
    def success_rate(self) -> float:
        if not self._episode_results:
            return 0.0
        window = self._episode_results[-self.eval_window:]
        return sum(window) / len(window)

    def sample_goal(self, rng: np.random.Generator = None) -> Tuple[float, float]:
        """Sample a goal position (distance, angle) for the current level.

        Returns:
            (distance, angle_radians) tuple for goal placement relative to snake head.
        """
        if rng is None:
            rng = np.random.default_rng()

        spec = self.level_spec
        distance = rng.uniform(spec.distance_min, spec.distance_max)
        angle_deg = rng.uniform(-spec.angle_max, spec.angle_max)
        angle_rad = np.radians(angle_deg)
        return distance, angle_rad

    def goal_to_xy(
        self,
        distance: float,
        angle_rad: float,
        head_x: float,
        head_y: float,
        head_yaw: float,
    ) -> Tuple[float, float]:
        """Convert (distance, angle) goal to world (x, y) coordinates.

        Args:
            distance: Distance from head to goal.
            angle_rad: Angle offset from head heading.
            head_x, head_y: Head position in world frame.
            head_yaw: Head heading angle in radians.

        Returns:
            (goal_x, goal_y) in world coordinates.
        """
        world_angle = head_yaw + angle_rad
        goal_x = head_x + distance * np.cos(world_angle)
        goal_y = head_y + distance * np.sin(world_angle)
        return goal_x, goal_y

    def report_episode(self, success: bool) -> bool:
        """Report episode outcome and check for level advancement.

        Args:
            success: Whether the episode was successful (reached goal).

        Returns:
            True if the curriculum level was advanced.
        """
        self._episode_results.append(success)

        if len(self._episode_results) >= self.eval_window:
            if self.success_rate >= self.success_threshold:
                if self._current_level < self.max_level:
                    self._current_level += 1
                    self._episode_results.clear()
                    return True
        return False

    def reset(self) -> None:
        """Reset to level 0."""
        self._current_level = 0
        self._episode_results.clear()
