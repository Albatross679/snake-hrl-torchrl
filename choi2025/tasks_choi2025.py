"""Task logic for soft manipulator tasks (Choi & Tong, 2025).

Provides target generation and obstacle management for the four tasks:
- FOLLOW_TARGET: moving target that the tip must track
- INVERSE_KINEMATICS: static target (position + orientation)
- TIGHT_OBSTACLES: navigate through a narrow gap to reach target
- RANDOM_OBSTACLES: reach target while avoiding random spheres
"""

import numpy as np

from choi2025.configs_choi2025 import (
    ObstacleConfig,
    TargetConfig,
    TaskType,
)


class TargetGenerator:
    """Generates and updates targets within the manipulator workspace.

    The workspace is a disk/sphere of radius ≈ rod length, centered at the
    clamped base.  Targets are sampled in reachable regions.
    """

    def __init__(self, config: TargetConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0])  # tangent direction
        self._velocity = np.zeros(3)

    def sample(self, task: TaskType) -> None:
        """Sample a new target appropriate for the given task."""
        r = self.rng.uniform(self.config.min_radius, self.config.max_radius)
        theta = self.rng.uniform(0, 2 * np.pi)
        phi = self.rng.uniform(0, np.pi)

        # 3D spherical coordinates
        self.position = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])

        if task == TaskType.INVERSE_KINEMATICS:
            # Random orientation for IK task
            orient = self.rng.standard_normal(3)
            orient /= np.linalg.norm(orient) + 1e-8
            self.orientation = orient

        if task == TaskType.FOLLOW_TARGET:
            # Random initial velocity direction
            vel_dir = self.rng.standard_normal(3)
            vel_dir /= np.linalg.norm(vel_dir) + 1e-8
            self._velocity = vel_dir * self.config.target_speed

    def step(self, dt: float) -> None:
        """Move target for follow_target task.

        The target bounces off the workspace boundary to stay reachable.
        """
        new_pos = self.position + self._velocity * dt
        r = np.linalg.norm(new_pos)

        if r > self.config.max_radius:
            # Reflect velocity off workspace boundary
            normal = new_pos / (r + 1e-8)
            self._velocity -= 2 * np.dot(self._velocity, normal) * normal
            new_pos = self.position + self._velocity * dt

        self.position = new_pos


class ObstacleManager:
    """Manages obstacles for obstacle avoidance tasks.

    For TIGHT_OBSTACLES: creates a fixed narrow gap configuration.
    For RANDOM_OBSTACLES: samples random spherical obstacles.
    """

    def __init__(self, config: ObstacleConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        # Each obstacle: (position, radius)
        self.positions: np.ndarray = np.zeros((0, 3))
        self.radii: np.ndarray = np.zeros(0)

    def setup(self, task: TaskType) -> None:
        """Generate obstacle layout for the given task."""
        if task == TaskType.TIGHT_OBSTACLES:
            self._setup_tight_obstacles()
        elif task == TaskType.RANDOM_OBSTACLES:
            self._setup_random_obstacles()
        else:
            self.positions = np.zeros((0, 3))
            self.radii = np.zeros(0)

    def _setup_tight_obstacles(self) -> None:
        """Create a narrow gap from two large obstacles.

        Two obstacles are placed symmetrically about the x-axis at mid-range,
        forming a gap of width `gap_width`.
        """
        gap_half = self.config.gap_width / 2
        r = self.config.obstacle_radius
        dist = (self.config.min_distance + self.config.max_distance) / 2

        self.positions = np.array([
            [dist, gap_half + r, 0.0],
            [dist, -(gap_half + r), 0.0],
        ])
        self.radii = np.array([r, r])

    def _setup_random_obstacles(self) -> None:
        """Sample random spherical obstacles in the workspace."""
        n = self.config.num_obstacles
        positions = []
        radii = []

        for _ in range(n):
            r = self.rng.uniform(self.config.min_distance, self.config.max_distance)
            theta = self.rng.uniform(0, 2 * np.pi)
            phi = self.rng.uniform(0.3, np.pi - 0.3)  # avoid poles

            pos = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ])
            positions.append(pos)
            radii.append(self.config.obstacle_radius)

        self.positions = np.array(positions) if positions else np.zeros((0, 3))
        self.radii = np.array(radii) if radii else np.zeros(0)

    def compute_penetrations(self, rod_positions: np.ndarray) -> np.ndarray:
        """Compute penetration depths between rod nodes and obstacles.

        Args:
            rod_positions: Rod node positions, shape (num_nodes, 3).

        Returns:
            Total penetration depth per node, shape (num_nodes,).
            Positive values indicate penetration.
        """
        if len(self.positions) == 0:
            return np.zeros(rod_positions.shape[0])

        penetrations = np.zeros(rod_positions.shape[0])

        for obs_pos, obs_r in zip(self.positions, self.radii):
            # Distance from each rod node to obstacle center
            diffs = rod_positions - obs_pos[np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=1)

            # Penetration = max(0, obstacle_radius - distance)
            pen = np.maximum(0.0, obs_r - dists)
            penetrations += pen

        return penetrations
