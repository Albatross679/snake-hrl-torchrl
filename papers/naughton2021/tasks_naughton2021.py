"""Benchmark task definitions for Elastica-RL (Naughton et al., 2021).

Four benchmark cases:

Case 1 — 3D Tracking: Track a randomly moving target in 3D space.
Case 2 — Reaching with Orientation: Reach a stationary target with
    both position and orientation matching.
Case 3 — Structured Obstacles: Navigate tip through a 2x4 grid of
    8 cylindrical obstacles to reach a target.
Case 4 — Unstructured Obstacles: Navigate through 12 randomly placed
    cylindrical obstacles.
"""

from typing import List, Tuple

import numpy as np

from naughton2021.configs_naughton2021 import (
    BenchmarkCase,
    ObstacleLayoutConfig,
    TargetConfig,
)


class TargetManager:
    """Generates and manages targets for each benchmark case."""

    def __init__(self, config: TargetConfig, rng: np.random.Generator):
        self.config = config
        self._rng = rng

        self.position = np.zeros(3)
        self.orientation = np.array([0.0, 0.0, 1.0])
        self._velocity = np.zeros(3)

    def sample(self, case: BenchmarkCase, rod_length: float = 1.0) -> None:
        """Sample a new target for the given case."""
        if case == BenchmarkCase.CASE1_TRACKING:
            self._sample_moving_target(rod_length)
        elif case == BenchmarkCase.CASE2_REACHING:
            self._sample_static_target_with_orientation(rod_length)
        else:
            self._sample_static_target(rod_length)

    def _sample_moving_target(self, rod_length: float) -> None:
        """Sample initial position and velocity for moving target."""
        r = self.config.workspace_radius * rod_length
        # Random position within workspace
        theta = self._rng.uniform(0, 2 * np.pi)
        phi = self._rng.uniform(0, np.pi)
        radius = self._rng.uniform(0.3 * r, r)
        self.position = np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ])
        # Random velocity direction
        v_dir = self._rng.standard_normal(3)
        v_dir /= np.linalg.norm(v_dir) + 1e-8
        self._velocity = v_dir * self.config.target_speed

    def _sample_static_target(self, rod_length: float) -> None:
        """Sample stationary target position."""
        r = self.config.workspace_radius * rod_length
        theta = self._rng.uniform(0, 2 * np.pi)
        phi = self._rng.uniform(np.pi / 6, np.pi / 2)
        radius = self._rng.uniform(0.3 * r, r)
        self.position = np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ])
        self._velocity = np.zeros(3)

    def _sample_static_target_with_orientation(self, rod_length: float) -> None:
        """Sample stationary target with desired orientation."""
        self._sample_static_target(rod_length)
        # Random target orientation (unit vector)
        orient = self._rng.standard_normal(3)
        self.orientation = orient / (np.linalg.norm(orient) + 1e-8)

    def step(self, dt: float, workspace_radius: float = 0.8) -> None:
        """Move target (for tracking case)."""
        self.position += self._velocity * dt

        # Bounce off workspace boundary
        r = np.linalg.norm(self.position)
        if r > workspace_radius:
            # Reflect velocity
            normal = self.position / r
            self._velocity -= 2 * np.dot(self._velocity, normal) * normal
            self.position = normal * workspace_radius * 0.99


class ObstacleManager:
    """Manages obstacle layout for Cases 3 and 4."""

    def __init__(self, config: ObstacleLayoutConfig, rng: np.random.Generator):
        self.config = config
        self._rng = rng
        self.positions: List[np.ndarray] = []
        self.radii: List[float] = []

    def setup(self, case: BenchmarkCase) -> None:
        """Set up obstacles for the given case."""
        self.positions = []
        self.radii = []

        if case == BenchmarkCase.CASE3_STRUCTURED:
            self._setup_structured()
        elif case == BenchmarkCase.CASE4_UNSTRUCTURED:
            self._setup_unstructured()

    def _setup_structured(self) -> None:
        """Create 2x4 grid of obstacles (Case 3)."""
        cfg = self.config
        spacing = cfg.grid_spacing

        for row in range(cfg.grid_rows):
            for col in range(cfg.grid_cols):
                pos = np.array([
                    (col - cfg.grid_cols / 2 + 0.5) * spacing,
                    (row - cfg.grid_rows / 2 + 0.5) * spacing,
                    0.3 + col * 0.15,  # Staggered height
                ])
                self.positions.append(pos)
                self.radii.append(cfg.obstacle_radius)

    def _setup_unstructured(self) -> None:
        """Create 12 randomly placed obstacles (Case 4)."""
        cfg = self.config
        for _ in range(12):
            pos = np.array([
                self._rng.uniform(-0.3, 0.3),
                self._rng.uniform(-0.3, 0.3),
                self._rng.uniform(0.2, 0.8),
            ])
            self.positions.append(pos)
            self.radii.append(cfg.obstacle_radius)

    def compute_penetrations(self, node_positions: np.ndarray) -> np.ndarray:
        """Compute penetration depths for all rod nodes vs all obstacles.

        Args:
            node_positions: Rod node positions, shape (num_nodes, 3).

        Returns:
            Total penetration per node, shape (num_nodes,).
        """
        if not self.positions:
            return np.zeros(node_positions.shape[0])

        penetrations = np.zeros(node_positions.shape[0])
        for obs_pos, obs_r in zip(self.positions, self.radii):
            dists = np.linalg.norm(node_positions - obs_pos, axis=1)
            pen = np.maximum(0, obs_r - dists)
            penetrations += pen

        return penetrations
