"""Snake and prey geometry for physics simulation.

This module provides geometry utilities for the snake robot simulation.
Snake geometry is now managed internally by DisMech, but this module
provides prey geometry and contact/wrapping utilities.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Protocol
import numpy as np

from src.configs.physics import PhysicsConfig


class SnakeGeometryProtocol(Protocol):
    """Protocol defining the interface for snake geometry.

    This is used for duck-typing with both the legacy SnakeGeometry
    dataclass and the new SnakeGeometryAdapter from DisMech.
    """

    positions: np.ndarray
    num_segments: int

    @property
    def num_nodes(self) -> int:
        ...

    @property
    def radii(self) -> np.ndarray:
        ...


@dataclass
class SnakeGeometry:
    """Geometry representation of a snake body.

    Note: This class is kept for backward compatibility. The new DisMech-based
    implementation uses SnakeGeometryAdapter internally, which implements
    the same interface.
    """

    # Node positions (n_nodes, 3)
    positions: np.ndarray

    # Rest lengths between nodes
    rest_lengths: np.ndarray

    # Physical properties
    radii: np.ndarray  # Radius at each node
    masses: np.ndarray  # Mass at each node

    # Connectivity
    num_segments: int

    @property
    def num_nodes(self) -> int:
        return len(self.positions)

    @property
    def total_length(self) -> float:
        return float(np.sum(self.rest_lengths))

    def get_segment_vectors(self) -> np.ndarray:
        """Get vectors along each segment."""
        return np.diff(self.positions, axis=0)

    def get_segment_lengths(self) -> np.ndarray:
        """Get current length of each segment."""
        vectors = self.get_segment_vectors()
        return np.linalg.norm(vectors, axis=1)

    def get_curvatures(self) -> np.ndarray:
        """Compute discrete curvature at each internal node."""
        if self.num_nodes < 3:
            return np.array([])

        curvatures = []
        for i in range(1, self.num_nodes - 1):
            v1 = self.positions[i] - self.positions[i - 1]
            v2 = self.positions[i + 1] - self.positions[i]

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm < 1e-8 or v2_norm < 1e-8:
                curvatures.append(0.0)
                continue

            v1 = v1 / v1_norm
            v2 = v2 / v2_norm

            # Curvature from angle between consecutive segments
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            avg_length = (v1_norm + v2_norm) / 2

            curvature = angle / avg_length if avg_length > 1e-8 else 0.0
            curvatures.append(curvature)

        return np.array(curvatures)


@dataclass
class PreyGeometry:
    """Geometry representation of cylindrical prey."""

    position: np.ndarray  # Center position (3,)
    orientation: np.ndarray  # Orientation axis (3,)
    radius: float
    length: float
    mass: float

    def get_surface_points(self, num_points: int = 32) -> np.ndarray:
        """Get points on cylinder surface for collision detection."""
        # Create points around the cylinder
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        z = np.linspace(-self.length / 2, self.length / 2, num_points // 4)

        points = []
        for zi in z:
            for ti in theta:
                # Local coordinates
                x = self.radius * np.cos(ti)
                y = self.radius * np.sin(ti)

                # Transform to world coordinates
                point = self._local_to_world(np.array([x, y, zi]))
                points.append(point)

        return np.array(points)

    def _local_to_world(self, local_point: np.ndarray) -> np.ndarray:
        """Transform point from local to world coordinates."""
        # Simplified: assumes orientation is along z-axis
        # For full implementation, compute rotation matrix from orientation
        return local_point + self.position

    def distance_to_point(self, point: np.ndarray) -> float:
        """Compute distance from point to cylinder surface."""
        # Project point onto cylinder axis
        to_point = point - self.position

        # Distance along axis
        axis = self.orientation / np.linalg.norm(self.orientation)
        along_axis = np.dot(to_point, axis)

        # Clamp to cylinder length
        along_axis_clamped = np.clip(along_axis, -self.length / 2, self.length / 2)

        # Closest point on axis
        closest_on_axis = self.position + along_axis_clamped * axis

        # Radial distance
        radial_vec = point - closest_on_axis
        radial_dist = np.linalg.norm(radial_vec)

        # Distance to surface
        return radial_dist - self.radius


def create_snake_geometry(
    config: PhysicsConfig,
    initial_position: Optional[np.ndarray] = None,
    initial_direction: Optional[np.ndarray] = None,
) -> SnakeGeometry:
    """Create snake geometry from configuration.

    .. deprecated::
        This function is deprecated. Snake geometry is now managed internally
        by DisMech in the SnakeRobot class. This function is kept for backward
        compatibility and testing purposes only.

    Args:
        config: Physics configuration with snake parameters
        initial_position: Starting position of snake head (default: origin)
        initial_direction: Initial direction of snake body (default: +x)

    Returns:
        SnakeGeometry object with initialized positions and properties
    """
    import warnings
    warnings.warn(
        "create_snake_geometry is deprecated. Snake geometry is now managed "
        "internally by DisMech in the SnakeRobot class.",
        DeprecationWarning,
        stacklevel=2,
    )
    if initial_position is None:
        initial_position = np.array([0.0, 0.0, config.snake_radius])

    if initial_direction is None:
        initial_direction = np.array([1.0, 0.0, 0.0])

    # Normalize direction
    direction = initial_direction / np.linalg.norm(initial_direction)

    # Create node positions along the snake
    num_nodes = config.num_segments + 1
    segment_length = config.snake_length / config.num_segments

    positions = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        positions[i] = initial_position + i * segment_length * direction

    # Rest lengths
    rest_lengths = np.full(config.num_segments, segment_length)

    # Radii (constant for now, could vary along body)
    radii = np.full(num_nodes, config.snake_radius)

    # Masses (computed from density and volume)
    segment_volume = np.pi * config.snake_radius**2 * segment_length
    segment_mass = config.density * segment_volume
    masses = np.full(num_nodes, segment_mass)
    # Distribute mass to nodes (half from each adjacent segment)
    masses[0] /= 2
    masses[-1] /= 2

    return SnakeGeometry(
        positions=positions,
        rest_lengths=rest_lengths,
        radii=radii,
        masses=masses,
        num_segments=config.num_segments,
    )


def create_prey_geometry(
    config: PhysicsConfig,
    position: Optional[np.ndarray] = None,
    orientation: Optional[np.ndarray] = None,
) -> PreyGeometry:
    """Create prey geometry from configuration.

    Args:
        config: Physics configuration with prey parameters
        position: Center position of prey (default: in front of snake)
        orientation: Orientation axis of cylinder (default: vertical)

    Returns:
        PreyGeometry object
    """
    if position is None:
        position = np.array([config.snake_length + config.prey_radius * 2, 0.0, config.prey_length / 2])

    if orientation is None:
        orientation = np.array([0.0, 0.0, 1.0])

    # Compute mass from density and volume
    volume = np.pi * config.prey_radius**2 * config.prey_length
    mass = config.density * volume

    return PreyGeometry(
        position=position.copy(),
        orientation=orientation / np.linalg.norm(orientation),
        radius=config.prey_radius,
        length=config.prey_length,
        mass=mass,
    )


def compute_contact_points(
    snake: SnakeGeometry, prey: PreyGeometry, contact_threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute contact points between snake and prey.

    Args:
        snake: Snake geometry
        prey: Prey geometry
        contact_threshold: Distance threshold for contact

    Returns:
        Tuple of (contact_mask, contact_distances) for each snake node
    """
    contact_mask = np.zeros(snake.num_nodes, dtype=bool)
    contact_distances = np.zeros(snake.num_nodes)

    for i, pos in enumerate(snake.positions):
        dist = prey.distance_to_point(pos)
        contact_distances[i] = dist
        contact_mask[i] = dist < contact_threshold + snake.radii[i]

    return contact_mask, contact_distances


def compute_wrap_angle(snake: SnakeGeometry, prey: PreyGeometry) -> float:
    """Compute total angle wrapped around prey.

    Returns angle in radians (2*pi = one full wrap).
    """
    # Project snake onto plane perpendicular to prey axis
    axis = prey.orientation / np.linalg.norm(prey.orientation)

    total_angle = 0.0
    for i in range(len(snake.positions) - 1):
        # Get positions relative to prey center
        p1 = snake.positions[i] - prey.position
        p2 = snake.positions[i + 1] - prey.position

        # Project onto plane perpendicular to prey axis
        p1_proj = p1 - np.dot(p1, axis) * axis
        p2_proj = p2 - np.dot(p2, axis) * axis

        # Compute angle between projected points
        norm1 = np.linalg.norm(p1_proj)
        norm2 = np.linalg.norm(p2_proj)

        if norm1 < 1e-8 or norm2 < 1e-8:
            continue

        p1_proj /= norm1
        p2_proj /= norm2

        # Signed angle using cross product
        cross = np.cross(p1_proj, p2_proj)
        sin_angle = np.dot(cross, axis)
        cos_angle = np.clip(np.dot(p1_proj, p2_proj), -1.0, 1.0)

        angle = np.arctan2(sin_angle, cos_angle)
        total_angle += angle

    return total_angle
