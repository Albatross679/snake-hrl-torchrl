"""Snake body geometry configuration."""

from dataclasses import dataclass


@dataclass
class GeometryConfig:
    """Snake body geometry (Body Representation).

    Defines the physical dimensions and discretization of the snake body.
    Prey parameters are NOT included here — they are scene setup on PhysicsConfig.
    """

    snake_length: float = 1.0  # Total length of snake (meters)
    snake_radius: float = 0.001  # Radius of snake body (meters)
    num_segments: int = 20  # Number of discrete segments

    @property
    def num_nodes(self) -> int:
        """Number of nodes (num_segments + 1)."""
        return self.num_segments + 1

    @property
    def num_joints(self) -> int:
        """Number of internal joints (num_segments - 1)."""
        return self.num_segments - 1

    @property
    def segment_length(self) -> float:
        """Length of each segment."""
        return self.snake_length / self.num_segments
