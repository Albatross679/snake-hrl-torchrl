"""Virtual chassis and goal-relative feature extraction.

These extractors compute body-frame state representations:
- VirtualChassisExtractor: Center of gravity, orientation, angular velocity
- GoalRelativeExtractor: Direction and distance to prey in body frame
"""

from typing import Any, Dict

import numpy as np

from .extractors import FeatureExtractor


class VirtualChassisExtractor(FeatureExtractor):
    """Extract virtual chassis state from snake positions and velocities.

    The virtual chassis is a body-fixed reference frame defined by:
    - Center of gravity (CoG) of the snake body
    - Principal axis (head-to-tail direction)
    - Angular velocity around vertical axis

    Output features (9 dims):
        [0:3]: Center of gravity position (x, y, z)
        [3:6]: Body orientation (forward direction unit vector)
        [6:9]: Angular velocity (approximated from position changes)

    Note: Angular velocity computation requires velocity information.
    """

    def __init__(self, normalize: bool = True, workspace_size: float = 2.0):
        """Initialize virtual chassis extractor.

        Args:
            normalize: If True, normalize positions to workspace scale
            workspace_size: Expected workspace size for normalization
        """
        self.normalize = normalize
        self.workspace_size = workspace_size
        self._max_angular_velocity = 5.0  # rad/s expected max

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (CoG + orientation + angular_vel)."""
        return 9

    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract virtual chassis features from state.

        Args:
            state: State dict containing 'positions' and 'velocities' keys

        Returns:
            Feature vector of shape (9,)
        """
        positions = state.get("positions", None)
        velocities = state.get("velocities", None)

        if positions is None:
            return np.zeros(9, dtype=np.float32)

        positions = np.asarray(positions, dtype=np.float64)
        n_nodes = len(positions)

        if n_nodes == 0:
            return np.zeros(9, dtype=np.float32)

        # Compute center of gravity (assuming uniform mass distribution)
        cog = np.mean(positions, axis=0)

        # Compute body orientation: head-to-tail direction
        head = positions[0]
        tail = positions[-1]
        body_axis = head - tail
        body_length = np.linalg.norm(body_axis)
        if body_length > 1e-6:
            orientation = body_axis / body_length
        else:
            orientation = np.array([1.0, 0.0, 0.0])

        # Compute angular velocity from velocity field
        angular_velocity = np.zeros(3)
        if velocities is not None:
            velocities = np.asarray(velocities, dtype=np.float64)
            if len(velocities) == n_nodes:
                angular_velocity = self._compute_angular_velocity(
                    positions, velocities, cog
                )

        # Build feature vector
        features = np.zeros(9, dtype=np.float32)
        features[0:3] = cog
        features[3:6] = orientation
        features[6:9] = angular_velocity

        if self.normalize:
            # Normalize CoG to workspace
            features[0:3] = features[0:3] / self.workspace_size
            # Orientation is already unit vector
            # Normalize angular velocity
            features[6:9] = np.clip(
                features[6:9] / self._max_angular_velocity, -1.0, 1.0
            )

        return features

    def _compute_angular_velocity(
        self, positions: np.ndarray, velocities: np.ndarray, cog: np.ndarray
    ) -> np.ndarray:
        """Compute angular velocity from position and velocity fields.

        Uses the relationship: v = omega x r, where r is position relative to CoG.
        Solves for omega in least-squares sense.

        Args:
            positions: Node positions (n_nodes, 3)
            velocities: Node velocities (n_nodes, 3)
            cog: Center of gravity (3,)

        Returns:
            Angular velocity vector (3,)
        """
        # Relative positions from CoG
        r = positions - cog

        # For each node: v = omega x r
        # This gives us: [v_x, v_y, v_z] = omega x [r_x, r_y, r_z]
        # Which can be written as: v = R @ omega, where R is skew-symmetric

        # Build system: A @ omega = b
        n_nodes = len(positions)
        A = np.zeros((3 * n_nodes, 3))
        b = velocities.flatten()

        for i in range(n_nodes):
            # Skew-symmetric matrix for r[i]
            # omega x r = -r x omega = R(r) @ omega
            rx, ry, rz = r[i]
            A[3 * i : 3 * i + 3] = np.array(
                [
                    [0, rz, -ry],
                    [-rz, 0, rx],
                    [ry, -rx, 0],
                ]
            )

        # Solve least squares
        try:
            omega, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            omega = np.zeros(3)

        return omega.astype(np.float64)


class GoalRelativeExtractor(FeatureExtractor):
    """Extract goal-relative features (direction and distance to prey).

    Computes the relative position of the prey with respect to the snake's
    head, both in world frame and normalized by distance.

    Output features (4 dims):
        [0:3]: Direction to prey (unit vector from head to prey center)
        [3]: Distance to prey (normalized)
    """

    def __init__(self, normalize: bool = True, max_distance: float = 2.0):
        """Initialize goal-relative extractor.

        Args:
            normalize: If True, normalize distance to [0, 1] range
            max_distance: Maximum expected distance for normalization
        """
        self.normalize = normalize
        self.max_distance = max_distance

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (direction + distance)."""
        return 4

    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract goal-relative features from state.

        Args:
            state: State dict containing 'positions', 'prey_position',
                   and optionally 'prey_distance' keys

        Returns:
            Feature vector of shape (4,)
        """
        positions = state.get("positions", None)
        prey_position = state.get("prey_position", None)

        if positions is None or prey_position is None:
            return np.zeros(4, dtype=np.float32)

        positions = np.asarray(positions, dtype=np.float64)
        prey_position = np.asarray(prey_position, dtype=np.float64)

        if len(positions) == 0:
            return np.zeros(4, dtype=np.float32)

        # Head position (first node)
        head = positions[0]

        # Vector from head to prey
        to_prey = prey_position - head
        distance = np.linalg.norm(to_prey)

        # Direction to prey (unit vector)
        if distance > 1e-6:
            direction = to_prey / distance
        else:
            direction = np.zeros(3)

        # Use provided prey_distance if available (more accurate, accounts for prey geometry)
        if "prey_distance" in state:
            distance = state["prey_distance"]

        # Build feature vector
        features = np.zeros(4, dtype=np.float32)
        features[0:3] = direction
        features[3] = distance

        if self.normalize:
            # Direction is already unit vector
            features[3] = np.clip(distance / self.max_distance, 0.0, 1.0)

        return features


class BodyFrameGoalExtractor(FeatureExtractor):
    """Extract goal position in body-frame coordinates.

    Transforms prey position into the snake's body reference frame,
    making the representation invariant to absolute position and orientation.

    Output features (4 dims):
        [0]: Forward distance to prey (along body axis)
        [1]: Lateral distance to prey (perpendicular to body axis, in XY plane)
        [2]: Vertical distance to prey (Z axis)
        [3]: Total distance to prey
    """

    def __init__(self, normalize: bool = True, max_distance: float = 2.0):
        """Initialize body-frame goal extractor.

        Args:
            normalize: If True, normalize distances
            max_distance: Maximum expected distance for normalization
        """
        self.normalize = normalize
        self.max_distance = max_distance

    @property
    def feature_dim(self) -> int:
        """Return feature dimension."""
        return 4

    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract body-frame goal features.

        Args:
            state: State dict containing 'positions' and 'prey_position'

        Returns:
            Feature vector of shape (4,)
        """
        positions = state.get("positions", None)
        prey_position = state.get("prey_position", None)

        if positions is None or prey_position is None:
            return np.zeros(4, dtype=np.float32)

        positions = np.asarray(positions, dtype=np.float64)
        prey_position = np.asarray(prey_position, dtype=np.float64)

        if len(positions) < 2:
            return np.zeros(4, dtype=np.float32)

        # Compute body frame
        head = positions[0]
        # Use second node to define forward direction
        forward = positions[0] - positions[1]
        forward_length = np.linalg.norm(forward[:2])  # XY plane only

        if forward_length > 1e-6:
            forward_xy = forward[:2] / forward_length
        else:
            forward_xy = np.array([1.0, 0.0])

        # Lateral direction (perpendicular to forward in XY plane)
        lateral = np.array([-forward_xy[1], forward_xy[0]])

        # Vector from head to prey
        to_prey = prey_position - head
        total_distance = np.linalg.norm(to_prey)

        # Project onto body frame
        forward_dist = np.dot(to_prey[:2], forward_xy)
        lateral_dist = np.dot(to_prey[:2], lateral)
        vertical_dist = to_prey[2]

        features = np.array(
            [forward_dist, lateral_dist, vertical_dist, total_distance],
            dtype=np.float32,
        )

        if self.normalize:
            features = np.clip(features / self.max_distance, -1.0, 1.0)
            features[3] = np.clip(features[3], 0.0, 1.0)  # Distance is non-negative

        return features
