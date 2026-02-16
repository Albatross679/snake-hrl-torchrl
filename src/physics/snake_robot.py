"""Snake robot physics simulation using DisMech.

This module provides a physics simulation for a soft snake robot using
the DisMech library for discrete elastic rod dynamics with implicit
time integration.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

from configs.physics import PhysicsConfig
from configs.env import StateRepresentation
from observations import (
    CompositeFeatureExtractor,
    CurvatureModeExtractor,
    VirtualChassisExtractor,
    GoalRelativeExtractor,
    ContactFeatureExtractor,
)
from physics.geometry import (
    PreyGeometry,
    create_prey_geometry,
    compute_contact_points,
    compute_wrap_angle,
)

# DisMech imports
import dismech
from dismech import (
    SoftRobot,
    Geometry,
    GeomParams,
    Material,
    SimParams,
    Environment,
    ImplicitEulerTimeStepper,
)


def _create_rod_geometry(
    num_segments: int,
    snake_length: float,
    initial_position: Optional[np.ndarray] = None,
    initial_direction: Optional[np.ndarray] = None,
) -> Geometry:
    """Create a rod geometry for DisMech.

    Args:
        num_segments: Number of segments in the rod
        snake_length: Total length of the rod
        initial_position: Starting position of first node
        initial_direction: Direction of the rod

    Returns:
        DisMech Geometry object
    """
    if initial_position is None:
        initial_position = np.array([0.0, 0.0, 0.0])

    if initial_direction is None:
        initial_direction = np.array([1.0, 0.0, 0.0])

    # Normalize direction
    direction = initial_direction / np.linalg.norm(initial_direction)

    # Create node positions
    num_nodes = num_segments + 1
    segment_length = snake_length / num_segments

    nodes = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        nodes[i] = initial_position + i * segment_length * direction

    # Create edges (connectivity)
    edges = np.array([[i, i + 1] for i in range(num_segments)], dtype=np.int64)

    # No faces (rod only, no shell)
    face_nodes = np.empty((0, 3), dtype=np.int64)

    # Create Geometry without plotting
    return Geometry(nodes, edges, face_nodes, plot_from_txt=False)


class SnakeGeometryAdapter:
    """Adapter class to provide snake geometry interface from DisMech state."""

    def __init__(self, positions: np.ndarray, config: PhysicsConfig):
        """Initialize adapter with positions from DisMech.

        Args:
            positions: Node positions (n_nodes, 3)
            config: Physics configuration
        """
        self.positions = positions
        self.num_segments = config.num_segments
        self._radii = np.full(len(positions), config.snake_radius)

        # Compute rest lengths
        segment_length = config.snake_length / config.num_segments
        self._rest_lengths = np.full(config.num_segments, segment_length)

        # Compute masses
        segment_volume = np.pi * config.snake_radius**2 * segment_length
        segment_mass = config.density * segment_volume
        self._masses = np.full(len(positions), segment_mass)
        self._masses[0] /= 2
        self._masses[-1] /= 2

    @property
    def num_nodes(self) -> int:
        return len(self.positions)

    @property
    def radii(self) -> np.ndarray:
        return self._radii

    @property
    def masses(self) -> np.ndarray:
        return self._masses

    @property
    def rest_lengths(self) -> np.ndarray:
        return self._rest_lengths

    @property
    def total_length(self) -> float:
        return float(np.sum(self._rest_lengths))

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


class SnakeRobot:
    """Physics simulation for soft snake robot using DisMech.

    This class wraps DisMech's SoftRobot and TimeStepper to provide
    discrete elastic rod simulation with controllable curvature at each joint.
    """

    def __init__(
        self,
        config: PhysicsConfig,
        initial_snake_position: Optional[np.ndarray] = None,
        initial_prey_position: Optional[np.ndarray] = None,
    ):
        """Initialize snake robot simulation.

        Args:
            config: Physics configuration
            initial_snake_position: Starting position of snake head
            initial_prey_position: Starting position of prey
        """
        self.config = config

        # Store initial positions for reset
        self._initial_snake_position = (
            initial_snake_position.copy() if initial_snake_position is not None else None
        )
        self._initial_prey_position = (
            initial_prey_position.copy() if initial_prey_position is not None else None
        )

        # Create prey geometry (not part of DisMech simulation)
        self.prey = create_prey_geometry(config, initial_prey_position)

        # Initialize DisMech components
        self._init_dismech(initial_snake_position)

        # Control input (target curvatures)
        self.target_curvatures = np.zeros(config.num_segments - 1)

        # Time tracking
        self.time = 0.0

        # Contact state (computed from positions)
        self._contact_mask = np.zeros(config.num_segments + 1, dtype=bool)
        self._contact_forces = np.zeros((config.num_segments + 1, 3))

        # Cache for velocities (computed from position differences)
        self._prev_positions = None
        self.velocities = np.zeros((config.num_segments + 1, 3))

        # Solver convergence: residual norm from last Newton solve
        self._last_residual_norm = None

    def _init_dismech(self, initial_position: Optional[np.ndarray] = None) -> None:
        """Initialize DisMech simulation components.

        Args:
            initial_position: Starting position for snake head
        """
        # Geometry parameters
        geom_params = GeomParams(
            rod_r0=self.config.snake_radius,
            shell_h=0,  # No shell elements
        )

        # Material parameters
        material = Material(
            density=self.config.density,
            youngs_rod=self.config.youngs_modulus,
            youngs_shell=0,  # No shell
            poisson_rod=self.config.poisson_ratio,
            poisson_shell=0,  # No shell
        )

        # Simulation parameters
        sim_params = SimParams(
            static_sim=False,
            two_d_sim=True,  # Snake moves primarily in 2D plane
            use_mid_edge=False,
            use_line_search=True,
            log_data=False,
            log_step=1,
            show_floor=False,
            dt=self.config.dt,
            max_iter=self.config.max_iter,
            total_time=1000.0,  # Not used directly
            plot_step=1,
            tol=self.config.tol,
            ftol=self.config.ftol,
            dtol=self.config.dtol,
        )

        # Environment with forces
        env = Environment()

        # Add gravity if enabled
        if self.config.enable_gravity:
            env.add_force('gravity', g=np.array(self.config.gravity))

        # Add RFT for ground interaction
        if self.config.use_rft:
            env.add_force('rft', ct=self.config.rft_ct, cn=self.config.rft_cn)

        # Create rod geometry
        geometry = _create_rod_geometry(
            self.config.num_segments,
            self.config.snake_length,
            initial_position,
        )

        # Create DisMech SoftRobot
        self._dismech_robot = SoftRobot(
            geom_params, material, geometry, sim_params, env
        )

        # Fix first node if needed (optional - snake can translate freely)
        # self._dismech_robot = self._dismech_robot.fix_nodes([0])

        # Create time stepper
        self._time_stepper = ImplicitEulerTimeStepper(self._dismech_robot)

        # Create snake geometry adapter from initial positions
        self._update_snake_adapter()

    def _update_snake_adapter(self) -> None:
        """Update the snake geometry adapter from DisMech state."""
        # Extract positions from DisMech state
        q = self._dismech_robot.state.q
        num_nodes = self.config.num_segments + 1
        positions = q[:3 * num_nodes].reshape(num_nodes, 3)

        self.snake = SnakeGeometryAdapter(positions, self.config)

    def _get_positions_from_state(self) -> np.ndarray:
        """Extract node positions from DisMech state."""
        q = self._dismech_robot.state.q
        num_nodes = self.config.num_segments + 1
        return q[:3 * num_nodes].reshape(num_nodes, 3)

    def _get_velocities_from_state(self) -> np.ndarray:
        """Extract node velocities from DisMech state."""
        u = self._dismech_robot.state.u
        num_nodes = self.config.num_segments + 1
        return u[:3 * num_nodes].reshape(num_nodes, 3)

    def reset(
        self,
        snake_position: Optional[np.ndarray] = None,
        prey_position: Optional[np.ndarray] = None,
    ) -> None:
        """Reset simulation to initial state.

        Args:
            snake_position: Optional new starting position for snake
            prey_position: Optional new position for prey
        """
        # Reset prey
        if prey_position is not None:
            self.prey.position = prey_position.copy()
        else:
            self.prey = create_prey_geometry(self.config)

        # Reinitialize DisMech with new position
        self._init_dismech(snake_position)

        # Reset control and state
        self.target_curvatures = np.zeros(self.config.num_segments - 1)
        self.time = 0.0
        self._contact_mask = np.zeros(self.config.num_segments + 1, dtype=bool)
        self._contact_forces = np.zeros((self.config.num_segments + 1, 3))
        self._prev_positions = None
        self.velocities = np.zeros((self.config.num_segments + 1, 3))

    def set_curvature_control(self, curvatures: np.ndarray) -> None:
        """Set target curvatures for control.

        Args:
            curvatures: Target curvature at each internal joint
                       Shape: (num_segments - 1,)
        """
        assert len(curvatures) == self.config.num_segments - 1
        self.target_curvatures = np.clip(curvatures, -10.0, 10.0)

        # Apply curvature control to DisMech bend springs
        # DisMech uses natural curvature in bend springs
        self._apply_curvature_to_dismech()

    def _apply_curvature_to_dismech(self) -> None:
        """Apply target curvatures to DisMech bend springs via natural strain.

        DisMech uses natural strain to control the rod shape. By modifying
        the `nat_strain` of bend springs, we set the target curvature that
        the elastic energy will drive the rod toward.

        The bend springs have nat_strain of shape (Nb, 2) for two curvature
        components (kappa1, kappa2). For planar motion, we primarily use kappa1.
        """
        bend_springs = self._dismech_robot.bend_springs

        if bend_springs.N > 0 and hasattr(bend_springs, 'nat_strain'):
            # Map target curvatures to bend spring natural strain
            # nat_strain has shape (Nb, 2) for two curvature components
            num_springs = min(len(self.target_curvatures), bend_springs.N)

            # Set the first curvature component (kappa1) for planar bending
            # The second component (kappa2) is for out-of-plane bending
            for i in range(num_springs):
                bend_springs.nat_strain[i, 0] = self.target_curvatures[i]
                # Keep kappa2 at zero for planar motion
                bend_springs.nat_strain[i, 1] = 0.0

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Advance simulation by one timestep.

        Args:
            dt: Timestep (uses config.dt if not specified)

        Returns:
            Dictionary with simulation state and metrics
        """
        if dt is None:
            dt = self.config.dt

        # Store previous positions for velocity computation
        self._prev_positions = self._get_positions_from_state().copy()

        # Apply curvature control before stepping
        self._apply_curvature_to_dismech()

        # Step DisMech simulation
        try:
            self._dismech_robot, f_norm = self._time_stepper.step(
                self._dismech_robot, debug=False
            )
            self._last_residual_norm = f_norm
        except ValueError as e:
            # Handle convergence issues gracefully
            self._last_residual_norm = None
            print(f"DisMech step warning: {e}")

        # Update snake adapter and velocities
        self._update_snake_adapter()
        self.velocities = self._get_velocities_from_state()

        self.time += dt

        # Compute contact with prey
        self._update_contact_state()

        return self.get_state()

    def _update_contact_state(self) -> None:
        """Update contact state between snake and prey."""
        self._contact_mask, distances = compute_contact_points(
            self.snake, self.prey, contact_threshold=0.01
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state.

        Returns:
            Dictionary containing:
                - positions: Snake node positions (n_nodes, 3)
                - velocities: Snake node velocities (n_nodes, 3)
                - curvatures: Current curvatures at joints
                - prey_position: Prey center position
                - prey_distance: Distance from snake head to prey
                - contact_mask: Which nodes are in contact with prey
                - wrap_angle: Total angle wrapped around prey
                - time: Current simulation time
        """
        curvatures = self.snake.get_curvatures()

        # Distance from head to prey
        head_pos = self.snake.positions[0]
        prey_distance = self.prey.distance_to_point(head_pos)

        # Wrap angle
        wrap_angle = compute_wrap_angle(self.snake, self.prey)

        return {
            "positions": self.snake.positions.copy(),
            "velocities": self.velocities.copy(),
            "curvatures": curvatures,
            "prey_position": self.prey.position.copy(),
            "prey_orientation": self.prey.orientation.copy(),
            "prey_distance": prey_distance,
            "contact_mask": self._contact_mask.copy(),
            "contact_fraction": np.mean(self._contact_mask),
            "wrap_angle": wrap_angle,
            "wrap_count": wrap_angle / (2 * np.pi),
            "time": self.time,
        }

    def get_observation(
        self,
        include_curvatures: bool = True,
        state_representation: StateRepresentation = StateRepresentation.FULL,
    ) -> np.ndarray:
        """Get observation vector for RL.

        Args:
            include_curvatures: Whether to include current curvatures in observation
                (only used for FULL representation)
            state_representation: Which representation to use (FULL, REDUCED, or REDUCED_COIL)

        Returns:
            For FULL representation (151-dim default):
                - Snake positions (flattened): 21 * 3 = 63
                - Snake velocities (flattened): 21 * 3 = 63
                - Current curvatures (if enabled): 19
                - Prey position: 3
                - Prey orientation: 3

            For REDUCED representation (16-dim):
                - Curvature modes: 3 (amplitude, wave number, phase)
                - Virtual chassis: 9 (CoG position, orientation, angular velocity)
                - Goal-relative: 4 (direction to prey + distance)

            For REDUCED_COIL representation (22-dim):
                - Curvature modes: 3 (amplitude, wave number, phase)
                - Virtual chassis: 9 (CoG position, orientation, angular velocity)
                - Goal-relative: 4 (direction to prey + distance)
                - Contact features: 6 (contact_fraction, wrap_count, regional contacts, continuity)
        """
        state = self.get_state()

        if state_representation == StateRepresentation.REDUCED:
            return self._get_reduced_observation(state)
        elif state_representation == StateRepresentation.REDUCED_APPROACH:
            return self._get_reduced_approach_observation(state)
        elif state_representation == StateRepresentation.REDUCED_COIL:
            return self._get_reduced_coil_observation(state)
        else:
            return self._get_full_observation(state, include_curvatures)

    def _get_full_observation(
        self, state: Dict[str, Any], include_curvatures: bool = True
    ) -> np.ndarray:
        """Get full high-dimensional observation.

        Args:
            state: State dictionary from get_state()
            include_curvatures: Whether to include current curvatures

        Returns:
            Full observation array (151-dim with default settings)
        """
        components = [
            state["positions"].flatten(),
            state["velocities"].flatten(),
        ]

        if include_curvatures:
            components.append(state["curvatures"])

        components.extend([
            state["prey_position"],
            state["prey_orientation"],
        ])

        obs = np.concatenate(components)
        return obs.astype(np.float32)

    def _get_reduced_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Get compact feature-based observation.

        Uses CompositeFeatureExtractor with:
            - CurvatureModeExtractor: 3 dims (amplitude, wave number, phase)
            - VirtualChassisExtractor: 9 dims (CoG position, orientation, angular velocity)
            - GoalRelativeExtractor: 4 dims (direction to prey + distance)

        Args:
            state: State dictionary from get_state()

        Returns:
            Reduced observation array (16-dim)
        """
        # Lazily create the feature extractor if not exists
        if not hasattr(self, "_feature_extractor"):
            self._feature_extractor = CompositeFeatureExtractor([
                CurvatureModeExtractor(),
                VirtualChassisExtractor(),
                GoalRelativeExtractor(),
            ])

        return self._feature_extractor.extract(state).astype(np.float32)

    def _get_reduced_approach_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Get minimal feature-based observation for approach task.

        Uses CompositeFeatureExtractor with:
            - CurvatureModeExtractor: 3 dims (amplitude, wave number, phase)
            - VirtualChassisExtractor: 6 dims (orientation + angular velocity, skip CoG)
            - GoalRelativeExtractor: 4 dims (direction to prey + distance)

        Total: 13 dims (excludes CoG position which is irrelevant for approach task)

        Args:
            state: State dictionary from get_state()

        Returns:
            Reduced approach observation array (13-dim)
        """
        # Lazily create the approach feature extractor if not exists
        if not hasattr(self, "_approach_feature_extractor"):
            self._approach_feature_extractor = CompositeFeatureExtractor([
                CurvatureModeExtractor(),
                VirtualChassisExtractor(),  # Full VC (9 dims) - we'll slice it
                GoalRelativeExtractor(),
            ])

        # Extract full features (16 dims)
        full_features = self._approach_feature_extractor.extract(state)

        # Build 13-dim output by skipping CoG position (first 3 dims of VC)
        # Layout: curvature_modes[0:3] + orientation[3:6] + angular_vel[6:9] + goal[9:13]
        # From full: curvature_modes[0:3] + vc_cog[3:6] + vc_orient[6:9] + vc_angular[9:12] + goal[12:16]
        approach_features = np.zeros(13, dtype=np.float32)
        approach_features[0:3] = full_features[0:3]    # Curvature modes
        approach_features[3:6] = full_features[6:9]    # Orientation (skip CoG at 3:6)
        approach_features[6:9] = full_features[9:12]   # Angular velocity
        approach_features[9:13] = full_features[12:16] # Goal direction + distance

        return approach_features

    def _get_reduced_coil_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Get compact feature-based observation with contact features for coiling.

        Uses CompositeFeatureExtractor with:
            - CurvatureModeExtractor: 3 dims (amplitude, wave number, phase)
            - VirtualChassisExtractor: 9 dims (CoG position, orientation, angular velocity)
            - GoalRelativeExtractor: 4 dims (direction to prey + distance)
            - ContactFeatureExtractor: 6 dims (contact_fraction, wrap_count, regional contacts, continuity)

        Args:
            state: State dictionary from get_state()

        Returns:
            Reduced coil observation array (22-dim)
        """
        # Lazily create the coil feature extractor if not exists
        if not hasattr(self, "_coil_feature_extractor"):
            self._coil_feature_extractor = CompositeFeatureExtractor([
                CurvatureModeExtractor(),
                VirtualChassisExtractor(),
                GoalRelativeExtractor(),
                ContactFeatureExtractor(),
            ])

        return self._coil_feature_extractor.extract(state).astype(np.float32)

    def get_energy(self) -> Dict[str, float]:
        """Compute energy metrics.

        Returns:
            Dictionary with kinetic and potential energy
        """
        # Kinetic energy
        kinetic = 0.5 * np.sum(
            self.snake.masses[:, np.newaxis] * self.velocities**2
        )

        # Gravitational potential energy
        if self.config.enable_gravity:
            heights = self.snake.positions[:, 2]
            gravitational = np.sum(
                self.snake.masses * np.abs(self.config.gravity[2]) * heights
            )
        else:
            gravitational = 0.0

        # Elastic potential energy from DisMech
        try:
            elastic = float(self._time_stepper.compute_total_elastic_energy(
                self._dismech_robot.state
            ))
        except Exception:
            # Fallback if DisMech energy computation fails
            current_lengths = self.snake.get_segment_lengths()
            stretch = current_lengths - self.snake.rest_lengths
            stiffness = self.config.youngs_modulus * np.pi * self.config.snake_radius**2
            elastic = 0.5 * stiffness * np.sum(stretch**2)

        return {
            "kinetic": float(kinetic),
            "gravitational": float(gravitational),
            "elastic": float(elastic),
            "total": float(kinetic + gravitational + elastic),
        }
