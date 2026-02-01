"""Snake robot physics simulation using PyElastica.

This module provides a physics simulation for a soft snake robot using
PyElastica (Cosserat rod theory) with symplectic time integration.
"""

from typing import Optional, Dict, Any
import numpy as np

from snake_hrl.configs.env import PhysicsConfig, StateRepresentation, ElasticaGroundContact

# PyElastica imports
from elastica import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    Damping,
    CosseratRod,
    PositionVerlet,
    PEFRL,
    GravityForces,
    AnalyticalLinearDamper,
    extend_stepper_interface,
)

from snake_hrl.physics.geometry import (
    PreyGeometry,
    create_prey_geometry,
    compute_contact_points,
    compute_wrap_angle,
)


class RFTForcing:
    """Custom Resistive Force Theory forcing for PyElastica.

    Implements anisotropic drag forces for ground interaction,
    matching the DisMech RFT implementation.
    """

    def __init__(self, ct: float, cn: float):
        """Initialize RFT forcing.

        Args:
            ct: Tangential drag coefficient
            cn: Normal drag coefficient
        """
        self.ct = ct
        self.cn = cn

    def apply_forces(self, system, time: float = 0.0):
        """Apply RFT forces to the rod.

        Args:
            system: CosseratRod system
            time: Current simulation time
        """
        # Get velocities at nodes
        velocities = system.velocity_collection

        # Get tangent vectors (direction along rod)
        # For each element, tangent is direction from node i to node i+1
        positions = system.position_collection
        tangents = np.diff(positions, axis=1)
        tangent_norms = np.linalg.norm(tangents, axis=0, keepdims=True)
        tangents = tangents / (tangent_norms + 1e-10)

        # Interpolate tangents to nodes (average neighboring element tangents)
        n_nodes = velocities.shape[1]
        node_tangents = np.zeros_like(velocities)
        node_tangents[:, 0] = tangents[:, 0]  # First node uses first element tangent
        node_tangents[:, -1] = tangents[:, -1]  # Last node uses last element tangent
        for i in range(1, n_nodes - 1):
            node_tangents[:, i] = 0.5 * (tangents[:, i - 1] + tangents[:, i])
        # Normalize
        norms = np.linalg.norm(node_tangents, axis=0, keepdims=True)
        node_tangents = node_tangents / (norms + 1e-10)

        # Compute velocity components
        v_tangent = np.sum(velocities * node_tangents, axis=0, keepdims=True) * node_tangents
        v_normal = velocities - v_tangent

        # RFT forces: F = -ct * v_t - cn * v_n
        forces = -self.ct * v_tangent - self.cn * v_normal

        # Apply forces to external forces array
        system.external_forces += forces


class SnakeSimulator(BaseSystemCollection, Constraints, Forcing, Damping):
    """PyElastica simulator for snake robot."""
    pass


class ElasticaSnakeGeometryAdapter:
    """Adapter class to provide snake geometry interface from PyElastica state."""

    def __init__(self, positions: np.ndarray, config: PhysicsConfig):
        """Initialize adapter with positions from PyElastica.

        Args:
            positions: Node positions (3, n_nodes) - PyElastica format
            config: Physics configuration
        """
        # Convert from PyElastica format (3, n_nodes) to (n_nodes, 3)
        self.positions = positions.T.copy()
        self.num_segments = config.num_segments
        self._radii = np.full(len(self.positions), config.snake_radius)

        # Compute rest lengths
        segment_length = config.snake_length / config.num_segments
        self._rest_lengths = np.full(config.num_segments, segment_length)

        # Compute masses
        segment_volume = np.pi * config.snake_radius**2 * segment_length
        segment_mass = config.density * segment_volume
        self._masses = np.full(len(self.positions), segment_mass)
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


class ElasticaSnakeRobot:
    """Physics simulation for soft snake robot using PyElastica.

    This class wraps PyElastica's CosseratRod simulation to provide
    Cosserat rod dynamics with controllable curvature via rest_kappa.
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

        # Create prey geometry (not part of PyElastica simulation)
        self.prey = create_prey_geometry(config, initial_prey_position)

        # Initialize PyElastica components
        self._init_elastica(initial_snake_position)

        # Control input (target curvatures)
        self.target_curvatures = np.zeros(config.num_segments - 1)

        # Time tracking
        self.time = 0.0

        # Contact state (computed from positions)
        self._contact_mask = np.zeros(config.num_segments + 1, dtype=bool)

        # Velocities (from PyElastica state)
        self.velocities = np.zeros((config.num_segments + 1, 3))

    def _init_elastica(self, initial_position: Optional[np.ndarray] = None) -> None:
        """Initialize PyElastica simulation components.

        Args:
            initial_position: Starting position for snake head
        """
        # Set starting position
        if initial_position is None:
            initial_position = np.array([0.0, 0.0, 0.0])

        # Direction of the rod (along x-axis by default)
        direction = np.array([1.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])

        # Create simulator
        self._simulator = SnakeSimulator()

        # Compute rod properties
        n_elements = self.config.num_segments
        base_length = self.config.snake_length
        base_radius = self.config.snake_radius
        density = self.config.density
        youngs_modulus = self.config.youngs_modulus
        poisson_ratio = self.config.poisson_ratio

        # Shear modulus from Young's modulus and Poisson ratio
        shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))

        # Create straight rod
        self._rod = CosseratRod.straight_rod(
            n_elements=n_elements,
            start=initial_position,
            direction=direction,
            normal=normal,
            base_length=base_length,
            base_radius=base_radius,
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )

        # Add rod to simulator
        self._simulator.append(self._rod)

        # Add gravity if enabled
        if self.config.enable_gravity:
            self._simulator.add_forcing_to(self._rod).using(
                GravityForces,
                acc_gravity=np.array(self.config.gravity),
            )

        # Add ground contact / RFT forcing
        if self.config.use_rft:
            if self.config.elastica_ground_contact == ElasticaGroundContact.RFT:
                # Custom RFT forcing (applied manually in step())
                self._rft_forcing = RFTForcing(
                    ct=self.config.rft_ct,
                    cn=self.config.rft_cn,
                )
            elif self.config.elastica_ground_contact == ElasticaGroundContact.DAMPING:
                # Use built-in damping instead
                dt_substep = self.config.dt / self.config.elastica_substeps
                self._simulator.dampen(self._rod).using(
                    AnalyticalLinearDamper,
                    damping_constant=self.config.elastica_damping,
                    time_step=dt_substep,
                )
                self._rft_forcing = None
            else:
                self._rft_forcing = None
        else:
            self._rft_forcing = None

        # Add numerical damping (if not already added via DAMPING ground contact)
        if self.config.elastica_damping > 0 and self.config.elastica_ground_contact != ElasticaGroundContact.DAMPING:
            dt_substep = self.config.dt / self.config.elastica_substeps
            self._simulator.dampen(self._rod).using(
                AnalyticalLinearDamper,
                damping_constant=self.config.elastica_damping,
                time_step=dt_substep,
            )

        # Finalize simulator
        self._simulator.finalize()

        # Create time stepper and extend interface for manual stepping
        if self.config.elastica_time_stepper == "PEFRL":
            self._time_stepper = PEFRL()
        else:
            self._time_stepper = PositionVerlet()

        # Get do_step function for manual integration
        self._do_step, self._stages_and_updates = extend_stepper_interface(
            self._time_stepper, self._simulator
        )

        # Update snake adapter
        self._update_snake_adapter()

    def _update_snake_adapter(self) -> None:
        """Update the snake geometry adapter from PyElastica state."""
        self.snake = ElasticaSnakeGeometryAdapter(
            self._rod.position_collection.copy(),
            self.config,
        )

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

        # Reinitialize PyElastica with new position
        self._init_elastica(snake_position)

        # Reset control and state
        self.target_curvatures = np.zeros(self.config.num_segments - 1)
        self.time = 0.0
        self._contact_mask = np.zeros(self.config.num_segments + 1, dtype=bool)
        self.velocities = np.zeros((self.config.num_segments + 1, 3))

    def set_curvature_control(self, curvatures: np.ndarray) -> None:
        """Set target curvatures for control.

        Args:
            curvatures: Target curvature at each internal joint
                       Shape: (num_segments - 1,)
        """
        assert len(curvatures) == self.config.num_segments - 1
        self.target_curvatures = np.clip(curvatures, -10.0, 10.0)

        # Apply curvature control to PyElastica via rest_kappa
        self._apply_curvature_to_elastica()

    def _apply_curvature_to_elastica(self) -> None:
        """Apply target curvatures to PyElastica rod via rest curvature.

        PyElastica uses rest_kappa to control the rod shape. By modifying
        the rest curvature, we set the target curvature that the elastic
        energy will drive the rod toward.

        rest_kappa has shape (3, n_elements - 1) for:
        - kappa1: curvature in normal direction
        - kappa2: curvature in binormal direction
        - twist: torsion
        """
        # For planar motion, set kappa1 (first component)
        # The number of curvature values is n_elements - 1 = num_segments - 1
        n_kappa = self._rod.rest_kappa.shape[1]
        n_curvatures = len(self.target_curvatures)

        # Map target curvatures to rest_kappa
        # Use min of available slots
        n_apply = min(n_kappa, n_curvatures)

        for i in range(n_apply):
            self._rod.rest_kappa[0, i] = self.target_curvatures[i]
            # Keep other components at zero for planar motion
            self._rod.rest_kappa[1, i] = 0.0
            # No twist
            if self._rod.rest_kappa.shape[0] > 2:
                self._rod.rest_kappa[2, i] = 0.0

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Advance simulation by one timestep.

        Args:
            dt: Timestep (uses config.dt if not specified)

        Returns:
            Dictionary with simulation state and metrics
        """
        if dt is None:
            dt = self.config.dt

        # Apply curvature control before stepping
        self._apply_curvature_to_elastica()

        # PyElastica uses smaller internal timesteps
        dt_substep = dt / self.config.elastica_substeps
        total_steps = self.config.elastica_substeps

        # Integrate with substeps using manual stepping
        for _ in range(total_steps):
            # Apply custom RFT forcing if enabled
            if self._rft_forcing is not None:
                self._rft_forcing.apply_forces(self._rod, self.time)

            # Integrate one substep using do_step
            self.time = self._do_step(
                self._time_stepper,
                self._stages_and_updates,
                self._simulator,
                self.time,
                dt_substep,
            )

        # Update snake adapter and velocities
        self._update_snake_adapter()
        self.velocities = self._rod.velocity_collection.T.copy()

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
            state_representation: Which representation to use

        Returns:
            Observation array matching the specified representation
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
        """Get full high-dimensional observation."""
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
        """Get compact feature-based observation (16-dim)."""
        # Import here to avoid circular imports
        from snake_hrl.features import (
            CompositeFeatureExtractor,
            CurvatureModeExtractor,
            VirtualChassisExtractor,
            GoalRelativeExtractor,
        )

        if not hasattr(self, "_feature_extractor"):
            self._feature_extractor = CompositeFeatureExtractor([
                CurvatureModeExtractor(),
                VirtualChassisExtractor(),
                GoalRelativeExtractor(),
            ])

        return self._feature_extractor.extract(state).astype(np.float32)

    def _get_reduced_approach_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Get minimal observation for approach task (13-dim)."""
        from snake_hrl.features import (
            CompositeFeatureExtractor,
            CurvatureModeExtractor,
            VirtualChassisExtractor,
            GoalRelativeExtractor,
        )

        if not hasattr(self, "_approach_feature_extractor"):
            self._approach_feature_extractor = CompositeFeatureExtractor([
                CurvatureModeExtractor(),
                VirtualChassisExtractor(),
                GoalRelativeExtractor(),
            ])

        full_features = self._approach_feature_extractor.extract(state)

        # Build 13-dim output by skipping CoG position
        approach_features = np.zeros(13, dtype=np.float32)
        approach_features[0:3] = full_features[0:3]    # Curvature modes
        approach_features[3:6] = full_features[6:9]    # Orientation
        approach_features[6:9] = full_features[9:12]   # Angular velocity
        approach_features[9:13] = full_features[12:16] # Goal direction + distance

        return approach_features

    def _get_reduced_coil_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Get observation with contact features for coiling (22-dim)."""
        from snake_hrl.features import (
            CompositeFeatureExtractor,
            CurvatureModeExtractor,
            VirtualChassisExtractor,
            GoalRelativeExtractor,
            ContactFeatureExtractor,
        )

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

        # Elastic potential energy (simplified estimate)
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
