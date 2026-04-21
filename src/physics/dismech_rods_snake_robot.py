"""Snake robot physics simulation using dismech-rods (C++).

This module provides a physics simulation for a soft snake robot using
the dismech-rods library (C++ with pybind11 bindings) for discrete elastic
rod dynamics via the SimulationManager API.
"""

from typing import Optional, Dict, Any
import os
import numpy as np

from src.configs.physics import PhysicsConfig, FrictionModel
from src.configs.env import StateRepresentation

# PARDISO solver in dismech-rods requires single-threaded MKL to avoid
# symbolic factorization errors when other libraries (e.g. torch) have
# already initialized MKL in multi-threaded mode.
os.environ.setdefault("MKL_NUM_THREADS", "1")

import py_dismech

from .geometry import (
    PreyGeometry,
    create_prey_geometry,
    compute_contact_points,
    compute_wrap_angle,
)


class DismechRodsSnakeGeometryAdapter:
    """Adapter class to provide snake geometry interface from dismech-rods state."""

    def __init__(self, positions: np.ndarray, config: PhysicsConfig):
        """Initialize adapter with positions from dismech-rods.

        Args:
            positions: Node positions (n_nodes, 3)
            config: Physics configuration
        """
        self.positions = positions.copy()
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


class DismechRodsSnakeRobot:
    """Physics simulation for soft snake robot using dismech-rods (C++).

    This class wraps the py_dismech SimulationManager to provide
    discrete elastic rod simulation with controllable curvature.
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

        # Create prey geometry (not part of dismech-rods simulation)
        self.prey = create_prey_geometry(config, initial_prey_position)

        # Initialize dismech-rods components
        self._init_dismech_rods(initial_snake_position)

        # Control input (target curvatures)
        self.target_curvatures = np.zeros(config.num_segments - 1)

        # Time tracking
        self.time = 0.0

        # Contact state (computed from positions)
        self._contact_mask = np.zeros(config.num_segments + 1, dtype=bool)

        # Velocities
        self.velocities = np.zeros((config.num_segments + 1, 3))

    def _init_dismech_rods(self, initial_position: Optional[np.ndarray] = None) -> None:
        """Initialize dismech-rods simulation components.

        Args:
            initial_position: Starting position for snake head
        """
        if initial_position is None:
            initial_position = np.array([0.0, 0.0, 0.0])

        # Compute rod start and end points
        direction = np.array([1.0, 0.0, 0.0])
        start = initial_position.copy()
        end = initial_position + self.config.snake_length * direction

        # Create SimulationManager
        self._sim_manager = py_dismech.SimulationManager()

        # Configure simulation parameters
        sim_params = self._sim_manager.sim_params
        sim_params.dt = self.config.dt
        max_iter = py_dismech.MaxIterations()
        max_iter.num_iters = self.config.max_iter
        sim_params.max_iter = max_iter
        sim_params.ftol = self.config.ftol
        sim_params.dtol = self.config.dtol
        sim_params.adaptive_time_stepping = self.config.dismech_rods_adaptive_time_stepping

        # Set integrator
        integrator_map = {
            "BACKWARD_EULER": py_dismech.BACKWARD_EULER,
            "IMPLICIT_MIDPOINT": py_dismech.IMPLICIT_MIDPOINT,
        }
        integrator_enum = integrator_map.get(
            self.config.dismech_rods_integrator, py_dismech.BACKWARD_EULER
        )
        sim_params.integrator = integrator_enum

        # Set headless rendering
        render_params = self._sim_manager.render_params
        render_params.renderer = py_dismech.HEADLESS

        # Get soft_robots handle and add limb
        soft_robots = self._sim_manager.soft_robots
        num_nodes = self.config.num_segments + 1
        soft_robots.addLimb(
            start,
            end,
            num_nodes,
            self.config.density,
            self.config.snake_radius,
            self.config.youngs_modulus,
            self.config.poisson_ratio,
        )

        # Add gravity if enabled
        if self.config.enable_gravity:
            gravity_vec = np.array(self.config.gravity)
            force = py_dismech.GravityForce(soft_robots, gravity_vec)
            self._sim_manager.forces.addForce(force)

        # Add ground interaction forces based on friction config
        friction = self.config.friction
        if friction.model in (FrictionModel.RFT, FrictionModel.COULOMB, FrictionModel.STRIBECK):
            raise NotImplementedError(
                f"FrictionModel.{friction.model.name} is not supported by the "
                f"dismech-rods (C++) backend. Use NATIVE or NONE."
            )

        # Add damping if configured (used by both NATIVE and NONE modes)
        if self.config.dismech_rods_damping_viscosity > 0:
            damping = py_dismech.DampingForce(
                soft_robots, self.config.dismech_rods_damping_viscosity
            )
            self._sim_manager.forces.addForce(damping)

        # Initialize the simulation (requires argv-style args, empty list for headless)
        self._sim_manager.initialize([])

        # Store limb reference for position/velocity queries
        self._limb = soft_robots.limbs[0]

        # Update snake adapter
        self._update_snake_adapter()

    def _update_snake_adapter(self) -> None:
        """Update the snake geometry adapter from dismech-rods state."""
        positions = np.array(self._limb.getVertices())
        self.snake = DismechRodsSnakeGeometryAdapter(positions, self.config)

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

        # Reinitialize dismech-rods with new position
        self._init_dismech_rods(snake_position)

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

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Advance simulation by one timestep.

        Args:
            dt: Timestep (uses config.dt if not specified)

        Returns:
            Dictionary with simulation state and metrics
        """
        if dt is None:
            dt = self.config.dt

        # Build curvature BC matrix: [limb_idx, edge_idx, cx, cy]
        # edge_idx ranges from 1 to num_segments-1 (internal edges)
        n_joints = self.config.num_segments - 1
        curvature_bc = np.zeros((n_joints, 4))
        curvature_bc[:, 0] = 0  # limb index
        curvature_bc[:, 1] = np.arange(1, n_joints + 1)  # edge indices (1-based)
        curvature_bc[:, 2] = self.target_curvatures  # cx (planar curvature)
        curvature_bc[:, 3] = 0.0  # cy = 0 for planar

        # Step dismech-rods simulation
        try:
            self._sim_manager.step_simulation({"curvature": curvature_bc})
        except RuntimeError as e:
            print(f"dismech-rods step warning: {e}")

        # Update snake adapter
        self._update_snake_adapter()

        # Update velocities from dismech-rods state
        try:
            self.velocities = np.array(self._limb.getVelocities())
        except (AttributeError, RuntimeError):
            # Fallback: estimate from positions if getVelocities not available
            self.velocities = np.zeros((self.config.num_segments + 1, 3))

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
        from src.observations import (
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
        from src.observations import (
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
        from src.observations import (
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

        # Elastic potential energy (fallback formula from positions)
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
