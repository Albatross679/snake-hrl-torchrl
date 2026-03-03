"""Snake robot physics simulation using MuJoCo.

This module provides a physics simulation for a snake robot using MuJoCo
rigid-body simulation. The snake is modeled as a chain of rigid capsule
links connected by hinge joints with position actuators for curvature control.
"""

from typing import Optional, Dict, Any
import numpy as np

import mujoco

from configs.physics import PhysicsConfig, FrictionModel
from configs.env import StateRepresentation

from physics.geometry import (
    PreyGeometry,
    create_prey_geometry,
    compute_contact_points,
    compute_wrap_angle,
)


def _build_mjcf_xml(
    config: PhysicsConfig,
    snake_pos: np.ndarray,
    prey_pos: np.ndarray,
) -> str:
    """Build MJCF XML string for the snake-prey scene.

    Args:
        config: Physics configuration
        snake_pos: Initial snake head position (3,)
        prey_pos: Initial prey center position (3,)

    Returns:
        MJCF XML string
    """
    seg_len = config.snake_length / config.num_segments
    radius = config.snake_radius
    damping = config.mujoco_joint_damping
    kp = config.mujoco_joint_stiffness

    # Determine friction tuple from friction config
    fc = config.friction
    if fc.model == FrictionModel.NONE:
        friction = "0 0 0"
    elif fc.model == FrictionModel.NATIVE:
        friction = f"{config.mujoco_friction[0]} {config.mujoco_friction[1]} {config.mujoco_friction[2]}"
    elif fc.model in (FrictionModel.COULOMB, FrictionModel.STRIBECK):
        # Map mu_kinetic to MuJoCo's slide friction (1st element)
        # Keep torsional and rolling at small defaults
        friction = f"{fc.mu_kinetic} 0.005 0.0001"
    elif fc.model == FrictionModel.RFT:
        # RFT doesn't map naturally to MuJoCo rigid-body contact;
        # use a low friction as approximation for viscous drag regime
        friction = f"{fc.rft_cn} 0.005 0.0001"

    # Initial z so snake sits on ground
    init_z = snake_pos[2] if snake_pos[2] > radius else radius

    # Build body chain XML
    body_xml_parts = []
    indent = "        "

    # Root body with freejoint
    body_xml_parts.append(
        f'{indent}<body name="seg_0" pos="{snake_pos[0]:.6f} {snake_pos[1]:.6f} {init_z:.6f}">'
    )
    body_xml_parts.append(f'{indent}  <freejoint name="root"/>')
    body_xml_parts.append(
        f'{indent}  <geom name="g_0" type="capsule" size="{radius}" '
        f'fromto="0 0 0 {seg_len} 0 0" mass="{_segment_mass(config)}"/>'
    )

    # Nested child bodies with hinge joints
    for i in range(1, config.num_segments):
        body_xml_parts.append(
            f'{indent}  <body name="seg_{i}" pos="{seg_len} 0 0">'
        )
        body_xml_parts.append(
            f'{indent}    <joint name="j_{i}" type="hinge" axis="0 0 1" '
            f'damping="{damping}"/>'
        )
        body_xml_parts.append(
            f'{indent}    <geom name="g_{i}" type="capsule" size="{radius}" '
            f'fromto="0 0 0 {seg_len} 0 0" mass="{_segment_mass(config)}"/>'
        )
        indent_str = indent + "  "

    # Close all nested bodies (innermost first is wrong - we need to close after actuators)
    # Actually, we need to close them in reverse order
    close_tags = ""
    for i in range(config.num_segments - 1):
        close_tags += f"{indent}  {'  ' * (config.num_segments - 2 - i)}</body>\n"
    close_tags += f"{indent}</body>"

    # Actuator XML
    actuator_parts = []
    for i in range(1, config.num_segments):
        actuator_parts.append(
            f'        <position name="a_{i}" joint="j_{i}" kp="{kp}" ctrlrange="-3.14 3.14"/>'
        )

    # Build complete XML
    grav = f"{config.gravity[0]} {config.gravity[1]} {config.gravity[2]}" if config.enable_gravity else "0 0 0"

    xml = f"""<mujoco model="snake">
  <option timestep="{config.mujoco_timestep}" gravity="{grav}">
    <flag contact="enable"/>
  </option>

  <default>
    <geom contype="1" conaffinity="1" friction="{friction}" condim="3"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="ground" type="plane" size="10 10 0.01" pos="0 0 0"
          contype="1" conaffinity="1" friction="{friction}"/>

"""

    # Add body chain
    xml += "\n".join(body_xml_parts) + "\n"
    xml += close_tags + "\n"

    # Add prey body (no contact - analytical contact detection)
    xml += f"""
    <body name="prey" pos="{prey_pos[0]:.6f} {prey_pos[1]:.6f} {prey_pos[2]:.6f}">
      <geom name="g_prey" type="cylinder" size="{config.prey_radius} {config.prey_length / 2}"
            contype="0" conaffinity="0" mass="1.0"/>
    </body>
  </worldbody>

  <actuator>
{chr(10).join(actuator_parts)}
  </actuator>
</mujoco>
"""
    return xml


def _segment_mass(config: PhysicsConfig) -> float:
    """Compute mass of one segment."""
    seg_len = config.snake_length / config.num_segments
    volume = np.pi * config.snake_radius**2 * seg_len
    return config.density * volume


class MujocoSnakeGeometryAdapter:
    """Adapter class to provide snake geometry interface from MuJoCo state."""

    def __init__(self, positions: np.ndarray, config: PhysicsConfig):
        """Initialize adapter with positions from MuJoCo.

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


class MujocoSnakeRobot:
    """Physics simulation for snake robot using MuJoCo.

    The snake is modeled as a chain of rigid capsule links connected
    by hinge joints with position actuators for curvature control.
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

        # Create prey geometry (analytical contact, not part of MuJoCo sim)
        self.prey = create_prey_geometry(config, initial_prey_position)

        # Initialize MuJoCo
        self._init_mujoco(initial_snake_position)

        # Control input (target curvatures)
        self.target_curvatures = np.zeros(config.num_segments - 1)

        # Time tracking
        self.time = 0.0

        # Contact state (computed from positions)
        self._contact_mask = np.zeros(config.num_segments + 1, dtype=bool)

        # Velocities
        self.velocities = np.zeros((config.num_segments + 1, 3))

    def _init_mujoco(self, initial_position: Optional[np.ndarray] = None) -> None:
        """Initialize MuJoCo simulation.

        Args:
            initial_position: Starting position for snake head
        """
        if initial_position is None:
            initial_position = np.array([0.0, 0.0, 0.0])

        # Ensure z >= radius so snake sits on ground
        snake_pos = initial_position.copy()
        if snake_pos[2] < self.config.snake_radius:
            snake_pos[2] = self.config.snake_radius

        prey_pos = self.prey.position.copy()

        # Build MJCF and load model
        xml = _build_mjcf_xml(self.config, snake_pos, prey_pos)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)

        # Cache body IDs
        self._body_ids = []
        for i in range(self.config.num_segments):
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"seg_{i}")
            self._body_ids.append(bid)

        # Cache joint IDs (for hinge joints, not freejoint)
        self._joint_ids = []
        for i in range(1, self.config.num_segments):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"j_{i}")
            self._joint_ids.append(jid)

        # Cache actuator IDs
        self._actuator_ids = []
        for i in range(1, self.config.num_segments):
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{i}")
            self._actuator_ids.append(aid)

        # Forward kinematics to initialize positions
        mujoco.mj_forward(self._model, self._data)

        # Update snake adapter from MuJoCo state
        self._update_snake_adapter()

    def _extract_node_positions(self) -> np.ndarray:
        """Extract node positions from MuJoCo body origins.

        Nodes 0..N-1 correspond to body origins. Node N (tail tip) is
        computed as the last body position plus segment_length along
        its local x-axis.

        Returns:
            Node positions (num_segments + 1, 3)
        """
        seg_len = self.config.snake_length / self.config.num_segments
        num_nodes = self.config.num_segments + 1
        positions = np.zeros((num_nodes, 3))

        # Body origins for nodes 0..N-1
        for i, bid in enumerate(self._body_ids):
            positions[i] = self._data.xpos[bid].copy()

        # Tail tip: last body pos + seg_len along local x-axis
        last_bid = self._body_ids[-1]
        # xmat is stored as 9-element flat array (row-major 3x3)
        rot = self._data.xmat[last_bid].reshape(3, 3)
        local_x = rot[:, 0]  # First column = local x-axis in world frame
        positions[-1] = positions[-2] + seg_len * local_x

        return positions

    def _extract_node_velocities(self) -> np.ndarray:
        """Extract node velocities from MuJoCo.

        Uses mj_objectVelocity for each body. Tail node velocity
        is approximated from the last body's velocity.

        Returns:
            Node velocities (num_segments + 1, 3)
        """
        num_nodes = self.config.num_segments + 1
        velocities = np.zeros((num_nodes, 3))

        # 6D velocity buffer: [angular(3), linear(3)]
        vel6 = np.zeros(6)

        for i, bid in enumerate(self._body_ids):
            mujoco.mj_objectVelocity(
                self._model, self._data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0
            )
            velocities[i] = vel6[3:6]  # Linear velocity

        # Tail node: approximate from last body velocity
        velocities[-1] = velocities[-2].copy()

        return velocities

    def _update_snake_adapter(self) -> None:
        """Update the snake geometry adapter from MuJoCo state."""
        positions = self._extract_node_positions()
        self.snake = MujocoSnakeGeometryAdapter(positions, self.config)

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

        # Reinitialize MuJoCo with new position
        self._init_mujoco(snake_position)

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

        Converts target curvatures to joint angles and steps MuJoCo
        for the configured number of substeps.

        Args:
            dt: Timestep (uses config.dt if not specified, only for time tracking)

        Returns:
            Dictionary with simulation state
        """
        if dt is None:
            dt = self.config.dt

        seg_len = self.config.snake_length / self.config.num_segments

        # Convert curvatures to joint angles: angle = curvature * segment_length
        target_angles = self.target_curvatures * seg_len

        # Set actuator controls
        for i, aid in enumerate(self._actuator_ids):
            self._data.ctrl[aid] = target_angles[i]

        # Step MuJoCo for substeps
        for _ in range(self.config.mujoco_substeps):
            mujoco.mj_step(self._model, self._data)

        # Update snake adapter
        self._update_snake_adapter()

        # Update velocities
        self.velocities = self._extract_node_velocities()

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
            Dictionary containing all 11 standard state keys.
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
        from observations import (
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
        from observations import (
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
        from observations import (
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
            Dictionary with kinetic, gravitational, elastic, and total energy.
            Elastic energy uses actuator spring energy: 0.5 * kp * (q - q_target)^2
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

        # Elastic energy from actuator springs: 0.5 * kp * (q - q_target)^2
        kp = self.config.mujoco_joint_stiffness
        seg_len = self.config.snake_length / self.config.num_segments
        target_angles = self.target_curvatures * seg_len
        elastic = 0.0
        for i, jid in enumerate(self._joint_ids):
            # Get joint angle from qpos
            qpos_adr = self._model.jnt_qposadr[jid]
            q = self._data.qpos[qpos_adr]
            elastic += 0.5 * kp * (q - target_angles[i]) ** 2

        return {
            "kinetic": float(kinetic),
            "gravitational": float(gravitational),
            "elastic": float(elastic),
            "total": float(kinetic + gravitational + elastic),
        }
