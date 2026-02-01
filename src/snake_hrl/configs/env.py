"""Environment configuration dataclasses."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, List


class ControlMethod(str, Enum):
    """Control method for the snake robot.

    DIRECT: Direct curvature control (19-dim action space, one per joint)
        - Full control over each joint's curvature
        - Most flexible but hardest to learn

    CPG: Central Pattern Generator (4-dim action space)
        - amplitude, frequency, wave_number, phase
        - Uses neural oscillators to generate coordinated curvatures
        - Cannot steer (only forward/backward along body axis)

    SERPENOID: Direct serpenoid formula (4-dim action space)
        - amplitude, frequency, wave_number, phase
        - κ(s,t) = A × sin(k × s - ω × t + φ)
        - Cannot steer (only forward/backward along body axis)

    SERPENOID_STEERING: Serpenoid with turn bias (5-dim action space)
        - amplitude, frequency, wave_number, phase, turn_bias
        - κ(s,t) = A × sin(k × s - ω × t + φ) + κ_turn
        - CAN steer: turn_bias > 0 curves left, < 0 curves right
        - Turn radius R = 1/|turn_bias|
    """
    DIRECT = "direct"
    CPG = "cpg"
    SERPENOID = "serpenoid"
    SERPENOID_STEERING = "serpenoid_steering"


class SolverFramework(str, Enum):
    """Physics solver framework selection.

    DISMECH: DisMech library with implicit Euler integration
        - Uses discrete elastic rod model
        - Implicit time stepping for stability at larger dt
        - Control via bend spring natural strain

    ELASTICA: PyElastica (Cosserat rod theory)
        - Full Cosserat rod dynamics
        - Explicit symplectic integrators (PositionVerlet, PEFRL)
        - Control via rest curvature (rest_kappa)
    """
    DISMECH = "dismech"
    ELASTICA = "elastica"


class ElasticaGroundContact(str, Enum):
    """Ground contact method for PyElastica.

    RFT: Resistive Force Theory (custom implementation matching DisMech)
    DAMPING: Use Elastica's built-in damping mechanisms
    NONE: No ground contact forces
    """
    RFT = "rft"
    DAMPING = "damping"
    NONE = "none"


class StateRepresentation(str, Enum):
    """State representation mode for observations.

    FULL: High-dimensional state (151-dim default with 20 segments)
        - Snake positions: 21 * 3 = 63
        - Snake velocities: 21 * 3 = 63
        - Current curvatures: 19
        - Prey position: 3
        - Prey orientation: 3

    REDUCED: Compact feature-based representation (16-dim) - legacy
        - Curvature modes: 3 (amplitude, wave number, phase from FFT)
        - Virtual chassis: 9 (CoG position, orientation, angular velocity)
        - Goal-relative: 4 (direction to prey + distance)

    REDUCED_APPROACH: Minimal representation for approach task (13-dim)
        - Curvature modes: 3 (amplitude, wave number, phase from FFT)
        - Orientation: 3 (snake's facing direction, unit vector)
        - Angular velocity: 3 (rotational dynamics)
        - Goal direction: 3 (world-frame direction to goal, unit vector)
        - Goal distance: 1 (scalar distance to goal)
        Note: CoG position excluded (absolute position irrelevant for approach)

    REDUCED_COIL: Representation designed for coiling task (22-dim)
        - Curvature modes: 3 (amplitude, wave number, phase from FFT)
        - Virtual chassis: 9 (CoG position, orientation, angular velocity)
        - Goal-relative: 4 (direction to prey + distance)
        - Contact features: 6 (contact_fraction, wrap_count, head/mid/tail contact, continuity)
        Note: Includes contact information critical for successful coiling behavior.
    """
    FULL = "full"
    REDUCED = "reduced"
    REDUCED_APPROACH = "reduced_approach"
    REDUCED_COIL = "reduced_coil"


@dataclass
class PhysicsConfig:
    """Configuration for physics simulation parameters.

    This uses DisMech (dismech-python) for physics simulation with implicit
    time integration and resistive force theory for ground interaction.
    """

    # Time stepping (DisMech implicit integrator)
    dt: float = 5e-2  # Simulation timestep (seconds) - larger due to implicit integration
    max_iter: int = 25  # Maximum Newton iterations per timestep
    tol: float = 1e-4  # Force tolerance for convergence
    ftol: float = 1e-4  # Relative force tolerance
    dtol: float = 1e-2  # Displacement tolerance

    # Snake geometry
    snake_length: float = 1.0  # Total length of snake (meters)
    snake_radius: float = 0.001  # Radius of snake body (meters) - DisMech rod radius
    num_segments: int = 20  # Number of discrete segments

    # Material properties (DisMech parameters)
    youngs_modulus: float = 2e6  # Young's modulus for rod (Pa)
    density: float = 1200.0  # Material density (kg/m^3)
    poisson_ratio: float = 0.5  # Poisson's ratio for rod

    # Resistive Force Theory (RFT) parameters for ground interaction
    use_rft: bool = True  # Enable RFT for ground forces
    rft_ct: float = 0.01  # Tangential drag coefficient
    rft_cn: float = 0.1  # Normal drag coefficient

    # Prey geometry
    prey_radius: float = 0.1  # Radius of cylindrical prey (meters)
    prey_length: float = 0.3  # Length of prey (meters)

    # Gravity
    enable_gravity: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Solver framework selection
    solver_framework: SolverFramework = SolverFramework.DISMECH

    # Elastica-specific parameters
    elastica_damping: float = 0.1  # Numerical damping coefficient
    elastica_time_stepper: str = "PositionVerlet"  # "PositionVerlet" or "PEFRL"
    elastica_substeps: int = 50  # Internal substeps per RL step (for smaller dt)
    elastica_ground_contact: ElasticaGroundContact = ElasticaGroundContact.RFT


@dataclass
class EnvConfig:
    """Configuration for the reinforcement learning environment."""

    # Physics
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # Environment bounds
    workspace_size: Tuple[float, float, float] = (2.0, 2.0, 1.0)  # x, y, z bounds

    # Episode settings
    max_episode_steps: int = 1000

    # Observation settings
    include_velocities: bool = True
    include_prey_state: bool = True
    include_curvatures: bool = True  # Include current curvatures in observation
    history_length: int = 1  # Number of past observations to stack
    state_representation: "StateRepresentation" = StateRepresentation.FULL

    # Action settings
    action_scale: float = 1.0  # Scale factor for actions
    action_repeat: int = 1  # Number of times to repeat each action

    # Reward settings
    use_reward_shaping: bool = True
    reward_scale: float = 1.0

    # Randomization
    randomize_initial_state: bool = True
    randomize_prey_position: bool = True
    prey_position_range: Tuple[float, float] = (0.3, 0.8)  # Min/max distance from snake

    # Device
    device: str = "cpu"

    # Control method configuration (CPG/Serpenoid/Direct)
    cpg: "CPGConfig" = field(default_factory=lambda: CPGConfig())

    @property
    def obs_dim(self) -> int:
        """Calculate observation dimension based on config.

        Returns:
            For REDUCED_APPROACH representation: 13 * history_length
                - Curvature modes: 3 (amplitude, wave number, phase)
                - Orientation: 3 (snake's facing direction)
                - Angular velocity: 3 (rotational dynamics)
                - Goal direction: 3 (world-frame direction to goal)
                - Goal distance: 1 (scalar distance to goal)

            For REDUCED representation: 16 * history_length
                - Curvature modes: 3 (amplitude, wave number, phase)
                - Virtual chassis: 9 (CoG position, orientation, angular velocity)
                - Goal-relative: 4 (direction to prey + distance)

            For REDUCED_COIL representation: 22 * history_length
                - Curvature modes: 3 (amplitude, wave number, phase)
                - Virtual chassis: 9 (CoG position, orientation, angular velocity)
                - Goal-relative: 4 (direction to prey + distance)
                - Contact features: 6 (contact_fraction, wrap_count, regional contacts, continuity)

            For FULL representation: (positions + velocities + curvatures + prey) * history_length
                - Snake positions: 21 * 3 = 63
                - Snake velocities: 21 * 3 = 63 (if include_velocities)
                - Current curvatures: 19 (if include_curvatures)
                - Prey state: 6 (if include_prey_state)
        """
        # Handle REDUCED_APPROACH representation (13-dim for approach task)
        if self.state_representation == StateRepresentation.REDUCED_APPROACH:
            # Curvature modes (3) + Orientation (3) + Angular vel (3) + Goal (4) = 13
            return 13 * self.history_length

        # Handle reduced representation
        if self.state_representation == StateRepresentation.REDUCED:
            # Curvature modes (3) + Virtual chassis (9) + Goal-relative (4) = 16
            return 16 * self.history_length

        # Handle reduced coil representation (includes contact features)
        if self.state_representation == StateRepresentation.REDUCED_COIL:
            # Curvature modes (3) + Virtual chassis (9) + Goal-relative (4) + Contact (6) = 22
            return 22 * self.history_length

        # Full representation
        num_nodes = self.physics.num_segments + 1  # 21 nodes for 20 segments
        num_joints = self.physics.num_segments - 1  # 19 internal joints

        # Snake positions (3 * num_nodes)
        dim = 3 * num_nodes

        if self.include_velocities:
            # Snake velocities (3 * num_nodes)
            dim += 3 * num_nodes

        if self.include_curvatures:
            # Current curvatures at internal joints
            dim += num_joints

        if self.include_prey_state:
            # Prey position (3) + orientation (3)
            dim += 6

        return dim * self.history_length

    @property
    def action_dim(self) -> int:
        """Calculate action dimension based on control method."""
        num_joints = self.physics.num_segments - 1  # 19 internal joints
        return self.cpg.get_action_dim(num_joints)


@dataclass
class GaitConfig:
    """Configuration for gait-based reward shaping from demonstrations."""

    # Enable/disable gait potential
    use_gait_potential: bool = False

    # Demonstration source
    demo_path: Optional[str] = None  # Path to saved demonstrations
    num_generated_demos: int = 100  # Number of demos to generate if no path

    # Gait potential parameters
    sigma: float = 1.0  # Gaussian kernel width for fixed potential
    sigma_init: float = 2.0  # Initial sigma for curriculum
    sigma_final: float = 0.5  # Final sigma for curriculum
    gait_weight: float = 0.5  # Weight for gait shaping reward

    # Potential type selection
    potential_type: str = "gaussian"  # "gaussian" or "curriculum"
    curriculum_schedule: str = "linear"  # "linear", "cosine", or "exponential"

    # Demo generation parameters (used if generating demos)
    amplitude_range: Tuple[float, float] = (0.5, 2.0)
    wave_number_range: Tuple[float, float] = (1.0, 3.0)
    frequency_range: Tuple[float, float] = (0.5, 2.0)
    demo_duration: float = 5.0

    # Feature extraction settings
    feature_extractors: List[str] = field(default_factory=lambda: [
        "CurvatureModeExtractor",
        "VirtualChassisExtractor",
        "GoalRelativeExtractor",
    ])


@dataclass
class CPGConfig:
    """Configuration for action transformation and control method.

    Supports four control methods:
    - DIRECT: 19-dim action space, direct curvature control per joint
    - CPG: 4-dim action space, uses neural oscillators to generate curvatures
    - SERPENOID: 4-dim action space, uses analytical serpenoid formula (no steering)
    - SERPENOID_STEERING: 5-dim action space, serpenoid with turn bias (can steer)

    SERPENOID vs SERPENOID_STEERING:
        SERPENOID:          κ(s,t) = A × sin(k × s - ω × t + φ)
        SERPENOID_STEERING: κ(s,t) = A × sin(k × s - ω × t + φ) + κ_turn

        The κ_turn (turn_bias) parameter enables steering:
        - turn_bias = 0: straight path
        - turn_bias > 0: curves left (counterclockwise)
        - turn_bias < 0: curves right (clockwise)
        - Turn radius R = 1/|turn_bias|
    """

    # Control method selection
    control_method: ControlMethod = ControlMethod.DIRECT

    # CPG network architecture (used when control_method=CPG)
    num_oscillators: int = 4  # Number of coupled oscillators
    coupling_strength: float = 1.0  # Inter-oscillator coupling

    # Oscillator parameters (used when control_method=CPG)
    oscillator_type: str = "matsuoka"  # "matsuoka" or "hopf"
    time_constant: float = 0.1  # Neural time constant (tau)
    adaptation_constant: float = 0.5  # Adaptation time constant

    # Gait parameter ranges (used when control_method=CPG, SERPENOID, or SERPENOID_STEERING)
    gait_action_dim: int = 4  # Base RL action dimension (amplitude, freq, wave_num, phase)
    amplitude_range: Tuple[float, float] = (0.0, 2.0)
    frequency_range: Tuple[float, float] = (0.5, 3.0)

    # Turn bias range (used when control_method=SERPENOID_STEERING)
    # Typical values: [-2.0, 2.0] where |turn_bias| = 1/turn_radius
    turn_bias_range: Tuple[float, float] = (-2.0, 2.0)

    # Interpolation from oscillators to joints (used when control_method=CPG)
    interpolation_method: str = "linear"  # "linear" or "cubic"

    @property
    def use_cpg(self) -> bool:
        """Backward compatibility: check if using CPG control."""
        return self.control_method == ControlMethod.CPG

    @property
    def use_serpenoid(self) -> bool:
        """Check if using serpenoid control (4-dim, no steering)."""
        return self.control_method == ControlMethod.SERPENOID

    @property
    def use_serpenoid_steering(self) -> bool:
        """Check if using serpenoid steering control (5-dim, with turn bias)."""
        return self.control_method == ControlMethod.SERPENOID_STEERING

    @property
    def use_any_serpenoid(self) -> bool:
        """Check if using any serpenoid-based control (4-dim or 5-dim)."""
        return self.control_method in (ControlMethod.SERPENOID, ControlMethod.SERPENOID_STEERING)

    @property
    def use_direct(self) -> bool:
        """Check if using direct curvature control."""
        return self.control_method == ControlMethod.DIRECT

    def get_action_dim(self, num_joints: int = 19) -> int:
        """Get action dimension based on control method.

        Args:
            num_joints: Number of joints (used for direct control)

        Returns:
            Action dimension for the selected control method:
            - DIRECT: num_joints (19)
            - CPG: 4 (amplitude, frequency, wave_number, phase)
            - SERPENOID: 4 (amplitude, frequency, wave_number, phase)
            - SERPENOID_STEERING: 5 (amplitude, frequency, wave_number, phase, turn_bias)
        """
        if self.control_method == ControlMethod.DIRECT:
            return num_joints
        elif self.control_method == ControlMethod.SERPENOID_STEERING:
            return 5  # 4 base params + turn_bias
        else:
            return self.gait_action_dim  # 4 for CPG and SERPENOID


@dataclass
class ApproachEnvConfig(EnvConfig):
    """Configuration specific to the approach skill environment."""

    # Success criteria
    approach_distance_threshold: float = 0.15  # Distance to consider "approached"

    # Base reward weights (true objectives)
    energy_penalty_weight: float = 0.01  # Energy efficiency penalty
    success_bonus: float = 1.0  # Bonus for being in success zone

    # PBRS potential weights (used by ApproachPotential, not direct rewards)
    distance_reward_weight: float = 1.0  # Used by ApproachPotential
    velocity_reward_weight: float = 0.1  # Used by ApproachPotential

    # Termination
    terminate_on_success: bool = True
    success_hold_steps: int = 10  # Steps to hold position to confirm success

    # Gait-based reward shaping
    gait: GaitConfig = field(default_factory=GaitConfig)


@dataclass
class CoilEnvConfig(EnvConfig):
    """Configuration specific to the coil skill environment."""

    # Use REDUCED_COIL representation by default for coiling tasks
    # This includes contact features critical for successful coiling
    state_representation: "StateRepresentation" = StateRepresentation.REDUCED_COIL

    # Success criteria
    min_coil_wraps: float = 1.5  # Minimum number of wraps around prey
    contact_fraction_threshold: float = 0.6  # Fraction of snake in contact with prey

    # Base reward weights (true objectives)
    stability_reward_weight: float = 0.5  # Maintain low velocity while in contact
    energy_penalty_weight: float = 0.001  # Energy efficiency penalty
    success_bonus: float = 2.0  # Bonus for achieving success criteria

    # PBRS potential weights (used by CoilPotential, not direct rewards)
    contact_reward_weight: float = 1.0  # Used by CoilPotential
    wrap_reward_weight: float = 2.0  # Used by CoilPotential
    constriction_reward_weight: float = 1.5  # Used by CoilPotential

    # Termination
    terminate_on_success: bool = True
    success_hold_steps: int = 20

    # Gait-based reward shaping
    gait: GaitConfig = field(default_factory=GaitConfig)


@dataclass
class HRLEnvConfig(EnvConfig):
    """Configuration for hierarchical RL environment."""

    # Skill configurations
    approach_config: ApproachEnvConfig = field(default_factory=ApproachEnvConfig)
    coil_config: CoilEnvConfig = field(default_factory=CoilEnvConfig)

    # Manager settings
    num_skills: int = 2  # Approach, Coil
    skill_duration: int = 50  # Steps before manager can switch skills
    allow_early_termination: bool = True  # Allow skill to signal completion early

    # Reward settings
    task_completion_bonus: float = 100.0
    skill_switch_penalty: float = 0.1
