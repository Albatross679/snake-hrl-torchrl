"""Configuration dataclasses for Elastica-RL-control benchmark (Naughton et al., 2021).

Paper: Elastica: A Compliant Mechanics Environment for Soft Robotic Control
(IEEE Robotics and Automation Letters, 6(2), 3389-3396)

Hierarchy:
    ElasticaConfig           -> Naughton2021PhysicsConfig (3D Cosserat rod)
    SACConfig                -> Naughton2021Config (top-level, SAC as best performer)

Composable pieces:
    TorqueControlConfig       -- control point layout and torque scaling
    BenchmarkCase (enum)      -- 4 benchmark tasks
    ObstacleLayoutConfig      -- obstacle positions for Cases 3 & 4
    Naughton2021EnvConfig     -- composes physics + control + task
    Naughton2021NetworkConfig -- SB defaults (2x64 tanh)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from src.configs.base import Checkpointing, TensorBoard
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.physics import ElasticaConfig, FrictionConfig, FrictionModel, GeometryConfig
from src.configs.training import SACConfig


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------


class BenchmarkCase(str, Enum):
    """Four benchmark tasks from Naughton et al. (2021).

    CASE1_TRACKING: 3D tracking of a randomly moving target.
        - 6 control points, normal + binormal torques = 12 DOF.
    CASE2_REACHING: Reaching a stationary target with orientation matching.
        - 6 control points, normal + binormal + tangent = 18 DOF.
    CASE3_STRUCTURED: Navigate through structured array of 8 obstacles.
        - 2 control points, normal direction = 2 DOF.
    CASE4_UNSTRUCTURED: Navigate through unstructured nest of 12 obstacles.
        - 2 control points, normal + binormal = 4 DOF.
    """

    CASE1_TRACKING = "case1_tracking"
    CASE2_REACHING = "case2_reaching"
    CASE3_STRUCTURED = "case3_structured"
    CASE4_UNSTRUCTURED = "case4_unstructured"


# Map from case -> (num_control_points, torque_directions)
CASE_CONTROL_SPECS = {
    BenchmarkCase.CASE1_TRACKING: (6, ["normal", "binormal"]),
    BenchmarkCase.CASE2_REACHING: (6, ["normal", "binormal", "tangent"]),
    BenchmarkCase.CASE3_STRUCTURED: (2, ["normal"]),
    BenchmarkCase.CASE4_UNSTRUCTURED: (2, ["normal", "binormal"]),
}


# ---------------------------------------------------------------------------
# Control config
# ---------------------------------------------------------------------------


@dataclass
class TorqueControlConfig:
    """Configuration for distributed torque control along the rod.

    Torques are applied at equidistant control points along the rod.
    Each control point can apply torques in normal, binormal, and/or
    tangent directions depending on the benchmark case.
    """

    num_control_points: int = 6
    torque_directions: List[str] = field(
        default_factory=lambda: ["normal", "binormal"]
    )
    torque_scaling: float = 1.0  # alpha scaling factor

    @property
    def action_dim(self) -> int:
        return self.num_control_points * len(self.torque_directions)


# ---------------------------------------------------------------------------
# Obstacle config
# ---------------------------------------------------------------------------


@dataclass
class ObstacleLayoutConfig:
    """Obstacle configuration for Cases 3 and 4.

    Case 3: 8 obstacles in a structured grid array.
    Case 4: 12 obstacles in an unstructured (random) arrangement.
    """

    num_obstacles: int = 8
    obstacle_radius: float = 0.025  # meters

    # Structured layout (Case 3)
    grid_spacing: float = 0.1
    grid_rows: int = 2
    grid_cols: int = 4

    # Obstacle stiffness for contact
    contact_stiffness: float = 1e5
    contact_damping: float = 10.0

    # Penalty for collision
    collision_penalty: float = 5.0


# ---------------------------------------------------------------------------
# Physics config
# ---------------------------------------------------------------------------


@dataclass
class Naughton2021PhysicsConfig(ElasticaConfig):
    """PyElastica physics for compliant rod control benchmark.

    A single Cosserat rod with one end clamped, controlled by distributed
    torques applied at equidistant control points.
    """

    # Rod geometry (default: 1m rod, 50 elements)
    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(
            snake_length=1.0,
            snake_radius=0.025,
            num_segments=50,
        )
    )

    # Material
    youngs_modulus: float = 1e6  # Pa
    poisson_ratio: float = 0.5
    density: float = 1000.0  # kg/m^3
    dt: float = 2.5e-5  # Fine timestep for stability

    # Clamped boundary condition
    clamp_first_node: bool = True

    # Gravity
    enable_gravity: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Elastica-specific
    elastica_substeps: int = 100
    elastica_damping: float = 0.1

    # No ground friction (rod is clamped vertically)
    friction: FrictionConfig = field(
        default_factory=lambda: FrictionConfig(model=FrictionModel.NONE)
    )


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class TargetConfig:
    """Target configuration for tracking/reaching tasks."""

    # Random target motion (Case 1)
    target_speed: float = 0.05  # m/s
    workspace_radius: float = 0.8  # Maximum reach

    # Reaching (Case 2)
    match_orientation: bool = False


@dataclass
class Naughton2021EnvConfig:
    """Environment configuration for Elastica-RL benchmark."""

    physics: Naughton2021PhysicsConfig = field(
        default_factory=Naughton2021PhysicsConfig
    )
    control: TorqueControlConfig = field(default_factory=TorqueControlConfig)
    obstacles: ObstacleLayoutConfig = field(default_factory=ObstacleLayoutConfig)
    target: TargetConfig = field(default_factory=TargetConfig)

    # Benchmark case
    case: BenchmarkCase = BenchmarkCase.CASE1_TRACKING

    # Episode settings
    max_episode_steps: int = 500
    timesteps_per_batch: int = 2048

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Set control config based on benchmark case."""
        if self.case in CASE_CONTROL_SPECS:
            n_cp, dirs = CASE_CONTROL_SPECS[self.case]
            self.control.num_control_points = n_cp
            self.control.torque_directions = dirs

        # Set obstacle count for obstacle cases
        if self.case == BenchmarkCase.CASE3_STRUCTURED:
            self.obstacles.num_obstacles = 8
        elif self.case == BenchmarkCase.CASE4_UNSTRUCTURED:
            self.obstacles.num_obstacles = 12

        # Enable orientation matching for Case 2
        if self.case == BenchmarkCase.CASE2_REACHING:
            self.target.match_orientation = True


# ---------------------------------------------------------------------------
# Network config (SB defaults: 2x64 tanh)
# ---------------------------------------------------------------------------


@dataclass
class Naughton2021NetworkConfig(NetworkConfig):
    """Network config matching Stable Baselines defaults."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[64, 64],
            activation="tanh",
            distribution="tanh_normal",
            min_std=0.01,
            max_std=1.0,
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[64, 64],
            activation="tanh",
        )
    )


# ---------------------------------------------------------------------------
# Top-level config (SAC — best performer in the paper)
# ---------------------------------------------------------------------------


@dataclass
class Naughton2021Config(SACConfig):
    """Top-level config for Elastica-RL benchmark (SAC).

    SAC and TD3 converge fastest among the five algorithms tested.
    Off-policy: 5M timesteps. On-policy (PPO/TRPO): 10M timesteps.
    """

    name: str = "naughton2021"
    experiment_name: str = "naughton2021_elastica_rl"

    # SAC hyperparameters (SB defaults + paper settings)
    total_frames: int = 5_000_000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    warmup_steps: int = 10000

    # Composed configs
    env: Naughton2021EnvConfig = field(default_factory=Naughton2021EnvConfig)
    network: Naughton2021NetworkConfig = field(
        default_factory=Naughton2021NetworkConfig
    )
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)

    num_envs: int = 1

    def __post_init__(self):
        """Set name and experiment_name from benchmark case."""
        case = self.env.case.value if isinstance(self.env.case, BenchmarkCase) else self.env.case
        self.name = f"naughton2021_{case}"
        self.experiment_name = self.name
