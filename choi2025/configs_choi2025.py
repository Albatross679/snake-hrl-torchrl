"""Configuration dataclasses for soft manipulator control (Choi & Tong, 2025).

Hierarchy:
    DismechConfig         → Choi2025PhysicsConfig (3D clamped manipulator)
    SACConfig             → Choi2025Config (top-level project config)

Composable pieces:
    DeltaCurvatureControlConfig  -- control point interpolation
    TargetConfig                 -- target sampling ranges
    ObstacleConfig               -- obstacle layout
    Choi2025EnvConfig            -- composes physics + control + target + obstacles
    Choi2025NetworkConfig        -- 3×256 ReLU MLP
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from configs.base import TensorBoard
from configs.network import ActorConfig, CriticConfig, NetworkConfig
from configs.physics import DismechConfig, FrictionConfig, FrictionModel, GeometryConfig
from configs.training import SACConfig


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Manipulation task types from Choi & Tong (2025)."""

    FOLLOW_TARGET = "follow_target"
    INVERSE_KINEMATICS = "inverse_kinematics"
    TIGHT_OBSTACLES = "tight_obstacles"
    RANDOM_OBSTACLES = "random_obstacles"


# ---------------------------------------------------------------------------
# Physics config (inherits DisMech, overrides for 3D clamped manipulator)
# ---------------------------------------------------------------------------


@dataclass
class Choi2025PhysicsConfig(DismechConfig):
    """DisMech physics for clamped soft manipulator.

    Overrides DismechConfig defaults for the Choi & Tong (2025) setup:
    - 3D rod (two_d_sim=False)
    - First node clamped (clamp_first_node=True)
    - 1m rod, 21 nodes (20 segments)
    - Young's modulus 2e6 Pa, Poisson's ratio 0.5
    - dt=0.01s
    """

    # Manipulator-specific
    clamp_first_node: bool = True
    two_d_sim: bool = False

    # Contact parameters (for obstacle tasks)
    contact_stiffness: float = 1e6
    contact_delta: float = 0.005
    max_newton_iter_contact: int = 25
    max_newton_iter_noncontact: int = 15

    # Override rod defaults for this paper
    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(
            snake_length=1.0,
            snake_radius=0.001,
            num_segments=20,
        )
    )
    dt: float = 0.01
    youngs_modulus: float = 2e6
    poisson_ratio: float = 0.5
    density: float = 1200.0

    # No gravity in 2D manipulator tasks (rod is horizontal)
    enable_gravity: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # No ground friction for manipulator
    friction: FrictionConfig = field(
        default_factory=lambda: FrictionConfig(model=FrictionModel.NONE)
    )


# ---------------------------------------------------------------------------
# Control config
# ---------------------------------------------------------------------------


@dataclass
class DeltaCurvatureControlConfig:
    """Configuration for delta curvature control with Voronoi smoothing.

    The agent outputs delta natural curvature at `num_control_points` locations
    along the rod. These are interpolated to all bend springs via a Voronoi
    weight matrix.
    """

    num_control_points: int = 5
    max_delta_curvature: float = 1.0
    voronoi_smoothing: bool = True


# ---------------------------------------------------------------------------
# Target and obstacle configs
# ---------------------------------------------------------------------------


@dataclass
class TargetConfig:
    """Target sampling configuration."""

    # Workspace radius (reachable region for a 1m rod)
    min_radius: float = 0.3
    max_radius: float = 0.9

    # For follow_target: target velocity
    target_speed: float = 0.05

    # For inverse_kinematics: include orientation
    match_orientation: bool = False


@dataclass
class ObstacleConfig:
    """Obstacle configuration for obstacle avoidance tasks."""

    num_obstacles: int = 3
    obstacle_radius: float = 0.05

    # Position sampling ranges (relative to workspace)
    min_distance: float = 0.2
    max_distance: float = 0.8

    # Tight obstacles: narrow gap width
    gap_width: float = 0.15

    # Penalty scaling for contact
    contact_penalty: float = 10.0


# ---------------------------------------------------------------------------
# Environment config (composes physics + control + target + obstacles)
# ---------------------------------------------------------------------------


@dataclass
class Choi2025EnvConfig:
    """Environment configuration for soft manipulator tasks."""

    physics: Choi2025PhysicsConfig = field(default_factory=Choi2025PhysicsConfig)
    control: DeltaCurvatureControlConfig = field(default_factory=DeltaCurvatureControlConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)

    # Task selection
    task: TaskType = TaskType.FOLLOW_TARGET

    # Episode settings
    max_episode_steps: int = 200

    # Device — "auto" → GPU when available, else CPU
    device: str = "auto"


# ---------------------------------------------------------------------------
# Network config (3×256 ReLU MLP matching paper)
# ---------------------------------------------------------------------------


@dataclass
class Choi2025NetworkConfig(NetworkConfig):
    """Network config matching Choi & Tong (2025): 3×256 ReLU."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[256, 256, 256],
            activation="relu",
            ortho_init=True,
            init_gain=0.01,
            min_std=0.01,
            max_std=1.0,
            init_std=0.5,
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[256, 256, 256],
            activation="relu",
            ortho_init=True,
            init_gain=1.0,
        )
    )


# ---------------------------------------------------------------------------
# Top-level config (SAC hyperparameters from paper)
# ---------------------------------------------------------------------------


@dataclass
class Choi2025Config(SACConfig):
    """Top-level config for soft manipulator SAC training.

    Overrides SAC defaults to match Choi & Tong (2025):
    - lr=0.001, batch=2048, buffer=2M, UTD=4
    """

    name: str = "choi2025"
    experiment_name: str = "choi2025"

    # SAC hyperparameters from paper
    total_frames: int = 1_000_000
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    alpha_lr: float = 0.001
    batch_size: int = 2048
    buffer_size: int = 2_000_000
    num_updates: int = 4  # UTD ratio
    warmup_steps: int = 1000

    # Compose env + network + logging
    env: Choi2025EnvConfig = field(default_factory=Choi2025EnvConfig)
    network: Choi2025NetworkConfig = field(default_factory=Choi2025NetworkConfig)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)

    # Parallelism
    num_envs: int = 1  # Number of parallel environments (paper uses 500)

    def __post_init__(self):
        """Set name and experiment_name from task."""
        task = self.env.task.value if isinstance(self.env.task, TaskType) else self.env.task
        self.name = f"choi2025_{task}"
        self.experiment_name = self.name
