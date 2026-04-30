"""Configuration dataclasses for hierarchical RL path following (Liu, Guo & Fang, 2022).

Paper: A Reinforcement Learning-Based Strategy of Path Following for
Snake Robots with an Onboard Camera (Sensors 2022, 22, 9867)

Hierarchical control:
    1. RL Policy Training Layer: outputs gait offset φ_o
    2. Gait Execution Layer: φⁱ(t) = α·sin(ωt + (i-1)δ) + φ_o

The RL agent modifies the lateral undulatory gait equation in real-time
to follow desired paths (lines, sinusoids, circles).

Hierarchy:
    MujocoPhysicsConfig -> Liu2022PhysicsConfig (9-link wheeled)
    PPOConfig           -> Liu2022Config (top-level)

Composable pieces:
    GaitConfig             -- lateral undulatory gait parameters
    PathConfig             -- desired path definitions
    VisualLocConfig        -- pan-tilt visual localization
    Liu2022EnvConfig       -- composes physics + gait + path
    Liu2022NetworkConfig   -- 3-layer Tanh [64, 32, 32]
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from src.configs.base import TensorBoard
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.physics import MujocoPhysicsConfig
from src.configs.training import PPOConfig


class PathType(str, Enum):
    """Desired path types for path following."""
    STRAIGHT_LINE = "straight_line"
    SINUSOIDAL = "sinusoidal"
    CIRCLE = "circle"


# ---------------------------------------------------------------------------
# Gait parameters
# ---------------------------------------------------------------------------


@dataclass
class GaitConfig:
    """Lateral undulatory gait equation parameters.

    φⁱ(t) = α · sin(ωt + (i-1)δ) + φ_o

    where:
        α = amplitude (fixed during training)
        ω = angular frequency
        δ = phase difference between adjacent joints
        φ_o = gait offset (output of RL policy — the action)
    """

    amplitude: float = 0.5       # α: gait amplitude (radians)
    angular_freq: float = 2.0    # ω: angular frequency (rad/s)
    phase_diff: float = 0.5      # δ: inter-joint phase difference (radians)

    # Gait offset bounds (action range for RL)
    max_offset: float = 0.5  # Maximum |φ_o| (radians)


# ---------------------------------------------------------------------------
# Path definitions
# ---------------------------------------------------------------------------


@dataclass
class PathConfig:
    """Desired path configuration for path following.

    Straight lines: y = y* ∈ [-1.5, 1.5]
    Sinusoidal: y = A·sin(ωx + φ), A ∈ [0.2, 1.0], ω ∈ [π/2, π], φ ∈ [-1.5, 1.5]
    Circles: x² + y² = R², R ∈ [1.5, 3.0]
    """

    path_type: PathType = PathType.STRAIGHT_LINE

    # Straight line parameters
    line_y: float = 1.0  # y = y*

    # Sinusoidal parameters
    sin_amplitude: float = 0.5  # A
    sin_omega: float = 1.57     # ω (≈ π/2)
    sin_phase: float = 0.0     # φ

    # Circle parameters
    circle_radius: float = 2.0  # R

    # Target point (random point on desired path ahead of robot)
    target_x_range: Tuple[float, float] = (4.0, 5.0)


# ---------------------------------------------------------------------------
# Visual localization
# ---------------------------------------------------------------------------


@dataclass
class VisualLocConfig:
    """Pan-tilt visual localization configuration.

    The camera on the head uses a pan-tilt to compensate for head swings.
    The visual stabilization term p_h penalizes excessive head swings
    to maintain visual marker tracking.
    """

    # Head swing penalty
    angle_threshold: float = 0.3  # φ_* (radians): threshold for penalty
    penalty_coefficient: float = -0.1  # c_h: negative constant


# ---------------------------------------------------------------------------
# Physics config (9-link MuJoCo wheeled snake)
# ---------------------------------------------------------------------------


@dataclass
class Liu2022PhysicsConfig(MujocoPhysicsConfig):
    """MuJoCo physics for 9-link wheeled snake robot.

    9 connection modules with passive wheels and 8 yaw joints.
    """

    num_links: int = 9
    num_joints: int = 8
    link_length: float = 0.1    # meters per link module
    link_radius: float = 0.02   # meters

    # Simulation
    mujoco_timestep: float = 0.01   # 100 Hz sim
    mujoco_substeps: int = 10       # 10 Hz control

    # Joint limits
    joint_range_deg: float = 90.0


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class Liu2022EnvConfig:
    """Environment configuration for hierarchical path following."""

    physics: Liu2022PhysicsConfig = field(default_factory=Liu2022PhysicsConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    path: PathConfig = field(default_factory=PathConfig)
    visual: VisualLocConfig = field(default_factory=VisualLocConfig)

    # Episode settings
    max_episode_steps: int = 1000

    # Reward weights
    c_p: float = 1.0   # Path tracking reward weight
    c_e: float = 0.5   # Endpoint progress reward weight
    d_1: float = 0.1    # Inner distance threshold (full reward)
    d_2: float = 0.5    # Outer distance threshold (zero reward)

    # Device
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Network config (3 Tanh hidden layers [64, 32, 32])
# ---------------------------------------------------------------------------


@dataclass
class Liu2022NetworkConfig(NetworkConfig):
    """Network config: 3-layer Tanh [64, 32, 32] from the paper."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[64, 32, 32],
            activation="tanh",
            distribution="tanh_normal",
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[64, 32, 32],
            activation="tanh",
        )
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class Liu2022Config(PPOConfig):
    """Top-level config for hierarchical path following PPO training.

    Converges in ~1M timesteps (proposed method) vs ~2M (end-to-end baseline).
    """

    name: str = "liu2022"
    experiment_name: str = "liu2022_path_following"

    # PPO hyperparameters
    total_frames: int = 10_000_000
    frames_per_batch: int = 2048
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    num_epochs: int = 10

    # Composed configs
    env: Liu2022EnvConfig = field(default_factory=Liu2022EnvConfig)
    network: Liu2022NetworkConfig = field(default_factory=Liu2022NetworkConfig)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)

    num_envs: int = 1
