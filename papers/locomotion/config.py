"""Configuration dataclasses for free-body snake locomotion.

Hierarchy:
    DismechConfig         -> LocomotionPhysicsConfig (3D free-body with gravity + Coulomb friction)
    PPOConfig             -> LocomotionConfig (top-level project config)

Composable pieces:
    GaitType                 -- locomotion gait selection
    SerpenoidControlConfig   -- amplitude/frequency/turn_bias ranges, substeps
    GoalConfig               -- goal placement and termination
    LocomotionRewardConfig   -- Liu 2023 potential-field reward coefficients
    LocomotionEnvConfig      -- composes physics + control + goal + reward + episode settings
    LocomotionNetworkConfig  -- 2x128 tanh MLP
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.physics import DismechConfig, FrictionConfig, FrictionModel, GeometryConfig
from src.configs.training import PPOConfig


# ---------------------------------------------------------------------------
# Gait selection
# ---------------------------------------------------------------------------


class GaitType(str, Enum):
    """Locomotion gait types."""

    FORWARD = "forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    U_TURN = "u_turn"


# ---------------------------------------------------------------------------
# Physics config (inherits DisMech, overrides for free-body ground locomotion)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionPhysicsConfig(DismechConfig):
    """DisMech physics for free-body snake locomotion.

    Overrides DismechConfig defaults:
    - 2D simulation with RFT friction (anisotropic drag)
    - Free body (no clamped nodes)
    - 0.5m snake, radius 0.01m, 20 segments
    """

    # Free body (not clamped)
    clamp_first_node: bool = False
    two_d_sim: bool = False

    # Override geometry for locomotion snake
    # Thin rod (radius=0.001) gives low bending stiffness EI ∝ r^4,
    # enabling dynamic oscillation needed for RFT-based locomotion
    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(
            snake_length=0.5,
            snake_radius=0.001,
            num_segments=20,
        )
    )

    # Time stepping (dt=0.05 matches default DisMech config)
    dt: float = 0.05

    # Material
    youngs_modulus: float = 2e6
    poisson_ratio: float = 0.5
    density: float = 1200.0

    # No gravity — snake moves in XY ground plane
    enable_gravity: bool = False

    # RFT friction (anisotropic drag: ct < cn for anisotropic locomotion)
    friction: FrictionConfig = field(
        default_factory=lambda: FrictionConfig(
            model=FrictionModel.RFT,
            rft_ct=0.01,
            rft_cn=0.1,
        )
    )


# ---------------------------------------------------------------------------
# Control config
# ---------------------------------------------------------------------------


@dataclass
class SerpenoidControlConfig:
    """Configuration for serpenoid steering control.

    The agent outputs 5-dim action [amplitude, frequency, wave_number, phase, turn_bias]
    which is transformed to joint curvatures via DirectSerpenoidSteeringTransform.
    """

    # Action dimension (5: amp, freq, wave_num, phase, turn_bias)
    action_dim: int = 5

    # Parameter ranges (physical values mapped from [-1, 1])
    amplitude_range: Tuple[float, float] = (0.0, 2.0)
    frequency_range: Tuple[float, float] = (0.5, 3.0)
    turn_bias_range: Tuple[float, float] = (-2.0, 2.0)

    # Physics substeps per RL action (10 × 0.05s = 0.5s per RL step,
    # ~1.2 serpenoid wave cycles at default frequency)
    substeps_per_action: int = 10


# ---------------------------------------------------------------------------
# Goal placement
# ---------------------------------------------------------------------------


@dataclass
class GoalConfig:
    """Goal placement and termination for forward locomotion.

    The goal is placed along the snake's initial heading direction at
    a fixed distance. Episode terminates on goal reach or starvation.
    """

    goal_distance: float = 2.0  # Distance from CoM along initial heading (meters)
    goal_radius: float = 0.1  # Reach threshold (meters)
    starvation_timeout: int = 60  # Consecutive steps with v_g < 0 → terminate


# ---------------------------------------------------------------------------
# Reward weights (Liu et al. 2023 potential-field reward)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionRewardConfig:
    """Potential-field reward coefficients (Liu et al. 2023, eq. 14).

    R = c_v * v_g + c_g * v_g * cos(theta_g) / dist
    """

    c_v: float = 1.0  # Velocity-toward-goal coefficient
    c_g: float = 0.5  # Potential-field (proximity) coefficient


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class LocomotionEnvConfig:
    """Environment configuration for locomotion tasks."""

    physics: LocomotionPhysicsConfig = field(default_factory=LocomotionPhysicsConfig)
    control: SerpenoidControlConfig = field(default_factory=SerpenoidControlConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    rewards: LocomotionRewardConfig = field(default_factory=LocomotionRewardConfig)

    # Gait selection
    gait: GaitType = GaitType.FORWARD

    # Episode settings
    max_episode_steps: int = 500

    # Randomize initial heading in XY plane
    randomize_initial_heading: bool = True

    # Device
    device: str = "auto"


# ---------------------------------------------------------------------------
# Network config (2x128 tanh MLP)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionNetworkConfig(NetworkConfig):
    """Network config for locomotion: 3x256 tanh MLP (scaled for 16GB GPU)."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[256, 256, 256],
            activation="tanh",
            ortho_init=True,
            init_gain=0.01,
            min_std=0.1,
            max_std=1.0,
            init_std=0.5,
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[256, 256, 256],
            activation="tanh",
            ortho_init=True,
            init_gain=1.0,
        )
    )


# ---------------------------------------------------------------------------
# Top-level config (PPO hyperparameters)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionConfig(PPOConfig):
    """Top-level config for locomotion PPO training.

    Scaled for 16GB GPU: 3x256 network, larger batches.
    Defaults: 2M frames, lr=3e-4, batch=8192, minibatch=512, clip=0.2.
    """

    name: str = "locomotion"
    experiment_name: str = "locomotion"

    # PPO hyperparameters
    total_frames: int = 2_000_000
    learning_rate: float = 3e-4
    frames_per_batch: int = 8192   # 4x larger — keeps GPU busy during PPO update
    clip_epsilon: float = 0.2
    num_epochs: int = 10
    mini_batch_size: int = 512     # 2x larger — better GPU utilization per minibatch
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01

    # Compose env + network + logging
    env: LocomotionEnvConfig = field(default_factory=LocomotionEnvConfig)
    network: LocomotionNetworkConfig = field(default_factory=LocomotionNetworkConfig)

    # Parallelism
    num_envs: int = 1

    def __post_init__(self):
        """Set name from gait type."""
        gait = self.env.gait.value if isinstance(self.env.gait, GaitType) else self.env.gait
        self.name = f"locomotion_{gait}"
        self.experiment_name = self.name
