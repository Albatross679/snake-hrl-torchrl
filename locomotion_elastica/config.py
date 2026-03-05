"""Configuration dataclasses for free-body snake locomotion (PyElastica backend).

Hierarchy:
    ElasticaConfig        -> LocomotionElasticaPhysicsConfig (3D free-body with RFT friction)
    PPOConfig             -> LocomotionElasticaConfig (top-level project config)

Composable pieces:
    GaitType                 -- locomotion gait selection
    SerpenoidControlConfig   -- amplitude/frequency/turn_bias ranges, substeps
    GoalConfig               -- goal placement and termination
    LocomotionRewardConfig   -- Liu 2023 potential-field reward coefficients
    LocomotionElasticaEnvConfig -- composes physics + control + goal + reward + episode settings
    LocomotionElasticaNetworkConfig -- 3x256 tanh MLP
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.physics import ElasticaConfig, FrictionConfig, FrictionModel, GeometryConfig
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
# Physics config (inherits ElasticaConfig, overrides for free-body ground locomotion)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionElasticaPhysicsConfig(ElasticaConfig):
    """PyElastica physics for free-body snake locomotion.

    Overrides ElasticaConfig defaults:
    - 2D simulation with RFT friction (anisotropic drag)
    - Free body (no clamped nodes)
    - 0.5m snake, radius 0.001m, 20 segments
    """

    # Free body (not clamped)
    clamp_first_node: bool = False
    two_d_sim: bool = False

    # Override geometry for locomotion snake
    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(
            snake_length=0.5,
            snake_radius=0.001,
            num_segments=20,
        )
    )

    # Time stepping
    dt: float = 0.05

    # Material
    youngs_modulus: float = 2e6
    poisson_ratio: float = 0.5
    density: float = 1200.0

    # No gravity — snake moves in XY ground plane
    enable_gravity: bool = False

    # PyElastica-specific
    elastica_damping: float = 0.1
    elastica_time_stepper: str = "PositionVerlet"
    elastica_substeps: int = 50

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

    # Physics substeps per RL action (10 × 0.05s = 0.5s per RL step)
    substeps_per_action: int = 10


# ---------------------------------------------------------------------------
# Goal placement
# ---------------------------------------------------------------------------


@dataclass
class GoalConfig:
    """Goal placement and termination for forward locomotion."""

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
class LocomotionElasticaEnvConfig:
    """Environment configuration for locomotion tasks (PyElastica backend)."""

    physics: LocomotionElasticaPhysicsConfig = field(
        default_factory=LocomotionElasticaPhysicsConfig
    )
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
# Network config (3x256 tanh MLP)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionElasticaNetworkConfig(NetworkConfig):
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
class LocomotionElasticaConfig(PPOConfig):
    """Top-level config for locomotion PPO training (PyElastica backend).

    Scaled for 16GB GPU: 3x256 network, larger batches.
    Defaults: 2M frames, lr=3e-4, batch=8192, minibatch=512, clip=0.2.
    """

    name: str = "locomotion_elastica"
    experiment_name: str = "locomotion_elastica"

    # PPO hyperparameters
    total_frames: int = 2_000_000
    learning_rate: float = 3e-4
    frames_per_batch: int = 8192
    clip_epsilon: float = 0.2
    num_epochs: int = 10
    mini_batch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01

    # Compose env + network + logging
    env: LocomotionElasticaEnvConfig = field(default_factory=LocomotionElasticaEnvConfig)
    network: LocomotionElasticaNetworkConfig = field(
        default_factory=LocomotionElasticaNetworkConfig
    )

    # Parallelism (default: CPU cores - 1, minimum 1)
    num_envs: int = field(default_factory=lambda: min(max(1, os.cpu_count() - 1), 40))

    def __post_init__(self):
        """Set name from gait type."""
        gait = self.env.gait.value if isinstance(self.env.gait, GaitType) else self.env.gait
        self.name = f"locomotion_elastica_{gait}"
        self.experiment_name = self.name
