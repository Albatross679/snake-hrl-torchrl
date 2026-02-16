"""Configuration dataclasses for locomotion task (Bing et al., IJCAI 2019)."""

from dataclasses import dataclass, field
from typing import List, Optional

from snake_hrl.configs.training import PPOConfig
from snake_hrl.configs.network import ActorConfig, CriticConfig, NetworkConfig


@dataclass
class LocomotionPhysicsConfig:
    """Physics parameters matching the original MuJoCo model."""

    timestep: float = 0.0125
    frame_skip: int = 4
    num_joints: int = 8
    max_episode_steps: int = 1000


@dataclass
class LocomotionEnvConfig:
    """Environment configuration for locomotion tasks."""

    physics: LocomotionPhysicsConfig = field(default_factory=LocomotionPhysicsConfig)

    # Task selection
    task: str = "power_velocity"  # "power_velocity" or "target_tracking"
    track_type: str = "line"  # line/wave/zigzag/circle/random

    # Power-velocity task parameters
    target_v: float = 0.1
    target_v_array: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.15, 0.20, 0.25]
    )
    head_target_dist: float = 2.0

    # Target-tracking task parameters
    target_distance: float = 4.0
    target_dist_min: float = 2.0
    target_dist_max: float = 6.0
    track_target_v: float = 0.3

    # Camera parameters (for target_tracking task)
    camera_width: int = 32
    camera_height: int = 20

    # Device
    device: str = "cpu"


@dataclass
class LocomotionPPOConfig(PPOConfig):
    """PPO config matching original paper hyperparameters."""

    total_frames: int = 3_000_000
    frames_per_batch: int = 2048
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0  # Paper uses 0
    value_coef: float = 0.5
    num_epochs: int = 10
    mini_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"
    target_kl: Optional[float] = None  # No early stopping
    normalize_advantage: bool = True
    max_grad_norm: float = 0.5
    experiment_name: str = "locomotion"


@dataclass
class LocomotionNetworkConfig(NetworkConfig):
    """Network config matching original 2x64 MLP."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[64, 64],
            activation="tanh",
            ortho_init=True,
            init_gain=0.01,
            min_std=0.01,
            max_std=1.0,
            init_std=0.5,
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[64, 64],
            activation="tanh",
            ortho_init=True,
            init_gain=1.0,
        )
    )
