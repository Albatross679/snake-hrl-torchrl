"""Configuration dataclasses for locomotion task (Bing et al., IJCAI 2019).

Hierarchy:
    MujocoPhysicsConfig → LocomotionPhysicsConfig
    PPOConfig           → LocomotionConfig (top-level project config)

Composable pieces:
    LocomotionPhysicsConfig  -- MuJoCo physics (overrides timestep, substeps)
    LocomotionEnvConfig      -- task selection, reward params, camera
    LocomotionNetworkConfig  -- 2x64 MLP matching original paper
    Checkpointing            -- model saving
    TensorBoard              -- metric logging
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.configs.base import Checkpointing, TensorBoard
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.physics import MujocoPhysicsConfig
from src.configs.training import PPOConfig


# ---------------------------------------------------------------------------
# Physics config (inherits MuJoCo backend, overrides for locomotion model)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionPhysicsConfig(MujocoPhysicsConfig):
    """MuJoCo physics for planar snake locomotion.

    Overrides MujocoPhysicsConfig defaults to match the original
    Bing et al. (IJCAI 2019) MuJoCo model.
    """

    mujoco_timestep: float = 0.0125
    mujoco_substeps: int = 4
    num_joints: int = 8

    @property
    def timestep(self) -> float:
        """Alias for mujoco_timestep (used by env)."""
        return self.mujoco_timestep

    @property
    def frame_skip(self) -> int:
        """Alias for mujoco_substeps (used by env)."""
        return self.mujoco_substeps


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class LocomotionEnvConfig:
    """Environment configuration for locomotion tasks."""

    physics: LocomotionPhysicsConfig = field(default_factory=LocomotionPhysicsConfig)

    # Episode settings
    max_episode_steps: int = 1000

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


# ---------------------------------------------------------------------------
# Network config (overrides for 2x64 MLP matching original paper)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionNetworkConfig(NetworkConfig):
    """Network config matching original 2x64 MLP."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[128, 64],
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
            hidden_dims=[128, 64],
            activation="tanh",
            ortho_init=True,
            init_gain=1.0,
        )
    )


# ---------------------------------------------------------------------------
# Project-level config (composes env + network + training + logging)
# ---------------------------------------------------------------------------


@dataclass
class LocomotionConfig(PPOConfig):
    """Top-level config for locomotion training (Bing et al., IJCAI 2019).

    Inherits PPO hyperparameters from PPOConfig. Only overrides values
    that differ from the parent defaults.
    """

    name: str = "locomotion"

    # Override PPO defaults for this task
    total_frames: int = 3_000_000
    frames_per_batch: int = 2048
    mini_batch_size: int = 64
    entropy_coef: float = 0.0  # Paper uses 0
    target_kl: Optional[float] = None  # No early stopping
    experiment_name: str = "locomotion"

    # Compose env + network + logging
    env: LocomotionEnvConfig = field(default_factory=LocomotionEnvConfig)
    network: LocomotionNetworkConfig = field(default_factory=LocomotionNetworkConfig)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)


# Backward-compat alias (scripts use this name)
LocomotionPPOConfig = LocomotionConfig
