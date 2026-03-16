"""Configuration dataclasses for Liu et al. (2023) replication.

Hierarchy:
    MujocoPhysicsConfig → Liu2023PhysicsConfig
    Liu2023CPGConfig    (standalone)
    Liu2023CurriculumConfig (standalone)
    Liu2023EnvConfig    (composition of physics + cpg + curriculum)
    NetworkConfig → Liu2023NetworkConfig
    PPOConfig → Liu2023Config (top-level)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.configs.base import Checkpointing, TensorBoard
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.physics import MujocoPhysicsConfig
from src.configs.training import PPOConfig


# ---------------------------------------------------------------------------
# Physics config
# ---------------------------------------------------------------------------


@dataclass
class Liu2023PhysicsConfig(MujocoPhysicsConfig):
    """MuJoCo physics for 4-link soft snake (Liu et al. 2023)."""

    mujoco_timestep: float = 0.002
    mujoco_substeps: int = 25  # control dt = 0.002 * 25 = 0.05s
    num_links: int = 4
    num_joints: int = 3  # 4 links → 3 joints

    # Domain randomization ranges (Table IV)
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.5, 1.5)
    randomize_mass: bool = True
    mass_scale_range: Tuple[float, float] = (0.8, 1.2)
    randomize_gravity: bool = False
    gravity_range: Tuple[float, float] = (9.0, 10.0)
    randomize_max_pressure: bool = True
    max_pressure_range: Tuple[float, float] = (0.7, 1.0)

    @property
    def timestep(self) -> float:
        return self.mujoco_timestep

    @property
    def frame_skip(self) -> int:
        return self.mujoco_substeps


# ---------------------------------------------------------------------------
# CPG config
# ---------------------------------------------------------------------------


@dataclass
class Liu2023CPGConfig:
    """CPG parameters from Table II of Liu et al. (2023)."""

    # Matsuoka oscillator parameters
    a_psi: float = 2.0935
    b: float = 10.0355
    tau_r: float = 0.7696
    tau_a: float = 1.7728
    a_i: float = 4.6062
    w_ij: float = 8.8669
    w_ji: float = 0.7844
    c: float = 0.75
    u_max: float = 5.0

    # Frequency scaling options for PPOC (future)
    kf_options: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5]
    )
    kf_default: float = 1.0  # Fixed K_f for Phase 1 (PPO+CPG)


# ---------------------------------------------------------------------------
# Curriculum config
# ---------------------------------------------------------------------------


@dataclass
class Liu2023CurriculumConfig:
    """Curriculum settings from Table III."""

    num_levels: int = 12
    success_threshold: float = 0.9
    eval_window: int = 100
    enabled: bool = True


# ---------------------------------------------------------------------------
# Environment config (composition)
# ---------------------------------------------------------------------------


@dataclass
class Liu2023EnvConfig:
    """Environment configuration for Liu 2023 goal-tracking task."""

    physics: Liu2023PhysicsConfig = field(default_factory=Liu2023PhysicsConfig)
    cpg: Liu2023CPGConfig = field(default_factory=Liu2023CPGConfig)
    curriculum: Liu2023CurriculumConfig = field(default_factory=Liu2023CurriculumConfig)

    # Episode limits
    max_episode_steps: int = 1200
    starvation_timeout_steps: int = 60

    # Reward coefficients
    reward_c_v: float = 1.0
    reward_c_g: float = 0.5

    device: str = "cpu"


# ---------------------------------------------------------------------------
# Network config (4-layer 128×128 MLP)
# ---------------------------------------------------------------------------


@dataclass
class Liu2023NetworkConfig(NetworkConfig):
    """Network config: 4-layer 128-wide MLP with tanh, matching the paper."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[128, 128, 128, 128],
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
            hidden_dims=[128, 128, 128, 128],
            activation="tanh",
            ortho_init=True,
            init_gain=1.0,
        )
    )


# ---------------------------------------------------------------------------
# Top-level training config
# ---------------------------------------------------------------------------


@dataclass
class Liu2023Config(PPOConfig):
    """Top-level config for Liu 2023 replication (Phase 1: PPO+CPG).

    Inherits PPO hyperparameters from PPOConfig.
    """

    name: str = "liu2023"

    # PPO overrides
    total_frames: int = 5_000_000
    frames_per_batch: int = 2048
    mini_batch_size: int = 64
    learning_rate: float = 5e-4
    entropy_coef: float = 0.01
    target_kl: Optional[float] = None
    experiment_name: str = "liu2023_cpg_locomotion"

    # Composed configs
    env: Liu2023EnvConfig = field(default_factory=Liu2023EnvConfig)
    network: Liu2023NetworkConfig = field(default_factory=Liu2023NetworkConfig)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)

    # Parallelism
    num_workers: int = 4
