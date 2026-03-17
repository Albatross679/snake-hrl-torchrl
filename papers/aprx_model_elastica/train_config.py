"""Configuration for surrogate model training and RL training."""

from dataclasses import dataclass, field
from typing import List

from locomotion_elastica.config import (
    GoalConfig,
    LocomotionElasticaPhysicsConfig,
    LocomotionRewardConfig,
    SerpenoidControlConfig,
)
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.training import PPOConfig


# ---------------------------------------------------------------------------
# Surrogate model architecture
# ---------------------------------------------------------------------------


@dataclass
class SurrogateModelConfig:
    """Architecture config for the surrogate MLP.

    Uses the 130-dim relative state representation:
        CoM (2) + heading sin/cos (2) + CoM velocity (2) + relative positions (42) +
        relative velocities (42) + yaw (20) + omega_z (20) = 130.
    """

    state_dim: int = 130             # relative state representation
    action_dim: int = 5
    time_encoding_dim: int = 4       # sin/cos phase (2) + sin/cos n_cycles (2)
    input_dim: int = 139             # state(130) + action(5) + time_encoding(4)
    output_dim: int = 130            # state delta prediction (relative repr)
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 1024, 1024, 1024])
    activation: str = "silu"
    use_layer_norm: bool = True
    dropout: float = 0.1
    predict_delta: bool = True       # next_state = current + model(input)

    # Architecture selection: "mlp", "residual", or "transformer"
    arch: str = "mlp"

    # Transformer-specific (only used when arch="transformer")
    n_layers: int = 6                # number of transformer encoder layers
    n_heads: int = 8                 # number of attention heads
    d_model: int = 256               # transformer embedding dimension


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------


@dataclass
class SurrogateTrainConfig:
    """Training hyperparameters for the surrogate model."""

    name: str = "surrogate"

    # Data (raw 124-dim, converted to 130-dim on-the-fly via raw_to_relative())
    data_dir: str = "data/surrogate_rl_step"
    val_fraction: float = 0.1

    # Training
    batch_size: int = 4096
    auto_batch_size: bool = True          # Probe for largest batch that fits 85% VRAM
    gradient_accumulation_steps: int = 1  # effective batch = batch_size * this
    use_amp: bool = True                  # bf16 autocast (Ampere+ GPUs)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 999_999  # Effectively infinite; early stopping or manual stop decides
    lr_schedule: str = "cosine"
    warmup_epochs: int = 5

    # Multi-step rollout loss (disabled — flat data has 1 step per episode)
    rollout_steps: int = 8
    rollout_loss_weight: float = 0.0
    rollout_start_epoch: int = 20

    # Noise injection for robustness
    state_noise_std: float = 0.001

    # Normalization
    normalize_inputs: bool = True
    normalize_targets: bool = True

    # Inverse density weighting (upweight rare states, downweight common ones)
    use_density_weighting: bool = True
    density_bins: int = 20           # bins per feature dimension
    density_clip_max: float = 10.0   # max sample weight (prevents outlier domination)

    # Checkpointing
    save_dir: str = "output/surrogate"
    patience: int = 30

    # Architecture variants (Phase 3.1)
    use_residual: bool = False       # Use ResidualSurrogateModel instead of base SurrogateModel
    history_k: int = 0               # History window size K for HistorySurrogateModel (0=disabled)

    # Model
    model: SurrogateModelConfig = field(default_factory=SurrogateModelConfig)

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "snake-hrl-surrogate"
    wandb_entity: str = ""


# ---------------------------------------------------------------------------
# Surrogate environment
# ---------------------------------------------------------------------------


@dataclass
class SurrogateEnvConfig:
    """Config for the GPU-batched surrogate TorchRL environment."""

    surrogate_checkpoint: str = ""   # path to trained surrogate model dir
    device: str = "auto"

    # Inherited from locomotion_elastica (same physics/control/goal/reward)
    physics: LocomotionElasticaPhysicsConfig = field(
        default_factory=LocomotionElasticaPhysicsConfig
    )
    control: SerpenoidControlConfig = field(default_factory=SerpenoidControlConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    rewards: LocomotionRewardConfig = field(default_factory=LocomotionRewardConfig)
    max_episode_steps: int = 500
    randomize_initial_heading: bool = True

    # Batched env
    batch_size: int = 256


# ---------------------------------------------------------------------------
# RL training with surrogate
# ---------------------------------------------------------------------------


@dataclass
class SurrogateNetworkConfig(NetworkConfig):
    """Network config for RL training with surrogate (same as locomotion_elastica)."""

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


@dataclass
class SurrogateRLConfig(PPOConfig):
    """Top-level config for PPO training with the surrogate environment."""

    name: str = "surrogate_locomotion"
    experiment_name: str = "surrogate_locomotion"

    # Same PPO hyperparameters as locomotion_elastica
    total_frames: int = 2_000_000
    learning_rate: float = 1e-4
    frames_per_batch: int = 8192
    clip_epsilon: float = 0.2
    num_epochs: int = 4
    mini_batch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.02

    # Surrogate env config
    env: SurrogateEnvConfig = field(default_factory=SurrogateEnvConfig)
    network: SurrogateNetworkConfig = field(default_factory=SurrogateNetworkConfig)

    # No ParallelEnv needed — surrogate is internally batched
    num_envs: int = 1
