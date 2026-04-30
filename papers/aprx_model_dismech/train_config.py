"""Configuration for surrogate model training and RL training (DisMech backend)."""

from dataclasses import dataclass, field
from typing import List

from src.configs.physics import PhysicsConfig, DismechConfig
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.training import PPOConfig


# ---------------------------------------------------------------------------
# Surrogate model architecture
# ---------------------------------------------------------------------------


@dataclass
class SurrogateModelConfig:
    """Architecture config for the surrogate MLP.

    Uses the 128-dim relative state representation:
        CoM (2) + heading sin/cos (2) + relative positions (42) +
        velocities (42) + yaw (20) + omega_z (20) = 128.
    """

    state_dim: int = 128             # relative state representation
    action_dim: int = 5
    time_encoding_dim: int = 4       # sin/cos phase (2) + sin/cos n_cycles (2)
    input_dim: int = 137             # state(128) + action(5) + time_encoding(4)
    output_dim: int = 128            # state delta prediction (relative repr)
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 1024, 1024, 1024])
    activation: str = "silu"
    use_layer_norm: bool = True
    dropout: float = 0.0
    predict_delta: bool = True       # next_state = current + model(input)


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------


@dataclass
class SurrogateTrainConfig:
    """Training hyperparameters for the surrogate model (DisMech data)."""

    name: str = "surrogate_dismech"

    # Data (default: pre-processed 128-dim relative representation)
    data_dir: str = "data/surrogate_dismech_rl_step_rel128"
    val_fraction: float = 0.1

    # Training
    batch_size: int = 4096
    auto_batch_size: bool = True          # Probe for largest batch that fits 85% VRAM
    gradient_accumulation_steps: int = 1  # effective batch = batch_size * this
    use_amp: bool = True                  # bf16 autocast (Ampere+ GPUs)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 999_999  # Effectively infinite; early stopping or manual stop decides
    lr_schedule: str = "cosine"
    warmup_epochs: int = 5

    # Multi-step rollout loss
    rollout_steps: int = 8
    rollout_loss_weight: float = 0.1
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
    save_dir: str = "output/surrogate_dismech"
    patience: int = 30

    # Architecture variants (Phase 3.1)
    use_residual: bool = False       # Use ResidualSurrogateModel instead of base SurrogateModel
    history_k: int = 0               # History window size K for HistorySurrogateModel (0=disabled)

    # Model
    model: SurrogateModelConfig = field(default_factory=SurrogateModelConfig)

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "snake-hrl-surrogate-dismech"
    wandb_entity: str = ""


# ---------------------------------------------------------------------------
# Surrogate environment
# ---------------------------------------------------------------------------


@dataclass
class SurrogateEnvConfig:
    """Config for the GPU-batched surrogate TorchRL environment (DisMech)."""

    surrogate_checkpoint: str = ""   # path to trained surrogate model dir
    device: str = "auto"

    # DisMech physics config
    physics: DismechConfig = field(default_factory=DismechConfig)
    max_episode_steps: int = 500
    randomize_initial_heading: bool = True

    # Batched env
    batch_size: int = 256


# ---------------------------------------------------------------------------
# RL training with surrogate
# ---------------------------------------------------------------------------


@dataclass
class SurrogateNetworkConfig(NetworkConfig):
    """Network config for RL training with surrogate (same as locomotion)."""

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
    """Top-level config for PPO training with the surrogate environment (DisMech)."""

    name: str = "surrogate_locomotion_dismech"
    experiment_name: str = "surrogate_locomotion_dismech"

    # Same PPO hyperparameters as locomotion
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

    # No ParallelEnv needed -- surrogate is internally batched
    num_envs: int = 1
