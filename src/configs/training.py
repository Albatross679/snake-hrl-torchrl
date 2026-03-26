"""Training configuration dataclasses for PPO, SAC, DDPG, and HRL.

Hierarchy:
    MLBaseConfig (name, seed, device, output_dir)
    └── RLConfig (total_frames, gamma, lr, num_envs, ...)
        ├── PPOConfig (clip_epsilon, gae_lambda, entropy/value coefs)
        ├── SACConfig (tau, alpha, replay buffer, ...)
        ├── DDPGConfig (tau, noise, replay buffer, ...)
        └── HRLConfig (skill configs, curriculum, ...)
"""

from dataclasses import dataclass, field
from typing import Optional, List

from .base import MLBaseConfig, WandB, Output, Console


@dataclass
class RLConfig(MLBaseConfig):
    """Base RL training configuration.

    Inherits name, seed, device, output_dir from MLBaseConfig.
    Adds all RL-specific training parameters.
    """

    # Override MLBaseConfig defaults — "auto" → GPU when available, else CPU
    device: str = "auto"

    # Training duration
    total_frames: int = 1_000_000
    max_wall_time: Optional[float] = None  # Wall-clock limit in seconds (None = no limit)
    frames_per_batch: int = 4096

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    weight_decay: float = 0.0

    # Batch processing
    num_epochs: int = 10
    mini_batch_size: int = 256

    # Value function
    gamma: float = 0.99  # Discount factor

    # Logging
    log_interval: int = 1
    eval_interval: int = 50
    save_interval: int = 100

    # Paths
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    experiment_name: str = "snake_hrl"

    # Parallelism
    num_workers: int = 1
    num_envs: int = 1  # Number of parallel environments (vectorized)

    # Mixed precision
    use_amp: bool = True  # bf16 autocast (disable for pre-Ampere GPUs)

    # Weights & Biases
    wandb: WandB = field(default_factory=WandB)

    # Output directory
    output: Output = field(default_factory=Output)

    # Console logging
    console: Console = field(default_factory=Console)

    @property
    def total_timesteps(self) -> int:
        """Alias: total_timesteps → total_frames."""
        return self.total_frames

    @total_timesteps.setter
    def total_timesteps(self, value: int) -> None:
        self.total_frames = value


# Backward-compat alias
TrainingConfig = RLConfig


@dataclass
class PPOConfig(RLConfig):
    """PPO-specific training configuration."""

    # PPO hyperparameters
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # GAE (Generalized Advantage Estimation)
    gae_lambda: float = 0.95
    normalize_advantage: bool = True

    # Value function clipping
    clip_value_loss: bool = True
    value_clip_epsilon: float = 0.2

    # Policy constraints
    target_kl: Optional[float] = 0.01  # Early stopping if KL exceeds this

    # Learning rate schedule
    lr_schedule: str = "linear"  # constant, linear, cosine
    lr_end: float = 1e-5

    # Early stopping (patience-based, measured in batches without reward improvement)
    # 0 = disabled. Default 200 batches.
    patience_batches: int = 200


@dataclass
class MMRKHSConfig(RLConfig):
    """MM-RKHS configuration (Gupta & Mahajan, 2026).

    Based on MM-RKHS algorithm (Gupta & Mahajan, 2026, arXiv:2603.17875).
    Adapts the majorization-minimization framework to continuous action
    spaces with neural network function approximation.

    Loss = -E[ratio * A] + beta * MMD^2(pi_new, pi_old) + (1/eta) * KL + value_coef * critic_loss
    """

    # Majorization bound coefficient (beta in paper Eq 6.1)
    beta: float = 1.0

    # Mirror descent step size (eta_k in paper Eq 7.1)
    eta: float = 1.0

    # MMD kernel configuration
    mmd_kernel: str = "rbf"
    mmd_bandwidth: float = 1.0
    mmd_num_samples: int = 16

    # GAE (shared with PPO)
    gae_lambda: float = 0.95
    normalize_advantage: bool = True

    # Critic loss weight
    value_coef: float = 0.5

    # Learning rate schedule
    lr_schedule: str = "linear"  # constant, linear, cosine
    lr_end: float = 1e-5

    # Early stopping (patience-based, measured in batches without reward improvement)
    # 0 = disabled. Default 200 batches.
    patience_batches: int = 200


@dataclass
class SACConfig(RLConfig):
    """SAC-specific training configuration."""

    # SAC hyperparameters
    tau: float = 0.005  # Target network update rate
    alpha: float = 0.2  # Entropy coefficient (initial)

    # Automatic entropy tuning
    auto_alpha: bool = True
    target_entropy: Optional[float] = None  # If None, set to -action_dim
    alpha_lr: float = 3e-4

    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256

    # Training
    warmup_steps: int = 10000  # Random actions before training
    update_frequency: int = 1  # How often to update (in env steps)
    num_updates: int = 1  # Updates per update step

    # Target network update frequency
    soft_update_period: int = 1  # Update target every N critic updates (1 = every update)

    # Critic
    num_critics: int = 2  # Number of Q-networks for min-Q trick
    critic_lr: float = 3e-4

    # Actor
    actor_lr: float = 3e-4
    actor_update_frequency: int = 2  # Update actor every N critic updates
    actor_max_grad_norm: Optional[float] = None  # Separate actor grad clip (None = use max_grad_norm)


@dataclass
class DDPGConfig(RLConfig):
    """DDPG-specific training configuration.

    Deterministic policy gradient with exploration noise and soft target updates.
    """

    # Soft target update rate
    tau: float = 0.001

    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256

    # Training
    warmup_steps: int = 10000
    update_frequency: int = 1
    num_updates: int = 1

    # Exploration noise
    noise_type: str = "ou"  # "ou" or "gaussian"
    noise_sigma: float = 0.2
    noise_theta: float = 0.15  # OU mean-reversion rate

    # Learning rates
    critic_lr: float = 1e-3
    actor_lr: float = 1e-4


@dataclass
class HRLConfig(RLConfig):
    """Hierarchical RL training configuration."""

    # Skill training configs
    approach_config: PPOConfig = field(default_factory=PPOConfig)
    coil_config: PPOConfig = field(default_factory=PPOConfig)

    # Manager training config
    manager_config: PPOConfig = field(default_factory=PPOConfig)

    # Training strategy
    training_strategy: str = "sequential"  # sequential, joint, pretrain_skills

    # Sequential training phases
    approach_frames: int = 500_000
    coil_frames: int = 500_000
    manager_frames: int = 500_000

    # Skill freezing
    freeze_skills_during_manager_training: bool = True

    # Intrinsic motivation
    use_intrinsic_reward: bool = False
    intrinsic_reward_scale: float = 0.1

    # Skill curriculum
    use_curriculum: bool = True
    curriculum_stages: List[str] = field(
        default_factory=lambda: ["approach_only", "coil_only", "full"]
    )
    curriculum_thresholds: List[float] = field(
        default_factory=lambda: [0.8, 0.8]  # Success rate to advance
    )

    def __post_init__(self):
        """Configure sub-configs."""
        # Adjust manager for discrete action space
        self.manager_config.entropy_coef = 0.05  # Higher entropy for exploration


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation."""

    # Evaluation settings
    num_episodes: int = 100
    deterministic: bool = True
    render: bool = False
    save_videos: bool = False
    video_dir: str = "./videos"

    # Metrics
    compute_success_rate: bool = True
    compute_efficiency: bool = True  # Steps to success

    # Checkpoints to evaluate
    checkpoint_path: Optional[str] = None
    evaluate_all_checkpoints: bool = False
