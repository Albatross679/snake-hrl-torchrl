"""Training configuration dataclasses for PPO, SAC, and HRL."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Base training configuration."""

    # Training duration
    total_frames: int = 1_000_000
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
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100

    # Paths
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    experiment_name: str = "snake_hrl"

    # Device
    device: str = "cpu"
    num_workers: int = 1

    # Reproducibility
    seed: Optional[int] = 42


@dataclass
class PPOConfig(TrainingConfig):
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


@dataclass
class SACConfig(TrainingConfig):
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

    # Critic
    num_critics: int = 2  # Number of Q-networks for min-Q trick
    critic_lr: float = 3e-4

    # Actor
    actor_lr: float = 3e-4
    actor_update_frequency: int = 2  # Update actor every N critic updates


@dataclass
class HRLConfig(TrainingConfig):
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
