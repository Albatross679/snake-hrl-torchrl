"""Snake Hierarchical Reinforcement Learning Package.

A hierarchical RL framework for snake predation behavior using TorchRL.
"""

__version__ = "0.1.0"

from snake_hrl.envs import ApproachEnv, CoilEnv, HRLEnv, BaseSnakeEnv
from snake_hrl.trainers import PPOTrainer, SACTrainer, HRLTrainer

__all__ = [
    "ApproachEnv",
    "CoilEnv",
    "HRLEnv",
    "BaseSnakeEnv",
    "PPOTrainer",
    "SACTrainer",
    "HRLTrainer",
]
