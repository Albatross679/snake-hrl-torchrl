"""Configuration dataclasses for environments, networks, and training."""

from snake_hrl.configs.env import (
    EnvConfig,
    PhysicsConfig,
    ApproachEnvConfig,
    CoilEnvConfig,
    HRLEnvConfig,
    GaitConfig,
    CPGConfig,
)
from snake_hrl.configs.network import NetworkConfig, ActorConfig, CriticConfig
from snake_hrl.configs.training import TrainingConfig, PPOConfig, SACConfig, HRLConfig

__all__ = [
    # Environment configs
    "EnvConfig",
    "PhysicsConfig",
    "ApproachEnvConfig",
    "CoilEnvConfig",
    "HRLEnvConfig",
    # Gait and CPG configs
    "GaitConfig",
    "CPGConfig",
    # Network configs
    "NetworkConfig",
    "ActorConfig",
    "CriticConfig",
    # Training configs
    "TrainingConfig",
    "PPOConfig",
    "SACConfig",
    "HRLConfig",
]
