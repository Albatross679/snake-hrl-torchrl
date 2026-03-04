"""TorchRL Environment implementations for snake predation."""

from .base_env import BaseSnakeEnv

try:
    from .approach_env import ApproachEnv
    from .coil_env import CoilEnv
    from .hrl_env import HRLEnv
except ImportError:
    pass

__all__ = ["BaseSnakeEnv", "ApproachEnv", "CoilEnv", "HRLEnv"]
