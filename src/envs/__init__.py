"""TorchRL Environment implementations for snake predation."""

from envs.base_env import BaseSnakeEnv

try:
    from envs.approach_env import ApproachEnv
    from envs.coil_env import CoilEnv
    from envs.hrl_env import HRLEnv
except ImportError:
    pass

__all__ = ["BaseSnakeEnv", "ApproachEnv", "CoilEnv", "HRLEnv"]
