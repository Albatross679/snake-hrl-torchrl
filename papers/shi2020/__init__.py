"""DQN gait learning via geometric mechanics (Shi, Dear & Kelly, 2020)."""

from shi2020.env_shi2020 import WheeledSnakeEnv, SwimmingSnakeEnv
from shi2020.kinematics_shi2020 import ConnectionForm

__all__ = ["WheeledSnakeEnv", "SwimmingSnakeEnv", "ConnectionForm"]
