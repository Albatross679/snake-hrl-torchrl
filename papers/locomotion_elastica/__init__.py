"""Free-body snake locomotion via RL with serpenoid steering (PyElastica backend)."""

from locomotion_elastica.config import GaitType
from locomotion_elastica.env import LocomotionElasticaEnv

__all__ = ["LocomotionElasticaEnv", "GaitType"]
