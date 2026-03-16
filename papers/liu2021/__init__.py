"""Contact-aware CPG-based locomotion for soft snake robot (Liu, Onal & Fu, 2021)."""

from liu2021.env_liu2021 import ContactAwareSoftSnakeEnv
from liu2021.cpg_liu2021 import MatsuokaCPG

__all__ = ["ContactAwareSoftSnakeEnv", "MatsuokaCPG"]
