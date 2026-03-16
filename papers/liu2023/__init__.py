"""CPG-regulated locomotion for soft snake robot (Liu et al., 2023)."""

from liu2023.env_liu2023 import SoftSnakeEnv
from liu2023.cpg_liu2023 import LiuCPGNetwork
from liu2023.curriculum_liu2023 import CurriculumManager

__all__ = ["SoftSnakeEnv", "LiuCPGNetwork", "CurriculumManager"]
