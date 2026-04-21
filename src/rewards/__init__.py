"""Reward shaping functions for RL training.

This module provides potential-based reward shaping (PBRS) implementations
for guiding snake locomotion learning.

Key classes:
    - PotentialFunction: Abstract base for potential functions
    - PotentialBasedRewardShaping: PBRS wrapper that computes gamma*Phi(s') - Phi(s)
    - ApproachPotential: Distance and velocity-based potential for approach task
    - CoilPotential: Contact and wrap-based potential for coil task
    - GaitPotential: Demonstration-based Gaussian potential for gait matching
    - CurriculumGaitPotential: Gait potential with sigma annealing

Factory functions:
    - create_approach_shaper: Create PBRS for approach task
    - create_coil_shaper: Create PBRS for coil task
    - create_gait_shaper: Create PBRS with gait potential
    - create_curriculum_gait_shaper: Create PBRS with curriculum gait potential
"""

from .shaping import (
    PotentialFunction,
    PotentialBasedRewardShaping,
    CompositeRewardShaping,
    ApproachPotential,
    CoilPotential,
    create_approach_shaper,
    create_coil_shaper,
    create_full_task_shaper,
    create_gait_shaper,
    create_curriculum_gait_shaper,
)
from .gait_potential import (
    GaitPotential,
    CurriculumGaitPotential,
    AdaptiveGaitPotential,
)

__all__ = [
    # Base classes
    "PotentialFunction",
    "PotentialBasedRewardShaping",
    "CompositeRewardShaping",
    # Task potentials
    "ApproachPotential",
    "CoilPotential",
    # Gait potentials
    "GaitPotential",
    "CurriculumGaitPotential",
    "AdaptiveGaitPotential",
    # Factory functions
    "create_approach_shaper",
    "create_coil_shaper",
    "create_full_task_shaper",
    "create_gait_shaper",
    "create_curriculum_gait_shaper",
]
