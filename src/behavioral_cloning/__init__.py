"""Behavioral cloning module for demonstration generation and pretraining.

This module provides:
    - DemonstrationBuffer: Storage with KDTree for efficient nearest-neighbor queries
    - SerpenoidGenerator: Generate locomotion demos using serpenoid controllers
    - I/O utilities: Save and load demonstration data

Example:
    >>> from behavioral_cloning import (
    ...     DemonstrationBuffer,
    ...     SerpenoidGenerator,
    ...     save_demonstrations,
    ...     load_demonstrations,
    ... )
    >>> from observations import CompositeFeatureExtractor, CurvatureModeExtractor
    >>> from configs.env import PhysicsConfig
    >>>
    >>> # Generate demonstrations
    >>> generator = SerpenoidGenerator(PhysicsConfig())
    >>> trajectories = generator.generate_batch(num_demos=10, duration=5.0)
    >>>
    >>> # Save to disk
    >>> save_demonstrations(trajectories, "demos.pkl")
    >>>
    >>> # Create buffer and populate
    >>> extractor = CompositeFeatureExtractor([CurvatureModeExtractor()])
    >>> buffer = DemonstrationBuffer(extractor)
    >>> for i, traj in enumerate(trajectories):
    ...     buffer.add_trajectory(traj, trajectory_id=i)
    >>> buffer.build_index()
"""

from .buffer import DemonstrationBuffer
from .generators import (
    SerpenoidGenerator,
    LateralUndulationGenerator,
    SidewindingGenerator,
)
from .io import (
    save_demonstrations,
    load_demonstrations,
    save_buffer,
    load_buffer,
    export_to_json,
    load_from_json,
    populate_buffer_from_trajectories,
)
from .fitness import (
    compute_displacement_vector,
    compute_displacement_magnitude,
    compute_displacement_direction,
    evaluate_trajectory,
    compute_direction_bin,
    get_direction_bin_name,
    filter_successful_trajectories,
    compute_direction_coverage,
    get_best_parameters_per_direction,
)
from .approach_experiences import (
    ApproachExperienceBuffer,
    ApproachExperienceGenerator,
)
from .curvature_action_experiences import (
    CurvatureActionExperienceBuffer,
    CurvatureActionExperienceGenerator,
    analyze_trajectory_distribution,
)

__all__ = [
    # Buffer
    "DemonstrationBuffer",
    # Generators
    "SerpenoidGenerator",
    "LateralUndulationGenerator",
    "SidewindingGenerator",
    # I/O
    "save_demonstrations",
    "load_demonstrations",
    "save_buffer",
    "load_buffer",
    "export_to_json",
    "load_from_json",
    "populate_buffer_from_trajectories",
    # Fitness
    "compute_displacement_vector",
    "compute_displacement_magnitude",
    "compute_displacement_direction",
    "evaluate_trajectory",
    "compute_direction_bin",
    "get_direction_bin_name",
    "filter_successful_trajectories",
    "compute_direction_coverage",
    "get_best_parameters_per_direction",
    # Approach Experiences
    "ApproachExperienceBuffer",
    "ApproachExperienceGenerator",
    # Curvature Action Experiences
    "CurvatureActionExperienceBuffer",
    "CurvatureActionExperienceGenerator",
    "analyze_trajectory_distribution",
]
