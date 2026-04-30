"""Feature extraction module for compact state representations.

This module provides feature extractors that transform high-dimensional snake
robot states into compact, semantically meaningful features suitable for
demonstration matching and gait-based reward shaping.

Feature Extractors:
    - CurvatureModeExtractor: Serpenoid parameters (amplitude, wave_number, phase)
    - VirtualChassisExtractor: Body-frame state (CoG, orientation, angular velocity)
    - GoalRelativeExtractor: Goal-relative features (direction to prey, distance)
    - CompositeFeatureExtractor: Combines multiple extractors into single vector

Example:
    >>> from observations import (
    ...     CompositeFeatureExtractor,
    ...     CurvatureModeExtractor,
    ...     VirtualChassisExtractor,
    ...     GoalRelativeExtractor,
    ... )
    >>> extractor = CompositeFeatureExtractor([
    ...     CurvatureModeExtractor(),
    ...     VirtualChassisExtractor(),
    ...     GoalRelativeExtractor(),
    ... ])
    >>> features = extractor.extract(state)  # Shape: (16,)
"""

from .extractors import FeatureExtractor, CompositeFeatureExtractor
from .curvature_modes import CurvatureModeExtractor, ExtendedCurvatureModeExtractor
from .virtual_chassis import (
    VirtualChassisExtractor,
    GoalRelativeExtractor,
    BodyFrameGoalExtractor,
)
from .contact_features import ContactFeatureExtractor, ExtendedContactFeatureExtractor

__all__ = [
    "FeatureExtractor",
    "CompositeFeatureExtractor",
    "CurvatureModeExtractor",
    "ExtendedCurvatureModeExtractor",
    "VirtualChassisExtractor",
    "GoalRelativeExtractor",
    "BodyFrameGoalExtractor",
    "ContactFeatureExtractor",
    "ExtendedContactFeatureExtractor",
]
