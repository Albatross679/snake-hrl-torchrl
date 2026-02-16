"""Feature extraction base classes and composite extractors.

This module provides the foundation for compact feature extraction from snake
robot state dictionaries. Features are used for demonstration matching and
gait-based reward shaping.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors.

    Feature extractors transform high-dimensional state dictionaries from
    SnakeRobot.get_state() into compact, semantically meaningful feature
    vectors suitable for demonstration matching.

    State dict structure (from SnakeRobot.get_state()):
        - positions: (21, 3) - Snake node positions
        - velocities: (21, 3) - Snake node velocities
        - curvatures: (19,) - Current curvatures at joints
        - prey_position: (3,) - Prey center position
        - prey_distance: float - Distance from head to prey surface
        - contact_mask: (21,) bool - Which nodes contact prey
        - wrap_count: float - Number of wraps around prey
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        pass

    @abstractmethod
    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract compact features from state dictionary.

        Args:
            state: State dictionary from SnakeRobot.get_state() or sim.step()

        Returns:
            Feature vector of shape (feature_dim,)
        """
        pass

    def __call__(self, state: Dict[str, Any]) -> np.ndarray:
        """Convenience method - equivalent to extract()."""
        return self.extract(state)


class CompositeFeatureExtractor(FeatureExtractor):
    """Combines multiple feature extractors into a single feature vector.

    This allows modular composition of different feature types (e.g., curvature
    modes, virtual chassis, goal-relative features) into a unified representation.

    Example:
        extractor = CompositeFeatureExtractor([
            CurvatureModeExtractor(),
            VirtualChassisExtractor(),
            GoalRelativeExtractor(),
        ])
        features = extractor.extract(state)  # Shape: (16,)
    """

    def __init__(self, extractors: List[FeatureExtractor]):
        """Initialize with a list of extractors to compose.

        Args:
            extractors: List of feature extractors to combine
        """
        self.extractors = extractors
        self._feature_dim = sum(e.feature_dim for e in extractors)

        # Precompute slice indices for efficient extraction
        self._slices: List[slice] = []
        start = 0
        for extractor in extractors:
            end = start + extractor.feature_dim
            self._slices.append(slice(start, end))
            start = end

    @property
    def feature_dim(self) -> int:
        """Return total feature dimension across all extractors."""
        return self._feature_dim

    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract and concatenate features from all extractors.

        Args:
            state: State dictionary from SnakeRobot

        Returns:
            Concatenated feature vector of shape (feature_dim,)
        """
        features = np.zeros(self._feature_dim, dtype=np.float32)
        for extractor, slc in zip(self.extractors, self._slices):
            features[slc] = extractor.extract(state)
        return features

    def extract_components(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract features as a dictionary of named components.

        Useful for debugging and analysis.

        Args:
            state: State dictionary from SnakeRobot

        Returns:
            Dict mapping extractor class names to their feature vectors
        """
        return {
            type(extractor).__name__: extractor.extract(state)
            for extractor in self.extractors
        }
