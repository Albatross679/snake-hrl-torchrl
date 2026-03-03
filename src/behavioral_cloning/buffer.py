"""Demonstration buffer with KDTree for efficient nearest-neighbor queries.

This module provides storage and retrieval of demonstration trajectories
using a KDTree for fast distance queries in feature space.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import KDTree

from observations.extractors import FeatureExtractor


class DemonstrationBuffer:
    """Buffer for storing and querying demonstration trajectories.

    Stores demonstrations as feature vectors and provides efficient
    nearest-neighbor queries using a KDTree.

    Example:
        >>> from observations import CompositeFeatureExtractor, CurvatureModeExtractor
        >>> extractor = CompositeFeatureExtractor([CurvatureModeExtractor()])
        >>> buffer = DemonstrationBuffer(extractor)
        >>> buffer.add_trajectory(states, trajectory_id=0)
        >>> buffer.build_index()
        >>> distance = buffer.query_distance(new_state)
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        """Initialize demonstration buffer.

        Args:
            feature_extractor: Feature extractor for converting states to features
        """
        self.feature_extractor = feature_extractor
        self.features: List[np.ndarray] = []
        self.trajectory_ids: List[int] = []
        self.timestamps: List[float] = []
        self._kdtree: Optional[KDTree] = None
        self._features_array: Optional[np.ndarray] = None

    @property
    def feature_dim(self) -> int:
        """Return the feature dimensionality."""
        return self.feature_extractor.feature_dim

    @property
    def num_samples(self) -> int:
        """Return total number of stored samples."""
        return len(self.features)

    @property
    def num_trajectories(self) -> int:
        """Return number of unique trajectories."""
        if not self.trajectory_ids:
            return 0
        return len(set(self.trajectory_ids))

    def add_trajectory(
        self,
        states: List[Dict[str, Any]],
        trajectory_id: int,
        timestamps: Optional[List[float]] = None,
    ) -> int:
        """Add a trajectory of states to the buffer.

        Args:
            states: List of state dictionaries from SnakeRobot.get_state() or step()
            trajectory_id: Unique identifier for this trajectory
            timestamps: Optional list of timestamps for each state

        Returns:
            Number of states added
        """
        if not states:
            return 0

        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = [float(i) for i in range(len(states))]

        # Extract features for each state
        for i, state in enumerate(states):
            try:
                feature = self.feature_extractor.extract(state)
                self.features.append(feature)
                self.trajectory_ids.append(trajectory_id)
                self.timestamps.append(timestamps[i])
            except Exception as e:
                # Skip states that fail feature extraction
                print(f"Warning: Failed to extract features from state {i}: {e}")
                continue

        # Invalidate cached KDTree
        self._kdtree = None
        self._features_array = None

        return len(states)

    def add_state(
        self,
        state: Dict[str, Any],
        trajectory_id: int,
        timestamp: float = 0.0,
    ) -> None:
        """Add a single state to the buffer.

        Args:
            state: State dictionary from SnakeRobot.get_state() or step()
            trajectory_id: Trajectory identifier
            timestamp: Timestamp for the state
        """
        feature = self.feature_extractor.extract(state)
        self.features.append(feature)
        self.trajectory_ids.append(trajectory_id)
        self.timestamps.append(timestamp)

        # Invalidate cached KDTree
        self._kdtree = None
        self._features_array = None

    def build_index(self) -> None:
        """Build or rebuild the KDTree index.

        Call this after adding all demonstrations to enable fast queries.
        The index is built lazily on first query if not explicitly called.
        """
        if not self.features:
            self._kdtree = None
            self._features_array = None
            return

        self._features_array = np.array(self.features, dtype=np.float64)
        self._kdtree = KDTree(self._features_array)

    def query_distance(self, state: Dict[str, Any], k: int = 1) -> float:
        """Query distance to nearest demonstration in feature space.

        Args:
            state: State dictionary to query
            k: Number of nearest neighbors to consider (returns mean distance)

        Returns:
            Distance to nearest demonstration (or mean of k nearest)
        """
        if not self.features:
            return float("inf")

        # Build index if needed
        if self._kdtree is None:
            self.build_index()

        # Extract features from query state
        query_features = self.feature_extractor.extract(state)

        # Query KDTree
        distances, _ = self._kdtree.query(query_features, k=min(k, len(self.features)))

        if k == 1:
            return float(distances)
        else:
            return float(np.mean(distances))

    def query_nearest(
        self, state: Dict[str, Any], k: int = 1
    ) -> List[Dict[str, Any]]:
        """Query the k nearest demonstrations to a state.

        Args:
            state: State dictionary to query
            k: Number of nearest neighbors to return

        Returns:
            List of dicts with 'distance', 'trajectory_id', 'timestamp', 'features'
        """
        if not self.features:
            return []

        # Build index if needed
        if self._kdtree is None:
            self.build_index()

        # Extract features from query state
        query_features = self.feature_extractor.extract(state)

        # Query KDTree
        k = min(k, len(self.features))
        distances, indices = self._kdtree.query(query_features, k=k)

        # Handle single result case
        if k == 1:
            distances = [distances]
            indices = [indices]

        results = []
        for dist, idx in zip(distances, indices):
            results.append({
                "distance": float(dist),
                "trajectory_id": self.trajectory_ids[idx],
                "timestamp": self.timestamps[idx],
                "features": self.features[idx].copy(),
            })

        return results

    def clear(self) -> None:
        """Clear all stored demonstrations."""
        self.features.clear()
        self.trajectory_ids.clear()
        self.timestamps.clear()
        self._kdtree = None
        self._features_array = None

    def get_feature_statistics(self) -> Dict[str, np.ndarray]:
        """Compute statistics over stored features.

        Returns:
            Dict with 'mean', 'std', 'min', 'max' arrays
        """
        if not self.features:
            dim = self.feature_dim
            return {
                "mean": np.zeros(dim),
                "std": np.ones(dim),
                "min": np.zeros(dim),
                "max": np.ones(dim),
            }

        features_array = np.array(self.features)
        return {
            "mean": np.mean(features_array, axis=0),
            "std": np.std(features_array, axis=0),
            "min": np.min(features_array, axis=0),
            "max": np.max(features_array, axis=0),
        }

    def __len__(self) -> int:
        """Return number of stored samples."""
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"DemonstrationBuffer(num_samples={self.num_samples}, "
            f"num_trajectories={self.num_trajectories}, "
            f"feature_dim={self.feature_dim})"
        )
