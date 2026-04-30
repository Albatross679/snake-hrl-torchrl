"""Contact feature extraction for coiling behavior.

This module provides feature extractors that capture contact state between
the snake and prey, designed specifically for the coiling skill.
"""

from typing import Any, Dict

import numpy as np

from .extractors import FeatureExtractor


class ContactFeatureExtractor(FeatureExtractor):
    """Extract contact features between snake and prey.

    Captures the contact state which is critical for coiling behavior:
    - What fraction of the snake body is in contact with prey
    - How many times the snake has wrapped around the prey
    - Distribution of contact along the body (head, middle, tail regions)

    Output features (6 dims):
        [0]: contact_fraction - fraction of nodes in contact with prey [0, 1]
        [1]: wrap_count - number of complete wraps around prey (normalized)
        [2]: head_contact - contact density in head region (first 1/3 of body)
        [3]: mid_contact - contact density in middle region (middle 1/3)
        [4]: tail_contact - contact density in tail region (last 1/3)
        [5]: contact_continuity - measure of how continuous the contact is [0, 1]

    The contact distribution features help the policy understand which
    parts of the body are engaged with the prey, enabling more sophisticated
    coiling strategies.
    """

    def __init__(self, normalize: bool = True, max_wraps: float = 3.0):
        """Initialize contact feature extractor.

        Args:
            normalize: If True, normalize features to approximately [0, 1] range
            max_wraps: Maximum expected wraps for normalization
        """
        self.normalize = normalize
        self.max_wraps = max_wraps

    @property
    def feature_dim(self) -> int:
        """Return feature dimension."""
        return 6

    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract contact features from state.

        Args:
            state: State dict containing 'contact_mask', 'contact_fraction',
                   'wrap_count' or 'wrap_angle' keys

        Returns:
            Feature vector of shape (6,)
        """
        features = np.zeros(6, dtype=np.float32)

        # Get contact fraction (already in [0, 1])
        contact_fraction = state.get("contact_fraction", 0.0)
        features[0] = float(contact_fraction)

        # Get wrap count
        wrap_count = state.get("wrap_count", 0.0)
        if wrap_count == 0.0 and "wrap_angle" in state:
            wrap_count = state["wrap_angle"] / (2 * np.pi)
        features[1] = float(wrap_count)

        # Get contact mask for regional analysis
        contact_mask = state.get("contact_mask", None)
        if contact_mask is not None:
            contact_mask = np.asarray(contact_mask, dtype=bool)
            n_nodes = len(contact_mask)

            if n_nodes >= 3:
                # Divide body into three regions
                third = n_nodes // 3

                # Head region (first third) - index 0 is the head
                head_region = contact_mask[:third]
                features[2] = np.mean(head_region) if len(head_region) > 0 else 0.0

                # Middle region
                mid_region = contact_mask[third:2*third]
                features[3] = np.mean(mid_region) if len(mid_region) > 0 else 0.0

                # Tail region (last third)
                tail_region = contact_mask[2*third:]
                features[4] = np.mean(tail_region) if len(tail_region) > 0 else 0.0

                # Contact continuity: measure how continuous the contact is
                # High continuity = contacts are clustered together
                # Low continuity = contacts are scattered
                features[5] = self._compute_contact_continuity(contact_mask)
            else:
                # Fallback for very few nodes
                features[2:5] = contact_fraction
                features[5] = 1.0 if contact_fraction > 0 else 0.0
        else:
            # No contact mask available, use contact_fraction for all regions
            features[2:5] = contact_fraction
            features[5] = 1.0 if contact_fraction > 0 else 0.0

        # Normalize if requested
        if self.normalize:
            # contact_fraction already in [0, 1]
            # wrap_count normalized by max_wraps
            features[1] = np.clip(features[1] / self.max_wraps, 0.0, 1.0)
            # Regional contacts already in [0, 1]
            # Continuity already in [0, 1]

        return features

    def _compute_contact_continuity(self, contact_mask: np.ndarray) -> float:
        """Compute how continuous/clustered the contacts are.

        Returns a value in [0, 1] where:
        - 1.0 = all contacts are in one continuous segment
        - 0.0 = contacts are maximally scattered

        Args:
            contact_mask: Boolean array indicating contact at each node

        Returns:
            Continuity score in [0, 1]
        """
        if not np.any(contact_mask):
            return 0.0

        n_contacts = np.sum(contact_mask)
        if n_contacts <= 1:
            return 1.0

        # Count the number of contact segments (runs of True values)
        # Ideal coil has one continuous contact segment
        transitions = np.diff(contact_mask.astype(int))
        n_segments = np.sum(transitions == 1)  # Count rising edges
        if contact_mask[0]:
            n_segments += 1  # Account for segment starting at index 0

        if n_segments == 0:
            return 0.0

        # Continuity: 1 segment is ideal, more segments = less continuous
        # Score = 1/n_segments, but capped at [0, 1]
        continuity = 1.0 / n_segments

        return float(continuity)


class ExtendedContactFeatureExtractor(FeatureExtractor):
    """Extended contact feature extractor with additional coiling metrics.

    Provides more detailed contact information for advanced coiling policies:
    - Basic contact features (6 dims from ContactFeatureExtractor)
    - Constriction pressure estimate
    - Contact velocity (how fast contact is being established/lost)

    Output features (8 dims):
        [0:6]: Basic contact features (see ContactFeatureExtractor)
        [6]: constriction_tightness - estimate of how tightly wrapped [0, 1]
        [7]: contact_velocity - rate of change of contact_fraction [-1, 1]
    """

    def __init__(
        self,
        normalize: bool = True,
        max_wraps: float = 3.0,
        dt: float = 0.05,
    ):
        """Initialize extended contact feature extractor.

        Args:
            normalize: If True, normalize features
            max_wraps: Maximum expected wraps for normalization
            dt: Timestep for velocity computation
        """
        self.normalize = normalize
        self.max_wraps = max_wraps
        self.dt = dt

        # Base extractor
        self._base_extractor = ContactFeatureExtractor(
            normalize=normalize, max_wraps=max_wraps
        )

        # State for computing velocity
        self._prev_contact_fraction = 0.0

    @property
    def feature_dim(self) -> int:
        """Return feature dimension."""
        return 8

    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract extended contact features.

        Args:
            state: State dict with contact information

        Returns:
            Feature vector of shape (8,)
        """
        features = np.zeros(8, dtype=np.float32)

        # Get base features
        features[:6] = self._base_extractor.extract(state)

        # Constriction tightness: combination of wrap count and contact fraction
        # High tightness = wrapped AND in contact
        wrap_count = state.get("wrap_count", 0.0)
        contact_fraction = state.get("contact_fraction", 0.0)
        # Geometric mean emphasizes that both are needed
        features[6] = np.sqrt(
            np.clip(wrap_count / self.max_wraps, 0, 1) * contact_fraction
        )

        # Contact velocity: rate of change of contact fraction
        contact_velocity = (contact_fraction - self._prev_contact_fraction) / self.dt
        self._prev_contact_fraction = contact_fraction

        # Normalize velocity to approximately [-1, 1]
        # Assume max change is full contact in one step
        if self.normalize:
            features[7] = np.clip(contact_velocity, -1.0, 1.0)
        else:
            features[7] = contact_velocity

        return features

    def reset(self) -> None:
        """Reset internal state (call at episode start)."""
        self._prev_contact_fraction = 0.0
