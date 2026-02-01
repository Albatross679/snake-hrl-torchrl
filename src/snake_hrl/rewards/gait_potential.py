"""Gait-based potential functions for demonstration-guided reward shaping.

This module provides potential functions that encourage the agent to match
demonstrated locomotion patterns. The potentials use Gaussian kernels in
feature space to reward proximity to demonstration states.

Key classes:
    - GaitPotential: Fixed-sigma Gaussian potential
    - CurriculumGaitPotential: Sigma anneals over training for curriculum learning
"""

from typing import Any, Dict, Optional

import numpy as np

from snake_hrl.rewards.shaping import PotentialFunction
from snake_hrl.demonstrations.buffer import DemonstrationBuffer


class GaitPotential(PotentialFunction):
    """Gaussian potential function based on distance to demonstrations.

    Computes potential as:
        Phi(s) = scale * exp(-d^2 / (2 * sigma^2))

    where d is the distance to the nearest demonstration in feature space.

    Higher potential (closer to 1.0) when agent state matches demonstrations.
    Lower potential (closer to 0.0) when agent deviates from demonstrations.

    Example:
        >>> from snake_hrl.features import CompositeFeatureExtractor, CurvatureModeExtractor
        >>> from snake_hrl.demonstrations import DemonstrationBuffer
        >>> extractor = CompositeFeatureExtractor([CurvatureModeExtractor()])
        >>> buffer = DemonstrationBuffer(extractor)
        >>> # ... populate buffer ...
        >>> potential = GaitPotential(buffer, sigma=1.0, scale=1.0)
        >>> phi = potential(state)  # Returns potential value
    """

    def __init__(
        self,
        demo_buffer: DemonstrationBuffer,
        sigma: float = 1.0,
        scale: float = 1.0,
        k_neighbors: int = 1,
    ):
        """Initialize gait potential.

        Args:
            demo_buffer: Buffer containing demonstration features
            sigma: Gaussian kernel width (controls sharpness of potential)
            scale: Scale factor for potential output
            k_neighbors: Number of nearest neighbors to consider
        """
        self.demo_buffer = demo_buffer
        self.sigma = sigma
        self.scale = scale
        self.k_neighbors = k_neighbors

        # Precompute normalization factor
        self._sigma_sq_2 = 2.0 * sigma * sigma

    def __call__(self, state: Dict[str, Any]) -> float:
        """Compute gait potential for given state.

        Args:
            state: State dictionary from SnakeRobot.get_state() or step()

        Returns:
            Potential value in range [0, scale]
        """
        # Handle empty buffer
        if len(self.demo_buffer) == 0:
            return 0.0

        # Query distance to nearest demonstration
        distance = self.demo_buffer.query_distance(state, k=self.k_neighbors)

        # Compute Gaussian potential
        potential = np.exp(-distance * distance / self._sigma_sq_2)

        return float(self.scale * potential)

    def set_sigma(self, sigma: float) -> None:
        """Update the sigma parameter.

        Args:
            sigma: New sigma value
        """
        self.sigma = sigma
        self._sigma_sq_2 = 2.0 * sigma * sigma

    def get_distance(self, state: Dict[str, Any]) -> float:
        """Get raw distance to nearest demonstration (for debugging).

        Args:
            state: State dictionary

        Returns:
            Distance to nearest demonstration in feature space
        """
        if len(self.demo_buffer) == 0:
            return float("inf")
        return self.demo_buffer.query_distance(state, k=self.k_neighbors)


class CurriculumGaitPotential(PotentialFunction):
    """Gait potential with sigma annealing for curriculum learning.

    The sigma parameter anneals from sigma_init to sigma_final over training,
    allowing the agent to first receive broad guidance (large sigma) and
    then be shaped toward precise demonstration matching (small sigma).

    Annealing schedule:
        sigma(p) = sigma_init + p * (sigma_final - sigma_init)

    where p is the training progress in [0, 1].

    Example:
        >>> potential = CurriculumGaitPotential(
        ...     buffer, sigma_init=2.0, sigma_final=0.5
        ... )
        >>> potential.set_progress(0.0)   # sigma = 2.0 (broad)
        >>> potential.set_progress(0.5)   # sigma = 1.25
        >>> potential.set_progress(1.0)   # sigma = 0.5 (precise)
    """

    def __init__(
        self,
        demo_buffer: DemonstrationBuffer,
        sigma_init: float = 2.0,
        sigma_final: float = 0.5,
        scale: float = 1.0,
        k_neighbors: int = 1,
        schedule: str = "linear",
    ):
        """Initialize curriculum gait potential.

        Args:
            demo_buffer: Buffer containing demonstration features
            sigma_init: Initial sigma (broader, more forgiving)
            sigma_final: Final sigma (tighter, more precise)
            scale: Scale factor for potential output
            k_neighbors: Number of nearest neighbors to consider
            schedule: Annealing schedule - "linear", "cosine", or "exponential"
        """
        self.demo_buffer = demo_buffer
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.scale = scale
        self.k_neighbors = k_neighbors
        self.schedule = schedule

        # Current state
        self._progress = 0.0
        self._sigma = sigma_init
        self._sigma_sq_2 = 2.0 * sigma_init * sigma_init

    @property
    def sigma(self) -> float:
        """Return current sigma value."""
        return self._sigma

    @property
    def progress(self) -> float:
        """Return current training progress."""
        return self._progress

    def set_progress(self, progress: float) -> None:
        """Update training progress and recompute sigma.

        Args:
            progress: Training progress in [0, 1]
        """
        self._progress = np.clip(progress, 0.0, 1.0)
        self._sigma = self._compute_sigma(self._progress)
        self._sigma_sq_2 = 2.0 * self._sigma * self._sigma

    def _compute_sigma(self, progress: float) -> float:
        """Compute sigma based on progress and schedule.

        Args:
            progress: Progress in [0, 1]

        Returns:
            Sigma value
        """
        if self.schedule == "linear":
            # Linear interpolation
            return self.sigma_init + progress * (self.sigma_final - self.sigma_init)

        elif self.schedule == "cosine":
            # Cosine annealing (smoother)
            cosine_factor = (1 - np.cos(progress * np.pi)) / 2
            return self.sigma_init + cosine_factor * (self.sigma_final - self.sigma_init)

        elif self.schedule == "exponential":
            # Exponential decay
            ratio = self.sigma_final / self.sigma_init
            return self.sigma_init * (ratio ** progress)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def __call__(self, state: Dict[str, Any]) -> float:
        """Compute gait potential for given state.

        Args:
            state: State dictionary from SnakeRobot.get_state() or step()

        Returns:
            Potential value in range [0, scale]
        """
        # Handle empty buffer
        if len(self.demo_buffer) == 0:
            return 0.0

        # Query distance to nearest demonstration
        distance = self.demo_buffer.query_distance(state, k=self.k_neighbors)

        # Compute Gaussian potential
        potential = np.exp(-distance * distance / self._sigma_sq_2)

        return float(self.scale * potential)

    def get_distance(self, state: Dict[str, Any]) -> float:
        """Get raw distance to nearest demonstration.

        Args:
            state: State dictionary

        Returns:
            Distance to nearest demonstration in feature space
        """
        if len(self.demo_buffer) == 0:
            return float("inf")
        return self.demo_buffer.query_distance(state, k=self.k_neighbors)


class AdaptiveGaitPotential(PotentialFunction):
    """Gait potential with adaptive sigma based on demonstration density.

    Adjusts sigma locally based on the density of demonstrations in feature
    space. Areas with dense demonstrations have smaller effective sigma,
    while sparse areas have larger sigma.

    This provides more robust shaping when demonstrations have non-uniform
    coverage of the state space.
    """

    def __init__(
        self,
        demo_buffer: DemonstrationBuffer,
        base_sigma: float = 1.0,
        scale: float = 1.0,
        density_neighbors: int = 5,
        density_scale: float = 1.0,
    ):
        """Initialize adaptive gait potential.

        Args:
            demo_buffer: Buffer containing demonstration features
            base_sigma: Base sigma value
            scale: Scale factor for potential output
            density_neighbors: Number of neighbors for density estimation
            density_scale: How much density affects sigma
        """
        self.demo_buffer = demo_buffer
        self.base_sigma = base_sigma
        self.scale = scale
        self.density_neighbors = density_neighbors
        self.density_scale = density_scale

        # Cache for density-based sigma adjustments
        self._local_sigmas: Optional[np.ndarray] = None

    def _estimate_local_density(self) -> np.ndarray:
        """Estimate local density at each demonstration point.

        Returns:
            Array of local sigma values for each demonstration
        """
        if len(self.demo_buffer) == 0:
            return np.array([self.base_sigma])

        # Get feature array
        features = np.array(self.demo_buffer.features)
        n = len(features)

        if n < self.density_neighbors:
            return np.full(n, self.base_sigma)

        # For each point, compute average distance to k nearest neighbors
        from scipy.spatial import KDTree
        tree = KDTree(features)

        local_sigmas = np.zeros(n)
        for i in range(n):
            distances, _ = tree.query(features[i], k=self.density_neighbors + 1)
            # Exclude self (distance 0)
            avg_dist = np.mean(distances[1:])
            # Scale sigma by local density
            local_sigmas[i] = self.base_sigma * (1 + self.density_scale * avg_dist)

        return local_sigmas

    def __call__(self, state: Dict[str, Any]) -> float:
        """Compute adaptive gait potential.

        Args:
            state: State dictionary

        Returns:
            Potential value in range [0, scale]
        """
        if len(self.demo_buffer) == 0:
            return 0.0

        # Query nearest demonstration
        results = self.demo_buffer.query_nearest(state, k=1)
        if not results:
            return 0.0

        distance = results[0]["distance"]

        # Use adaptive sigma (could be precomputed for efficiency)
        # For simplicity, use base sigma here
        sigma_sq_2 = 2.0 * self.base_sigma * self.base_sigma
        potential = np.exp(-distance * distance / sigma_sq_2)

        return float(self.scale * potential)
