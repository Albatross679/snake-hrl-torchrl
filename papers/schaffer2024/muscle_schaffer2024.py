"""Muscle adaptation model for biohybrid worm (Schaffer et al., 2024).

Each of the 42 muscle rods has an adaptive force ceiling that increases
with exercise (strain and force experienced during episodes). This models
biological muscle strengthening from repeated activation.

Force ceiling update per episode:
    lambda_{m,i} = min(alpha_{m,i} * lambda_{m,i-1}, 2 * lambda_0)
    alpha_{m,i} = 1 + beta * |epsilon_{m,i-1}| + gamma * |F_{m,i-1}|

where:
    lambda_0 = initial force ceiling (2000 mN)
    beta = strain coefficient (1e-6)
    gamma = force coefficient (4e-8)
    epsilon = accumulated strain during episode
    F = accumulated force during episode
"""

import numpy as np

from schaffer2024.configs_schaffer2024 import MuscleConfig


class MuscleAdaptationModel:
    """Tracks and updates adaptive force ceilings for muscle rods.

    The force ceiling for each muscle increases based on accumulated
    strain and force during the previous episode, capped at 2x the
    initial value.
    """

    def __init__(self, config: MuscleConfig = None):
        self.config = config or MuscleConfig()
        n = self.config.num_muscles

        self._initial_ceiling = self.config.initial_force_ceiling
        self._max_ceiling = (
            self._initial_ceiling * self.config.max_force_ceiling_multiplier
        )

        # Per-muscle state
        self.force_ceilings = np.full(n, self._initial_ceiling, dtype=np.float64)
        self._episode_strains = np.zeros(n, dtype=np.float64)
        self._episode_forces = np.zeros(n, dtype=np.float64)

    @property
    def num_muscles(self) -> int:
        return self.config.num_muscles

    def reset_episode(self) -> None:
        """Reset per-episode accumulators (call at start of each episode)."""
        self._episode_strains[:] = 0.0
        self._episode_forces[:] = 0.0

    def accumulate(
        self, strains: np.ndarray, forces: np.ndarray
    ) -> None:
        """Accumulate strain and force magnitudes during a timestep.

        Args:
            strains: Muscle strains this step, shape (num_muscles,).
            forces: Muscle forces this step, shape (num_muscles,).
        """
        self._episode_strains += np.abs(strains)
        self._episode_forces += np.abs(forces)

    def update_ceilings(self) -> np.ndarray:
        """Update force ceilings at end of episode and return new values.

        Returns:
            Updated force ceilings, shape (num_muscles,).
        """
        beta = self.config.strain_coefficient
        gamma = self.config.force_coefficient

        alpha = (
            1.0
            + beta * self._episode_strains
            + gamma * self._episode_forces
        )
        self.force_ceilings = np.minimum(
            alpha * self.force_ceilings, self._max_ceiling
        )

        return self.force_ceilings.copy()

    def get_normalized_ceilings(self) -> np.ndarray:
        """Return ceilings normalized to [0, 1] range for observations."""
        return self.force_ceilings / self._max_ceiling

    def scale_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        """Scale raw actions [0, 1] by force ceilings.

        Args:
            raw_actions: Agent outputs in [0, 1], shape (num_muscles,).

        Returns:
            Scaled forces in [0, ceiling_m], shape (num_muscles,).
        """
        return raw_actions * self.force_ceilings
