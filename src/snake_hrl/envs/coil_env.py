"""Coil skill environment - snake coils around prey."""

from typing import Optional, Dict, Any
import torch
import numpy as np

from tensordict import TensorDictBase

from snake_hrl.envs.base_env import BaseSnakeEnv
from snake_hrl.configs.env import CoilEnvConfig
from snake_hrl.rewards.shaping import (
    PotentialBasedRewardShaping,
    CoilPotential,
)


class CoilEnv(BaseSnakeEnv):
    """Environment for the coil skill.

    The snake must wrap around and constrict the prey.
    Success is achieving sufficient contact fraction and wrap count.

    Optionally includes gait-based reward shaping to encourage demonstration-like
    locomotion patterns.
    """

    def __init__(
        self,
        config: Optional[CoilEnvConfig] = None,
        device: str = "cpu",
        batch_size: Optional[torch.Size] = None,
    ):
        """Initialize coil environment.

        Args:
            config: Coil-specific configuration
            device: Device for tensors
            batch_size: Batch size for vectorized environments
        """
        self.coil_config = config or CoilEnvConfig()

        super().__init__(
            config=self.coil_config,
            device=device,
            batch_size=batch_size,
        )

        # Task reward shaping (contact + wrap + constriction)
        if self.coil_config.use_reward_shaping:
            self.reward_shaper = PotentialBasedRewardShaping(
                CoilPotential(
                    contact_weight=self.coil_config.contact_reward_weight,
                    wrap_weight=self.coil_config.wrap_reward_weight,
                    constriction_weight=self.coil_config.constriction_reward_weight,
                    target_wraps=self.coil_config.min_coil_wraps,
                ),
                gamma=0.99,
            )
        else:
            self.reward_shaper = None

        # Gait-based reward shaping
        self.gait_shaper: Optional[PotentialBasedRewardShaping] = None
        self.feature_extractor = None
        self.demo_buffer = None

        if self.coil_config.gait.use_gait_potential:
            self._setup_gait_potential()

        # Success tracking
        self._success_steps = 0

        # Start snake closer to prey for coil task
        self.coil_config.prey_position_range = (0.15, 0.25)

    def _setup_gait_potential(self) -> None:
        """Initialize gait-based reward shaping from demonstrations.

        Sets up feature extraction, demonstration buffer, and gait potential.
        Loads demonstrations from file or generates them if no path specified.
        """
        from snake_hrl.features import (
            CompositeFeatureExtractor,
            CurvatureModeExtractor,
            VirtualChassisExtractor,
            GoalRelativeExtractor,
        )
        from snake_hrl.demonstrations.buffer import DemonstrationBuffer
        from snake_hrl.demonstrations.generators import SerpenoidGenerator
        from snake_hrl.demonstrations.io import load_demonstrations, populate_buffer_from_trajectories
        from snake_hrl.rewards.gait_potential import GaitPotential, CurriculumGaitPotential

        gait_config = self.coil_config.gait

        # Build feature extractor based on config
        extractors = []
        extractor_mapping = {
            "CurvatureModeExtractor": CurvatureModeExtractor,
            "VirtualChassisExtractor": VirtualChassisExtractor,
            "GoalRelativeExtractor": GoalRelativeExtractor,
        }

        for name in gait_config.feature_extractors:
            if name in extractor_mapping:
                extractors.append(extractor_mapping[name]())

        if not extractors:
            # Default to all extractors
            extractors = [
                CurvatureModeExtractor(),
                VirtualChassisExtractor(),
                GoalRelativeExtractor(),
            ]

        self.feature_extractor = CompositeFeatureExtractor(extractors)

        # Create demonstration buffer
        self.demo_buffer = DemonstrationBuffer(self.feature_extractor)

        # Load or generate demonstrations
        if gait_config.demo_path:
            # Load from file
            demo_data = load_demonstrations(gait_config.demo_path)
            populate_buffer_from_trajectories(
                self.demo_buffer,
                demo_data["trajectories"],
                build_index=True,
            )
        else:
            # Generate demonstrations using serpenoid controller
            generator = SerpenoidGenerator(self.coil_config.physics)
            trajectories = generator.generate_batch(
                num_demos=gait_config.num_generated_demos,
                amplitude_range=gait_config.amplitude_range,
                wave_number_range=gait_config.wave_number_range,
                frequency_range=gait_config.frequency_range,
                duration=gait_config.demo_duration,
            )
            populate_buffer_from_trajectories(
                self.demo_buffer,
                trajectories,
                build_index=True,
            )

        # Create potential function
        if gait_config.potential_type == "curriculum":
            potential = CurriculumGaitPotential(
                self.demo_buffer,
                sigma_init=gait_config.sigma_init,
                sigma_final=gait_config.sigma_final,
                scale=1.0,
                schedule=gait_config.curriculum_schedule,
            )
        else:
            potential = GaitPotential(
                self.demo_buffer,
                sigma=gait_config.sigma,
                scale=1.0,
            )

        # Create PBRS wrapper
        self.gait_shaper = PotentialBasedRewardShaping(
            potential,
            gamma=0.99,
            scale=gait_config.gait_weight,
        )

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        """Reset coil environment.

        Starts snake close to prey (as if approach already completed).
        """
        result = super()._reset(tensordict)
        self._success_steps = 0

        if self.reward_shaper:
            self.reward_shaper.reset()

        if self.gait_shaper:
            self.gait_shaper.reset()

        return result

    def _compute_reward(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        action: np.ndarray,
    ) -> float:
        """Compute coil-specific reward.

        Clean architecture: Base rewards define true objectives,
        PBRS handles dense guidance (contact + wrap + constriction + gait).

        Args:
            prev_state: State before action
            curr_state: State after action
            action: Action taken

        Returns:
            Reward value
        """
        reward = 0.0
        contact_fraction = curr_state["contact_fraction"]

        # Stability reward (continuous incentive for maintaining low velocity while in contact)
        if contact_fraction > 0.5:
            velocity_mag = np.linalg.norm(curr_state["velocities"])
            stability = 1.0 / (1.0 + velocity_mag)
            reward += self.coil_config.stability_reward_weight * stability

        # Success bonus (sparse base reward)
        if self.is_success():
            reward += self.coil_config.success_bonus

        # Energy penalty (true objective)
        energy = np.sum(action**2)
        reward -= self.coil_config.energy_penalty_weight * energy

        # Task PBRS (contact + wrap + constriction)
        if self.reward_shaper:
            shaping = self.reward_shaper.compute_shaping_reward(
                prev_state, curr_state, done=False
            )
            reward += shaping

        # Gait PBRS (demonstration matching)
        if self.gait_shaper:
            gait_shaping = self.gait_shaper.compute_shaping_reward(
                prev_state, curr_state, done=False
            )
            reward += gait_shaping

        return reward * self.coil_config.reward_scale

    def set_training_progress(self, progress: float) -> None:
        """Update training progress for curriculum-based reward shaping.

        For curriculum gait potential, this anneals sigma from sigma_init
        to sigma_final as progress goes from 0 to 1.

        Args:
            progress: Training progress in [0, 1]
        """
        if self.gait_shaper is not None:
            potential = self.gait_shaper.potential_fn
            if hasattr(potential, "set_progress"):
                potential.set_progress(progress)

    def _check_terminated(self) -> bool:
        """Check if coil task is complete."""
        if not self.coil_config.terminate_on_success:
            return False

        if self.is_success():
            self._success_steps += 1
            if self._success_steps >= self.coil_config.success_hold_steps:
                return True
        else:
            self._success_steps = 0

        return False

    def is_success(self) -> bool:
        """Check if current state is a success state."""
        if self._current_state is None:
            return False

        contact_fraction = self._current_state["contact_fraction"]
        wrap_count = abs(self._current_state["wrap_count"])

        return (
            contact_fraction >= self.coil_config.contact_fraction_threshold
            and wrap_count >= self.coil_config.min_coil_wraps
        )

    def get_metrics(self) -> Dict[str, float]:
        """Get coil-specific metrics.

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "contact_fraction": self._current_state["contact_fraction"],
            "wrap_count": abs(self._current_state["wrap_count"]),
            "is_success": float(self.is_success()),
            "success_steps": self._success_steps,
            "episode_reward": self._episode_reward,
        }

        # Add gait metrics if available
        if self.gait_shaper is not None and self.demo_buffer is not None:
            gait_distance = self.demo_buffer.query_distance(self._current_state)
            metrics["gait_distance"] = gait_distance

            # Add curriculum sigma if using curriculum potential
            potential = self.gait_shaper.potential_fn
            if hasattr(potential, "sigma"):
                metrics["gait_sigma"] = potential.sigma

        return metrics

    def get_gait_potential(self, state: Optional[Dict[str, Any]] = None) -> float:
        """Get the current gait potential value for debugging.

        Args:
            state: State to evaluate (uses current state if None)

        Returns:
            Gait potential value, or 0.0 if gait shaping not enabled
        """
        if self.gait_shaper is None:
            return 0.0

        state = state or self._current_state
        if state is None:
            return 0.0

        return self.gait_shaper.potential_fn(state)
