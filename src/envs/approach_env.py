"""Approach skill environment - snake approaches prey."""

from typing import Optional, Dict, Any
import torch
import numpy as np

from tensordict import TensorDictBase

from envs.base_env import BaseSnakeEnv
from configs.env import ApproachEnvConfig
from rewards.shaping import (
    PotentialBasedRewardShaping,
    ApproachPotential,
)


class ApproachEnv(BaseSnakeEnv):
    """Environment for the approach skill.

    The snake must approach the prey until it's within striking distance.
    Success is reaching the approach_distance_threshold.

    Optionally includes gait-based reward shaping to encourage demonstration-like
    locomotion patterns.
    """

    def __init__(
        self,
        config: Optional[ApproachEnvConfig] = None,
        device: str = "cpu",
        batch_size: Optional[torch.Size] = None,
    ):
        """Initialize approach environment.

        Args:
            config: Approach-specific configuration
            device: Device for tensors
            batch_size: Batch size for vectorized environments
        """
        self.approach_config = config or ApproachEnvConfig()

        super().__init__(
            config=self.approach_config,
            device=device,
            batch_size=batch_size,
        )

        # Task reward shaping (distance + velocity)
        if self.approach_config.use_reward_shaping:
            self.reward_shaper = PotentialBasedRewardShaping(
                ApproachPotential(
                    max_distance=self.approach_config.prey_position_range[1] * 1.5,
                    distance_scale=self.approach_config.distance_reward_weight,
                    velocity_bonus_scale=self.approach_config.velocity_reward_weight,
                ),
                gamma=0.99,
            )
        else:
            self.reward_shaper = None

        # Gait-based reward shaping
        self.gait_shaper: Optional[PotentialBasedRewardShaping] = None
        self.feature_extractor = None
        self.demo_buffer = None

        if self.approach_config.gait.use_gait_potential:
            self._setup_gait_potential()

        # Success tracking
        self._success_steps = 0

    def _setup_gait_potential(self) -> None:
        """Initialize gait-based reward shaping from demonstrations.

        Sets up feature extraction, demonstration buffer, and gait potential.
        Loads demonstrations from file or generates them if no path specified.
        """
        from observations import (
            CompositeFeatureExtractor,
            CurvatureModeExtractor,
            VirtualChassisExtractor,
            GoalRelativeExtractor,
        )
        from demonstrations.buffer import DemonstrationBuffer
        from demonstrations.generators import SerpenoidGenerator
        from demonstrations.io import load_demonstrations, populate_buffer_from_trajectories
        from rewards.gait_potential import GaitPotential, CurriculumGaitPotential

        gait_config = self.approach_config.gait

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
            generator = SerpenoidGenerator(self.approach_config.physics)
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
        """Reset approach environment."""
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
        """Compute approach-specific reward.

        Clean architecture: Base rewards define true objectives,
        PBRS handles dense guidance (distance + velocity + gait).

        Args:
            prev_state: State before action
            curr_state: State after action
            action: Action taken

        Returns:
            Reward value
        """
        reward = 0.0

        # Energy penalty (true objective - want efficient behavior)
        energy = np.sum(action**2)
        reward -= self.approach_config.energy_penalty_weight * energy

        # Success bonus (sparse base reward)
        curr_dist = curr_state["prey_distance"]
        if curr_dist < self.approach_config.approach_distance_threshold:
            reward += self.approach_config.success_bonus

        # Task PBRS (distance + velocity)
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

        return reward * self.approach_config.reward_scale

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
        """Check if approach task is complete."""
        if not self.approach_config.terminate_on_success:
            return False

        curr_dist = self._current_state["prey_distance"]

        if curr_dist < self.approach_config.approach_distance_threshold:
            self._success_steps += 1
            if self._success_steps >= self.approach_config.success_hold_steps:
                return True
        else:
            self._success_steps = 0

        return False

    def is_success(self) -> bool:
        """Check if current state is a success state."""
        if self._current_state is None:
            return False
        return (
            self._current_state["prey_distance"]
            < self.approach_config.approach_distance_threshold
        )

    def get_metrics(self) -> Dict[str, float]:
        """Get approach-specific metrics.

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "prey_distance": self._current_state["prey_distance"],
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
