"""Hierarchical RL environment - manager coordinates skills."""

from typing import Optional, Dict, Any
import torch
import numpy as np

from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from tensordict import TensorDictBase

from snake_hrl.envs.base_env import BaseSnakeEnv
from snake_hrl.configs.env import HRLEnvConfig
from snake_hrl.rewards.shaping import (
    CompositeRewardShaping,
    ApproachPotential,
    CoilPotential,
)


class HRLEnv(BaseSnakeEnv):
    """Hierarchical RL environment for full snake predation task.

    The manager policy selects between skills (approach, coil),
    and the selected skill executes for a fixed duration.

    The task is successful when the snake:
    1. Approaches the prey
    2. Coils around and constricts the prey
    """

    def __init__(
        self,
        config: Optional[HRLEnvConfig] = None,
        device: str = "cpu",
        batch_size: Optional[torch.Size] = None,
    ):
        """Initialize HRL environment.

        Args:
            config: HRL-specific configuration
            device: Device for tensors
            batch_size: Batch size for vectorized environments
        """
        self.hrl_config = config or HRLEnvConfig()

        super().__init__(
            config=self.hrl_config,
            device=device,
            batch_size=batch_size,
        )

        # Task phase tracking
        self._current_skill = 0  # 0=approach, 1=coil
        self._skill_steps = 0
        self._approach_complete = False
        self._coil_complete = False

        # Reward shaping (PBRS for dense guidance)
        if self.hrl_config.use_reward_shaping:
            self.reward_shaper = CompositeRewardShaping(gamma=0.99)
            self.reward_shaper.add_shaper(
                "approach",
                ApproachPotential(
                    distance_scale=self.hrl_config.approach_config.distance_reward_weight,
                    velocity_bonus_scale=self.hrl_config.approach_config.velocity_reward_weight,
                ),
                weight=1.0,
            )
            self.reward_shaper.add_shaper(
                "coil",
                CoilPotential(
                    contact_weight=self.hrl_config.coil_config.contact_reward_weight,
                    wrap_weight=self.hrl_config.coil_config.wrap_reward_weight,
                    constriction_weight=self.hrl_config.coil_config.constriction_reward_weight,
                ),
                weight=1.5,
            )
        else:
            self.reward_shaper = None

        # Update specs for HRL
        self._make_hrl_spec()

    def _make_hrl_spec(self) -> None:
        """Add HRL-specific specs."""
        # Add skill selection to observation
        self.observation_spec["current_skill"] = DiscreteTensorSpec(
            n=self.hrl_config.num_skills,
            shape=(1,),
            dtype=torch.int64,
            device=self._device,
        )

        self.observation_spec["task_progress"] = UnboundedContinuousTensorSpec(
            shape=(2,),  # [approach_progress, coil_progress]
            dtype=torch.float32,
            device=self._device,
        )

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        """Reset HRL environment."""
        result = super()._reset(tensordict)

        # Reset task tracking
        self._current_skill = 0
        self._skill_steps = 0
        self._approach_complete = False
        self._coil_complete = False

        # Reset reward shaper
        if self.reward_shaper:
            self.reward_shaper.reset()

        # Add HRL-specific observations
        result["current_skill"] = torch.tensor(
            [self._current_skill], dtype=torch.int64, device=self._device
        )
        result["task_progress"] = torch.tensor(
            [0.0, 0.0], dtype=torch.float32, device=self._device
        )

        return result

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        result = super()._step(tensordict)

        self._skill_steps += 1

        # Update task progress
        approach_progress = self._compute_approach_progress()
        coil_progress = self._compute_coil_progress()

        result["current_skill"] = torch.tensor(
            [self._current_skill], dtype=torch.int64, device=self._device
        )
        result["task_progress"] = torch.tensor(
            [approach_progress, coil_progress], dtype=torch.float32, device=self._device
        )

        # Check for skill transitions
        if self._should_transition_skill():
            self._transition_skill()

        return result

    def _compute_reward(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        action: np.ndarray,
    ) -> float:
        """Compute HRL-specific reward.

        Clean architecture: Base rewards define true objectives,
        PBRS handles dense guidance for both phases.
        """
        reward = 0.0

        if not self._approach_complete:
            # Approach phase: sparse completion bonus only
            curr_dist = curr_state["prey_distance"]
            if curr_dist < self.hrl_config.approach_config.approach_distance_threshold:
                if not self._approach_complete:
                    reward += 10.0  # One-time bonus
                    self._approach_complete = True
        else:
            # Coil phase: stability + sparse completion
            contact_fraction = curr_state["contact_fraction"]

            # Stability reward (maintain low velocity while in contact)
            if contact_fraction > 0.5:
                velocity_mag = np.linalg.norm(curr_state["velocities"])
                stability = 1.0 / (1.0 + velocity_mag)
                reward += 0.5 * stability

            # Coil completion bonus
            if self._check_coil_success():
                if not self._coil_complete:
                    reward += self.hrl_config.task_completion_bonus
                    self._coil_complete = True

        # Energy penalty
        energy = np.sum(action**2)
        reward -= 0.001 * energy

        # Skill switch penalty
        if self._skill_steps == 1 and self._current_skill > 0:
            reward -= self.hrl_config.skill_switch_penalty

        # PBRS for dense guidance
        if self.reward_shaper:
            shaping_rewards = self.reward_shaper.compute_shaping_reward(
                prev_state, curr_state, done=False
            )
            reward += shaping_rewards["total"]

        return reward * self.hrl_config.reward_scale

    def _compute_approach_progress(self) -> float:
        """Compute progress towards approach goal (0 to 1)."""
        if self._current_state is None:
            return 0.0

        max_dist = self.hrl_config.prey_position_range[1]
        min_dist = self.hrl_config.approach_config.approach_distance_threshold
        curr_dist = self._current_state["prey_distance"]

        progress = (max_dist - curr_dist) / (max_dist - min_dist)
        return np.clip(progress, 0.0, 1.0)

    def _compute_coil_progress(self) -> float:
        """Compute progress towards coil goal (0 to 1)."""
        if self._current_state is None:
            return 0.0

        contact_fraction = self._current_state["contact_fraction"]
        wrap_count = abs(self._current_state["wrap_count"])

        # Weighted combination of contact and wrap progress
        contact_progress = contact_fraction / self.hrl_config.coil_config.contact_fraction_threshold
        wrap_progress = wrap_count / self.hrl_config.coil_config.min_coil_wraps

        progress = 0.5 * contact_progress + 0.5 * wrap_progress
        return np.clip(progress, 0.0, 1.0)

    def _should_transition_skill(self) -> bool:
        """Check if skill should transition."""
        if not self.hrl_config.allow_early_termination:
            return self._skill_steps >= self.hrl_config.skill_duration

        # Early termination conditions
        if self._current_skill == 0:  # Approach
            if self._approach_complete:
                return True
        elif self._current_skill == 1:  # Coil
            if self._coil_complete:
                return True

        return self._skill_steps >= self.hrl_config.skill_duration

    def _transition_skill(self) -> None:
        """Transition to next skill."""
        self._skill_steps = 0

        if self._current_skill == 0 and self._approach_complete:
            self._current_skill = 1  # Move to coil
        # Manager will select skill in HRL training

    def _check_coil_success(self) -> bool:
        """Check if coil task is successful."""
        if self._current_state is None:
            return False

        contact_fraction = self._current_state["contact_fraction"]
        wrap_count = abs(self._current_state["wrap_count"])

        return (
            contact_fraction >= self.hrl_config.coil_config.contact_fraction_threshold
            and wrap_count >= self.hrl_config.coil_config.min_coil_wraps
        )

    def _check_terminated(self) -> bool:
        """Check if full task is complete."""
        return self._approach_complete and self._coil_complete

    def is_success(self) -> bool:
        """Check if current state represents task success."""
        return self._approach_complete and self._coil_complete

    def get_metrics(self) -> Dict[str, float]:
        """Get HRL-specific metrics."""
        return {
            "approach_complete": float(self._approach_complete),
            "coil_complete": float(self._coil_complete),
            "task_success": float(self.is_success()),
            "current_skill": self._current_skill,
            "approach_progress": self._compute_approach_progress(),
            "coil_progress": self._compute_coil_progress(),
            "prey_distance": self._current_state["prey_distance"],
            "contact_fraction": self._current_state["contact_fraction"],
            "wrap_count": abs(self._current_state["wrap_count"]),
            "episode_reward": self._episode_reward,
        }

    def set_skill(self, skill_idx: int) -> None:
        """Set current skill (for manager policy).

        Args:
            skill_idx: Skill index (0=approach, 1=coil)
        """
        if skill_idx != self._current_skill:
            self._current_skill = skill_idx
            self._skill_steps = 0

    @property
    def current_skill(self) -> int:
        """Get current skill index."""
        return self._current_skill

    @property
    def approach_complete(self) -> bool:
        """Check if approach phase is complete."""
        return self._approach_complete

    @property
    def coil_complete(self) -> bool:
        """Check if coil phase is complete."""
        return self._coil_complete
