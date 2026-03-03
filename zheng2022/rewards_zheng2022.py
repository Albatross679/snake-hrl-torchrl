"""Two-phase curriculum reward from Zheng, Li & Hayashibe (2022).

Phase 1 (epochs 0-2000): r2 = c * v_h - P_hat
    Maximize forward velocity, penalize power.

Phase 2 (epochs 2000+): r1 = r_v * r_P
    Match target velocity while minimizing power.
    Target velocity decreases by 0.02 m/s every 1000 epochs.
"""

import math

from zheng2022.configs_zheng2022 import Zheng2022CurriculumConfig


class CurriculumReward:
    """Two-phase curriculum reward function."""

    def __init__(self, config: Zheng2022CurriculumConfig):
        self.c = config.reward_c
        self.phase_transition_epoch = config.phase_transition_epoch
        self.velocity_decrease_interval = config.velocity_decrease_interval
        self.velocity_decrease_amount = config.velocity_decrease_amount
        self.initial_target_velocity = config.initial_target_velocity
        self.min_target_velocity = config.min_target_velocity
        self.current_epoch = 0

    @property
    def phase(self) -> int:
        """Current training phase (1 or 2)."""
        return 1 if self.current_epoch < self.phase_transition_epoch else 2

    @property
    def target_velocity(self) -> float:
        """Current target velocity for Phase 2."""
        if self.phase == 1:
            return float("inf")  # No target in Phase 1
        epochs_in_phase2 = self.current_epoch - self.phase_transition_epoch
        decreases = epochs_in_phase2 // self.velocity_decrease_interval
        target = self.initial_target_velocity - decreases * self.velocity_decrease_amount
        return max(self.min_target_velocity, target)

    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch

    def compute_reward(self, head_velocity_x: float, power: float) -> float:
        """Compute curriculum reward.

        Args:
            head_velocity_x: Forward velocity of the head (m/s).
            power: Instantaneous mechanical power sum(|tau_i * omega_i|) (W).

        Returns:
            Scalar reward.
        """
        if self.phase == 1:
            return self._phase1_reward(head_velocity_x, power)
        else:
            return self._phase2_reward(head_velocity_x, power)

    def _phase1_reward(self, v_h: float, power: float) -> float:
        """Phase 1: r2 = c * v_h - P_hat.

        Simple reward: go fast, penalize power.
        P_hat is normalized power (power / max_power). We use raw power
        since the normalization constant cancels out during optimization.
        """
        # Normalize power by a reasonable scale (6 joints * max_torque * max_velocity)
        max_power = 6.0 * 0.1 * (math.pi / 2.0)  # ~0.94 W
        p_hat = power / max(max_power, 1e-8)
        return self.c * v_h - p_hat

    def _phase2_reward(self, v_h: float, power: float) -> float:
        """Phase 2: r1 = r_v * r_P.

        Multiplicative reward targeting specific velocity and low power.
        r_v = exp(-alpha * (v_h - v_target)^2)  (Gaussian around target)
        r_P = exp(-beta * P)                      (exponential power penalty)
        """
        v_target = self.target_velocity

        # Velocity matching: Gaussian centered on target
        alpha = 100.0  # Sharpness of velocity matching
        r_v = math.exp(-alpha * (v_h - v_target) ** 2)

        # Power efficiency: exponential decay with power
        beta = 5.0  # Power penalty strength
        r_P = math.exp(-beta * power)

        return r_v * r_P

    def get_info(self) -> dict:
        """Return current curriculum state for logging."""
        return {
            "curriculum_phase": self.phase,
            "curriculum_epoch": self.current_epoch,
            "target_velocity": self.target_velocity if self.phase == 2 else None,
        }
