"""Lateral undulatory gait equation for path following (Liu, Guo & Fang, 2022).

The gait execution layer generates joint angles from the lateral undulatory
gait equation:

    φⁱ(t) = α · sin(ωt + (i-1)δ) + φ_o

where φ_o is the gait offset output by the RL policy. This hierarchical
design ensures the resulting motion gait belongs to the lateral undulatory
family, enabling direct transfer to a physical robot.
"""

import numpy as np

from liu2022.configs_liu2022 import GaitConfig


class LateralUndulationGait:
    """Lateral undulatory gait generator.

    Produces joint angle commands from gait parameters + RL offset.
    The offset φ_o modifies the motion direction in real-time
    while maintaining the rhythmic undulatory pattern.
    """

    def __init__(self, num_joints: int = 8, config: GaitConfig = None):
        self.num_joints = num_joints
        self.config = config or GaitConfig()
        self._time = 0.0

    def reset(self):
        """Reset gait phase to zero."""
        self._time = 0.0

    def compute_joint_angles(
        self,
        phi_o: float,
        dt: float,
    ) -> np.ndarray:
        """Compute joint angles for current timestep.

        φⁱ(t) = α · sin(ωt + (i-1)δ) + φ_o

        Args:
            phi_o: Gait offset from RL policy (action).
            dt: Timestep to advance.

        Returns:
            Joint angles, shape (num_joints,).
        """
        self._time += dt
        cfg = self.config

        angles = np.zeros(self.num_joints)
        for i in range(self.num_joints):
            angles[i] = (
                cfg.amplitude * np.sin(
                    cfg.angular_freq * self._time + i * cfg.phase_diff
                )
                + phi_o
            )

        return angles

    def compute_head_angle(self, phi_o: float) -> float:
        """Compute the head (first joint) angle at current time.

        Used for visual localization stabilization term.

        Args:
            phi_o: Current gait offset.

        Returns:
            Head joint angle φ¹(t).
        """
        cfg = self.config
        return cfg.amplitude * np.sin(cfg.angular_freq * self._time) + phi_o

    @property
    def current_time(self) -> float:
        return self._time
