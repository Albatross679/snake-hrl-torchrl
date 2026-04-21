"""TorchRL environment for soft pneumatic actuator with DD-PINN MPC (Licher et al., 2025).

Wraps the Cosserat rod simulation of a 3-chamber soft pneumatic actuator.
The DD-PINN serves as a fast dynamics surrogate for the MPC controller,
but this environment uses the full Cosserat rod model as ground truth.

Observation:
    - tip position: (3,)
    - tip orientation: (3,)
    - estimated bending compliance: (3,)
    - target position: (3,)
    Total: 12

Action:
    - pressure inputs for 3 chambers: (3,) in [0, max_pressure]
"""

from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from licher2025.configs_licher2025 import Licher2025EnvConfig


class SoftPneumaticEnv(EnvBase):
    """Soft pneumatic actuator tip-tracking environment.

    A 3-chamber fiber-reinforced actuator must track a reference trajectory
    with its tip. In the full pipeline, the MPC controller uses the DD-PINN
    to plan optimal pressure sequences.
    """

    def __init__(
        self,
        config: Optional[Licher2025EnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Licher2025EnvConfig()
        self._device = device

        self._num_chambers = self.config.physics.num_chambers
        self._max_pressure = self.config.mpc.max_pressure

        # RNG
        self._rng = np.random.default_rng(42)

        # State
        self._tip_pos = np.zeros(3)
        self._tip_orient = np.array([0.0, 0.0, 1.0])
        self._compliance = np.ones(3) * 1e-4  # Initial bending compliance
        self._target_pos = np.zeros(3)

        # Trajectory state
        self._step_count = 0
        self._trajectory_time = 0.0

        # Observation: tip_pos(3) + tip_orient(3) + compliance(3) + target(3) = 12
        self._obs_dim = 12

        self._make_spec()

    def _make_spec(self):
        """Define observation, action, reward, and done specs."""
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self._obs_dim,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        self.action_spec = BoundedTensorSpec(
            low=0.0,
            high=float(self._max_pressure),
            shape=(self._num_chambers,),
            dtype=torch.float32,
            device=self._device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), dtype=torch.float32, device=self._device
        )

        self.done_spec = CompositeSpec(
            done=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.bool, device=self._device
            ),
            terminated=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.bool, device=self._device
            ),
            truncated=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.bool, device=self._device
            ),
            shape=(),
        )

    def _generate_target(self, t: float) -> np.ndarray:
        """Generate target position at time t based on trajectory type."""
        L = self.config.physics.actuator_length
        target_type = self.config.target_type

        if target_type == "circular":
            radius = 0.3 * L
            omega = 2 * np.pi * 0.2  # 0.2 Hz
            return np.array([
                radius * np.cos(omega * t),
                radius * np.sin(omega * t),
                L,
            ])
        elif target_type == "figure_eight":
            radius = 0.25 * L
            omega = 2 * np.pi * 0.15
            return np.array([
                radius * np.sin(omega * t),
                radius * np.sin(2 * omega * t),
                L,
            ])
        elif target_type == "step":
            # Step changes every 2 seconds
            period = 2.0
            phase = int(t / period) % 4
            offsets = [
                [0.02, 0.0, L],
                [0.0, 0.02, L],
                [-0.02, 0.0, L],
                [0.0, -0.02, L],
            ]
            return np.array(offsets[phase])
        else:
            return np.array([0.0, 0.0, L])

    def _step_actuator(self, pressures: np.ndarray) -> None:
        """Simulate one control step of the soft actuator.

        In a full implementation, this would integrate the Cosserat rod
        equations with the given pressure inputs. Here we use a simplified
        kinematic model based on the pneumatic-to-curvature mapping.
        """
        L = self.config.physics.actuator_length
        dt = self.config.mpc.control_dt

        # Simplified: pressure → curvature → tip displacement
        # 3 chambers at 120-degree angles
        angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        kappa_x = np.sum(pressures * np.cos(angles)) * self._compliance[0]
        kappa_y = np.sum(pressures * np.sin(angles)) * self._compliance[1]

        # Tip position from constant-curvature approximation
        kappa = np.sqrt(kappa_x ** 2 + kappa_y ** 2)
        if kappa > 1e-8:
            phi = np.arctan2(kappa_y, kappa_x)
            theta = kappa * L
            tip_x = (1 - np.cos(theta)) / kappa * np.cos(phi)
            tip_y = (1 - np.cos(theta)) / kappa * np.sin(phi)
            tip_z = np.sin(theta) / kappa
        else:
            tip_x, tip_y, tip_z = 0.0, 0.0, L

        # Smooth update (low-pass filter for dynamics)
        alpha = 0.3
        new_pos = np.array([tip_x, tip_y, tip_z])
        self._tip_pos = alpha * new_pos + (1 - alpha) * self._tip_pos

        # Update orientation (tangent at tip)
        if kappa > 1e-8:
            self._tip_orient = np.array([
                np.sin(kappa * L) * np.cos(phi),
                np.sin(kappa * L) * np.sin(phi),
                np.cos(kappa * L),
            ])
        else:
            self._tip_orient = np.array([0.0, 0.0, 1.0])

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        return np.concatenate([
            self._tip_pos,
            self._tip_orient,
            self._compliance,
            self._target_pos,
        ]).astype(np.float32)

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        L = self.config.physics.actuator_length
        self._tip_pos = np.array([0.0, 0.0, L])
        self._tip_orient = np.array([0.0, 0.0, 1.0])
        self._compliance = np.ones(3) * 1e-4
        self._step_count = 0
        self._trajectory_time = 0.0

        self._target_pos = self._generate_target(0.0)

        obs = self._get_obs()

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "done": torch.tensor([False], dtype=torch.bool, device=self._device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self._device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self._device),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict["action"].cpu().numpy().astype(np.float64)

        # Step simulation
        self._step_actuator(action)

        # Advance time
        self._trajectory_time += self.config.mpc.control_dt
        self._target_pos = self._generate_target(self._trajectory_time)

        # Tracking reward (negative L2 distance)
        dist = np.linalg.norm(self._tip_pos - self._target_pos)
        reward = -dist ** 2

        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps

        obs = self._get_obs()

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
                "done": torch.tensor([truncated], dtype=torch.bool, device=self._device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self._device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self._device),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def close(self):
        super().close()
