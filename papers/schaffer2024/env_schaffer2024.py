"""TorchRL environment for biohybrid lattice worm (Schaffer et al., 2024).

A 3D lattice worm with 40 structural rods and 42 muscle rods, modeled using
PyElastica Cosserat rod theory. The agent controls muscle activations to
steer the worm toward navigation targets.

Observation:
    - worm centroid position: (3,)
    - worm centroid velocity: (3,)
    - previous action: (42,)
    - adaptive force ceilings (normalized): (42,)
    - target position: (3,)
    Total: 93

Action:
    - muscle activations: (42,) in [0, 1]
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

from schaffer2024.configs_schaffer2024 import Schaffer2024EnvConfig
from schaffer2024.muscle_schaffer2024 import MuscleAdaptationModel
from schaffer2024.rewards_schaffer2024 import compute_nan_penalty, compute_navigation_reward


class LatticeWormEnv(EnvBase):
    """Biohybrid lattice worm navigation environment.

    A lattice structure of Cosserat rods is actuated by 42 muscle rods.
    The agent outputs activation levels for each muscle to steer the worm
    centroid toward a target position.
    """

    # 8 target positions arranged radially at 100mm distance
    TARGET_ANGLES = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    def __init__(
        self,
        config: Optional[Schaffer2024EnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Schaffer2024EnvConfig()
        self._device = device

        # Muscle model
        self._muscles = MuscleAdaptationModel(self.config.muscles)
        self._num_muscles = self._muscles.num_muscles

        # RNG
        self._rng = np.random.default_rng(42)

        # Target state
        self._target_pos = np.zeros(3)
        self._target_idx = 0

        # Episode state
        self._step_count = 0
        self._prev_action = np.zeros(self._num_muscles)
        self._worm_pos = np.zeros(3)
        self._worm_vel = np.zeros(3)
        self._prev_worm_pos = np.zeros(3)

        # PyElastica state (initialized in _reset)
        self._simulator = None

        # Observation dim: 3 + 3 + 42 + 42 + 3 = 93
        self._obs_dim = 3 + 3 + self._num_muscles + self._num_muscles + 3

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
            high=1.0,
            shape=(self._num_muscles,),
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

    def _sample_target(self) -> np.ndarray:
        """Sample one of 8 radial target positions."""
        self._target_idx = self._rng.integers(0, len(self.TARGET_ANGLES))
        angle = self.TARGET_ANGLES[self._target_idx]
        dist = self.config.target.target_distance
        return np.array([dist * np.cos(angle), dist * np.sin(angle), 0.0])

    def _init_elastica(self) -> None:
        """Initialize PyElastica simulator with lattice worm structure.

        Creates structural and muscle rods in a lattice configuration.
        The actual PyElastica rod assembly is abstracted here — a full
        implementation would construct the 40+42 rod lattice using
        elastica.CosseratRod and connect them with joint constraints.
        """
        try:
            import elastica
        except ImportError:
            pass  # PyElastica optional; env degrades to kinematic placeholder

        # Placeholder: a real implementation would build the lattice here
        # using elastica.CosseratRod for each structural and muscle rod,
        # connected via elastica.FreeJoint constraints.
        self._simulator = None

    def _get_worm_state(self) -> tuple:
        """Get worm centroid position and velocity.

        Returns:
            (position, velocity) each shape (3,).
        """
        # In a full implementation, this aggregates over all structural rod
        # node positions. Here we use the tracked state.
        return self._worm_pos.copy(), self._worm_vel.copy()

    def _step_simulation(self, muscle_forces: np.ndarray) -> bool:
        """Advance the PyElastica simulation one control step.

        Args:
            muscle_forces: Scaled muscle activations, shape (num_muscles,).

        Returns:
            True if step succeeded, False if NaN detected.
        """
        # Placeholder: a full implementation would apply muscle_forces
        # as external forces on each muscle rod and call
        # self._simulator.step() for elastica_substeps iterations.
        #
        # For now, simulate simple kinematic response to muscle forces.
        physics = self.config.physics
        dt = physics.dt * physics.elastica_substeps

        # Simplified dynamics: net force → acceleration → position update
        net_force = np.zeros(3)
        n = self._num_muscles
        for i in range(n):
            angle = 2 * np.pi * i / n
            net_force[0] += muscle_forces[i] * np.cos(angle)
            net_force[1] += muscle_forces[i] * np.sin(angle)

        # Scale force to reasonable acceleration (mass ~ 0.01 kg)
        mass = 0.01
        accel = net_force / mass * 1e-6  # Scale down for stability
        self._worm_vel += accel * dt
        self._worm_vel *= 0.95  # Simple damping
        self._worm_pos += self._worm_vel * dt

        return not np.any(np.isnan(self._worm_pos))

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        pos, vel = self._get_worm_state()
        ceilings = self._muscles.get_normalized_ceilings()

        return np.concatenate([
            pos,
            vel,
            self._prev_action,
            ceilings,
            self._target_pos,
        ]).astype(np.float32)

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Update muscle ceilings from previous episode
        if self.config.enable_adaptation and self._step_count > 0:
            self._muscles.update_ceilings()
        self._muscles.reset_episode()

        # Initialize simulation
        self._init_elastica()

        # Reset state
        self._worm_pos = np.zeros(3)
        self._worm_vel = np.zeros(3)
        self._prev_worm_pos = np.zeros(3)
        self._prev_action = np.zeros(self._num_muscles)
        self._step_count = 0

        # Sample target
        self._target_pos = self._sample_target()

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

        # Scale actions by force ceilings
        muscle_forces = self._muscles.scale_actions(action)

        # Record pre-step position
        self._prev_worm_pos = self._worm_pos.copy()

        # Step simulation
        success = self._step_simulation(muscle_forces)

        if not success:
            reward = compute_nan_penalty()
            terminated = True
        else:
            # Accumulate muscle strain/force for adaptation
            # (simplified: strain ~ action magnitude, force ~ scaled force)
            self._muscles.accumulate(action, muscle_forces)

            # Compute reward
            reward = compute_navigation_reward(
                self._worm_pos,
                self._target_pos,
                threshold=self.config.target.target_threshold,
            )
            terminated = False

        self._prev_action = action.copy()
        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps

        obs = self._get_obs()

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
                "done": torch.tensor(
                    [terminated or truncated], dtype=torch.bool, device=self._device
                ),
                "terminated": torch.tensor(
                    [terminated], dtype=torch.bool, device=self._device
                ),
                "truncated": torch.tensor(
                    [truncated], dtype=torch.bool, device=self._device
                ),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def close(self):
        self._simulator = None
        super().close()
