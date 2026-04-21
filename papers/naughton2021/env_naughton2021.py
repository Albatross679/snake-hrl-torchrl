"""TorchRL environment for Elastica-RL benchmark (Naughton et al., 2021).

A single Cosserat rod with one end clamped, controlled by distributed
torques at equidistant control points. Four benchmark tasks test
tracking, reaching, and obstacle navigation.

Observation varies by case:
    - rod node positions: (num_nodes * 3,)
    - rod node velocities: (num_nodes * 3,)
    - target position: (3,)
    - target orientation (Case 2 only): (3,)

Action:
    - torques at control points
    - Case 1: 12 DOF (6 pts * normal + binormal)
    - Case 2: 18 DOF (6 pts * normal + binormal + tangent)
    - Case 3: 2 DOF  (2 pts * normal)
    - Case 4: 4 DOF  (2 pts * normal + binormal)
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

from naughton2021.configs_naughton2021 import BenchmarkCase, Naughton2021EnvConfig
from naughton2021.rewards_naughton2021 import (
    compute_obstacle_reward,
    compute_reaching_reward,
    compute_tracking_reward,
)
from naughton2021.tasks_naughton2021 import ObstacleManager, TargetManager


class ElasticaControlEnv(EnvBase):
    """Elastica-RL compliant rod control benchmark environment.

    Wraps a PyElastica Cosserat rod simulation with distributed torque
    control for four benchmark control tasks.
    """

    def __init__(
        self,
        config: Optional[Naughton2021EnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Naughton2021EnvConfig()
        self._device = device

        physics = self.config.physics
        self._num_nodes = physics.geometry.num_nodes
        self._num_segments = physics.geometry.num_segments

        # RNG
        self._rng = np.random.default_rng(42)

        # Target and obstacle managers
        self._target = TargetManager(self.config.target, self._rng)
        self._obstacles = ObstacleManager(self.config.obstacles, self._rng)

        # Action dimension from control config
        self._action_dim = self.config.control.action_dim

        # Episode state
        self._step_count = 0

        # PyElastica state (initialized in _reset)
        self._simulator = None
        self._rod = None

        # Node state cache
        self._positions = np.zeros((self._num_nodes, 3))
        self._velocities = np.zeros((self._num_nodes, 3))

        self._make_spec()

    @property
    def _obs_dim(self) -> int:
        dim = self._num_nodes * 3  # positions
        dim += self._num_nodes * 3  # velocities
        dim += 3  # target position
        if self.config.case == BenchmarkCase.CASE2_REACHING:
            dim += 3  # target orientation
        return dim

    def _make_spec(self):
        """Define observation, action, reward, and done specs."""
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self._obs_dim,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        self.action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(self._action_dim,),
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

    def _init_elastica(self) -> None:
        """Initialize PyElastica simulator with a single clamped Cosserat rod.

        In a full implementation, this creates:
        - elastica.CosseratRod with the configured geometry and material
        - Clamped boundary condition at node 0
        - Gravity and damping forces
        - Contact with obstacles (Cases 3, 4)
        """
        try:
            import elastica
        except ImportError:
            pass  # PyElastica optional

        # Placeholder: initialize rod positions along z-axis (vertical rod)
        physics = self.config.physics
        segment_length = physics.geometry.snake_length / self._num_segments
        self._positions = np.zeros((self._num_nodes, 3))
        for i in range(self._num_nodes):
            self._positions[i, 2] = i * segment_length  # Vertical rod
        self._velocities = np.zeros((self._num_nodes, 3))

        self._simulator = None
        self._rod = None

    def _apply_torques(self, action: np.ndarray) -> None:
        """Apply torques at control points along the rod.

        Distributes the action vector to torques at equidistant points
        along the rod in the configured directions (normal, binormal,
        tangent).

        In a full implementation, this sets external_torques on the rod
        at the control point elements.
        """
        control = self.config.control
        n_cp = control.num_control_points
        dirs = control.torque_directions
        alpha = control.torque_scaling

        # Control point locations (equidistant along rod)
        cp_indices = np.linspace(0, self._num_segments - 1, n_cp, dtype=int)

        # Parse action into per-direction torques
        action = action * alpha
        idx = 0
        for cp_i in cp_indices:
            node_idx = min(cp_i, self._num_nodes - 2)
            for d in dirs:
                torque_val = action[idx]
                idx += 1
                # Simplified kinematic response
                if d == "normal":
                    self._positions[node_idx:, 0] += torque_val * 1e-4
                elif d == "binormal":
                    self._positions[node_idx:, 1] += torque_val * 1e-4
                elif d == "tangent":
                    self._positions[node_idx:, 2] += torque_val * 1e-5

        # Simple damping
        self._velocities *= 0.95

    def _get_tip_pos(self) -> np.ndarray:
        return self._positions[-1].copy()

    def _get_tip_tangent(self) -> np.ndarray:
        tangent = self._positions[-1] - self._positions[-2]
        norm = np.linalg.norm(tangent)
        return tangent / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

    def _get_obs(self) -> np.ndarray:
        parts = [
            self._positions.flatten(),
            self._velocities.flatten(),
            self._target.position,
        ]
        if self.config.case == BenchmarkCase.CASE2_REACHING:
            parts.append(self._target.orientation)
        return np.concatenate(parts).astype(np.float32)

    def _compute_reward(self) -> float:
        tip_pos = self._get_tip_pos()
        case = self.config.case

        if case == BenchmarkCase.CASE1_TRACKING:
            return compute_tracking_reward(tip_pos, self._target.position)

        elif case == BenchmarkCase.CASE2_REACHING:
            tip_tangent = self._get_tip_tangent()
            return compute_reaching_reward(
                tip_pos,
                tip_tangent,
                self._target.position,
                self._target.orientation,
            )

        elif case in (BenchmarkCase.CASE3_STRUCTURED, BenchmarkCase.CASE4_UNSTRUCTURED):
            penetrations = self._obstacles.compute_penetrations(self._positions)
            total_pen = float(np.sum(penetrations))
            return compute_obstacle_reward(
                tip_pos,
                self._target.position,
                total_pen,
                self.config.obstacles.collision_penalty,
            )

        return 0.0

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._target = TargetManager(self.config.target, self._rng)
            self._obstacles = ObstacleManager(self.config.obstacles, self._rng)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        self._init_elastica()

        # Setup target and obstacles
        rod_length = self.config.physics.geometry.snake_length
        self._target.sample(self.config.case, rod_length)
        self._obstacles.setup(self.config.case)

        self._step_count = 0

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

        # Apply torques and advance simulation
        self._apply_torques(action)

        # Move target (Case 1 only)
        if self.config.case == BenchmarkCase.CASE1_TRACKING:
            self._target.step(
                self.config.physics.dt * self.config.physics.elastica_substeps,
                self.config.target.workspace_radius,
            )

        # Reward
        reward = self._compute_reward()

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
        self._simulator = None
        self._rod = None
        super().close()
