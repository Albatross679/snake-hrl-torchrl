"""TorchRL environments for kinematic 3-link snake robots (Shi et al., 2020).

Two variants:
    1. WheeledSnakeEnv — nonholonomic wheeled robot on a plane
    2. SwimmingSnakeEnv — low Reynolds number swimmer in fluid

Both use the geometric mechanics kinematic model:
    ξ = -A(α) · α̇  (body velocity from shape velocity)

State: (α₁, α₂, θ) — joint angles + orientation (x,y removed by symmetry)
Action: discrete (α̇₁, α̇₂) — joint velocity commands
"""

from typing import Optional

import math
import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from shi2020.configs_shi2020 import (
    GaitTask,
    RobotType,
    Shi2020EnvConfig,
)
from shi2020.kinematics_shi2020 import ConnectionForm
from shi2020.rewards_shi2020 import compute_forward_reward, compute_reorientation_reward


def _build_action_table(a_max: float, a_interval: float) -> np.ndarray:
    """Build discrete action table (all joint velocity pairs, minus (0,0)).

    Returns:
        Action table, shape (num_actions, 2).
    """
    levels = np.arange(-a_max, a_max + a_interval / 2, a_interval)
    actions = []
    for a1 in levels:
        for a2 in levels:
            if abs(a1) < 1e-10 and abs(a2) < 1e-10:
                continue  # Remove null action
            actions.append([a1, a2])
    return np.array(actions)


class WheeledSnakeEnv(EnvBase):
    """Nonholonomic wheeled 3-link snake robot environment.

    The robot has 3 identical links on wheels. Nonholonomic constraints
    from the wheels define the kinematics via a fiber bundle connection.

    State: (α₁, α₂, θ) ∈ ℝ³
    Action: discrete index into joint velocity table
    """

    def __init__(
        self,
        config: Optional[Shi2020EnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Shi2020EnvConfig()
        self._device = device

        # Build action table
        act_cfg = self.config.action
        self._action_table = _build_action_table(act_cfg.a_max, act_cfg.a_interval)
        self._num_actions = len(self._action_table)

        # Connection form
        robot = self.config.robot
        self._connection = ConnectionForm(
            robot_type="wheeled",
            link_length=robot.link_length,
        )

        # State: (α₁, α₂, θ)
        self._state = np.zeros(3)
        # Position tracking (for reward/logging, not in state)
        self._position = np.zeros(2)  # (x, y) in world frame

        # Episode state
        self._step_count = 0
        self._rng = np.random.default_rng(42)

        self._make_spec()

    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(3,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        self.action_spec = DiscreteTensorSpec(
            n=self._num_actions,
            dtype=torch.int64,
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

    def _integrate_step(self, alpha_dot: np.ndarray) -> tuple:
        """Integrate kinematics for one action step.

        Returns:
            (delta_x_world, delta_y_world, delta_theta)
        """
        alpha = self._state[:2]
        theta = self._state[2]
        dt_action = self.config.action.t_interval

        # Euler integration of kinematic reconstruction equation
        num_substeps = 100
        dt = dt_action / num_substeps

        total_dx, total_dy, total_dtheta = 0.0, 0.0, 0.0

        for _ in range(num_substeps):
            world_vel = self._connection.world_velocity(alpha, alpha_dot, theta)
            dx = world_vel[0] * dt
            dy = world_vel[1] * dt
            dtheta = world_vel[2] * dt

            total_dx += dx
            total_dy += dy
            total_dtheta += dtheta

            # Update joint angles
            alpha = alpha + alpha_dot * dt
            theta = theta + dtheta

        # Apply joint limits
        limit = self.config.robot.joint_limit
        alpha = np.clip(alpha, -limit, limit)

        # Avoid singularity for wheeled robot
        if self.config.robot.avoid_singularity:
            alpha[0] = max(alpha[0], 0.01)
            alpha[1] = min(alpha[1], -0.01)

        self._state[:2] = alpha
        self._state[2] = theta
        self._position[0] += total_dx
        self._position[1] += total_dy

        return total_dx, total_dy, total_dtheta

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Random initial joint angles (away from singularity)
        self._state[0] = self._rng.uniform(0.5, 2.0)   # α₁ > 0
        self._state[1] = self._rng.uniform(-2.0, -0.5)  # α₂ < 0
        self._state[2] = 0.0  # θ = 0
        self._position[:] = 0.0
        self._step_count = 0

        obs = self._state.astype(np.float32)

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
        action_idx = tensordict["action"].item()
        alpha_dot = self._action_table[action_idx]

        dx, dy, dtheta = self._integrate_step(alpha_dot)

        # Compute reward based on task
        task = self.config.task
        dqn_cfg = None  # Reward weights come from DQNConfig at training time
        if task in (GaitTask.FORWARD, GaitTask.BACKWARD):
            sign = 1.0 if task == GaitTask.FORWARD else -1.0
            reward = compute_forward_reward(sign * dx, self._state[2])
        else:
            sign = 1.0 if task == GaitTask.ROTATE_LEFT else -1.0
            reward = compute_reorientation_reward(sign * dtheta)

        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps

        obs = self._state.astype(np.float32)

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


class SwimmingSnakeEnv(WheeledSnakeEnv):
    """Low Reynolds number swimming 3-link snake robot.

    Same structure as WheeledSnakeEnv but uses the swimming connection
    form (drag-based kinematics). No singularity avoidance needed.
    """

    def __init__(
        self,
        config: Optional[Shi2020EnvConfig] = None,
        device: str = "cpu",
    ):
        if config is None:
            config = Shi2020EnvConfig()
            config.robot.robot_type = RobotType.SWIMMING
            config.robot.avoid_singularity = False

        super().__init__(config=config, device=device)

        # Override connection form
        self._connection = ConnectionForm(
            robot_type="swimming",
            link_length=self.config.robot.link_length,
            drag_coefficient=self.config.robot.fluid_drag_coefficient,
        )

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Swimming robot: no singularity, free initial angles
        self._state[0] = self._rng.uniform(-1.5, 1.5)
        self._state[1] = self._rng.uniform(-1.5, 1.5)
        self._state[2] = 0.0
        self._position[:] = 0.0
        self._step_count = 0

        obs = self._state.astype(np.float32)

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
