"""TorchRL environment for hierarchical path following (Liu, Guo & Fang, 2022).

Hierarchical control:
    1. RL Policy Training Layer: outputs gait offset φ_o ∈ [-max_offset, max_offset]
    2. Gait Execution Layer: φⁱ(t) = α·sin(ωt + (i-1)δ) + φ_o

State s_t ∈ ℝ^{n+3}:
    d^p  — perpendicular distance to desired path
    d^c  — distance to target point on path
    θ_err — heading error relative to path tangent
    φ_{prev} — previous joint angles (n_joints,)

Action: scalar φ_o (gait offset)
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

from liu2022.configs_liu2022 import Liu2022EnvConfig, PathType
from liu2022.gait_liu2022 import LateralUndulationGait
from liu2022.rewards_liu2022 import compute_total_reward


class PathFollowingSnakeEnv(EnvBase):
    """9-link wheeled snake robot following desired paths.

    The RL agent outputs a scalar gait offset φ_o that modifies the lateral
    undulatory gait equation. The gait execution layer converts this into
    joint angle commands for the 9-link wheeled snake.

    Paths: straight lines, sinusoidal curves, circles.
    """

    def __init__(
        self,
        config: Optional[Liu2022EnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Liu2022EnvConfig()
        self._device = device

        # Gait generator
        self._gait = LateralUndulationGait(
            num_joints=self.config.physics.num_joints,
            config=self.config.gait,
        )

        # State
        self._head_pos = np.zeros(2)  # (x, y) world frame
        self._heading = 0.0           # θ orientation
        self._head_vel = np.zeros(2)
        self._prev_joint_angles = np.zeros(self.config.physics.num_joints)
        self._prev_head_angle = 0.0

        # Target point on path
        self._target_pos = np.zeros(2)

        # Episode
        self._step_count = 0
        self._rng = np.random.default_rng(42)

        # Observation dim: d^p + d^c + θ_err + prev_joint_angles
        self._obs_dim = 3 + self.config.physics.num_joints

        self._make_spec()

    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self._obs_dim,),
                dtype=torch.float32,
                device=self._device,
            ),
            shape=(),
        )

        # Action: scalar gait offset φ_o
        max_off = self.config.gait.max_offset
        self.action_spec = BoundedTensorSpec(
            low=-max_off,
            high=max_off,
            shape=(1,),
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

    # ------------------------------------------------------------------
    # Path geometry
    # ------------------------------------------------------------------

    def _path_y(self, x: float) -> float:
        """Evaluate desired path y-coordinate at given x."""
        path = self.config.path
        if path.path_type == PathType.STRAIGHT_LINE:
            return path.line_y
        elif path.path_type == PathType.SINUSOIDAL:
            return path.sin_amplitude * np.sin(path.sin_omega * x + path.sin_phase)
        elif path.path_type == PathType.CIRCLE:
            # Parametric circle: closest y given x
            r = path.circle_radius
            if abs(x) > r:
                return 0.0
            return np.sqrt(max(r**2 - x**2, 0.0))
        return 0.0

    def _path_tangent_angle(self, x: float) -> float:
        """Tangent angle of the desired path at given x."""
        path = self.config.path
        if path.path_type == PathType.STRAIGHT_LINE:
            return 0.0
        elif path.path_type == PathType.SINUSOIDAL:
            dy_dx = (
                path.sin_amplitude * path.sin_omega
                * np.cos(path.sin_omega * x + path.sin_phase)
            )
            return np.arctan(dy_dx)
        elif path.path_type == PathType.CIRCLE:
            r = path.circle_radius
            if abs(x) >= r:
                return np.pi / 2
            y = np.sqrt(max(r**2 - x**2, 0.0))
            return np.arctan2(-x, y) if y > 1e-8 else np.pi / 2
        return 0.0

    def _dist_to_path(self) -> float:
        """Signed perpendicular distance from head to desired path."""
        return self._head_pos[1] - self._path_y(self._head_pos[0])

    def _dist_to_target(self) -> float:
        """Euclidean distance from head to target point on path."""
        return np.linalg.norm(self._head_pos - self._target_pos)

    def _heading_error(self) -> float:
        """Heading error relative to path tangent."""
        tangent = self._path_tangent_angle(self._head_pos[0])
        err = self._heading - tangent
        # Wrap to [-π, π]
        return (err + np.pi) % (2 * np.pi) - np.pi

    def _sample_target(self) -> np.ndarray:
        """Sample a target point on the desired path ahead of the robot."""
        x_lo, x_hi = self.config.path.target_x_range
        tx = self._rng.uniform(x_lo, x_hi)
        ty = self._path_y(tx)
        return np.array([tx, ty])

    # ------------------------------------------------------------------
    # Simplified physics (9-link wheeled snake)
    # ------------------------------------------------------------------

    def _step_physics(self, joint_angles: np.ndarray, dt: float) -> None:
        """Advance simplified wheeled snake kinematics."""
        # Steering from mean joint angle offset
        mean_angle = np.mean(joint_angles)
        steering_rate = mean_angle * 0.5  # Simplified steering model

        self._heading += steering_rate * dt

        # Forward speed from oscillation amplitude
        osc_amplitude = np.std(joint_angles)
        speed = 0.1 + osc_amplitude * 0.3  # Base speed + oscillation contribution

        self._head_vel = np.array([
            speed * np.cos(self._heading),
            speed * np.sin(self._heading),
        ])
        self._head_pos += self._head_vel * dt

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build observation vector: [d^p, d^c, θ_err, prev_joint_angles]."""
        obs = np.concatenate([
            [self._dist_to_path()],
            [self._dist_to_target()],
            [self._heading_error()],
            self._prev_joint_angles,
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # EnvBase interface
    # ------------------------------------------------------------------

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        self._head_pos = np.zeros(2)
        self._heading = 0.0
        self._head_vel = np.zeros(2)
        self._prev_joint_angles = np.zeros(self.config.physics.num_joints)
        self._prev_head_angle = 0.0
        self._step_count = 0

        self._gait.reset()

        # Sample target on path
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
        phi_o = tensordict["action"].cpu().numpy().astype(np.float64).item()

        # Record previous state for reward
        prev_head_angle = self._prev_head_angle
        dist_to_target_prev = self._dist_to_target()

        # Gait execution layer: φⁱ(t) = α·sin(ωt + (i-1)δ) + φ_o
        dt = self.config.physics.mujoco_timestep * self.config.physics.mujoco_substeps
        joint_angles = self._gait.compute_joint_angles(phi_o, dt)
        head_angle = self._gait.compute_head_angle(phi_o)

        # Step physics
        self._step_physics(joint_angles, dt)

        # Update state
        self._prev_joint_angles = joint_angles.copy()
        self._prev_head_angle = head_angle

        # Compute reward
        dist_to_path = self._dist_to_path()
        dist_to_target_curr = self._dist_to_target()

        reward = compute_total_reward(
            dist_to_path=dist_to_path,
            dist_to_target_prev=dist_to_target_prev,
            dist_to_target_curr=dist_to_target_curr,
            head_angle_current=head_angle,
            head_angle_previous=prev_head_angle,
            c_p=self.config.c_p,
            c_e=self.config.c_e,
            d1=self.config.d_1,
            d2=self.config.d_2,
            angle_threshold=self.config.visual.angle_threshold,
            c_h=self.config.visual.penalty_coefficient,
        )

        # Check termination
        reached_target = dist_to_target_curr < 0.1
        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps

        # If target reached, sample new one (continuing episode)
        if reached_target:
            self._target_pos = self._sample_target()

        obs = self._get_obs()

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
                "done": torch.tensor(
                    [truncated], dtype=torch.bool, device=self._device
                ),
                "terminated": torch.tensor(
                    [False], dtype=torch.bool, device=self._device
                ),
                "truncated": torch.tensor(
                    [truncated], dtype=torch.bool, device=self._device
                ),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def close(self):
        super().close()
