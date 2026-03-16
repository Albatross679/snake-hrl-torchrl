"""TorchRL environment for contact-aware soft snake (Liu, Onal & Fu, 2021).

Two cooperative controllers (C1 + R2) jointly control a Matsuoka CPG
network to navigate a soft snake robot through an obstacle maze.

Observation (26 dims):
    ζ₁:₄  — dynamic state (ρ_g, ρ̇_g, θ_g, θ̇_g)
    ζ₅:₈  — body curvatures (κ₁, κ₂, κ₃, κ₄)
    ζ₉:₁₄ — previous actions + option + termination probability
    ζ₁₅:₂₄ — contact forces (10 values from 5 rigid bodies)
    ζ₂₅:₂₆ — distance and angle to nearest obstacle

Action (C1): 4 tonic inputs + 2 option [o, β] = 6 dims
Action (R2): 4 tonic input additions
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

from liu2021.configs_liu2021 import Liu2021EnvConfig
from liu2021.cpg_liu2021 import MatsuokaCPG
from liu2021.rewards_liu2021 import compute_apf_reward


class ContactAwareSoftSnakeEnv(EnvBase):
    """Contact-aware soft snake locomotion in obstacle maze.

    Implements the combined C1+R2 control scheme where:
    - C1 controls tonic inputs to the CPG + frequency ratio (steering/speed)
    - R2 is event-triggered when contact is detected, adding tonic modulation
    - The combined tonic input drives the Matsuoka CPG network
    """

    def __init__(
        self,
        config: Optional[Liu2021EnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Liu2021EnvConfig()
        self._device = device

        # CPG
        self._cpg = MatsuokaCPG(self.config.cpg)

        # RNG
        self._rng = np.random.default_rng(42)

        # State
        self._head_pos = np.zeros(2)
        self._head_vel = np.zeros(2)
        self._heading = 0.0
        self._curvatures = np.zeros(4)
        self._contact_forces = np.zeros(10)
        self._goal_pos = np.zeros(2)
        self._obstacles = np.zeros((0, 2))

        # Previous action state
        self._prev_c1_action = np.zeros(self.config.c1_action_dim)
        self._prev_option = np.zeros(2)

        # Episode
        self._step_count = 0
        self._starvation_timer = 0.0

        # Combined action dim for the joint C1+R2 output
        # In the cooperative game, each controller acts separately.
        # Here we expose the combined action space.
        self._total_action_dim = self.config.c1_action_dim + self.config.r2_action_dim

        self._make_spec()

    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self.config.obs_dim,),
                dtype=torch.float32,
                device=self._device,
            ),
            shape=(),
        )

        self.action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(self._total_action_dim,),
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

    def _setup_maze(self) -> None:
        """Generate obstacle maze for training."""
        maze = self.config.maze
        rows, cols = maze.train_grid_rows, maze.train_grid_cols
        spacing = maze.train_obstacle_spacing

        obstacles = []
        for r in range(rows):
            for c in range(cols):
                pos = np.array([
                    (c - cols / 2 + 0.5) * spacing,
                    (r + 1) * spacing,
                ])
                # Add noise
                pos += self._rng.normal(0, maze.obstacle_noise, 2)
                obstacles.append(pos)

        self._obstacles = np.array(obstacles)

    def _sample_goal(self) -> np.ndarray:
        """Sample goal position at fixed distance with random deviation angle."""
        maze = self.config.maze
        angle_range = np.radians(maze.deviation_angle_range)
        angle = self._rng.uniform(-angle_range, angle_range)
        dist = maze.train_goal_distance
        return np.array([dist * np.sin(angle), dist * np.cos(angle)])

    def _detect_contact(self) -> bool:
        """Check if any contact sensor detects force or obstacle proximity."""
        if np.any(np.abs(self._contact_forces) > self.config.contact.force_threshold):
            return True
        # Check distance to nearest obstacle
        if len(self._obstacles) > 0:
            dists = np.linalg.norm(self._obstacles - self._head_pos, axis=1)
            if np.min(dists) < self.config.contact.detection_distance:
                return True
        return False

    def _get_nearest_obstacle_info(self) -> np.ndarray:
        """Get distance and angle to nearest obstacle."""
        if len(self._obstacles) == 0:
            return np.array([1.0, 0.0])
        diffs = self._obstacles - self._head_pos
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.argmin(dists)
        angle = np.arctan2(diffs[idx, 1], diffs[idx, 0]) - self._heading
        return np.array([dists[idx], angle])

    def _get_obs(self) -> np.ndarray:
        """Build 26-dim observation vector."""
        goal_diff = self._goal_pos - self._head_pos
        rho_g = np.linalg.norm(goal_diff)
        theta_g = np.arctan2(goal_diff[1], goal_diff[0]) - self._heading

        # Dynamic state: ρ_g, ρ̇_g, θ_g, θ̇_g
        rho_dot = -np.dot(self._head_vel, goal_diff / (rho_g + 1e-8))
        theta_dot = 0.0  # Simplified
        dynamic = np.array([rho_g, rho_dot, theta_g, theta_dot])

        # Previous actions + option
        prev = np.concatenate([
            self._prev_c1_action[:4],
            self._prev_option,
        ])

        # Nearest obstacle
        obstacle_info = self._get_nearest_obstacle_info()

        obs = np.concatenate([
            dynamic,          # ζ₁:₄
            self._curvatures,  # ζ₅:₈
            prev,             # ζ₉:₁₄
            self._contact_forces,  # ζ₁₅:₂₄
            obstacle_info,    # ζ₂₅:₂₆
        ])

        return obs.astype(np.float32)

    def _step_physics(self, cpg_output: np.ndarray) -> None:
        """Advance simplified soft snake dynamics one step."""
        dt = 0.02  # 50 Hz control

        # CPG outputs → curvature → velocity (simplified)
        self._curvatures = cpg_output

        # Direction from curvature-induced steering
        steering = np.mean(cpg_output[:2]) - np.mean(cpg_output[2:])
        self._heading += steering * 0.1 * dt

        # Forward velocity from CPG amplitude
        speed = np.mean(np.abs(cpg_output)) * 0.1
        self._head_vel = np.array([
            speed * np.cos(self._heading),
            speed * np.sin(self._heading),
        ])
        self._head_pos += self._head_vel * dt

        # Simulate contact forces (from obstacle proximity)
        self._contact_forces[:] = 0.0
        if len(self._obstacles) > 0:
            for i, obs_pos in enumerate(self._obstacles):
                diff = self._head_pos - obs_pos
                dist = np.linalg.norm(diff)
                if dist < self.config.maze.obstacle_radius * 2:
                    # Generate contact force on nearest sensors
                    force_idx = min(i % 5, 4) * 2
                    self._contact_forces[force_idx] = (
                        self.config.maze.obstacle_radius * 2 - dist
                    )

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        self._head_pos = np.zeros(2)
        self._head_vel = np.zeros(2)
        self._heading = 0.0
        self._curvatures = np.zeros(4)
        self._contact_forces = np.zeros(10)
        self._prev_c1_action = np.zeros(self.config.c1_action_dim)
        self._prev_option = np.zeros(2)
        self._step_count = 0
        self._starvation_timer = 0.0

        self._cpg.reset(self._rng)
        self._setup_maze()
        self._goal_pos = self._sample_goal()

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

        # Split into C1 and R2 actions
        c1_action = action[:self.config.c1_action_dim]
        r2_action = action[self.config.c1_action_dim:]

        # C1: tonic inputs (4) + option (2)
        c1_tonic = c1_action[:4]
        c1_option = c1_action[4:6]
        K_f = max(0.1, 1.0 + c1_option[1])  # Frequency ratio from option β

        # R2: event-triggered tonic modulation
        contact_detected = self._detect_contact()
        if contact_detected:
            combined_tonic = self._cpg.compose_tonic_inputs(
                c1_tonic, r2_action,
                self.config.w1, self.config.w2,
            )
        else:
            combined_tonic = 1.0 / (1.0 + np.exp(-c1_tonic))

        # Step CPG
        cpg_output = self._cpg.step(combined_tonic * 2 - 1, K_f, dt=0.02)

        # Step physics
        self._step_physics(cpg_output)

        # Store previous action
        self._prev_c1_action = c1_action.copy()
        self._prev_option = c1_option.copy()

        # Check termination
        goal_diff = self._goal_pos - self._head_pos
        dist_to_goal = np.linalg.norm(goal_diff)
        heading_to_goal = np.arctan2(goal_diff[1], goal_diff[0]) - self._heading

        reached_goal = dist_to_goal < self.config.maze.goal_radius

        # Compute reward
        reward = compute_apf_reward(
            self._head_pos,
            self._head_vel,
            self._goal_pos,
            self._obstacles,
            heading_to_goal,
            self.config.reward,
        )

        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps

        obs = self._get_obs()

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
                "done": torch.tensor(
                    [reached_goal or truncated], dtype=torch.bool, device=self._device
                ),
                "terminated": torch.tensor(
                    [reached_goal], dtype=torch.bool, device=self._device
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
