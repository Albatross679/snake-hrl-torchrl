"""TorchRL environment for COBRA snake robot navigation (Jiang et al., 2024).

Two environment classes:
- CobraNavigationEnv: Local waypoint navigation (the RL training environment)
- CobraMazeEnv: Full pipeline with A* planner + procedural Kruskal maze

Four-layer control hierarchy:
    Layer 1: A* global planner (maze env only)
    Layer 2: RL (DDPG) local navigation -> 7-dim CPG action
    Layer 3: Dual CPG gait generation -> 11 joint targets
    Layer 4: PID gait tracking (MuJoCo position actuators)
"""

import math
import os
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import mujoco
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from jiang2024.configs_jiang2024 import CobraEnvConfig, CobraMazeEnvConfig
from jiang2024.cpg_jiang2024 import DualCPGController
from jiang2024.rewards_jiang2024 import compute_navigation_reward


class CobraNavigationEnv(EnvBase):
    """COBRA snake robot environment for local waypoint navigation.

    Obs (21-dim): joint_pos(11) + gyro(3) + displacement_to_waypoint(3)
                  + relative_rotation_axis_angle(4)
    Action (7-dim): [R1, R2, omega, theta1, theta2, delta1, delta2]
    RL frequency: 0.5 Hz (100 CPG steps * 20 MuJoCo substeps * 0.001s = 2s)
    """

    def __init__(
        self,
        config: Optional[CobraEnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or CobraEnvConfig()
        self._device = device

        # Load MuJoCo model
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        xml_path = os.path.join(asset_dir, "cobra.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache indices
        self._init_indices()

        # Store initial state
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # CPG controller
        self.cpg = DualCPGController(
            num_joints=self.config.physics.num_joints,
            cpg_dt=self.config.cpg.cpg_dt,
            num_cpg_steps=self.config.cpg.num_cpg_steps,
        )

        # Episode state
        self._step_count = 0
        self._prev_action = np.zeros(7)
        self._goal = np.array([5.0, 0.0, 0.0])

        # Build specs
        self._make_spec()

    def _init_indices(self):
        """Cache MuJoCo name-to-index mappings."""
        m = self.model

        def joint_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

        def sensor_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)

        def body_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

        def geom_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

        # Free body DOFs
        self._slider_ids = [joint_id("slider_x"), joint_id("slider_y"), joint_id("slider_z")]
        self._hinge_ids = [joint_id("hinge_x"), joint_id("hinge_y"), joint_id("hinge_z")]

        # Snake joints (11 joints)
        self._joint_names = [f"joint{i:02d}" for i in range(1, 12)]
        self._joint_ids = [joint_id(n) for n in self._joint_names]

        # Waypoint slider joints
        self._wp_slider_x_id = joint_id("waypoint_slider_x")
        self._wp_slider_y_id = joint_id("waypoint_slider_y")

        # Sensor indices for joint positions (11 sensors)
        self._sensor_jointpos_ids = [
            sensor_id(f"sensor_jointpos_joint{i:02d}") for i in range(1, 12)
        ]

        # Gyro sensor (3-dim)
        self._sensor_gyro_id = sensor_id("sensor_gyro")

        # Bodies
        self._body_head_id = body_id("link0")

        # Geoms
        self._geom_head_id = geom_id("head")
        self._geom_waypoint_id = geom_id("waypoint_geom")

        # Joint address mapping
        self._joint_qpos_ids = [m.jnt_qposadr[jid] for jid in self._joint_ids]
        self._slider_qpos_ids = [m.jnt_qposadr[jid] for jid in self._slider_ids]
        self._hinge_qpos_ids = [m.jnt_qposadr[jid] for jid in self._hinge_ids]
        self._wp_slider_x_qpos = m.jnt_qposadr[self._wp_slider_x_id]
        self._wp_slider_y_qpos = m.jnt_qposadr[self._wp_slider_y_id]

    def _make_spec(self):
        """Define observation, action, reward, and done specs."""
        obs_dim = 21  # joint_pos(11) + gyro(3) + displacement(3) + rotation(4)

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(obs_dim,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        # Action: 7-dim CPG parameters
        cpg = self.config.cpg
        action_low = torch.tensor(
            [cpg.R_min, cpg.R_min, cpg.omega_min, cpg.theta_min, cpg.theta_min,
             cpg.delta_min, cpg.delta_min],
            dtype=torch.float32, device=self._device,
        )
        action_high = torch.tensor(
            [cpg.R_max, cpg.R_max, cpg.omega_max, cpg.theta_max, cpg.theta_max,
             cpg.delta_max, cpg.delta_max],
            dtype=torch.float32, device=self._device,
        )
        self.action_spec = BoundedTensorSpec(
            low=action_low,
            high=action_high,
            shape=(7,),
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

    # === State accessors ===

    def _get_head_pos(self) -> np.ndarray:
        """Get head position [x, y, z] in world frame."""
        return self.data.geom_xpos[self._geom_head_id].copy()

    def _get_head_quat(self) -> np.ndarray:
        """Get head orientation as quaternion [w, x, y, z]."""
        return self.data.xquat[self._body_head_id].copy()

    def _get_joint_positions(self) -> np.ndarray:
        """Get 11 joint positions from sensors."""
        return self.data.sensordata[self._sensor_jointpos_ids].copy()

    def _get_gyro(self) -> np.ndarray:
        """Get 3-axis gyroscope reading from head IMU."""
        return self.data.sensordata[self._sensor_gyro_id:self._sensor_gyro_id + 3].copy()

    def _get_waypoint_pos(self) -> np.ndarray:
        """Get waypoint position [x, y, z]."""
        return self.data.geom_xpos[self._geom_waypoint_id].copy()

    def _set_waypoint(self, x: float, y: float):
        """Move waypoint marker to (x, y)."""
        self.data.qpos[self._wp_slider_x_qpos] = x
        self.data.qpos[self._wp_slider_y_qpos] = y
        mujoco.mj_forward(self.model, self.data)

    def _compute_displacement_to_waypoint(self) -> np.ndarray:
        """Compute displacement vector from head to waypoint in world frame."""
        head_pos = self._get_head_pos()
        wp_pos = self._get_waypoint_pos()
        return (wp_pos - head_pos).astype(np.float32)

    def _compute_relative_rotation(self) -> np.ndarray:
        """Compute relative rotation from head frame to waypoint as axis-angle (4-dim).

        Returns [ax, ay, az, angle] where (ax,ay,az) is the unit rotation axis
        and angle is the rotation magnitude.
        """
        head_pos = self._get_head_pos()
        wp_pos = self._get_waypoint_pos()
        direction = wp_pos - head_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.zeros(4, dtype=np.float32)

        # Target direction in world frame
        target_dir = direction / dist

        # Head forward direction from rotation matrix
        head_quat = self._get_head_quat()
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, head_quat)
        rot_mat = rot_mat.reshape(3, 3)
        head_forward = rot_mat[:, 0]  # x-axis of head frame

        # Rotation axis and angle
        cross = np.cross(head_forward, target_dir)
        sin_angle = np.linalg.norm(cross)
        cos_angle = np.dot(head_forward, target_dir)
        angle = np.arctan2(sin_angle, cos_angle)

        if sin_angle < 1e-6:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = cross / sin_angle

        return np.array([axis[0], axis[1], axis[2], angle], dtype=np.float32)

    def _calc_distance_to_waypoint(self) -> float:
        """Distance from head to current waypoint."""
        head_pos = self._get_head_pos()
        wp_pos = self._get_waypoint_pos()
        return float(np.linalg.norm(wp_pos[:2] - head_pos[:2]))

    # === Observation ===

    def _get_obs(self) -> np.ndarray:
        """21-dim observation."""
        joint_pos = self._get_joint_positions()  # (11,)
        gyro = self._get_gyro()  # (3,)
        displacement = self._compute_displacement_to_waypoint()  # (3,)
        rotation = self._compute_relative_rotation()  # (4,)

        obs = np.concatenate([joint_pos, gyro, displacement, rotation])
        return obs.astype(np.float32)

    # === EnvBase interface ===

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            np.random.seed(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Reset MuJoCo state
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = np.zeros(7)
        self.cpg.reset()

        # Random goal position in arena
        angle = np.random.uniform(0, 2 * math.pi)
        dist = np.random.uniform(self.config.min_goal_dist, self.config.max_goal_dist)
        gx = dist * math.cos(angle)
        gy = dist * math.sin(angle)
        self._goal = np.array([gx, gy, 0.0])
        self._set_waypoint(gx, gy)

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

        # Record pre-step distance
        dist_prev = self._calc_distance_to_waypoint()

        # Layer 3: CPG generates joint targets from 7-dim action
        joint_targets = self.cpg.step(action)

        # Layer 4: Apply joint targets to MuJoCo position actuators
        self.data.ctrl[:] = joint_targets
        for _ in range(self.config.cpg.num_cpg_steps * self.config.physics.mujoco_substeps):
            mujoco.mj_step(self.model, self.data)

        # Record post-step distance
        dist_curr = self._calc_distance_to_waypoint()

        # Compute reward (Eq. 12)
        reward = compute_navigation_reward(
            distance=dist_curr,
            dist_prev=dist_prev,
            action=action,
            prev_action=self._prev_action,
            w1=self.config.reward_w1,
            w2=self.config.reward_w2,
            w3=self.config.reward_w3,
        )

        # Handle NaN rewards
        if np.isnan(reward):
            reward = 0.0

        self._prev_action = action.copy()
        self._step_count += 1

        # Termination conditions
        terminated = dist_curr < self.config.waypoint_threshold
        truncated = self._step_count >= self.config.max_episode_steps
        done = terminated or truncated

        obs = self._get_obs()

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
                "done": torch.tensor([done], dtype=torch.bool, device=self._device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self._device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self._device),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def close(self):
        super().close()


class CobraMazeEnv(CobraNavigationEnv):
    """COBRA navigation with procedural Kruskal maze and A* path planning.

    At reset: generates maze, computes A* path, steps through waypoints sequentially.
    """

    def __init__(
        self,
        config: Optional[CobraMazeEnvConfig] = None,
        device: str = "cpu",
    ):
        config = config or CobraMazeEnvConfig()
        super().__init__(config=config, device=device)
        self._waypoints = []
        self._waypoint_idx = 0
        self._maze_generator = None
        self._maze_xml_bodies = ""

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        from jiang2024.maze_jiang2024 import KruskalMazeGenerator
        from jiang2024.planner_jiang2024 import AStarPlanner

        cfg = self.config

        # Reset base state
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = np.zeros(7)
        self.cpg.reset()

        # Generate maze
        maze = KruskalMazeGenerator(
            rows=cfg.maze_rows,
            cols=cfg.maze_cols,
            cell_size=cfg.maze_cell_size,
            wall_height=cfg.maze_wall_height,
            wall_thickness=cfg.maze_wall_thickness,
        )
        walls = maze.generate(seed=np.random.randint(0, 100000))
        self._maze_generator = maze

        # Plan path
        grid = maze.to_occupancy_grid(
            resolution=cfg.planner_resolution,
            inflation=cfg.planner_inflation,
        )
        planner = AStarPlanner(grid)

        # Start at cell (0,0), goal at cell (rows-1, cols-1)
        start = maze.get_cell_center(0, 0)
        goal = maze.get_cell_center(cfg.maze_rows - 1, cfg.maze_cols - 1)

        path = planner.plan(start, goal)
        if path is None:
            # Fallback: direct path
            path = [start, goal]

        self._waypoints = path
        self._waypoint_idx = 0

        # Set first waypoint
        if self._waypoints:
            wx, wy = self._waypoints[0]
            self._goal = np.array([wx, wy, 0.0])
            self._set_waypoint(wx, wy)

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
        result = super()._step(tensordict)

        # Check if current waypoint is reached
        dist = self._calc_distance_to_waypoint()
        if dist < self.config.waypoint_threshold and self._waypoint_idx < len(self._waypoints) - 1:
            # Advance to next waypoint
            self._waypoint_idx += 1
            wx, wy = self._waypoints[self._waypoint_idx]
            self._goal = np.array([wx, wy, 0.0])
            self._set_waypoint(wx, wy)

            # Update observation with new waypoint
            obs = self._get_obs()
            result["observation"] = torch.tensor(obs, dtype=torch.float32, device=self._device)

            # Not terminated until final waypoint reached
            result["terminated"] = torch.tensor([False], dtype=torch.bool, device=self._device)
            result["done"] = result["truncated"]

        # Terminal: final waypoint reached
        if self._waypoint_idx == len(self._waypoints) - 1 and dist < self.config.waypoint_threshold:
            result["terminated"] = torch.tensor([True], dtype=torch.bool, device=self._device)
            result["done"] = torch.tensor([True], dtype=torch.bool, device=self._device)

        return result
