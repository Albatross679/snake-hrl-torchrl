"""TorchRL environment for planar snake locomotion (Bing et al., IJCAI 2019).

Recreates the original MuJoCo-based snake robot environment using modern
MuJoCo (>=3.0) and TorchRL's EnvBase interface.
"""

import math
import os
from typing import Optional

# Ensure headless MuJoCo rendering works on servers without displays
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

from locomotion.configs import LocomotionEnvConfig
from locomotion.rewards import (
    compute_energy_normalized,
    compute_power_velocity_reward,
    compute_target_tracking_reward,
)
from locomotion.tracks import TracksGenerator


class PlanarSnakeEnv(EnvBase):
    """Planar snake robot environment for locomotion tasks.

    Supports two tasks:
    - power_velocity: Maximize velocity tracking while minimizing energy.
        Observation: 27-dim [joint_pos(8), joint_vel(8), angle(1),
                     head_velocity(1), actuator_forces(8), target_v(1)]
    - target_tracking: Follow a moving target using a head camera.
        Observation: 49-dim [joint_pos(8), joint_vel(8), head_velocity(1),
                     grayscale_pixels(32)]
    """

    def __init__(
        self,
        config: Optional[LocomotionEnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or LocomotionEnvConfig()
        self._device = device

        # Load MuJoCo model
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        if self.config.task == "target_tracking":
            xml_path = os.path.join(asset_dir, "planar_snake_tracking.xml")
        else:
            xml_path = os.path.join(asset_dir, "planar_snake.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache useful indices
        self._init_indices()

        # Store initial state for reset
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Track generator for target tracking task
        self.tracks_generator = TracksGenerator(
            target_v=self.config.track_target_v,
            head_target_dist=self.config.target_distance,
            target_dist_min=self.config.target_dist_min,
            target_dist_max=self.config.target_dist_max,
        )

        # Episode state
        self._step_count = 0
        self._update_count = 0  # Episode counter (for target_v cycling)
        self._target_v = self.config.target_v
        self._random_track_seed = 5

        # Renderer for camera observations (lazy init)
        self._renderer = None

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

        def camera_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)

        # Slider joints (free body DOFs for car1)
        self._slider_ids = [joint_id("slider_x"), joint_id("slider_y"), joint_id("slider_z")]
        self._hinge_ids = [joint_id("hinge_x"), joint_id("hinge_y"), joint_id("hinge_z")]

        # Snake joints
        self._joint_names = [f"joint0{i}" for i in range(1, 9)]
        self._joint_ids = [joint_id(n) for n in self._joint_names]

        # Target ball slider joints
        self._target_slider_x_id = joint_id("target_slider_x")
        self._target_slider_y_id = joint_id("target_slider_y")

        # Sensors
        self._sensor_actuatorfrc_ids = [
            sensor_id(f"sensor_actuatorfrc_joint0{i}") for i in range(1, 9)
        ]
        self._sensor_velocimeter_id = sensor_id("sensor_velocimeter")

        # Bodies
        self._body_car_ids = [body_id(f"car{i}") for i in range(1, 10)]

        # Geoms
        self._geom_head_id = geom_id("head")
        self._geom_target_ball_id = geom_id("target_ball_geom")

        # Cameras
        self._camera_head_id = camera_id("head")

        # Joint address mapping (qpos index for each joint)
        self._joint_qpos_ids = [m.jnt_qposadr[jid] for jid in self._joint_ids]
        self._joint_qvel_ids = [m.jnt_dofadr[jid] for jid in self._joint_ids]
        self._slider_qpos_ids = [m.jnt_qposadr[jid] for jid in self._slider_ids]
        self._hinge_qpos_ids = [m.jnt_qposadr[jid] for jid in self._hinge_ids]
        self._target_slider_x_qpos = m.jnt_qposadr[self._target_slider_x_id]
        self._target_slider_y_qpos = m.jnt_qposadr[self._target_slider_y_id]

    def _make_spec(self):
        """Define observation, action, reward, and done specs."""
        num_joints = self.config.physics.num_joints  # 8

        if self.config.task == "power_velocity":
            obs_dim = 27  # joint_pos(8) + joint_vel(8) + angle(1) + velocity(1) + forces(8) + target_v(1)
        else:
            obs_dim = 49  # joint_pos(8) + joint_vel(8) + velocity(1) + pixels(32)

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(obs_dim,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        self.action_spec = BoundedTensorSpec(
            low=-1.5,
            high=1.5,
            shape=(num_joints,),
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

    @property
    def dt(self) -> float:
        """Control timestep (physics_timestep * frame_skip)."""
        return self.config.physics.timestep * self.config.physics.frame_skip

    # === State accessors (matching original) ===

    def _get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._joint_qpos_ids].copy()

    def _get_joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self._joint_qvel_ids].copy()

    def _get_head_pos(self) -> tuple:
        """Get head geom world position (x, y)."""
        pos = self.data.geom_xpos[self._geom_head_id]
        return float(pos[0]), float(pos[1])

    def _get_target_pos(self) -> tuple:
        """Get target ball geom world position (x, y)."""
        pos = self.data.geom_xpos[self._geom_target_ball_id]
        return float(pos[0]), float(pos[1])

    def _get_body_pos(self) -> tuple:
        """Get car1 body position via slider joints."""
        head_x = float(self.data.qpos[self._slider_qpos_ids[0]])
        head_y = float(self.data.qpos[self._slider_qpos_ids[1]])
        return head_x, head_y

    def _get_sensor_actuatorfrcs(self) -> np.ndarray:
        return self.data.sensordata[self._sensor_actuatorfrc_ids].copy()

    def _get_sensor_head_velocity(self) -> float:
        return float(self.data.sensordata[self._sensor_velocimeter_id])

    def _get_head_euler_angles(self) -> np.ndarray:
        """Get head (car1) body orientation as euler angles."""
        quat = self.data.xquat[self._body_car_ids[0]].copy()
        return self._quat_to_euler(quat)

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."""
        w, x, y, z = quat
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def _get_head_degrees_z_angle(self) -> float:
        return math.degrees(self._get_head_euler_angles()[2])

    def _get_target_z_degree_angle(self) -> float:
        """Angle from head camera to target in degrees."""
        head_x, head_y = self._get_head_pos()
        target_x, target_y = self._get_target_pos()
        opposite = target_y - head_y
        adjacent = target_x - head_x
        if abs(adjacent) < 1e-10:
            return 90.0 if opposite > 0 else -90.0
        return math.degrees(math.atan(opposite / adjacent))

    def _get_head_to_target_degree_angle(self) -> float:
        return self._get_head_degrees_z_angle() - self._get_target_z_degree_angle()

    def _calc_distance(self) -> float:
        head_x, head_y = self._get_head_pos()
        target_x, target_y = self._get_target_pos()
        return math.sqrt((target_x - head_x) ** 2 + (target_y - head_y) ** 2)

    # === Ball control ===

    def _set_ball(self, x: float, y: float):
        """Move target ball to (x, y) via its slider joints."""
        self.data.qpos[self._target_slider_x_qpos] = x
        self.data.qpos[self._target_slider_y_qpos] = y
        mujoco.mj_forward(self.model, self.data)

    def _move_ball_power_velocity(self):
        """Place ball ahead of snake head (power-velocity task)."""
        head_x = float(self.data.qpos[self._slider_qpos_ids[0]])
        x = head_x + self.config.head_target_dist
        y = 0.0
        self._set_ball(x, y)

    def _move_ball_tracking(self):
        """Move ball according to track generator (tracking task)."""
        head_x, head_y = self._get_head_pos()
        target_x, target_y = self._get_target_pos()

        x, y = self.tracks_generator.step(
            self.config.track_type,
            head_x, head_y, target_x, target_y,
            self.dt,
            seed=self._random_track_seed,
        )
        self._set_ball(x, y)

    # === Camera ===

    def _get_head_camera_image(self) -> np.ndarray:
        """Render grayscale image from head camera, return middle row.

        Returns:
            1D array of shape (camera_width,) with grayscale pixel values.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.camera_height,
                width=self.config.camera_width,
            )
        self._renderer.update_scene(self.data, camera=self._camera_head_id)
        rgb = self._renderer.render()  # shape: (H, W, 3), uint8

        # Convert to grayscale: standard luminance weights
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        gray = gray / 255.0  # Normalize to [0, 1]

        # Select middle row (row index 9 for 32x20, matching original)
        middle_row = gray[self.config.camera_height // 2 - 1]
        return middle_row.astype(np.float32)

    # === Observation construction ===

    def _get_power_velocity_obs(self) -> np.ndarray:
        """27-dim observation for power-velocity task."""
        joint_pos = self._get_joint_positions()  # (8,)
        joint_vel = self._get_joint_velocities()  # (8,)
        angle = self._get_head_to_target_degree_angle()  # scalar
        sensor_velocity = self._get_sensor_head_velocity()  # scalar

        # Actuator forces scaled by gear (gear=1.0)
        sensor_actuatorfrcs = self._get_sensor_actuatorfrcs()  # (8,)
        actuator_gear = float(self.model.actuator_gear.max())
        actuator_torques = sensor_actuatorfrcs * actuator_gear  # (8,)

        ob = np.concatenate([
            joint_pos,
            joint_vel,
            [angle],
            [sensor_velocity],
            actuator_torques,
            [self._target_v],
        ])
        return ob.astype(np.float32)

    def _get_target_tracking_obs(self) -> np.ndarray:
        """49-dim observation for target-tracking task."""
        joint_pos = self._get_joint_positions()  # (8,)
        joint_vel = self._get_joint_velocities()  # (8,)
        sensor_velocity = self._get_sensor_head_velocity()  # scalar
        img_gray = self._get_head_camera_image()  # (32,)

        ob = np.concatenate([
            joint_pos,
            joint_vel,
            [sensor_velocity],
            img_gray,
        ])
        return ob.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        if self.config.task == "power_velocity":
            return self._get_power_velocity_obs()
        else:
            return self._get_target_tracking_obs()

    # === EnvBase interface ===

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            np.random.seed(seed)
            self._random_track_seed = seed

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Reset MuJoCo state
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        # Update target velocity for power-velocity task
        if self.config.task == "power_velocity":
            if self._update_count >= 100:
                idx = self._update_count % len(self.config.target_v_array)
                self._target_v = self.config.target_v_array[idx]
            else:
                self._target_v = 0.1  # default for first 100 episodes
            self._update_count += 1
            self._move_ball_power_velocity()
        else:
            # Target tracking: place ball at target_distance ahead
            head_x, _ = self._get_head_pos()
            self._set_ball(head_x + self.config.target_distance, 0.0)
            self.tracks_generator.reset()
            self._random_track_seed += 1
            self._update_count += 1

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
        # Extract action
        action = tensordict["action"].cpu().numpy().astype(np.float64)

        # Move ball before simulation step (matching original order)
        if self.config.task == "power_velocity":
            self._move_ball_power_velocity()
        else:
            self._move_ball_tracking()

        # Record pre-step state
        dist_before = self._calc_distance()
        head_x0, head_y0 = self._get_head_pos()

        # Apply action and simulate
        self.data.ctrl[:] = action
        for _ in range(self.config.physics.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Record post-step state
        dist_after = self._calc_distance()
        head_x1, head_y1 = self._get_head_pos()

        # Compute reward
        if self.config.task == "power_velocity":
            distance = dist_before - dist_after
            velocity = distance / self.dt

            sensor_actuatorfrcs = self._get_sensor_actuatorfrcs()
            joint_velocities = self._get_joint_velocities()
            actuator_gear = float(self.model.actuator_gear.max())
            force_max = float(self.model.actuator_forcerange.max())

            power_normalized, _ = compute_energy_normalized(
                sensor_actuatorfrcs, joint_velocities,
                actuator_gear=actuator_gear,
                force_max=force_max,
            )

            reward = compute_power_velocity_reward(
                self._target_v, velocity, power_normalized
            )
        else:
            reward = compute_target_tracking_reward(
                self.config.target_distance,
                dist_before,
                dist_after,
                self.config.target_dist_min,
                self.config.target_dist_max,
            )

        # Handle NaN rewards (can happen with extreme velocity/power values)
        if np.isnan(reward):
            reward = 0.0

        self._step_count += 1
        truncated = self._step_count >= self.config.physics.max_episode_steps

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
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
