"""TorchRL environment for CPG-regulated goal-tracking locomotion (Liu et al. 2023).

Observation (8-dim):
    [||rho_g||, v_g, theta_g, theta_dot_g, kappa_1, kappa_2, kappa_3, kappa_4]

Action (4-dim):
    alpha = [alpha_1, alpha_2, alpha_3, alpha_4] — tonic inputs to CPG oscillators

The CPG network runs at the physics timestep inside the environment,
converting RL tonic inputs into smooth actuator commands for the 3 hinge joints.
Note: 4 oscillators produce 4 outputs, but only 3 joints exist — the 4th output
controls the tail joint implicitly (the paper uses 4 actuated segments).
We map the first 3 CPG outputs to the 3 actuated joints.
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

from liu2023.configs_liu2023 import Liu2023EnvConfig
from liu2023.cpg_liu2023 import LiuCPGNetwork
from liu2023.curriculum_liu2023 import CurriculumManager, DEFAULT_LEVELS
from liu2023.rewards_liu2023 import (
    compute_goal_tracking_reward,
    check_goal_reached,
    check_starvation,
)


class SoftSnakeEnv(EnvBase):
    """4-link soft snake robot with CPG-regulated goal tracking.

    The RL agent outputs tonic inputs to a Matsuoka CPG network,
    which generates smooth actuator commands. A curriculum manager
    progressively increases task difficulty.
    """

    def __init__(
        self,
        config: Optional[Liu2023EnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Liu2023EnvConfig()
        self._device = device

        # Load MuJoCo model
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        xml_path = os.path.join(asset_dir, "soft_snake_4link.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache indices
        self._init_indices()

        # Store initial state
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()
        self._init_body_masses = self.model.body_mass.copy()

        # CPG network
        cpg = self.config.cpg
        self._cpg = LiuCPGNetwork(
            num_oscillators=4,
            a_psi=cpg.a_psi, b=cpg.b, tau_r=cpg.tau_r, tau_a=cpg.tau_a,
            a_i=cpg.a_i, w_ij=cpg.w_ij, w_ji=cpg.w_ji, c=cpg.c,
            u_max=cpg.u_max,
        )
        self._kf = cpg.kf_default

        # Curriculum manager
        if self.config.curriculum.enabled:
            self._curriculum = CurriculumManager(
                levels=DEFAULT_LEVELS[:self.config.curriculum.num_levels],
                success_threshold=self.config.curriculum.success_threshold,
                eval_window=self.config.curriculum.eval_window,
            )
        else:
            self._curriculum = None

        # Episode state
        self._step_count = 0
        self._v_g_history = []
        self._max_pressure = 1.0
        self._goal_x = 2.0
        self._goal_y = 0.0
        self._prev_head_x = 0.0
        self._prev_head_y = 0.0
        self._prev_head_yaw = 0.0
        self._rng = np.random.default_rng()

        # Build specs
        self._make_spec()

    def _init_indices(self):
        """Cache MuJoCo name-to-index mappings."""
        m = self.model

        def joint_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

        def body_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

        def geom_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

        def sensor_id(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)

        # Free body DOFs
        self._slider_ids = [joint_id("slider_x"), joint_id("slider_y"), joint_id("slider_z")]
        self._hinge_ids = [joint_id("hinge_x"), joint_id("hinge_y"), joint_id("hinge_z")]

        # Snake joints (3 joints for 4-link snake)
        self._joint_names = ["joint01", "joint02", "joint03"]
        self._joint_ids = [joint_id(n) for n in self._joint_names]

        # Goal slider joints
        self._goal_slider_x_id = joint_id("goal_slider_x")
        self._goal_slider_y_id = joint_id("goal_slider_y")

        # Bodies
        self._body_link_ids = [body_id(f"link{i}") for i in range(1, 5)]

        # Geoms
        self._geom_head_id = geom_id("head")
        self._geom_goal_id = geom_id("goal_geom")

        # Sensors
        self._sensor_velocimeter_id = sensor_id("sensor_velocimeter")
        self._sensor_gyro_id = sensor_id("sensor_gyro")

        # Address mappings
        self._joint_qpos_ids = [m.jnt_qposadr[jid] for jid in self._joint_ids]
        self._joint_qvel_ids = [m.jnt_dofadr[jid] for jid in self._joint_ids]
        self._slider_qpos_ids = [m.jnt_qposadr[jid] for jid in self._slider_ids]
        self._hinge_qpos_ids = [m.jnt_qposadr[jid] for jid in self._hinge_ids]
        self._goal_slider_x_qpos = m.jnt_qposadr[self._goal_slider_x_id]
        self._goal_slider_y_qpos = m.jnt_qposadr[self._goal_slider_y_id]

    def _make_spec(self):
        """Define observation, action, reward, and done specs."""
        obs_dim = 8  # [dist, v_g, theta_g, theta_dot_g, kappa_1..4]

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(obs_dim,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        # 4 tonic inputs to CPG
        self.action_spec = BoundedTensorSpec(
            low=-3.0,
            high=3.0,
            shape=(4,),
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
        """Control timestep."""
        return self.config.physics.timestep * self.config.physics.frame_skip

    # === State accessors ===

    def _get_head_pos(self) -> tuple:
        pos = self.data.geom_xpos[self._geom_head_id]
        return float(pos[0]), float(pos[1])

    def _get_head_yaw(self) -> float:
        """Get head yaw angle (rotation around z-axis) in radians."""
        quat = self.data.xquat[self._body_link_ids[0]].copy()
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _get_goal_pos(self) -> tuple:
        pos = self.data.geom_xpos[self._geom_goal_id]
        return float(pos[0]), float(pos[1])

    def _get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._joint_qpos_ids].copy()

    def _get_dist_to_goal(self) -> float:
        hx, hy = self._get_head_pos()
        gx, gy = self._get_goal_pos()
        return math.sqrt((gx - hx) ** 2 + (gy - hy) ** 2)

    def _get_heading_error(self) -> float:
        """Heading angle error to goal in radians, in [-pi, pi]."""
        hx, hy = self._get_head_pos()
        gx, gy = self._get_goal_pos()
        goal_angle = math.atan2(gy - hy, gx - hx)
        head_yaw = self._get_head_yaw()
        error = goal_angle - head_yaw
        # Wrap to [-pi, pi]
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi
        return error

    def _get_velocity_toward_goal(self) -> float:
        """Projected velocity toward goal."""
        hx, hy = self._get_head_pos()
        gx, gy = self._get_goal_pos()
        dist = math.sqrt((gx - hx) ** 2 + (gy - hy) ** 2)
        if dist < 1e-6:
            return 0.0

        # Unit vector toward goal
        ux = (gx - hx) / dist
        uy = (gy - hy) / dist

        # Head velocity from position difference
        vx = (hx - self._prev_head_x) / self.dt
        vy = (hy - self._prev_head_y) / self.dt

        return vx * ux + vy * uy

    def _get_angular_velocity(self) -> float:
        """Yaw angular velocity from gyro sensor."""
        # Gyro sensor gives angular velocity in body frame
        # z-component is yaw rate
        gyro_adr = self.model.sensor_adr[self._sensor_gyro_id]
        return float(self.data.sensordata[gyro_adr + 2])

    def _get_obs(self) -> np.ndarray:
        """8-dim observation vector."""
        dist = self._get_dist_to_goal()
        v_g = self._get_velocity_toward_goal()
        theta_g = self._get_heading_error()
        theta_dot_g = self._get_angular_velocity()
        joint_pos = self._get_joint_positions()  # (3,)

        # Pad to 4 curvatures (joint_pos has 3 values for 3 joints;
        # 4th "curvature" is 0 since it has no joint)
        kappas = np.zeros(4)
        kappas[:3] = joint_pos

        obs = np.array([
            dist, v_g, theta_g, theta_dot_g,
            kappas[0], kappas[1], kappas[2], kappas[3],
        ], dtype=np.float32)

        return obs

    # === Goal placement ===

    def _set_goal(self, x: float, y: float):
        self.data.qpos[self._goal_slider_x_qpos] = x
        self.data.qpos[self._goal_slider_y_qpos] = y
        self._goal_x = x
        self._goal_y = y
        mujoco.mj_forward(self.model, self.data)

    # === Domain randomization ===

    def _apply_domain_randomization(self):
        physics = self.config.physics

        if physics.randomize_friction:
            scale = self._rng.uniform(*physics.friction_range)
            # Scale all geom friction coefficients
            for i in range(self.model.ngeom):
                self.model.geom_friction[i, 0] = scale

        if physics.randomize_mass:
            scale = self._rng.uniform(*physics.mass_scale_range)
            self.model.body_mass[:] = self._init_body_masses * scale

        if physics.randomize_max_pressure:
            self._max_pressure = self._rng.uniform(*physics.max_pressure_range)
        else:
            self._max_pressure = 1.0

        self._cpg.max_pressure = self._max_pressure

    # === EnvBase interface ===

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Reset MuJoCo state
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._v_g_history.clear()

        # Reset CPG
        self._cpg.reset()

        # Apply domain randomization
        self._apply_domain_randomization()

        # Place goal based on curriculum
        hx, hy = self._get_head_pos()
        head_yaw = self._get_head_yaw()

        if self._curriculum is not None:
            dist, angle = self._curriculum.sample_goal(self._rng)
            gx, gy = self._curriculum.goal_to_xy(dist, angle, hx, hy, head_yaw)
        else:
            # Default: goal 2m ahead
            gx = hx + 2.0
            gy = hy
        self._set_goal(gx, gy)

        # Store previous head state for velocity computation
        self._prev_head_x, self._prev_head_y = self._get_head_pos()
        self._prev_head_yaw = self._get_head_yaw()

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

        # Record pre-step state
        self._prev_head_x, self._prev_head_y = self._get_head_pos()
        self._prev_head_yaw = self._get_head_yaw()

        # Run CPG at physics timestep within frame_skip loop
        physics_dt = self.config.physics.timestep
        for _ in range(self.config.physics.frame_skip):
            cpg_outputs = self._cpg.step(physics_dt, action, self._kf)
            # Apply first 3 CPG outputs to 3 joints
            self.data.ctrl[:] = cpg_outputs[:3]
            mujoco.mj_step(self.model, self.data)

        # Compute reward components
        dist = self._get_dist_to_goal()
        v_g = self._get_velocity_toward_goal()
        theta_g = self._get_heading_error()

        self._v_g_history.append(v_g)

        # Collect goal radii from all curriculum levels reached so far
        goal_radii = []
        if self._curriculum is not None:
            for i in range(self._curriculum.current_level + 1):
                goal_radii.append(self._curriculum.levels[i].goal_radius)
        else:
            goal_radii = [0.2]

        reward = compute_goal_tracking_reward(
            v_g=v_g,
            dist_to_goal=dist,
            theta_g=theta_g,
            goal_radii=goal_radii,
            c_v=self.config.reward_c_v,
            c_g=self.config.reward_c_g,
        )

        if np.isnan(reward):
            reward = 0.0

        self._step_count += 1

        # Check termination conditions
        current_radius = self._curriculum.level_spec.goal_radius if self._curriculum else 0.2
        goal_reached = check_goal_reached(dist, current_radius)
        starvation = check_starvation(
            self._v_g_history, self.config.starvation_timeout_steps
        )
        truncated = self._step_count >= self.config.max_episode_steps
        terminated = goal_reached or starvation
        done = terminated or truncated

        # Report to curriculum on episode end
        if done and self._curriculum is not None:
            self._curriculum.report_episode(goal_reached)

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

    @property
    def curriculum_level(self) -> int:
        """Current curriculum level (0-indexed)."""
        if self._curriculum is not None:
            return self._curriculum.current_level
        return 0

    def close(self):
        super().close()
