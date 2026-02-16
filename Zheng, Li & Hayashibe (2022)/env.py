"""TorchRL environment for underwater snake locomotion.

Reproduces Zheng, Li & Hayashibe (2022): 7-link rigid snake robot swimming
underwater using MuJoCo's built-in fluid model with torque control.
"""

from typing import Optional
import math

import numpy as np
import torch
import mujoco

from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)
from tensordict import TensorDict, TensorDictBase

from configs import ZhengConfig


def build_mjcf_xml(config: ZhengConfig) -> str:
    """Build MJCF XML string for the underwater snake robot.

    Creates a chain of capsule links connected by hinge joints with motor
    actuators, immersed in fluid via MuJoCo's density/viscosity options.

    Args:
        config: Snake and simulation parameters.

    Returns:
        MJCF XML string.
    """
    link_len = config.link_length
    radius = config.link_radius
    stiffness = config.joint_stiffness
    damping = config.joint_damping
    armature = config.armature
    joint_range = config.joint_range_deg
    gear = config.motor_gear
    force_range = config.motor_force_range

    # Build nested body chain
    indent = "      "
    bodies_open = ""
    bodies_close = ""
    actuators = ""

    # Head link (link_0) with freejoint
    bodies_open += f"""
    <body name="link_0" pos="0 0 0">
      <freejoint name="root"/>
      <geom name="g_0" type="capsule" fromto="0 0 0 {link_len} 0 0"
            size="{radius}" density="{config.link_density}"/>
      <site name="head_site" pos="0 0 0" size="0.001"/>
"""

    # Links 1 through N-1 (nested)
    for i in range(1, config.num_links):
        bodies_open += f"""{indent * i}<body name="link_{i}" pos="{link_len} 0 0">
{indent * i}  <joint name="j_{i}" type="hinge" axis="0 0 1"
{indent * i}         limited="true" range="-{joint_range} {joint_range}"
{indent * i}         stiffness="{stiffness}" damping="{damping}" armature="{armature}"/>
{indent * i}  <geom name="g_{i}" type="capsule" fromto="0 0 0 {link_len} 0 0"
{indent * i}        size="{radius}" density="{config.link_density}"/>
"""
        bodies_close = f"{indent * i}</body>\n" + bodies_close

        # Motor actuator for this joint
        actuators += (
            f'    <motor name="a_{i}" joint="j_{i}" '
            f'gear="{gear}" ctrllimited="true" '
            f'ctrlrange="-{force_range} {force_range}"/>\n'
        )

    # Close head body
    bodies_close += "    </body>\n"

    xml = f"""<mujoco model="underwater_snake_zheng2022">
  <option timestep="{config.sim_timestep}"
          gravity="0 0 0"
          density="{config.fluid_density}"
          viscosity="{config.fluid_viscosity}">
    <flag energy="enable"/>
  </option>

  <default>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
{bodies_open}{bodies_close}  </worldbody>

  <actuator>
{actuators}  </actuator>
</mujoco>
"""
    return xml


def _quat_to_yaw(quat: np.ndarray) -> float:
    """Extract yaw (z-rotation) from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class UnderwaterSnakeEnv(EnvBase):
    """TorchRL environment for underwater snake locomotion.

    Observation (16D):
        - Head angular velocity (z-axis): 1D
        - Joint angular velocities: 6D
        - Head rotation angle (yaw): 1D
        - Joint rotation angles: 6D
        - Head velocity (vx, vy): 2D

    Action (6D):
        - Motor torques in [-1, 1] (scaled by gear ratio to actual torque)
    """

    def __init__(
        self,
        config: Optional[ZhengConfig] = None,
        device: str = "cpu",
        batch_size: Optional[torch.Size] = None,
    ):
        self.config = config or ZhengConfig()
        self._device = device

        super().__init__(device=device, batch_size=batch_size or torch.Size([]))

        # Build and load MuJoCo model
        xml_string = build_mjcf_xml(self.config)
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # Cache joint/actuator indices
        self._num_joints = self.config.num_joints  # 6
        # Freejoint: qpos[0:7] (3 pos + 4 quat), qvel[0:6] (3 lin + 3 ang)
        # Hinge joints: qpos[7:13], qvel[6:12]
        self._joint_qpos_start = 7
        self._joint_qvel_start = 6

        # Episode tracking
        self._step_count = 0

        # Build specs
        self._make_spec()

    def _make_spec(self) -> None:
        """Define observation, action, reward, and done specs."""
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self.config.obs_dim,),
                dtype=torch.float32,
                device=self._device,
            ),
            # Extra fields for curriculum reward computation
            head_velocity_x=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self._device,
            ),
            power=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self._device,
            ),
            shape=self.batch_size,
            device=self._device,
        )

        self.action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(self.config.action_dim,),
            dtype=torch.float32,
            device=self._device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
            device=self._device,
        )

        self.done_spec = CompositeSpec(
            done=DiscreteTensorSpec(
                n=2, shape=(1,), dtype=torch.bool, device=self._device,
            ),
            terminated=DiscreteTensorSpec(
                n=2, shape=(1,), dtype=torch.bool, device=self._device,
            ),
            truncated=DiscreteTensorSpec(
                n=2, shape=(1,), dtype=torch.bool, device=self._device,
            ),
            shape=self.batch_size,
            device=self._device,
        )

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        """Reset to initial straight configuration."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._make_tensordict()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one control step (4 sim steps at 100Hz = 25Hz control)."""
        action = tensordict["action"].cpu().numpy()

        # Apply torques to actuators
        self.data.ctrl[:] = action

        # Step simulation (sim_steps_per_control times)
        for _ in range(self.config.sim_steps_per_control):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Check termination
        truncated = self._step_count >= self.config.max_episode_steps
        terminated = self._check_terminated()
        done = terminated or truncated

        return self._make_tensordict(
            terminated=terminated, truncated=truncated, done=done,
        )

    def _make_tensordict(
        self,
        terminated: bool = False,
        truncated: bool = False,
        done: bool = False,
    ) -> TensorDictBase:
        """Build TensorDict from current MuJoCo state."""
        obs = self._get_observation()
        head_vx = float(self.data.qvel[0])  # x-velocity of freejoint
        power = self._compute_power()

        # Reward is a placeholder — overwritten by curriculum in training loop
        reward = head_vx  # Simple default: forward velocity

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "head_velocity_x": torch.tensor([head_vx], dtype=torch.float32, device=self._device),
                "power": torch.tensor([power], dtype=torch.float32, device=self._device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
                "done": torch.tensor([done], dtype=torch.bool, device=self._device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self._device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self._device),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _get_observation(self) -> np.ndarray:
        """Build 16D observation vector.

        Components:
            [0]     Head angular velocity (z-axis rotation of freejoint)
            [1:7]   Joint angular velocities (6 hinge joints)
            [7]     Head rotation angle (yaw from quaternion)
            [8:14]  Joint rotation angles (6 hinge joints)
            [14:16] Head velocity (vx, vy)
        """
        # Head angular velocity: qvel[5] is wz of the freejoint
        head_ang_vel = self.data.qvel[5]

        # Joint angular velocities: qvel[6:12]
        joint_ang_vels = self.data.qvel[self._joint_qvel_start:self._joint_qvel_start + self._num_joints]

        # Head rotation angle (yaw): from quaternion qpos[3:7]
        head_yaw = _quat_to_yaw(self.data.qpos[3:7])

        # Joint rotation angles: qpos[7:13]
        joint_angles = self.data.qpos[self._joint_qpos_start:self._joint_qpos_start + self._num_joints]

        # Head velocity (vx, vy): qvel[0:2]
        head_vel_xy = self.data.qvel[0:2]

        obs = np.concatenate([
            [head_ang_vel],       # 1D
            joint_ang_vels,       # 6D
            [head_yaw],           # 1D
            joint_angles,         # 6D
            head_vel_xy,          # 2D
        ])  # Total: 16D

        return obs.astype(np.float32)

    def _compute_power(self) -> float:
        """Compute instantaneous mechanical power: sum(|tau_i * omega_i|).

        Uses actuator forces and joint velocities.
        """
        # Actuator forces (after gear scaling)
        actuator_forces = self.data.actuator_force  # (6,)
        # Joint velocities
        joint_vels = self.data.qvel[self._joint_qvel_start:self._joint_qvel_start + self._num_joints]
        # Power = sum of |force * velocity| for each joint
        power = np.sum(np.abs(actuator_forces * joint_vels))
        return float(power)

    def _check_terminated(self) -> bool:
        """Check for early termination (e.g., NaN in state)."""
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return True
        return False

    def _set_seed(self, seed: Optional[int]) -> None:
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
