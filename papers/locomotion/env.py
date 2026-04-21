"""TorchRL environment for free-body snake locomotion.

A soft snake robot moves in 2D with RFT (Resistive Force Theory) anisotropic
friction. The agent controls 5-dim serpenoid steering parameters which are
transformed to joint curvatures via DirectSerpenoidSteeringTransform.

The goal is placed along the snake's initial heading direction. The reward
follows Liu et al. (2023) potential-field formulation:
    R = c_v * v_g + c_g * v_g * cos(theta_g) / dist
"""

import math
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
try:
    from torchrl.data import (
        BoundedTensorSpec,
        CompositeSpec,
        UnboundedContinuousTensorSpec,
    )
except ImportError:
    from torchrl.data import (
        Bounded as BoundedTensorSpec,
        Composite as CompositeSpec,
        Unbounded as UnboundedContinuousTensorSpec,
    )
from torchrl.envs import EnvBase

from locomotion.config import LocomotionEnvConfig
from locomotion.rewards import compute_goal_reward
from src.physics.cpg.action_wrapper import DirectSerpenoidSteeringTransform

# DisMech imports
import dismech
from dismech import (
    Environment,
    Geometry,
    GeomParams,
    ImplicitEulerTimeStepper,
    Material,
    SimParams,
    SoftRobot,
)


class LocomotionEnv(EnvBase):
    """Free-body snake locomotion environment with goal-directed reward.

    The snake moves in 2D with RFT friction. A goal point is placed along the
    initial heading direction at a configurable distance.

    Observation (14 dims):
        - Curvature modes (amplitude, wave_number, phase): 3
        - Body heading (cos, sin): 2
        - Yaw angular velocity: 1
        - CoM velocity in body frame (forward, lateral): 2
        - Distance to goal: 1
        - Heading error to goal (radians): 1
        - Velocity toward goal: 1
        - Forward speed (scalar): 1
        - Energy rate: 1
        - Current turn bias: 1

    Action (5 dims, [-1, 1]):
        [0] amplitude, [1] frequency, [2] wave_number, [3] phase, [4] turn_bias
    """

    OBS_DIM = 14

    def __init__(
        self,
        config: Optional[LocomotionEnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or LocomotionEnvConfig()
        self._device = device

        physics = self.config.physics
        self._num_nodes = physics.geometry.num_nodes
        self._num_segments = physics.geometry.num_segments
        self._num_joints = physics.geometry.num_joints

        # Serpenoid steering transform (5-dim action -> num_joints curvatures)
        control = self.config.control
        self._serpenoid = DirectSerpenoidSteeringTransform(
            num_joints=self._num_joints,
            amplitude_range=control.amplitude_range,
            frequency_range=control.frequency_range,
            turn_bias_range=control.turn_bias_range,
        )

        # RNG
        self._rng = np.random.default_rng(42)

        # DisMech state (initialized in _reset)
        self._dismech_robot = None
        self._time_stepper = None

        # Episode state
        self._step_count = 0
        self._initial_heading = np.array([1.0, 0.0])
        self._goal_xy = np.array([2.0, 0.0])
        self._prev_com = np.zeros(2)
        self._prev_heading_angle = 0.0
        self._energy_rate = 0.0
        self._v_g_history = []

        # Build specs
        self._make_spec()

    def _make_spec(self):
        """Define observation, action, reward, and done specs."""
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self.OBS_DIM,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        self.action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(self.config.control.action_dim,),
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

    # === DisMech initialization ===

    def _init_dismech(self, initial_direction: np.ndarray) -> None:
        """Create DisMech rod and time stepper."""
        physics = self.config.physics

        geom_params = GeomParams(rod_r0=physics.snake_radius, shell_h=0)

        material = Material(
            density=physics.density,
            youngs_rod=physics.youngs_modulus,
            youngs_shell=0,
            poisson_rod=physics.poisson_ratio,
            poisson_shell=0,
        )

        sim_params = SimParams(
            static_sim=False,
            two_d_sim=physics.two_d_sim,
            use_mid_edge=False,
            use_line_search=True,
            log_data=False,
            log_step=1,
            show_floor=False,
            dt=physics.dt,
            max_iter=physics.max_iter,
            total_time=1000.0,
            plot_step=1,
            tol=physics.tol,
            ftol=physics.ftol,
            dtol=physics.dtol,
        )

        env = Environment()

        if physics.enable_gravity:
            env.add_force("gravity", g=np.array(physics.gravity))

        # Add friction forces
        friction = physics.friction
        from src.configs.physics import FrictionModel
        if friction.model == FrictionModel.RFT:
            env.add_force("rft", ct=friction.rft_ct, cn=friction.rft_cn)
        elif friction.model in (FrictionModel.COULOMB, FrictionModel.STRIBECK):
            env.add_force(
                "floorContact",
                ground_z=0.0,
                stiffness=friction.ground_stiffness,
                delta=friction.ground_delta,
                h=physics.snake_radius,
            )
            env.add_force(
                "floorFriction",
                mu=friction.mu_kinetic,
                vel_tol=friction.stribeck_velocity,
            )

        geometry = self._create_rod_geometry(initial_direction)
        self._dismech_robot = SoftRobot(geom_params, material, geometry, sim_params, env)
        self._time_stepper = ImplicitEulerTimeStepper(self._dismech_robot)

    def _create_rod_geometry(self, direction: np.ndarray) -> Geometry:
        """Create a straight rod along given direction in XY plane."""
        physics = self.config.physics
        num_nodes = self._num_nodes
        segment_length = physics.snake_length / self._num_segments

        dir_3d = np.array([direction[0], direction[1], 0.0])
        dir_3d = dir_3d / np.linalg.norm(dir_3d)

        start = -0.5 * physics.snake_length * dir_3d

        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            nodes[i] = start + i * segment_length * dir_3d

        edges = np.array([[i, i + 1] for i in range(self._num_segments)], dtype=np.int64)
        face_nodes = np.empty((0, 3), dtype=np.int64)
        return Geometry(nodes, edges, face_nodes, plot_from_txt=False)

    # === State accessors ===

    def _get_positions(self) -> np.ndarray:
        """Get node positions, shape (num_nodes, 3)."""
        q = self._dismech_robot.state.q
        return q[: 3 * self._num_nodes].reshape(self._num_nodes, 3).copy()

    def _get_velocities(self) -> np.ndarray:
        """Get node velocities, shape (num_nodes, 3)."""
        u = self._dismech_robot.state.u
        return u[: 3 * self._num_nodes].reshape(self._num_nodes, 3).copy()

    def _get_com(self, positions: np.ndarray) -> np.ndarray:
        """Get center of mass in XY plane, shape (2,)."""
        return positions[:, :2].mean(axis=0)

    def _get_heading(self, positions: np.ndarray) -> np.ndarray:
        """Get body heading as unit vector in XY (tail-to-head)."""
        head = positions[-1, :2]
        tail = positions[0, :2]
        direction = head - tail
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return self._initial_heading.copy()
        return direction / norm

    def _get_heading_angle(self, positions: np.ndarray) -> float:
        heading = self._get_heading(positions)
        return float(np.arctan2(heading[1], heading[0]))

    def _get_yaw_rate(self, positions: np.ndarray) -> float:
        """Estimate yaw angular velocity from heading change."""
        current_angle = self._get_heading_angle(positions)
        delta = current_angle - self._prev_heading_angle
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        dt = self.config.physics.dt * self.config.control.substeps_per_action
        return delta / dt if dt > 0 else 0.0

    def _get_curvature_modes(self, positions: np.ndarray) -> np.ndarray:
        """Extract curvature modes (amplitude, wave_number, phase) via FFT."""
        curvatures = []
        for i in range(1, self._num_nodes - 1):
            v1 = positions[i] - positions[i - 1]
            v2 = positions[i + 1] - positions[i]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm < 1e-8 or v2_norm < 1e-8:
                curvatures.append(0.0)
                continue
            cross_z = v1[0] * v2[1] - v1[1] * v2[0]
            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            avg_len = (v1_norm + v2_norm) / 2
            kappa = np.sign(cross_z) * angle / avg_len if avg_len > 1e-8 else 0.0
            curvatures.append(kappa)

        curvatures = np.array(curvatures)
        n = len(curvatures)
        if n < 2:
            return np.zeros(3, dtype=np.float32)

        fft = np.fft.rfft(curvatures)
        magnitudes = np.abs(fft)

        if len(magnitudes) > 1:
            dominant_idx = 1 + np.argmax(magnitudes[1:])
            amplitude = 2.0 * magnitudes[dominant_idx] / n
            wave_number = float(dominant_idx) / n * (2 * np.pi)
            phase = float(np.angle(fft[dominant_idx]))
        else:
            amplitude = wave_number = phase = 0.0

        return np.array([amplitude, wave_number, phase], dtype=np.float32)

    def _get_energy_rate(self, velocities: np.ndarray) -> float:
        """Compute normalized kinetic energy rate."""
        physics = self.config.physics
        seg_len = physics.snake_length / self._num_segments
        seg_mass = physics.density * np.pi * physics.snake_radius**2 * seg_len
        ke = 0.5 * seg_mass * np.sum(velocities**2)
        ref = seg_mass * self._num_nodes * 0.1**2
        return ke / ref if ref > 0 else 0.0

    # === Goal-relative quantities ===

    def _get_dist_to_goal(self, com: np.ndarray) -> float:
        return float(np.linalg.norm(self._goal_xy - com))

    def _get_heading_error_to_goal(self, com: np.ndarray, heading: np.ndarray) -> float:
        """Heading angle error to goal in radians, in [-pi, pi]."""
        to_goal = self._goal_xy - com
        goal_angle = math.atan2(to_goal[1], to_goal[0])
        head_angle = math.atan2(heading[1], heading[0])
        error = goal_angle - head_angle
        # Wrap to [-pi, pi]
        error = (error + math.pi) % (2 * math.pi) - math.pi
        return error

    def _get_velocity_toward_goal(self, com: np.ndarray, v_com_xy: np.ndarray) -> float:
        """Projected velocity toward goal."""
        to_goal = self._goal_xy - com
        dist = np.linalg.norm(to_goal)
        if dist < 1e-6:
            return 0.0
        unit = to_goal / dist
        return float(np.dot(v_com_xy, unit))

    # === Observation construction ===

    def _get_obs(self) -> np.ndarray:
        """Build 14-dim observation vector."""
        positions = self._get_positions()
        velocities = self._get_velocities()

        # 1. Curvature modes (3)
        curv_modes = self._get_curvature_modes(positions)

        # 2. Body heading (cos, sin) (2)
        heading = self._get_heading(positions)

        # 3. Yaw angular velocity (1)
        yaw_rate = self._get_yaw_rate(positions)

        # 4. CoM velocity in body frame (forward, lateral) (2)
        com = self._get_com(positions)
        dt_total = self.config.physics.dt * self.config.control.substeps_per_action
        v_com_xy = (com - self._prev_com) / dt_total if dt_total > 0 else np.zeros(2)
        v_forward = float(np.dot(v_com_xy, heading))
        lateral_dir = np.array([-heading[1], heading[0]])
        v_lateral = float(np.dot(v_com_xy, lateral_dir))

        # 5. Distance to goal (1)
        dist_to_goal = self._get_dist_to_goal(com)

        # 6. Heading error to goal (1)
        theta_g = self._get_heading_error_to_goal(com, heading)

        # 7. Velocity toward goal (1)
        v_g = self._get_velocity_toward_goal(com, v_com_xy)

        # 8. Forward speed (scalar) (1)
        speed = float(np.linalg.norm(v_com_xy))

        # 9. Energy rate (1)
        self._energy_rate = self._get_energy_rate(velocities)

        # 10. Current turn bias (1)
        turn_bias = self._serpenoid.current_parameters.get("turn_bias", 0.0)

        obs = np.array([
            curv_modes[0], curv_modes[1], curv_modes[2],  # 3: curvature modes
            heading[0], heading[1],                        # 2: heading
            yaw_rate,                                      # 1: yaw rate
            v_forward, v_lateral,                          # 2: body-frame velocity
            dist_to_goal,                                  # 1: distance to goal
            theta_g,                                       # 1: heading error to goal
            v_g,                                           # 1: velocity toward goal
            speed,                                         # 1: forward speed
            self._energy_rate,                             # 1: energy
            turn_bias,                                     # 1: turn bias
        ], dtype=np.float32)

        return obs

    # === Apply curvature control to DisMech ===

    def _apply_curvature_to_dismech(self, curvatures: np.ndarray) -> None:
        """Write curvatures to DisMech bend springs."""
        bend_springs = self._dismech_robot.bend_springs
        if bend_springs.N > 0 and hasattr(bend_springs, "nat_strain"):
            n = min(len(curvatures), bend_springs.N)
            for i in range(n):
                bend_springs.nat_strain[i, 0] = 0.0
                bend_springs.nat_strain[i, 1] = curvatures[i]

    # === EnvBase interface ===

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Randomize initial heading
        if self.config.randomize_initial_heading:
            angle = self._rng.uniform(0, 2 * np.pi)
            self._initial_heading = np.array([np.cos(angle), np.sin(angle)])
        else:
            self._initial_heading = np.array([1.0, 0.0])

        # Initialize DisMech
        self._init_dismech(self._initial_heading)

        # Reset serpenoid transform
        self._serpenoid.reset()

        # Initialize episode state
        self._step_count = 0
        self._v_g_history.clear()
        positions = self._get_positions()
        self._prev_com = self._get_com(positions)
        self._prev_heading_angle = self._get_heading_angle(positions)
        self._energy_rate = 0.0

        # Place goal along initial heading
        com = self._get_com(positions)
        self._goal_xy = com + self.config.goal.goal_distance * self._initial_heading

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
        positions = self._get_positions()
        self._prev_com = self._get_com(positions)
        self._prev_heading_angle = self._get_heading_angle(positions)

        # Apply serpenoid curvatures once, then let physics run for all substeps.
        # The dynamic response (rod trying to reach rest curvature) drives
        # locomotion through anisotropic RFT friction.
        # Reverse curvature order so wave travels head→tail for forward motion.
        dt_total = self.config.physics.dt * self.config.control.substeps_per_action
        curvatures = self._serpenoid.step(action, dt=dt_total)
        curvatures = curvatures[::-1]
        self._apply_curvature_to_dismech(curvatures)

        for _ in range(self.config.control.substeps_per_action):
            try:
                self._dismech_robot, _ = self._time_stepper.step(
                    self._dismech_robot, debug=False
                )
            except ValueError:
                pass

        # Post-step state
        positions = self._get_positions()
        velocities = self._get_velocities()
        com = self._get_com(positions)
        heading = self._get_heading(positions)
        dt_total = self.config.physics.dt * self.config.control.substeps_per_action
        v_com_xy = (com - self._prev_com) / dt_total if dt_total > 0 else np.zeros(2)

        # Goal-relative quantities for reward
        dist_to_goal = self._get_dist_to_goal(com)
        theta_g = self._get_heading_error_to_goal(com, heading)
        v_g = self._get_velocity_toward_goal(com, v_com_xy)
        self._energy_rate = self._get_energy_rate(velocities)

        # Track v_g history for starvation detection
        self._v_g_history.append(v_g)

        # Compute reward (Liu 2023 potential-field)
        reward = compute_goal_reward(
            config=self.config.rewards,
            v_g=v_g,
            dist_to_goal=dist_to_goal,
            theta_g=theta_g,
        )

        if np.isnan(reward):
            reward = 0.0

        self._step_count += 1

        # Termination conditions
        goal_reached = dist_to_goal <= self.config.goal.goal_radius
        starvation = self._check_starvation()

        terminated = goal_reached or starvation
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

    def _check_starvation(self) -> bool:
        """Check if v_g has been negative for too many consecutive steps."""
        timeout = self.config.goal.starvation_timeout
        if len(self._v_g_history) < timeout:
            return False
        return all(v < 0 for v in self._v_g_history[-timeout:])

    def close(self):
        self._dismech_robot = None
        self._time_stepper = None
        super().close()
