"""TorchRL environment for free-body snake locomotion (PyElastica backend).

A soft snake robot moves in 2D with RFT (Resistive Force Theory) anisotropic
friction. The agent controls 5-dim serpenoid steering parameters which are
transformed to joint curvatures via DirectSerpenoidSteeringTransform.

Uses PyElastica's Cosserat rod simulation instead of DisMech.

The goal is placed along the snake's initial heading direction. The reward
uses a distance-based potential formulation:
    R = c_dist * (prev_dist - curr_dist) + c_align * cos(theta_g) + goal_bonus
"""

import math
import time as _time
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

from locomotion_elastica.config import LocomotionElasticaEnvConfig
from locomotion_elastica.rewards import compute_goal_reward
from src.physics.cpg.action_wrapper import DirectSerpenoidSteeringTransform

# PyElastica imports
import elastica
from elastica import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Damping,
    Forcing,
    CallBacks,
    CosseratRod,
)
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import extend_stepper_interface


class SnakeSimulator(
    BaseSystemCollection, Constraints, Forcing, Damping, Connections, CallBacks
):
    """PyElastica simulator container for the snake rod."""
    pass


class AnisotropicRFTForce(elastica.external_forces.NoForces):
    """Resistive Force Theory (RFT) anisotropic drag for ground locomotion.

    Applies velocity-dependent drag with different coefficients for tangential
    (along body) and normal (perpendicular to body) directions:
        F_t = -c_t * v_t   (tangential drag, lower)
        F_n = -c_n * v_n   (normal drag, higher)

    The anisotropy (c_n > c_t) is what enables serpentine locomotion: lateral
    undulation pushes against the ground more than it slides along it.
    """

    def __init__(self, c_t: float, c_n: float):
        self.c_t = c_t
        self.c_n = c_n

    def apply_forces(self, system, time=0.0):
        # Element tangents: (3, n_elements)
        tangents = system.tangents  # (3, n_elem)
        # Element velocities from node velocities: average adjacent nodes
        vel_nodes = system.velocity_collection  # (3, n_nodes)
        vel_elem = 0.5 * (vel_nodes[:, :-1] + vel_nodes[:, 1:])  # (3, n_elem)

        # Vectorized projection: v_t_scalar = sum(v * t, axis=0)
        v_t_scalar = np.sum(vel_elem * tangents, axis=0)  # (n_elem,)
        v_t = v_t_scalar[np.newaxis, :] * tangents  # (3, n_elem)
        v_n = vel_elem - v_t  # (3, n_elem)

        # RFT drag force per element
        f_elem = -self.c_t * v_t - self.c_n * v_n  # (3, n_elem)

        # Distribute to adjacent nodes (half each)
        system.external_forces[:, :-1] += 0.5 * f_elem
        system.external_forces[:, 1:] += 0.5 * f_elem


class LocomotionElasticaEnv(EnvBase):
    """Free-body snake locomotion environment with goal-directed reward (PyElastica).

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
        config: Optional[LocomotionElasticaEnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or LocomotionElasticaEnvConfig()
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

        # PyElastica state (initialized in _reset)
        self._simulator = None
        self._rod = None
        self._time_stepper = None
        self._do_step = None
        self._steps_and_prefactors = None

        # Episode state
        self._step_count = 0
        self._initial_heading = np.array([1.0, 0.0])
        self._goal_xy = np.array([2.0, 0.0])
        self._prev_com = np.zeros(2)
        self._prev_heading_angle = 0.0
        self._prev_dist_to_goal = 2.0
        self._energy_rate = 0.0
        self._episode_start_time = 0.0
        self._no_progress_count = 0

        # Build specs
        self._make_spec()

    def _make_spec(self):
        """Define observation, action, reward, and done specs."""
        _scalar = lambda dtype=torch.float32: UnboundedContinuousTensorSpec(
            shape=(1,), dtype=dtype, device=self._device
        )

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self.OBS_DIM,), dtype=torch.float32, device=self._device
            ),
            # Reward diagnostics (preserved by collector)
            v_g=_scalar(),
            dist_to_goal=_scalar(),
            theta_g=_scalar(),
            reward_dist=_scalar(),
            reward_align=_scalar(),
            # Episode-end diagnostics
            episode_wall_time_s=_scalar(),
            final_dist_to_goal=_scalar(),
            goal_reached=_scalar(torch.bool),
            starvation=_scalar(torch.bool),
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

    # === PyElastica initialization ===

    def _init_elastica(self, initial_direction: np.ndarray) -> None:
        """Create PyElastica Cosserat rod and time stepper."""
        physics = self.config.physics

        # Build simulator
        self._simulator = SnakeSimulator()

        # Rod geometry: straight rod along initial_direction
        dir_3d = np.array([initial_direction[0], initial_direction[1], 0.0])
        dir_3d = dir_3d / np.linalg.norm(dir_3d)

        start = -0.5 * physics.geometry.snake_length * dir_3d

        # Create the Cosserat rod
        n_elements = self._num_segments
        self._rod = CosseratRod.straight_rod(
            n_elements=n_elements,
            start=start,
            direction=dir_3d,
            normal=np.array([0.0, 0.0, 1.0]),
            base_length=physics.geometry.snake_length,
            base_radius=physics.geometry.snake_radius,
            density=physics.density,
            youngs_modulus=physics.youngs_modulus,
            shear_modulus=physics.youngs_modulus / (2.0 * (1.0 + physics.poisson_ratio)),
        )

        self._simulator.append(self._rod)

        # Add gravity if enabled
        if physics.enable_gravity:
            from elastica.external_forces import GravityForces
            self._simulator.add_forcing_to(self._rod).using(
                GravityForces, acc_gravity=np.array(physics.gravity)
            )

        # Add damping (low value — high damping freezes the rod entirely)
        from elastica.dissipation import AnalyticalLinearDamper
        self._simulator.dampen(self._rod).using(
            AnalyticalLinearDamper,
            damping_constant=physics.elastica_damping,
            time_step=physics.dt / physics.elastica_substeps,
        )

        # Add RFT anisotropic friction (required for serpentine locomotion)
        friction = physics.friction
        self._rft_force = AnisotropicRFTForce(
            c_t=friction.rft_ct,
            c_n=friction.rft_cn,
        )
        self._simulator.add_forcing_to(self._rod).using(
            AnisotropicRFTForce,
            c_t=friction.rft_ct,
            c_n=friction.rft_cn,
        )

        # Finalize
        self._simulator.finalize()

        # Time stepper
        self._time_stepper = PositionVerlet()
        self._do_step, self._steps_and_prefactors = extend_stepper_interface(
            self._time_stepper, self._simulator
        )

    # === State accessors ===

    def _get_positions(self) -> np.ndarray:
        """Get node positions, shape (num_nodes, 3)."""
        return self._rod.position_collection.T.copy()

    def _get_velocities(self) -> np.ndarray:
        """Get node velocities, shape (num_nodes, 3)."""
        return self._rod.velocity_collection.T.copy()

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
        seg_len = physics.geometry.snake_length / self._num_segments
        seg_mass = physics.density * np.pi * physics.geometry.snake_radius**2 * seg_len
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

    # === Apply curvature control to PyElastica ===

    def _apply_curvature_to_elastica(self, curvatures: np.ndarray) -> None:
        """Write curvatures to PyElastica rod rest_kappa."""
        n = min(len(curvatures), self._rod.rest_kappa.shape[1])
        for i in range(n):
            self._rod.rest_kappa[0, i] = curvatures[i]

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

        # Initialize PyElastica
        self._init_elastica(self._initial_heading)

        # Reset serpenoid transform
        self._serpenoid.reset()

        # Initialize episode state
        self._step_count = 0
        self._episode_start_time = _time.monotonic()
        self._no_progress_count = 0
        positions = self._get_positions()
        self._prev_com = self._get_com(positions)
        self._prev_heading_angle = self._get_heading_angle(positions)
        self._energy_rate = 0.0

        # Place goal along initial heading
        com = self._get_com(positions)
        self._goal_xy = com + self.config.goal.goal_distance * self._initial_heading
        self._prev_dist_to_goal = self.config.goal.goal_distance

        obs = self._get_obs()

        _zero = torch.tensor([0.0], dtype=torch.float32, device=self._device)
        _false = torch.tensor([False], dtype=torch.bool, device=self._device)

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "done": _false.clone(),
                "terminated": _false.clone(),
                "truncated": _false.clone(),
                # Diagnostic defaults
                "v_g": _zero.clone(),
                "dist_to_goal": torch.tensor(
                    [self.config.goal.goal_distance], dtype=torch.float32, device=self._device
                ),
                "theta_g": _zero.clone(),
                "reward_dist": _zero.clone(),
                "reward_align": _zero.clone(),
                "episode_wall_time_s": _zero.clone(),
                "final_dist_to_goal": _zero.clone(),
                "goal_reached": _false.clone(),
                "starvation": _false.clone(),
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

        # Denormalize action to physical serpenoid parameters
        params = self._serpenoid.denormalize_action(action)
        self._serpenoid._current_params = params
        amplitude = params["amplitude"]
        frequency = params["frequency"]
        wave_number = params["wave_number"]
        phase = params["phase"]
        turn_bias = params["turn_bias"]

        omega = 2 * np.pi * frequency
        k = 2 * np.pi * wave_number

        # Step PyElastica: update curvature once per physics step, then run substeps
        physics = self.config.physics
        dt_sub = physics.dt / physics.elastica_substeps
        dt_physics = physics.dt
        joint_positions = self._serpenoid._joint_positions
        time = 0.0
        for _ in range(self.config.control.substeps_per_action):
            # Update curvature wave at the start of each physics step
            curvatures = (
                amplitude * np.sin(
                    k * joint_positions
                    + omega * self._serpenoid._time
                    + phase
                )
                + turn_bias
            )
            self._apply_curvature_to_elastica(curvatures)
            # Run elastica substeps with this curvature held constant
            for _ in range(physics.elastica_substeps):
                time = self._do_step(
                    self._time_stepper, self._steps_and_prefactors,
                    self._simulator, time, dt_sub
                )
            self._serpenoid._time += dt_physics

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

        # Termination: goal reached?
        goal_reached = dist_to_goal <= self.config.goal.goal_radius

        # Compute reward (distance-based potential)
        reward = compute_goal_reward(
            config=self.config.rewards,
            prev_dist=self._prev_dist_to_goal,
            curr_dist=dist_to_goal,
            theta_g=theta_g,
            goal_reached=goal_reached,
        )

        if np.isnan(reward):
            reward = 0.0

        # Reward components for logging
        c = self.config.rewards
        reward_dist = c.c_dist * (self._prev_dist_to_goal - dist_to_goal)
        reward_align = c.c_align * math.cos(theta_g)

        # Track starvation: no distance reduction for N consecutive steps
        if dist_to_goal >= self._prev_dist_to_goal:
            self._no_progress_count += 1
        else:
            self._no_progress_count = 0

        # Update previous distance
        self._prev_dist_to_goal = dist_to_goal

        self._step_count += 1

        # Termination conditions
        starvation = self._no_progress_count >= self.config.goal.starvation_timeout

        terminated = goal_reached or starvation
        truncated = self._step_count >= self.config.max_episode_steps
        done = terminated or truncated

        obs = self._get_obs()

        episode_wall_s = _time.monotonic() - self._episode_start_time

        step_dict = {
            "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
            "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
            "done": torch.tensor([done], dtype=torch.bool, device=self._device),
            "terminated": torch.tensor([terminated], dtype=torch.bool, device=self._device),
            "truncated": torch.tensor([truncated], dtype=torch.bool, device=self._device),
            # Reward diagnostics (per-step, always present for collector)
            "v_g": torch.tensor([v_g], dtype=torch.float32, device=self._device),
            "dist_to_goal": torch.tensor([dist_to_goal], dtype=torch.float32, device=self._device),
            "theta_g": torch.tensor([theta_g], dtype=torch.float32, device=self._device),
            "reward_dist": torch.tensor([reward_dist], dtype=torch.float32, device=self._device),
            "reward_align": torch.tensor([reward_align], dtype=torch.float32, device=self._device),
            # Episode-end diagnostics (always present; meaningful only when done)
            "episode_wall_time_s": torch.tensor(
                [episode_wall_s], dtype=torch.float32, device=self._device
            ),
            "final_dist_to_goal": torch.tensor(
                [dist_to_goal if done else 0.0], dtype=torch.float32, device=self._device
            ),
            "goal_reached": torch.tensor(
                [goal_reached], dtype=torch.bool, device=self._device
            ),
            "starvation": torch.tensor(
                [starvation], dtype=torch.bool, device=self._device
            ),
        }

        return TensorDict(
            step_dict,
            batch_size=self.batch_size,
            device=self._device,
        )

    def _check_starvation(self) -> bool:
        """Check if no distance progress for too many consecutive steps."""
        return self._no_progress_count >= self.config.goal.starvation_timeout

    def close(self, **kwargs):
        self._simulator = None
        self._rod = None
        self._time_stepper = None
        self._do_step = None
        self._steps_and_prefactors = None
        super().close(**kwargs)
