"""TorchRL environment for clamped soft manipulator (Choi & Tong, 2025).

Recreates the DisMech-based soft manipulator environment using TorchRL's
EnvBase interface.  A 1m rod is clamped at one end; the agent controls
delta natural curvature at sparse control points.
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

from choi2025.configs_choi2025 import Choi2025EnvConfig, TaskType
from choi2025.control_choi2025 import DeltaCurvatureController
from choi2025.rewards_choi2025 import (
    compute_follow_target_reward,
    compute_ik_reward,
    compute_obstacle_reward,
)
from choi2025.tasks_choi2025 import ObstacleManager, TargetGenerator

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


class SoftManipulatorEnv(EnvBase):
    """Clamped soft manipulator environment for control tasks.

    A 1m elastic rod is clamped at node 0.  The agent controls delta natural
    curvature at 5 (configurable) control points.  The tip must reach targets
    while optionally avoiding obstacles.

    Observation:
        - node positions flattened: (num_nodes * 3,)
        - node velocities flattened: (num_nodes * 3,)
        - curvatures at bend springs: (num_bend_springs,)
        - target position: (3,)
        - target orientation (IK only): (3,)
        Total dim depends on task.

    Action:
        - delta curvature at control points
        - Shape: (num_control_points * 2,) for 3D
    """

    def __init__(
        self,
        config: Optional[Choi2025EnvConfig] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.config = config or Choi2025EnvConfig()
        self._device = device

        physics = self.config.physics
        self._num_nodes = physics.geometry.num_nodes
        self._num_segments = physics.geometry.num_segments
        self._num_bend_springs = self._num_segments - 1

        # Controller
        self.controller = DeltaCurvatureController(
            num_bend_springs=self._num_bend_springs,
            config=self.config.control,
        )

        # RNG for reproducible target/obstacle sampling
        self._rng = np.random.default_rng(42)

        # Target generator and obstacle manager
        self._target = TargetGenerator(self.config.target, self._rng)
        self._obstacles = ObstacleManager(self.config.obstacles, self._rng)

        # DisMech state (initialized in _reset)
        self._dismech_robot = None
        self._time_stepper = None

        # Episode state
        self._step_count = 0
        self._prev_tip_pos = np.zeros(3)

        # Build specs
        self._make_spec()

    @property
    def _two_d_sim(self) -> bool:
        return self.config.physics.two_d_sim

    @property
    def _action_dim(self) -> int:
        n_cp = self.config.control.num_control_points
        return n_cp if self._two_d_sim else n_cp * 2

    @property
    def _obs_dim(self) -> int:
        """Compute observation dimension."""
        dim = 0
        dim += self._num_nodes * 3  # positions
        dim += self._num_nodes * 3  # velocities
        dim += self._num_bend_springs  # curvatures
        dim += 3  # target position
        if self.config.task == TaskType.INVERSE_KINEMATICS:
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

    # === DisMech initialization ===

    def _init_dismech(self) -> None:
        """Create DisMech rod and time stepper."""
        physics = self.config.physics

        geom_params = GeomParams(
            rod_r0=physics.snake_radius,
            shell_h=0,
        )

        material = Material(
            density=physics.density,
            youngs_rod=physics.youngs_modulus,
            youngs_shell=0,
            poisson_rod=physics.poisson_ratio,
            poisson_shell=0,
        )

        # Use fewer Newton iterations for non-contact tasks
        has_obstacles = self.config.task in (
            TaskType.TIGHT_OBSTACLES,
            TaskType.RANDOM_OBSTACLES,
        )
        max_iter = (
            physics.max_newton_iter_contact
            if has_obstacles
            else physics.max_newton_iter_noncontact
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
            max_iter=max_iter,
            total_time=1000.0,
            plot_step=1,
            tol=physics.tol,
            ftol=physics.ftol,
            dtol=physics.dtol,
        )

        env = Environment()

        # Add gravity if enabled
        if physics.enable_gravity:
            env.add_force("gravity", g=np.array(physics.gravity))

        # Create rod geometry (straight along x-axis from origin)
        geometry = self._create_rod_geometry()

        # Create SoftRobot
        self._dismech_robot = SoftRobot(geom_params, material, geometry, sim_params, env)

        # Clamp first node (boundary condition)
        if physics.clamp_first_node:
            self._dismech_robot = self._dismech_robot.fix_nodes([0])

        # Create time stepper
        self._time_stepper = ImplicitEulerTimeStepper(self._dismech_robot)

    def _create_rod_geometry(self) -> Geometry:
        """Create a straight rod along the x-axis."""
        physics = self.config.physics
        num_nodes = self._num_nodes
        segment_length = physics.snake_length / self._num_segments

        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            nodes[i, 0] = i * segment_length

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

    def _get_tip_pos(self) -> np.ndarray:
        """Get position of the last node (tip)."""
        return self._get_positions()[-1]

    def _get_tip_tangent(self) -> np.ndarray:
        """Get tangent vector at the tip (normalized)."""
        positions = self._get_positions()
        tangent = positions[-1] - positions[-2]
        norm = np.linalg.norm(tangent)
        return tangent / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])

    def _get_curvatures(self) -> np.ndarray:
        """Compute discrete curvatures at internal nodes."""
        positions = self._get_positions()
        curvatures = []

        for i in range(1, self._num_nodes - 1):
            v1 = positions[i] - positions[i - 1]
            v2 = positions[i + 1] - positions[i]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm < 1e-8 or v2_norm < 1e-8:
                curvatures.append(0.0)
                continue

            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            avg_length = (v1_norm + v2_norm) / 2
            curvatures.append(angle / avg_length if avg_length > 1e-8 else 0.0)

        return np.array(curvatures)

    # === Observation construction ===

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        positions = self._get_positions().flatten()
        velocities = self._get_velocities().flatten()
        curvatures = self._get_curvatures()
        target_pos = self._target.position

        parts = [positions, velocities, curvatures, target_pos]

        if self.config.task == TaskType.INVERSE_KINEMATICS:
            parts.append(self._target.orientation)

        return np.concatenate(parts).astype(np.float32)

    # === Apply curvature control to DisMech ===

    def _apply_curvature_to_dismech(self, curvature_state: np.ndarray) -> None:
        """Write curvature state to DisMech bend springs.

        Args:
            curvature_state: Shape (num_bend_springs, 2) for kappa1, kappa2.
        """
        bend_springs = self._dismech_robot.bend_springs

        if bend_springs.N > 0 and hasattr(bend_springs, "nat_strain"):
            n = min(len(curvature_state), bend_springs.N)
            for i in range(n):
                bend_springs.nat_strain[i, 0] = curvature_state[i, 0]
                bend_springs.nat_strain[i, 1] = curvature_state[i, 1]

    # === EnvBase interface ===

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._target = TargetGenerator(self.config.target, self._rng)
            self._obstacles = ObstacleManager(self.config.obstacles, self._rng)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Reinitialize DisMech from scratch
        self._init_dismech()

        # Reset controller
        self.controller.reset()

        # Sample target and obstacles
        self._target.sample(self.config.task)
        self._obstacles.setup(self.config.task)

        self._step_count = 0
        self._prev_tip_pos = self._get_tip_pos()

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

        # Record pre-step tip position
        self._prev_tip_pos = self._get_tip_pos()

        # Apply delta curvature control
        curvature_state = self.controller.apply_delta(action, two_d_sim=self._two_d_sim)
        self._apply_curvature_to_dismech(curvature_state)

        # Step DisMech simulation
        try:
            self._dismech_robot, _ = self._time_stepper.step(
                self._dismech_robot, debug=False
            )
        except ValueError:
            pass  # Convergence issue — state unchanged

        # Move target (for follow_target task)
        if self.config.task == TaskType.FOLLOW_TARGET:
            self._target.step(self.config.physics.dt)

        # Compute reward
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

    def _compute_reward(self) -> float:
        """Dispatch to task-specific reward function."""
        tip_pos = self._get_tip_pos()
        target_pos = self._target.position
        task = self.config.task

        if task == TaskType.FOLLOW_TARGET:
            return compute_follow_target_reward(tip_pos, target_pos, self._prev_tip_pos)

        elif task == TaskType.INVERSE_KINEMATICS:
            tip_tangent = self._get_tip_tangent()
            return compute_ik_reward(
                tip_pos, tip_tangent, target_pos, self._target.orientation
            )

        elif task in (TaskType.TIGHT_OBSTACLES, TaskType.RANDOM_OBSTACLES):
            positions = self._get_positions()
            penetrations = self._obstacles.compute_penetrations(positions)
            total_pen = float(np.sum(penetrations))
            return compute_obstacle_reward(
                tip_pos,
                target_pos,
                self._prev_tip_pos,
                total_pen,
                self.config.obstacles.contact_penalty,
            )

        return 0.0

    def close(self):
        self._dismech_robot = None
        self._time_stepper = None
        super().close()
