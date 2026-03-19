"""TorchRL environment for clamped soft manipulator (Choi & Tong, 2025).

Recreates the DisMech-based soft manipulator environment using TorchRL's
EnvBase interface.  A 1m rod is clamped at one end; the agent controls
delta natural curvature at sparse control points.

When DisMech is not installed, a lightweight mock physics backend is used
that provides the same observation/action/reward interface with simplified
rod dynamics.  This allows pipeline validation without the C++ dependency.
"""

from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    Bounded,
    Composite,
    Unbounded,
)
from torchrl.envs import EnvBase

from choi2025.config import Choi2025EnvConfig, TaskType
from choi2025.control import DeltaCurvatureController
from choi2025.rewards import (
    compute_follow_target_reward,
    compute_ik_reward,
    compute_obstacle_reward,
)
from choi2025.tasks import ObstacleManager, TargetGenerator

# Try importing DisMech; fall back to mock if unavailable
try:
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
    _HAS_DISMECH = True
except ImportError:
    _HAS_DISMECH = False


class _MockRodState:
    """Lightweight mock of DisMech rod state for pipeline validation.

    Maintains node positions and velocities as flat arrays (same layout as
    DisMech's state.q and state.u) and applies simplified curvature-driven
    dynamics: curvature deltas rotate segments around bend springs.
    """

    def __init__(self, num_nodes: int, segment_length: float, dt: float):
        self._num_nodes = num_nodes
        self._segment_length = segment_length
        self._dt = dt

        # Positions: straight rod along x-axis (same as _create_rod_geometry)
        self._positions = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            self._positions[i, 0] = i * segment_length

        # Velocities: initially zero
        self._velocities = np.zeros((num_nodes, 3))

    @property
    def positions(self) -> np.ndarray:
        return self._positions.copy()

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities.copy()

    def apply_curvature_and_step(self, curvature_state: np.ndarray) -> None:
        """Simplified dynamics: rotate segments based on curvature state.

        Each curvature value induces a small angular displacement at the
        corresponding bend spring, propagating to downstream nodes.  This
        gives non-trivial dynamics without a full physics engine.
        """
        prev_positions = self._positions.copy()
        n_springs = len(curvature_state)

        for i in range(n_springs):
            kappa1, kappa2 = curvature_state[i]
            # Convert curvature to small angle
            angle_xy = kappa1 * self._dt * 0.1  # Damped to keep stable
            angle_xz = kappa2 * self._dt * 0.1

            # Rotate all nodes downstream of spring i+1
            pivot = self._positions[i + 1].copy()
            for j in range(i + 2, self._num_nodes):
                rel = self._positions[j] - pivot
                # Rotation in xy plane (kappa1)
                c, s = np.cos(angle_xy), np.sin(angle_xy)
                new_x = c * rel[0] - s * rel[1]
                new_y = s * rel[0] + c * rel[1]
                # Rotation in xz plane (kappa2)
                c2, s2 = np.cos(angle_xz), np.sin(angle_xz)
                new_x2 = c2 * new_x - s2 * rel[2]
                new_z = s2 * new_x + c2 * rel[2]
                self._positions[j] = pivot + np.array([new_x2, new_y, new_z])

        # Finite-difference velocity
        self._velocities = (self._positions - prev_positions) / self._dt

    def reset(self) -> None:
        """Reset to straight rod along x-axis."""
        for i in range(self._num_nodes):
            self._positions[i] = [i * self._segment_length, 0.0, 0.0]
        self._velocities[:] = 0.0


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
        device = torch.device(device) if isinstance(device, str) else device
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

        # Physics backend state (initialized in _reset)
        self._use_dismech = _HAS_DISMECH
        self._dismech_robot = None
        self._time_stepper = None
        self._mock_state = None

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
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(self._obs_dim,), dtype=torch.float32, device=self._device
            ),
            shape=(),
        )

        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self._action_dim,),
            dtype=torch.float32,
            device=self._device,
        )

        self.reward_spec = Unbounded(
            shape=(1,), dtype=torch.float32, device=self._device
        )

        self.done_spec = Composite(
            done=Unbounded(
                shape=(1,), dtype=torch.bool, device=self._device
            ),
            terminated=Unbounded(
                shape=(1,), dtype=torch.bool, device=self._device
            ),
            truncated=Unbounded(
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

    def _create_rod_geometry(self):
        """Create a straight rod along the x-axis."""
        physics = self.config.physics
        num_nodes = self._num_nodes
        segment_length = physics.snake_length / self._num_segments

        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            nodes[i, 0] = i * segment_length

        edges = np.array([[i, i + 1] for i in range(self._num_segments)], dtype=np.int64)
        face_nodes = np.empty((0, 3), dtype=np.int64)

        if _HAS_DISMECH:
            return Geometry(nodes, edges, face_nodes, plot_from_txt=False)
        return nodes, edges, face_nodes

    def _init_mock_physics(self) -> None:
        """Initialize mock physics backend (fallback when DisMech unavailable)."""
        physics = self.config.physics
        segment_length = physics.snake_length / self._num_segments
        self._mock_state = _MockRodState(
            num_nodes=self._num_nodes,
            segment_length=segment_length,
            dt=physics.dt,
        )

    # === State accessors ===

    def _get_positions(self) -> np.ndarray:
        """Get node positions, shape (num_nodes, 3)."""
        if self._use_dismech:
            q = self._dismech_robot.state.q
            return q[: 3 * self._num_nodes].reshape(self._num_nodes, 3).copy()
        return self._mock_state.positions

    def _get_velocities(self) -> np.ndarray:
        """Get node velocities, shape (num_nodes, 3)."""
        if self._use_dismech:
            u = self._dismech_robot.state.u
            return u[: 3 * self._num_nodes].reshape(self._num_nodes, 3).copy()
        return self._mock_state.velocities

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

    # === Apply curvature control ===

    def _apply_curvature(self, curvature_state: np.ndarray) -> None:
        """Write curvature state to physics backend.

        Args:
            curvature_state: Shape (num_bend_springs, 2) for kappa1, kappa2.
        """
        if self._use_dismech:
            bend_springs = self._dismech_robot.bend_springs
            if bend_springs.N > 0 and hasattr(bend_springs, "nat_strain"):
                n = min(len(curvature_state), bend_springs.N)
                for i in range(n):
                    bend_springs.nat_strain[i, 0] = curvature_state[i, 0]
                    bend_springs.nat_strain[i, 1] = curvature_state[i, 1]
        else:
            # Mock: apply curvature via simplified dynamics
            self._mock_state.apply_curvature_and_step(curvature_state)

    # === EnvBase interface ===

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._target = TargetGenerator(self.config.target, self._rng)
            self._obstacles = ObstacleManager(self.config.obstacles, self._rng)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Initialize physics backend
        if self._use_dismech:
            self._init_dismech()
        else:
            self._init_mock_physics()

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

        # Apply delta curvature control (once per RL step)
        curvature_state = self.controller.apply_delta(action, two_d_sim=self._two_d_sim)

        # Step physics for control_period substeps (temporal smoothing)
        num_substeps = getattr(self.config, 'control_period', 1)
        for substep_i in range(num_substeps):
            if substep_i == 0 or self._use_dismech:
                self._apply_curvature(curvature_state)
            if self._use_dismech:
                try:
                    self._dismech_robot, _ = self._time_stepper.step(
                        self._dismech_robot, debug=False
                    )
                except ValueError:
                    pass  # Convergence issue - state unchanged

        # Move target (for follow_target task) — total time for all substeps
        if self.config.task == TaskType.FOLLOW_TARGET:
            self._target.step(self.config.physics.dt * num_substeps)

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
        self._mock_state = None
        super().close()
