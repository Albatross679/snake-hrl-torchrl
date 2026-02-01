"""Curvature action experience generation for direct control RL.

This module provides classes for generating experiences where:
- State: REDUCED_APPROACH (13-dim)
- Action: 19 joint curvatures (computed from serpenoid steering)

The key idea is to use DirectSerpenoidSteeringTransform (5 params) to generate diverse
locomotion behaviors, but record the 19-dim curvatures as the action. This allows
training an RL policy that outputs joint curvatures directly.

The generator uses "retrospective goal setting": since grid search has no predefined
target, we run the simulation, compute the actual displacement, and set the goal
direction as the normalized displacement direction.

State Representation: REDUCED_APPROACH (13-dim)
    [0:3]   - Curvature modes (amplitude, wave_number, phase from FFT)
    [3:6]   - Orientation (unit vector: snake's facing direction)
    [6:9]   - Angular velocity (how fast the snake is turning)
    [9:12]  - Goal direction (unit vector: world-frame direction to goal)
    [12]    - Goal distance (scalar: distance to goal)

Action Representation: CURVATURES (19-dim)
    [0:19] - Joint curvatures computed via: kappa(s,t) = A*sin(k*s - w*t + phi) + kappa_turn

Example:
    >>> from snake_hrl.demonstrations.curvature_action_experiences import (
    ...     CurvatureActionExperienceBuffer,
    ...     CurvatureActionExperienceGenerator,
    ... )
    >>> from snake_hrl.configs.env import PhysicsConfig
    >>>
    >>> # Generate experiences via grid search
    >>> generator = CurvatureActionExperienceGenerator(PhysicsConfig())
    >>> buffer, traj_info = generator.generate_from_grid(
    ...     amplitude_values=[0.5, 1.0, 1.5],
    ...     frequency_values=[0.5, 1.0, 1.5],
    ...     wave_number_values=[1.0, 2.0],
    ...     phase_values=[0.0, 3.14],
    ...     turn_bias_values=[-1.0, 0.0, 1.0],
    ...     duration=3.0,
    ...     sample_interval=0.1,
    ... )
    >>>
    >>> # Save experiences
    >>> buffer.save("data/demos/curvature_experiences.npz")
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from snake_hrl.configs.env import PhysicsConfig
from snake_hrl.cpg.action_wrapper import DirectSerpenoidSteeringTransform
from snake_hrl.demonstrations.fitness import compute_displacement_vector
from snake_hrl.features.curvature_modes import CurvatureModeExtractor
from snake_hrl.features.virtual_chassis import VirtualChassisExtractor
from snake_hrl.physics.snake_robot import SnakeRobot


def _create_snake_robot_with_direction(
    config: PhysicsConfig,
    initial_direction: np.ndarray,
    initial_position: Optional[np.ndarray] = None,
) -> SnakeRobot:
    """Create a SnakeRobot with custom initial direction.

    This is a factory function that creates a SnakeRobot with the snake
    oriented in a specific direction from the start.

    Args:
        config: Physics configuration
        initial_direction: Direction unit vector (3D)
        initial_position: Optional starting position

    Returns:
        SnakeRobot instance
    """
    from dismech import (
        SoftRobot,
        Geometry,
        GeomParams,
        Material,
        SimParams,
        Environment,
        ImplicitEulerTimeStepper,
    )
    from snake_hrl.physics.snake_robot import SnakeGeometryAdapter
    from snake_hrl.physics.geometry import create_prey_geometry

    if initial_position is None:
        initial_position = np.array([0.0, 0.0, 0.0])

    # Normalize direction
    direction = initial_direction / np.linalg.norm(initial_direction)

    # Create node positions with the specified direction
    num_nodes = config.num_segments + 1
    segment_length = config.snake_length / config.num_segments

    nodes = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        nodes[i] = initial_position + i * segment_length * direction

    # Create edges (connectivity)
    edges = np.array([[i, i + 1] for i in range(config.num_segments)], dtype=np.int64)

    # No faces (rod only, no shell)
    face_nodes = np.empty((0, 3), dtype=np.int64)

    # Create Geometry
    geometry = Geometry(nodes, edges, face_nodes, plot_from_txt=False)

    # Geometry parameters
    geom_params = GeomParams(
        rod_r0=config.snake_radius,
        shell_h=0,
    )

    # Material parameters
    material = Material(
        density=config.density,
        youngs_rod=config.youngs_modulus,
        youngs_shell=0,
        poisson_rod=config.poisson_ratio,
        poisson_shell=0,
    )

    # Simulation parameters
    sim_params = SimParams(
        static_sim=False,
        two_d_sim=True,
        use_mid_edge=False,
        use_line_search=True,
        log_data=False,
        log_step=1,
        show_floor=False,
        dt=config.dt,
        max_iter=config.max_iter,
        total_time=1000.0,
        plot_step=1,
        tol=config.tol,
        ftol=config.ftol,
        dtol=config.dtol,
    )

    # Environment with forces
    env = Environment()

    if config.enable_gravity:
        env.add_force('gravity', g=np.array(config.gravity))

    if config.use_rft:
        env.add_force('rft', ct=config.rft_ct, cn=config.rft_cn)

    # Create DisMech SoftRobot
    dismech_robot = SoftRobot(geom_params, material, geometry, sim_params, env)

    # Create time stepper
    time_stepper = ImplicitEulerTimeStepper(dismech_robot)

    # Create SnakeRobot instance and inject DisMech components
    sim = SnakeRobot.__new__(SnakeRobot)
    sim.config = config
    sim._initial_snake_position = initial_position.copy()
    sim._initial_prey_position = None
    sim.prey = create_prey_geometry(config, None)
    sim._dismech_robot = dismech_robot
    sim._time_stepper = time_stepper
    sim.target_curvatures = np.zeros(config.num_segments - 1)
    sim.time = 0.0
    sim._contact_mask = np.zeros(config.num_segments + 1, dtype=bool)
    sim._contact_forces = np.zeros((config.num_segments + 1, 3))
    sim._prev_positions = None
    sim.velocities = np.zeros((config.num_segments + 1, 3))

    # Create snake geometry adapter
    positions = dismech_robot.state.q[:3 * num_nodes].reshape(num_nodes, 3)
    sim.snake = SnakeGeometryAdapter(positions, config)

    return sim


class CurvatureActionExperienceBuffer:
    """Buffer for storing curvature action experiences for behavioral cloning.

    Stores (state, action) tuples where:
    - state: 13-dim REDUCED_APPROACH representation (goal embedded in dims 9-13)
    - action: 19-dim joint curvatures

    The buffer supports saving/loading from .npz files and conversion to
    PyTorch-compatible numpy arrays for training.
    """

    def __init__(self, state_dim: int = 13, action_dim: int = 19):
        """Initialize the experience buffer.

        Args:
            state_dim: Dimension of state vectors (default: 13 for REDUCED_APPROACH)
            action_dim: Dimension of action vectors (default: 19 for joint curvatures)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.trajectory_info: Dict[str, Any] = {
            "start_positions": [],
            "end_positions": [],
            "displacements": [],
            "directions": [],
            "l2_norms": [],
            "initial_orientations": [],
        }
        self.metadata: Dict[str, Any] = {
            "action_type": "curvatures",
            "state_type": "reduced_approach",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "state_components": [
                "curvature_modes (3)",
                "orientation (3)",
                "angular_velocity (3)",
                "goal_direction (3)",
                "goal_distance (1)",
            ],
            "num_trajectories": 0,
            "num_samples": 0,
        }

    def add(self, state: np.ndarray, action: np.ndarray) -> None:
        """Add a single (state, action) experience.

        Args:
            state: State vector of shape (state_dim,)
            action: Action vector of shape (action_dim,)
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"State shape {state.shape} doesn't match expected ({self.state_dim},)"
            )
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Action shape {action.shape} doesn't match expected ({self.action_dim},)"
            )

        self.states.append(state)
        self.actions.append(action)
        self.metadata["num_samples"] = len(self.states)

    def add_batch(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Add a batch of experiences.

        Args:
            states: State array of shape (N, state_dim)
            actions: Action array of shape (N, action_dim)
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                f"States shape {states.shape} doesn't match expected (N, {self.state_dim})"
            )
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"Actions shape {actions.shape} doesn't match expected (N, {self.action_dim})"
            )

        for state, action in zip(states, actions):
            self.states.append(state)
            self.actions.append(action)

        self.metadata["num_samples"] = len(self.states)

    def add_trajectory_info(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        initial_orientation: float,
    ) -> None:
        """Add trajectory metadata for distribution analysis.

        Args:
            start_pos: XY start position
            end_pos: XY end position
            initial_orientation: Initial snake direction in radians
        """
        start_pos = np.asarray(start_pos[:2], dtype=np.float32)
        end_pos = np.asarray(end_pos[:2], dtype=np.float32)
        displacement = end_pos - start_pos
        l2_norm = float(np.linalg.norm(displacement))
        direction = float(np.arctan2(displacement[1], displacement[0]))

        self.trajectory_info["start_positions"].append(start_pos)
        self.trajectory_info["end_positions"].append(end_pos)
        self.trajectory_info["displacements"].append(displacement)
        self.trajectory_info["directions"].append(direction)
        self.trajectory_info["l2_norms"].append(l2_norm)
        self.trajectory_info["initial_orientations"].append(initial_orientation)

    def save(self, path: str) -> None:
        """Save the experience buffer to a .npz file.

        Args:
            path: Path to save file (should end in .npz)
        """
        if not self.states:
            raise ValueError("Cannot save empty buffer")

        states_array = np.stack(self.states)
        actions_array = np.stack(self.actions)

        # Convert trajectory info lists to arrays
        traj_info_arrays = {
            "start_positions": np.array(self.trajectory_info["start_positions"]),
            "end_positions": np.array(self.trajectory_info["end_positions"]),
            "displacements": np.array(self.trajectory_info["displacements"]),
            "directions": np.array(self.trajectory_info["directions"]),
            "l2_norms": np.array(self.trajectory_info["l2_norms"]),
            "initial_orientations": np.array(self.trajectory_info["initial_orientations"]),
        }

        np.savez(
            path,
            states=states_array,
            actions=actions_array,
            trajectory_info=traj_info_arrays,
            metadata=self.metadata,
        )

    def load(self, path: str) -> None:
        """Load experiences from a .npz file.

        Args:
            path: Path to load file
        """
        data = np.load(path, allow_pickle=True)

        states_array = data["states"]
        actions_array = data["actions"]
        metadata = data["metadata"].item()

        self.state_dim = metadata.get("state_dim", 13)
        self.action_dim = metadata.get("action_dim", 19)
        self.metadata = metadata

        self.states = [states_array[i] for i in range(len(states_array))]
        self.actions = [actions_array[i] for i in range(len(actions_array))]

        # Load trajectory info if available
        if "trajectory_info" in data:
            traj_info = data["trajectory_info"].item()
            self.trajectory_info = {
                "start_positions": list(traj_info.get("start_positions", [])),
                "end_positions": list(traj_info.get("end_positions", [])),
                "displacements": list(traj_info.get("displacements", [])),
                "directions": list(traj_info.get("directions", [])),
                "l2_norms": list(traj_info.get("l2_norms", [])),
                "initial_orientations": list(traj_info.get("initial_orientations", [])),
            }

    def to_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert buffer to numpy arrays for training.

        Returns:
            Tuple of (states, actions) as numpy arrays:
            - states: shape (N, state_dim)
            - actions: shape (N, action_dim)
        """
        if not self.states:
            return np.zeros((0, self.state_dim)), np.zeros((0, self.action_dim))

        states = np.stack(self.states)
        actions = np.stack(self.actions)
        return states, actions

    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """Get statistics about trajectory distributions.

        Returns:
            Dictionary with direction and L2 norm statistics
        """
        if not self.trajectory_info["l2_norms"]:
            return {}

        l2_norms = np.array(self.trajectory_info["l2_norms"])
        directions = np.array(self.trajectory_info["directions"])

        return {
            "num_trajectories": len(l2_norms),
            "l2_norm": {
                "min": float(np.min(l2_norms)),
                "max": float(np.max(l2_norms)),
                "mean": float(np.mean(l2_norms)),
                "std": float(np.std(l2_norms)),
            },
            "direction": {
                "circular_mean": float(np.arctan2(
                    np.mean(np.sin(directions)),
                    np.mean(np.cos(directions))
                )),
                "circular_std": float(np.sqrt(-2 * np.log(
                    np.sqrt(np.mean(np.cos(directions))**2 + np.mean(np.sin(directions))**2)
                ))),
            },
        }

    def __len__(self) -> int:
        """Return number of experiences in buffer."""
        return len(self.states)


class CurvatureActionExperienceGenerator:
    """Generate curvature action experiences via grid search over steering parameters.

    Uses DirectSerpenoidSteeringTransform to generate 19-dim curvatures from 5-dim
    steering parameters (amplitude, frequency, wave_number, phase, turn_bias).

    The generator:
    1. Randomizes initial snake direction for each trajectory
    2. Runs physics simulation with steering parameters
    3. Records 19-dim curvatures at each timestep as the action
    4. Extracts 13-dim REDUCED_APPROACH state with retrospective goal labeling
    5. Tracks trajectory metadata for distribution analysis
    """

    def __init__(
        self,
        physics_config: PhysicsConfig,
        num_joints: int = 19,
    ):
        """Initialize the experience generator.

        Args:
            physics_config: Configuration for physics simulation
            num_joints: Number of joints for curvature output (default: 19)
        """
        self.physics_config = physics_config
        self.num_joints = num_joints

        # Create serpenoid steering transform
        self.transform = DirectSerpenoidSteeringTransform(num_joints=num_joints)

        # Feature extractors (without CoG position)
        self.curvature_extractor = CurvatureModeExtractor(normalize=True)
        self.chassis_extractor = VirtualChassisExtractor(normalize=True)

    def _create_initial_direction(self, angle: float) -> np.ndarray:
        """Create initial direction vector from angle.

        Args:
            angle: Direction angle in radians (0 = +X, pi/2 = +Y)

        Returns:
            3D direction vector
        """
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    def compute_retrospective_goal(
        self, trajectory: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Compute goal from actual trajectory outcome.

        Args:
            trajectory: List of state dictionaries from simulation

        Returns:
            Goal vector [direction_x, direction_y, direction_z, distance]
        """
        displacement = compute_displacement_vector(trajectory)
        distance = float(np.linalg.norm(displacement))

        if distance > 1e-8:
            direction = displacement / distance
        else:
            direction = np.array([1.0, 0.0, 0.0])

        return np.array(
            [direction[0], direction[1], direction[2], distance],
            dtype=np.float32,
        )

    def extract_reduced_state(
        self,
        state: Dict[str, Any],
        retrospective_goal: np.ndarray,
    ) -> np.ndarray:
        """Extract 13-dim REDUCED_APPROACH state with retrospective goal.

        Components:
        - Curvature modes (3): from FFT of curvature profile
        - Orientation (3): snake facing direction (from VirtualChassisExtractor)
        - Angular velocity (3): from VirtualChassisExtractor
        - Goal direction (3): from retrospective_goal
        - Goal distance (1): from retrospective_goal

        Args:
            state: State dictionary from simulation
            retrospective_goal: Goal vector [dir_x, dir_y, dir_z, distance]

        Returns:
            13-dim state vector
        """
        # Extract curvature modes
        curvature_modes = self.curvature_extractor.extract(state)  # 3-dim

        # Extract virtual chassis (9-dim), then take only orientation and angular vel
        chassis = self.chassis_extractor.extract(state)
        orientation = chassis[3:6]  # 3-dim (skip CoG position at 0:3)
        angular_vel = chassis[6:9]  # 3-dim

        # Use retrospective goal
        goal_direction = retrospective_goal[:3]  # 3-dim
        goal_distance = retrospective_goal[3:4]  # 1-dim

        return np.concatenate(
            [
                curvature_modes,  # [0:3]
                orientation,  # [3:6]
                angular_vel,  # [6:9]
                goal_direction,  # [9:12]
                goal_distance,  # [12:13]
            ]
        ).astype(np.float32)

    def _normalize_params(
        self,
        amplitude: float,
        frequency: float,
        wave_number: float,
        phase: float,
        turn_bias: float,
    ) -> np.ndarray:
        """Normalize physical parameters to [-1, 1] range for transform.

        Args:
            amplitude: Physical amplitude
            frequency: Physical frequency
            wave_number: Physical wave number
            phase: Physical phase (radians)
            turn_bias: Physical turn bias

        Returns:
            Normalized action array in [-1, 1]
        """
        # Get ranges from transform
        amp_range = self.transform.amplitude_range
        freq_range = self.transform.frequency_range
        turn_range = self.transform.turn_bias_range

        # Normalize amplitude: [amp_min, amp_max] -> [-1, 1]
        amp_norm = 2 * (amplitude - amp_range[0]) / (amp_range[1] - amp_range[0]) - 1

        # Normalize frequency: [freq_min, freq_max] -> [-1, 1]
        freq_norm = 2 * (frequency - freq_range[0]) / (freq_range[1] - freq_range[0]) - 1

        # Normalize wave_number: [0.5, 3.5] -> [-1, 1]
        wn_norm = 2 * (wave_number - 0.5) / 3.0 - 1

        # Normalize phase: [0, 2*pi] -> [-1, 1]
        phase_norm = phase / np.pi - 1

        # Normalize turn_bias: [turn_min, turn_max] -> [-1, 1]
        turn_norm = 2 * (turn_bias - turn_range[0]) / (turn_range[1] - turn_range[0]) - 1

        # Clip to ensure bounds
        return np.clip([amp_norm, freq_norm, wn_norm, phase_norm, turn_norm], -1.0, 1.0)

    def _run_trajectory(
        self,
        amplitude: float,
        frequency: float,
        wave_number: float,
        phase: float,
        turn_bias: float,
        duration: float,
        sample_interval: float,
        initial_direction: float,
    ) -> Tuple[List[Dict[str, Any]], List[np.ndarray], np.ndarray, np.ndarray]:
        """Run a single trajectory with the given parameters.

        Args:
            amplitude: Serpenoid amplitude
            frequency: Serpenoid frequency
            wave_number: Serpenoid wave number
            phase: Serpenoid phase
            turn_bias: Steering turn bias
            duration: Trajectory duration in seconds
            sample_interval: Time between recorded samples
            initial_direction: Initial snake direction in radians

        Returns:
            Tuple of:
            - trajectory: List of state dictionaries
            - curvatures_list: List of curvature arrays at each sample
            - start_pos: Start position (3D)
            - end_pos: End position (3D)
        """
        # Reset transform
        self.transform.reset()

        # Create simulation with rotated initial direction
        direction_vec = self._create_initial_direction(initial_direction)

        # Create a new SnakeRobot instance for this trajectory
        sim = _create_snake_robot_with_direction(self.physics_config, direction_vec)

        trajectory = []
        curvatures_list = []
        dt = self.physics_config.dt
        t = 0.0
        next_sample_time = 0.0

        # Normalize parameters to [-1, 1] for the transform
        norm_action = self._normalize_params(
            amplitude, frequency, wave_number, phase, turn_bias
        )

        # Record initial state and position
        initial_state = sim.get_state()
        trajectory.append(initial_state)
        start_pos = initial_state["positions"][0].copy()

        # Initial curvatures
        curvatures = self.transform.step(norm_action, dt=dt)
        curvatures_list.append(curvatures.copy())

        # Run simulation
        while t < duration:
            # Compute curvatures from steering params
            curvatures = self.transform.step(norm_action, dt=dt)

            # Apply curvatures and step physics
            sim.set_curvature_control(curvatures)
            state = sim.step()

            t += dt

            # Record state and curvatures at sample intervals
            if t >= next_sample_time:
                trajectory.append(state)
                curvatures_list.append(curvatures.copy())
                next_sample_time += sample_interval

        # Get end position
        end_state = trajectory[-1]
        end_pos = end_state["positions"][0].copy()

        return trajectory, curvatures_list, start_pos, end_pos

    def generate_from_grid(
        self,
        amplitude_values: List[float],
        frequency_values: List[float],
        wave_number_values: List[float],
        phase_values: List[float],
        turn_bias_values: List[float],
        duration: float = 3.0,
        sample_interval: float = 0.1,
        min_displacement: float = 0.01,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[CurvatureActionExperienceBuffer, Dict[str, Any]]:
        """Generate curvature action experiences via grid search.

        Args:
            amplitude_values: List of amplitude values to try
            frequency_values: List of frequency values to try
            wave_number_values: List of wave number values to try
            phase_values: List of phase values to try
            turn_bias_values: List of turn bias values to try
            duration: Duration for each trajectory in seconds
            sample_interval: Time between recorded samples
            min_displacement: Minimum displacement threshold (meters)
            seed: Random seed for reproducibility
            verbose: If True, print progress information

        Returns:
            Tuple of (buffer, trajectory_info_summary)
        """
        if seed is not None:
            np.random.seed(seed)

        # Calculate total combinations
        total = (
            len(amplitude_values)
            * len(frequency_values)
            * len(wave_number_values)
            * len(phase_values)
            * len(turn_bias_values)
        )

        if verbose:
            print(f"Generating {total} trajectories on parameter grid...")
            print(f"  Amplitudes: {amplitude_values}")
            print(f"  Frequencies: {frequency_values}")
            print(f"  Wave numbers: {wave_number_values}")
            print(f"  Phases: {phase_values}")
            print(f"  Turn biases: {turn_bias_values}")

        buffer = CurvatureActionExperienceBuffer()
        count = 0
        successful = 0

        for amp in amplitude_values:
            for freq in frequency_values:
                for wn in wave_number_values:
                    for phase in phase_values:
                        for turn_bias in turn_bias_values:
                            # Randomize initial direction
                            initial_direction = np.random.uniform(0, 2 * np.pi)

                            try:
                                # Run trajectory
                                trajectory, curvatures_list, start_pos, end_pos = (
                                    self._run_trajectory(
                                        amplitude=amp,
                                        frequency=freq,
                                        wave_number=wn,
                                        phase=phase,
                                        turn_bias=turn_bias,
                                        duration=duration,
                                        sample_interval=sample_interval,
                                        initial_direction=initial_direction,
                                    )
                                )

                                # Check displacement threshold
                                displacement = end_pos[:2] - start_pos[:2]
                                l2_norm = np.linalg.norm(displacement)

                                if l2_norm < min_displacement:
                                    count += 1
                                    if verbose and count % 20 == 0:
                                        print(f"  [{count}/{total}] Skipped (low displacement: {l2_norm:.4f}m)")
                                    continue

                                # Compute retrospective goal
                                retrospective_goal = self.compute_retrospective_goal(trajectory)

                                # Add trajectory info
                                buffer.add_trajectory_info(
                                    start_pos, end_pos, initial_direction
                                )

                                # Extract experiences for each timestep
                                for state_dict, curvatures in zip(trajectory, curvatures_list):
                                    state = self.extract_reduced_state(state_dict, retrospective_goal)
                                    buffer.add(state, curvatures)

                                successful += 1

                            except Exception as e:
                                if verbose:
                                    print(f"  Warning: Trajectory failed ({e})")

                            count += 1
                            if verbose and count % 20 == 0:
                                print(f"  [{count}/{total}] Generated {successful} successful trajectories")

        # Update metadata
        buffer.metadata["num_trajectories"] = successful
        buffer.metadata["sampling_config"] = {
            "amplitude_values": amplitude_values,
            "frequency_values": frequency_values,
            "wave_number_values": wave_number_values,
            "phase_values": phase_values,
            "turn_bias_values": turn_bias_values,
            "duration": duration,
            "sample_interval": sample_interval,
            "min_displacement": min_displacement,
        }

        if verbose:
            print(f"\nGeneration complete:")
            print(f"  Successful trajectories: {successful}/{total}")
            print(f"  Total experiences: {len(buffer)}")
            stats = buffer.get_trajectory_statistics()
            if stats:
                print(f"  L2 norm range: [{stats['l2_norm']['min']:.4f}, {stats['l2_norm']['max']:.4f}] m")
                print(f"  L2 norm mean: {stats['l2_norm']['mean']:.4f} m")

        return buffer, buffer.get_trajectory_statistics()


def analyze_trajectory_distribution(
    buffer: CurvatureActionExperienceBuffer,
    num_direction_bins: int = 8,
) -> Dict[str, Any]:
    """Analyze and summarize trajectory distributions.

    Args:
        buffer: Experience buffer with trajectory info
        num_direction_bins: Number of bins for direction histogram

    Returns:
        Dictionary with distribution analysis
    """
    if not buffer.trajectory_info["directions"]:
        return {}

    directions = np.array(buffer.trajectory_info["directions"])
    l2_norms = np.array(buffer.trajectory_info["l2_norms"])

    # Direction histogram
    bin_edges = np.linspace(-np.pi, np.pi, num_direction_bins + 1)
    direction_hist, _ = np.histogram(directions, bins=bin_edges)

    # Direction bin labels
    bin_labels = []
    for i in range(num_direction_bins):
        angle = (bin_edges[i] + bin_edges[i + 1]) / 2
        deg = np.degrees(angle)
        if -22.5 <= deg < 22.5:
            label = "E"
        elif 22.5 <= deg < 67.5:
            label = "NE"
        elif 67.5 <= deg < 112.5:
            label = "N"
        elif 112.5 <= deg < 157.5:
            label = "NW"
        elif deg >= 157.5 or deg < -157.5:
            label = "W"
        elif -157.5 <= deg < -112.5:
            label = "SW"
        elif -112.5 <= deg < -67.5:
            label = "S"
        else:
            label = "SE"
        bin_labels.append(label)

    return {
        "num_trajectories": len(directions),
        "direction_distribution": {
            "bins": bin_labels,
            "counts": direction_hist.tolist(),
            "percentages": (direction_hist / len(directions) * 100).tolist(),
        },
        "l2_norm_distribution": {
            "min": float(np.min(l2_norms)),
            "max": float(np.max(l2_norms)),
            "mean": float(np.mean(l2_norms)),
            "std": float(np.std(l2_norms)),
            "percentiles": {
                "25": float(np.percentile(l2_norms, 25)),
                "50": float(np.percentile(l2_norms, 50)),
                "75": float(np.percentile(l2_norms, 75)),
            },
        },
    }
