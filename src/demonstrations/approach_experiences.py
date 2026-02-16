"""Approach experience generation for behavioral cloning.

This module provides classes for generating and storing successful locomotion
experiences using grid search over serpenoid parameters. Experiences are used
to pretrain the approach worker policy via behavioral cloning (supervised learning).

The key insight is "retrospective goal setting": since grid search has no predefined
target, we run the simulation, compute the actual displacement, and set the goal
direction as the normalized displacement direction. This creates valid (state, action,
goal) tuples where the action actually achieves the goal.

State Representation: REDUCED_APPROACH (13-dim)
    [0:3]   - Curvature modes (amplitude, wave_number, phase from FFT)
    [3:6]   - Orientation (unit vector: snake's facing direction)
    [6:9]   - Angular velocity (how fast the snake is turning)
    [9:12]  - Goal direction (unit vector: world-frame direction to goal)
    [12]    - Goal distance (scalar: distance to goal)

Action Representation: SERPENOID (4-dim)
    [0] - amplitude
    [1] - frequency
    [2] - wave_number
    [3] - phase

Example:
    >>> from demonstrations.approach_experiences import (
    ...     ApproachExperienceBuffer,
    ...     ApproachExperienceGenerator,
    ... )
    >>> from configs.env import PhysicsConfig
    >>>
    >>> # Generate experiences via grid search
    >>> generator = ApproachExperienceGenerator(PhysicsConfig())
    >>> buffer = generator.generate_from_grid(
    ...     amplitude_values=[0.5, 1.0, 1.5],
    ...     frequency_values=[0.5, 1.0, 1.5],
    ...     wave_number_values=[1.0, 2.0],
    ...     phase_values=[0.0, 1.57],
    ...     duration=5.0,
    ...     min_displacement=0.05,
    ... )
    >>>
    >>> # Save experiences
    >>> buffer.save("data/approach_experiences.npz")
    >>>
    >>> # Load and use for training
    >>> buffer.load("data/approach_experiences.npz")
    >>> states, actions = buffer.to_dataset()
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.env import PhysicsConfig
from demonstrations.generators import SerpenoidGenerator
from demonstrations.fitness import (
    compute_displacement_vector,
    filter_successful_trajectories,
    compute_direction_coverage,
)
from observations.curvature_modes import CurvatureModeExtractor
from observations.virtual_chassis import VirtualChassisExtractor


class ApproachExperienceBuffer:
    """Buffer for storing approach experiences for behavioral cloning.

    Stores (state, action) tuples where:
    - state: 13-dim REDUCED_APPROACH representation (goal embedded in dims 9-13)
    - action: 4-dim serpenoid parameters

    The buffer supports saving/loading from .npz files and conversion to
    PyTorch-compatible numpy arrays for training.
    """

    def __init__(self, state_dim: int = 13, action_dim: int = 4):
        """Initialize the experience buffer.

        Args:
            state_dim: Dimension of state vectors (default: 13 for REDUCED_APPROACH)
            action_dim: Dimension of action vectors (default: 4 for serpenoid params)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.metadata: Dict[str, Any] = {
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

    def add_batch(
        self, states: np.ndarray, actions: np.ndarray
    ) -> None:
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

    def save(self, path: str) -> None:
        """Save the experience buffer to a .npz file.

        Args:
            path: Path to save file (should end in .npz)
        """
        if not self.states:
            raise ValueError("Cannot save empty buffer")

        states_array = np.stack(self.states)
        actions_array = np.stack(self.actions)

        np.savez(
            path,
            states=states_array,
            actions=actions_array,
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
        self.action_dim = metadata.get("action_dim", 4)
        self.metadata = metadata

        self.states = [states_array[i] for i in range(len(states_array))]
        self.actions = [actions_array[i] for i in range(len(actions_array))]

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

    def __len__(self) -> int:
        """Return number of experiences in buffer."""
        return len(self.states)


class ApproachExperienceGenerator:
    """Generate approach experiences via grid search over serpenoid parameters.

    Uses SerpenoidGenerator to run physics simulations with different parameters,
    filters for successful locomotion (by displacement threshold), and extracts
    REDUCED_APPROACH state features with retrospective goal labeling.

    The "retrospective goal" approach works as follows:
    1. Run simulation with serpenoid parameters (no predefined target)
    2. Compute actual displacement vector (origin -> final position)
    3. Set goal direction = normalized displacement direction
    4. Set goal distance = displacement magnitude
    5. This creates valid (state, action, goal) tuples
    """

    def __init__(self, physics_config: PhysicsConfig):
        """Initialize the experience generator.

        Args:
            physics_config: Configuration for physics simulation
        """
        self.physics_config = physics_config
        self.generator = SerpenoidGenerator(physics_config)

        # Feature extractors (without CoG position)
        self.curvature_extractor = CurvatureModeExtractor(normalize=True)
        self.chassis_extractor = VirtualChassisExtractor(normalize=True)

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

    def generate_from_grid(
        self,
        amplitude_values: List[float],
        wave_number_values: List[float],
        frequency_values: List[float],
        phase_values: Optional[List[float]] = None,
        duration: float = 5.0,
        min_displacement: float = 0.05,
        ensure_direction_diversity: bool = True,
        num_direction_bins: int = 8,
        top_k_per_bin: int = 3,
        sample_interval: int = 5,
        verbose: bool = False,
    ) -> ApproachExperienceBuffer:
        """Generate approach experiences via grid search.

        Args:
            amplitude_values: List of amplitude values to try
            wave_number_values: List of wave number values to try
            frequency_values: List of frequency values to try
            phase_values: List of phase values to try (default: [0.0])
            duration: Duration for each trajectory in seconds
            min_displacement: Minimum displacement threshold (meters)
            ensure_direction_diversity: If True, keep best per direction bin
            num_direction_bins: Number of direction bins for diversity
            top_k_per_bin: Number of best trajectories to keep per bin
            sample_interval: Sample every N states from trajectory
            verbose: If True, print progress information

        Returns:
            ApproachExperienceBuffer with extracted experiences
        """
        if phase_values is None:
            phase_values = [0.0]

        # Generate trajectories on grid
        if verbose:
            total = (
                len(amplitude_values)
                * len(wave_number_values)
                * len(frequency_values)
                * len(phase_values)
            )
            print(f"Generating {total} trajectories on parameter grid...")

        trajectories, parameters = self.generator.get_parameter_grid(
            amplitude_values=amplitude_values,
            wave_number_values=wave_number_values,
            frequency_values=frequency_values,
            phase_values=phase_values,
            duration=duration,
            verbose=verbose,
        )

        # Filter successful trajectories
        if verbose:
            print(f"\nFiltering by displacement threshold ({min_displacement}m)...")

        filtered_trajectories, filtered_params, fitness_info = filter_successful_trajectories(
            trajectories,
            parameters,
            min_displacement=min_displacement,
            ensure_direction_diversity=ensure_direction_diversity,
            num_direction_bins=num_direction_bins,
            top_k_per_bin=top_k_per_bin,
        )

        if verbose:
            print(f"  Successful trajectories: {len(filtered_trajectories)}/{len(trajectories)}")
            if fitness_info:
                coverage = compute_direction_coverage(fitness_info, num_direction_bins)
                print(f"  Direction coverage: {coverage['bins_covered']}/{coverage['total_bins']} bins")

        # Extract experiences
        buffer = ApproachExperienceBuffer()

        for traj_idx, (trajectory, params) in enumerate(
            zip(filtered_trajectories, filtered_params)
        ):
            # Compute retrospective goal
            retrospective_goal = self.compute_retrospective_goal(trajectory)

            # Create action vector from serpenoid parameters
            action = np.array(
                [
                    params["amplitude"],
                    params["frequency"],
                    params["wave_number"],
                    params["phase"],
                ],
                dtype=np.float32,
            )

            # Sample states from trajectory
            for state_idx in range(0, len(trajectory), sample_interval):
                state_dict = trajectory[state_idx]
                state = self.extract_reduced_state(state_dict, retrospective_goal)
                buffer.add(state, action)

        # Update metadata
        buffer.metadata["num_trajectories"] = len(filtered_trajectories)
        buffer.metadata["grid_params"] = {
            "amplitude_values": amplitude_values,
            "wave_number_values": wave_number_values,
            "frequency_values": frequency_values,
            "phase_values": phase_values,
            "duration": duration,
            "min_displacement": min_displacement,
            "ensure_direction_diversity": ensure_direction_diversity,
            "sample_interval": sample_interval,
        }

        if verbose:
            print(f"\nExtracted {len(buffer)} experiences from {len(filtered_trajectories)} trajectories")

        return buffer
