"""Demonstration generators for creating reference locomotion trajectories.

This module provides generators that create demonstration trajectories
using analytical controllers (e.g., serpenoid curves) running through
the physics simulation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.configs.env import PhysicsConfig
from src.physics.snake_robot import SnakeRobot


class SerpenoidGenerator:
    """Generate serpenoid locomotion demonstrations.

    Creates demonstration trajectories by running a serpenoid gait controller
    through the physics simulation. The serpenoid curve is defined by:

        kappa(s, t) = A * sin(k * s - omega * t + phi)

    where:
        - A: amplitude (peak curvature)
        - k: wave number (spatial frequency along body)
        - omega: temporal frequency
        - phi: phase offset

    Example:
        >>> config = PhysicsConfig()
        >>> generator = SerpenoidGenerator(config)
        >>> trajectory = generator.generate_physics_trajectory(
        ...     amplitude=1.0,
        ...     wave_number=2.0,
        ...     frequency=1.0,
        ...     duration=5.0,
        ... )
    """

    def __init__(
        self,
        physics_config: PhysicsConfig,
        initial_snake_position: Optional[np.ndarray] = None,
        initial_prey_position: Optional[np.ndarray] = None,
    ):
        """Initialize serpenoid generator.

        Args:
            physics_config: Configuration for physics simulation
            initial_snake_position: Optional starting position for snake
            initial_prey_position: Optional prey position (for state dict)
        """
        self.physics_config = physics_config
        self.initial_snake_position = initial_snake_position
        self.initial_prey_position = initial_prey_position

        # Create simulation instance
        self.sim = SnakeRobot(
            physics_config,
            initial_snake_position=initial_snake_position,
            initial_prey_position=initial_prey_position,
        )

        # Number of joints (curvature control points)
        self.num_joints = physics_config.num_segments - 1

    def compute_serpenoid_curvatures(
        self,
        t: float,
        amplitude: float,
        wave_number: float,
        frequency: float,
        phase: float = 0.0,
    ) -> np.ndarray:
        """Compute target curvatures from serpenoid parameters.

        Args:
            t: Current time
            amplitude: Peak curvature amplitude
            wave_number: Number of waves along body length
            frequency: Temporal frequency (Hz)
            phase: Phase offset (radians)

        Returns:
            Target curvatures for each joint, shape (num_joints,)
        """
        # Spatial positions along body (normalized to [0, 1])
        s = np.linspace(0, 1, self.num_joints)

        # Angular frequency
        omega = 2 * np.pi * frequency

        # Spatial frequency
        k = 2 * np.pi * wave_number

        # Serpenoid curvature profile
        curvatures = amplitude * np.sin(k * s - omega * t + phase)

        return curvatures

    def generate_physics_trajectory(
        self,
        amplitude: float,
        wave_number: float,
        frequency: float,
        duration: float = 5.0,
        phase: float = 0.0,
        sample_rate: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Generate trajectory by running serpenoid controller through physics.

        Args:
            amplitude: Peak curvature amplitude (rad/m)
            wave_number: Number of waves along body
            frequency: Temporal frequency (Hz)
            duration: Total duration in seconds
            phase: Initial phase offset (radians)
            sample_rate: How often to record states (Hz). If None, records every step.

        Returns:
            List of state dictionaries from simulation
        """
        # Reset simulation
        self.sim.reset(
            snake_position=self.initial_snake_position,
            prey_position=self.initial_prey_position,
        )

        states: List[Dict[str, Any]] = []
        dt = self.physics_config.dt
        t = 0.0

        # Determine sampling interval
        if sample_rate is not None:
            sample_interval = 1.0 / sample_rate
            next_sample_time = 0.0
        else:
            sample_interval = dt
            next_sample_time = 0.0

        # Record initial state
        states.append(self.sim.get_state())

        # Run simulation
        while t < duration:
            # Compute serpenoid curvatures
            curvatures = self.compute_serpenoid_curvatures(
                t, amplitude, wave_number, frequency, phase
            )

            # Apply curvatures and step
            self.sim.set_curvature_control(curvatures)
            state = self.sim.step()

            t += dt

            # Record state at sample rate
            if t >= next_sample_time:
                states.append(state)
                next_sample_time += sample_interval

        return states

    def generate_batch(
        self,
        num_demos: int,
        amplitude_range: Tuple[float, float] = (0.5, 2.0),
        wave_number_range: Tuple[float, float] = (1.0, 3.0),
        frequency_range: Tuple[float, float] = (0.5, 2.0),
        duration: float = 5.0,
        seed: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Generate a batch of demonstration trajectories with varied parameters.

        Args:
            num_demos: Number of demonstrations to generate
            amplitude_range: (min, max) amplitude range
            wave_number_range: (min, max) wave number range
            frequency_range: (min, max) frequency range
            duration: Duration of each trajectory
            seed: Random seed for reproducibility

        Returns:
            List of trajectory lists
        """
        if seed is not None:
            np.random.seed(seed)

        trajectories = []

        for i in range(num_demos):
            # Sample random parameters
            amplitude = np.random.uniform(*amplitude_range)
            wave_number = np.random.uniform(*wave_number_range)
            frequency = np.random.uniform(*frequency_range)
            phase = np.random.uniform(0, 2 * np.pi)

            # Generate trajectory
            trajectory = self.generate_physics_trajectory(
                amplitude=amplitude,
                wave_number=wave_number,
                frequency=frequency,
                duration=duration,
                phase=phase,
            )

            trajectories.append(trajectory)

        return trajectories

    def get_parameter_grid(
        self,
        amplitude_values: List[float],
        wave_number_values: List[float],
        frequency_values: List[float],
        phase_values: Optional[List[float]] = None,
        duration: float = 5.0,
        sample_rate: Optional[float] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, float]]]:
        """Generate demonstrations on a parameter grid.

        Args:
            amplitude_values: List of amplitude values
            wave_number_values: List of wave number values
            frequency_values: List of frequency values
            phase_values: List of phase offset values (default: [0.0])
            duration: Duration for each trajectory
            sample_rate: State recording rate in Hz (default: every step)
            verbose: If True, print progress information

        Returns:
            Tuple of (trajectories list, parameters list)
        """
        # Default phase values
        if phase_values is None:
            phase_values = [0.0]

        trajectories = []
        parameters = []

        # Calculate total combinations for progress reporting
        total = (
            len(amplitude_values)
            * len(wave_number_values)
            * len(frequency_values)
            * len(phase_values)
        )
        count = 0

        for amp in amplitude_values:
            for wn in wave_number_values:
                for freq in frequency_values:
                    for phase in phase_values:
                        traj = self.generate_physics_trajectory(
                            amplitude=amp,
                            wave_number=wn,
                            frequency=freq,
                            duration=duration,
                            phase=phase,
                            sample_rate=sample_rate,
                        )
                        trajectories.append(traj)
                        parameters.append({
                            "amplitude": amp,
                            "wave_number": wn,
                            "frequency": freq,
                            "phase": phase,
                        })

                        count += 1
                        if verbose and count % 50 == 0:
                            print(f"  Generated {count}/{total} trajectories...")

        return trajectories, parameters


class LateralUndulationGenerator(SerpenoidGenerator):
    """Specialized generator for lateral undulation gait.

    Lateral undulation is the most common snake locomotion mode,
    characterized by a traveling wave propagating from head to tail.
    """

    def __init__(
        self,
        physics_config: PhysicsConfig,
        default_amplitude: float = 1.0,
        default_wave_number: float = 2.0,
        default_frequency: float = 1.0,
    ):
        """Initialize lateral undulation generator with default parameters.

        Args:
            physics_config: Physics configuration
            default_amplitude: Default curvature amplitude
            default_wave_number: Default wave number
            default_frequency: Default temporal frequency
        """
        super().__init__(physics_config)
        self.default_amplitude = default_amplitude
        self.default_wave_number = default_wave_number
        self.default_frequency = default_frequency

    def generate_default(self, duration: float = 5.0) -> List[Dict[str, Any]]:
        """Generate trajectory with default lateral undulation parameters.

        Args:
            duration: Trajectory duration

        Returns:
            List of state dictionaries
        """
        return self.generate_physics_trajectory(
            amplitude=self.default_amplitude,
            wave_number=self.default_wave_number,
            frequency=self.default_frequency,
            duration=duration,
        )


class SidewindingGenerator(SerpenoidGenerator):
    """Specialized generator for sidewinding gait.

    Sidewinding combines horizontal and vertical body waves with
    a phase offset, creating a characteristic sideways locomotion.
    """

    def __init__(
        self,
        physics_config: PhysicsConfig,
        horizontal_amplitude: float = 1.0,
        vertical_amplitude: float = 0.5,
        wave_number: float = 1.5,
        frequency: float = 1.0,
        phase_offset: float = np.pi / 2,
    ):
        """Initialize sidewinding generator.

        Args:
            physics_config: Physics configuration
            horizontal_amplitude: Amplitude of horizontal wave
            vertical_amplitude: Amplitude of vertical wave
            wave_number: Wave number for both waves
            frequency: Temporal frequency
            phase_offset: Phase offset between horizontal and vertical waves
        """
        super().__init__(physics_config)
        self.horizontal_amplitude = horizontal_amplitude
        self.vertical_amplitude = vertical_amplitude
        self.wave_number = wave_number
        self.frequency = frequency
        self.phase_offset = phase_offset

    def compute_sidewinding_curvatures(
        self, t: float
    ) -> np.ndarray:
        """Compute sidewinding curvatures (simplified horizontal component).

        Note: Full 3D sidewinding requires additional control dimensions.
        This provides the horizontal wave component.

        Args:
            t: Current time

        Returns:
            Target curvatures for each joint
        """
        # For now, use standard serpenoid (horizontal wave)
        # Full sidewinding would need 3D curvature control
        return self.compute_serpenoid_curvatures(
            t,
            amplitude=self.horizontal_amplitude,
            wave_number=self.wave_number,
            frequency=self.frequency,
            phase=0.0,
        )

    def generate_default(self, duration: float = 5.0) -> List[Dict[str, Any]]:
        """Generate trajectory with sidewinding parameters.

        Args:
            duration: Trajectory duration

        Returns:
            List of state dictionaries
        """
        return self.generate_physics_trajectory(
            amplitude=self.horizontal_amplitude,
            wave_number=self.wave_number,
            frequency=self.frequency,
            duration=duration,
        )
