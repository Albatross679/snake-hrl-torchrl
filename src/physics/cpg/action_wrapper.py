"""CPG action transformation for reducing action space dimensionality.

This module provides wrappers that transform low-dimensional RL actions
(gait parameters) into high-dimensional curvature commands using CPG
oscillators.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from physics.cpg.oscillators import CPGNetwork, HopfOscillator
from configs.env import CPGConfig


class CPGActionTransform:
    """Transform low-dimensional RL actions to joint curvatures via CPG.

    This class acts as a policy wrapper that:
    1. Takes RL actions representing gait parameters (amplitude, frequency, etc.)
    2. Steps internal CPG oscillators
    3. Outputs interpolated curvature commands for all joints

    Action space reduction:
        Without CPG: 19 curvatures (one per joint)
        With CPG: 4-8 parameters (amplitude, frequency, wave_number, phases)

    Example:
        >>> config = CPGConfig(use_cpg=True, num_oscillators=4, action_dim=4)
        >>> transform = CPGActionTransform(config, num_joints=19)
        >>> rl_action = np.array([0.5, 0.3, 0.7, 0.0])  # amplitude, freq, wave_num, phase
        >>> curvatures = transform.step(rl_action, dt=0.01)  # shape: (19,)
    """

    def __init__(
        self,
        config: CPGConfig,
        num_joints: int = 19,
        dt: float = 0.01,
    ):
        """Initialize CPG action transform.

        Args:
            config: CPG configuration
            num_joints: Number of output joints (curvature control points)
            dt: Default timestep for CPG updates
        """
        self.config = config
        self.num_joints = num_joints
        self.dt = dt

        # Create CPG network
        self.cpg = CPGNetwork(
            num_oscillators=config.num_oscillators,
            oscillator_type=config.oscillator_type,
            coupling_strength=config.coupling_strength,
            base_frequency=1.0,
        )

        # Action dimension and interpretation
        self.action_dim = config.gait_action_dim

        # Parameter ranges
        self.amplitude_range = config.amplitude_range
        self.frequency_range = config.frequency_range

        # Current gait parameters (normalized to [0, 1] or [-1, 1])
        self._amplitude = 1.0
        self._frequency = 1.0
        self._wave_number = 2.0
        self._phase_offset = 0.0

    def reset(self) -> None:
        """Reset CPG state."""
        self.cpg.reset(initial_phase=self._phase_offset)

    def denormalize_action(self, action: np.ndarray) -> Dict[str, float]:
        """Convert normalized action to gait parameters.

        Args:
            action: Normalized action vector from RL policy, shape (action_dim,)
                   Values expected in [-1, 1] range

        Returns:
            Dictionary of gait parameters
        """
        action = np.clip(action, -1.0, 1.0)

        # Map action components to gait parameters
        # Default mapping assumes action_dim=4: [amplitude, frequency, wave_number, phase]

        if self.action_dim >= 1:
            # Amplitude: map [-1, 1] to amplitude_range
            amp_norm = (action[0] + 1) / 2  # [0, 1]
            amplitude = (
                self.amplitude_range[0] +
                amp_norm * (self.amplitude_range[1] - self.amplitude_range[0])
            )
        else:
            amplitude = 1.0

        if self.action_dim >= 2:
            # Frequency: map [-1, 1] to frequency_range
            freq_norm = (action[1] + 1) / 2
            frequency = (
                self.frequency_range[0] +
                freq_norm * (self.frequency_range[1] - self.frequency_range[0])
            )
        else:
            frequency = 1.0

        if self.action_dim >= 3:
            # Wave number: map [-1, 1] to [0.5, 3.0]
            wn_norm = (action[2] + 1) / 2
            wave_number = 0.5 + wn_norm * 2.5
        else:
            wave_number = 2.0

        if self.action_dim >= 4:
            # Phase offset: map [-1, 1] to [0, 2*pi]
            phase_offset = (action[3] + 1) * np.pi
        else:
            phase_offset = 0.0

        return {
            "amplitude": amplitude,
            "frequency": frequency,
            "wave_number": wave_number,
            "phase_offset": phase_offset,
        }

    def step(
        self,
        action: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Transform RL action to joint curvatures via CPG.

        Args:
            action: Normalized RL action, shape (action_dim,)
            dt: Timestep (uses default if None)

        Returns:
            Target curvatures for each joint, shape (num_joints,)
        """
        dt = dt or self.dt

        # Denormalize action to gait parameters
        params = self.denormalize_action(action)
        self._amplitude = params["amplitude"]
        self._frequency = params["frequency"]
        self._wave_number = params["wave_number"]
        self._phase_offset = params["phase_offset"]

        # Update CPG parameters
        self.cpg.set_parameters(
            amplitude=self._amplitude,
            frequency=self._frequency,
            wave_number=self._wave_number,
        )

        # Step CPG
        oscillator_outputs = self.cpg.step(dt, amplitude=self._amplitude)

        # Interpolate to joint curvatures
        curvatures = self.cpg.get_joint_curvatures(
            oscillator_outputs,
            self.num_joints,
            amplitude=1.0,  # Already applied in step()
        )

        return curvatures

    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get action space bounds.

        Returns:
            Tuple of (low, high) arrays for action bounds
        """
        low = np.full(self.action_dim, -1.0)
        high = np.full(self.action_dim, 1.0)
        return low, high

    @property
    def current_parameters(self) -> Dict[str, float]:
        """Get current gait parameters."""
        return {
            "amplitude": self._amplitude,
            "frequency": self._frequency,
            "wave_number": self._wave_number,
            "phase_offset": self._phase_offset,
        }


class CPGEnvWrapper:
    """Environment wrapper that applies CPG action transformation.

    This wrapper modifies the environment's action space and automatically
    transforms CPG parameters to joint curvatures.

    Note: This is a simple wrapper class. For TorchRL integration, use
    the TransformedEnv with a custom Transform.
    """

    def __init__(
        self,
        env,
        cpg_config: CPGConfig,
    ):
        """Initialize CPG environment wrapper.

        Args:
            env: Base environment (must have action_spec with shape (num_joints,))
            cpg_config: CPG configuration
        """
        self.env = env
        self.cpg_config = cpg_config

        # Get number of joints from environment
        original_action_dim = env.action_spec.shape[-1]
        self.num_joints = original_action_dim

        # Create CPG transform
        self.cpg_transform = CPGActionTransform(
            cpg_config,
            num_joints=self.num_joints,
            dt=env.config.physics.dt,
        )

        # New action dimension
        self.action_dim = cpg_config.action_dim

    def reset(self, **kwargs):
        """Reset environment and CPG."""
        self.cpg_transform.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step with CPG-transformed action.

        Args:
            action: Low-dimensional CPG action, shape (action_dim,)

        Returns:
            Environment step result
        """
        # Transform to curvatures
        curvatures = self.cpg_transform.step(
            action if isinstance(action, np.ndarray) else action.cpu().numpy()
        )

        # Convert to tensor if needed
        if isinstance(action, torch.Tensor):
            curvatures = torch.tensor(curvatures, dtype=action.dtype, device=action.device)

        # Step base environment
        return self.env.step(curvatures)

    def __getattr__(self, name):
        """Forward attribute access to base environment."""
        return getattr(self.env, name)


class DirectSerpenoidTransform:
    """Direct serpenoid action transformation (no oscillator dynamics).

    Instead of using CPG oscillators, this directly computes the serpenoid
    curvature profile from gait parameters. Simpler but less dynamic.

    Curvature profile:
        kappa(s, t) = A * sin(k * s - omega * t + phi)
    """

    def __init__(
        self,
        num_joints: int = 19,
        amplitude_range: Tuple[float, float] = (0.0, 2.0),
        frequency_range: Tuple[float, float] = (0.5, 3.0),
    ):
        """Initialize direct serpenoid transform.

        Args:
            num_joints: Number of output joints
            amplitude_range: (min, max) amplitude
            frequency_range: (min, max) frequency
        """
        self.num_joints = num_joints
        self.amplitude_range = amplitude_range
        self.frequency_range = frequency_range

        # Time accumulator
        self._time = 0.0

        # Joint positions along body (normalized to [0, 1])
        self._joint_positions = np.linspace(0, 1, num_joints)

    def reset(self) -> None:
        """Reset time accumulator."""
        self._time = 0.0

    def step(
        self,
        action: np.ndarray,
        dt: float = 0.01,
    ) -> np.ndarray:
        """Compute serpenoid curvatures from action.

        Args:
            action: [amplitude, frequency, wave_number, phase], normalized [-1, 1]
            dt: Timestep

        Returns:
            Target curvatures, shape (num_joints,)
        """
        action = np.clip(action, -1.0, 1.0)

        # Denormalize parameters
        amp_norm = (action[0] + 1) / 2
        amplitude = (
            self.amplitude_range[0] +
            amp_norm * (self.amplitude_range[1] - self.amplitude_range[0])
        )

        freq_norm = (action[1] + 1) / 2
        frequency = (
            self.frequency_range[0] +
            freq_norm * (self.frequency_range[1] - self.frequency_range[0])
        )

        wave_number = (action[2] + 1) / 2 * 3.0 + 0.5  # [0.5, 3.5]
        phase = (action[3] + 1) * np.pi  # [0, 2*pi]

        # Update time
        self._time += dt

        # Compute serpenoid profile
        omega = 2 * np.pi * frequency
        k = 2 * np.pi * wave_number

        curvatures = amplitude * np.sin(
            k * self._joint_positions - omega * self._time + phase
        )

        return curvatures


class DirectSerpenoidSteeringTransform:
    """Serpenoid with steering (turn bias) action transformation.

    Extends the basic serpenoid with a 5th parameter for steering:
        kappa(s, t) = A * sin(k * s - omega * t + phi) + kappa_turn

    The turn_bias (kappa_turn) creates a constant curvature offset that causes
    the snake to curve while moving:
        - turn_bias = 0: straight path
        - turn_bias > 0: curves left (counterclockwise)
        - turn_bias < 0: curves right (clockwise)
        - Turn radius R = 1 / |turn_bias|

    Action space (5-dim):
        [0] amplitude:   controls speed (how much body bends)
        [1] frequency:   controls speed (oscillation rate)
        [2] wave_number: controls efficiency (waves per body length)
        [3] phase:       controls wave timing
        [4] turn_bias:   controls steering (curvature offset)
    """

    def __init__(
        self,
        num_joints: int = 19,
        amplitude_range: Tuple[float, float] = (0.0, 2.0),
        frequency_range: Tuple[float, float] = (0.5, 3.0),
        turn_bias_range: Tuple[float, float] = (-2.0, 2.0),
    ):
        """Initialize serpenoid steering transform.

        Args:
            num_joints: Number of output joints
            amplitude_range: (min, max) amplitude
            frequency_range: (min, max) frequency
            turn_bias_range: (min, max) turn bias (kappa_turn)
                Typical values: (-2.0, 2.0) where |turn_bias| = 1/turn_radius
        """
        self.num_joints = num_joints
        self.amplitude_range = amplitude_range
        self.frequency_range = frequency_range
        self.turn_bias_range = turn_bias_range

        # Time accumulator
        self._time = 0.0

        # Joint positions along body (normalized to [0, 1])
        self._joint_positions = np.linspace(0, 1, num_joints)

        # Current parameters (for debugging/logging)
        self._current_params = {}

    def reset(self) -> None:
        """Reset time accumulator."""
        self._time = 0.0

    def denormalize_action(self, action: np.ndarray) -> Dict[str, float]:
        """Convert normalized action [-1, 1] to physical parameters.

        Args:
            action: Normalized action [amp, freq, wave_num, phase, turn_bias]

        Returns:
            Dictionary of physical parameters
        """
        action = np.clip(action, -1.0, 1.0)

        # Amplitude: [-1, 1] -> amplitude_range
        amp_norm = (action[0] + 1) / 2
        amplitude = (
            self.amplitude_range[0] +
            amp_norm * (self.amplitude_range[1] - self.amplitude_range[0])
        )

        # Frequency: [-1, 1] -> frequency_range
        freq_norm = (action[1] + 1) / 2
        frequency = (
            self.frequency_range[0] +
            freq_norm * (self.frequency_range[1] - self.frequency_range[0])
        )

        # Wave number: [-1, 1] -> [0.5, 3.5]
        wave_number = (action[2] + 1) / 2 * 3.0 + 0.5

        # Phase: [-1, 1] -> [0, 2*pi]
        phase = (action[3] + 1) * np.pi

        # Turn bias: [-1, 1] -> turn_bias_range
        turn_norm = (action[4] + 1) / 2
        turn_bias = (
            self.turn_bias_range[0] +
            turn_norm * (self.turn_bias_range[1] - self.turn_bias_range[0])
        )

        return {
            "amplitude": amplitude,
            "frequency": frequency,
            "wave_number": wave_number,
            "phase": phase,
            "turn_bias": turn_bias,
        }

    def step(
        self,
        action: np.ndarray,
        dt: float = 0.01,
    ) -> np.ndarray:
        """Compute serpenoid curvatures with steering from action.

        Args:
            action: [amplitude, frequency, wave_number, phase, turn_bias]
                    normalized to [-1, 1]
            dt: Timestep

        Returns:
            Target curvatures, shape (num_joints,)
        """
        # Denormalize parameters
        params = self.denormalize_action(action)
        self._current_params = params

        amplitude = params["amplitude"]
        frequency = params["frequency"]
        wave_number = params["wave_number"]
        phase = params["phase"]
        turn_bias = params["turn_bias"]

        # Update time
        self._time += dt

        # Compute serpenoid profile with steering
        # kappa(s, t) = A * sin(k * s - omega * t + phi) + kappa_turn
        omega = 2 * np.pi * frequency
        k = 2 * np.pi * wave_number

        curvatures = (
            amplitude * np.sin(k * self._joint_positions - omega * self._time + phase)
            + turn_bias  # This is the key addition for steering!
        )

        return curvatures

    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get action space bounds.

        Returns:
            Tuple of (low, high) arrays for action bounds
        """
        low = np.full(5, -1.0)
        high = np.full(5, 1.0)
        return low, high

    @property
    def current_parameters(self) -> Dict[str, float]:
        """Get current physical parameters."""
        return self._current_params.copy()
