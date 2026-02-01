"""Central Pattern Generator module for rhythmic locomotion control.

This module provides neural oscillators and action transformations for
generating coordinated locomotion patterns with reduced action spaces.

Key components:
    - Oscillators: MatsuokaOscillator, HopfOscillator, CPGNetwork
    - Action transforms: CPGActionTransform, DirectSerpenoidTransform, DirectSerpenoidSteeringTransform
    - Environment wrapper: CPGEnvWrapper

Action Space Reduction:
    Standard control requires 19 curvature values (one per joint).
    CPG-based control reduces this to 4-5 gait parameters:
        - amplitude: Peak curvature magnitude
        - frequency: Oscillation frequency (speed)
        - wave_number: Spatial frequency (wavelength)
        - phase_offset: Phase shift
        - turn_bias: Steering (only in SERPENOID_STEERING, 5-dim)

Control Methods:
    - DIRECT: 19-dim, full control over each joint
    - CPG: 4-dim, neural oscillators generate curvatures
    - SERPENOID: 4-dim, κ(s,t) = A·sin(k·s - ω·t + φ) - NO steering
    - SERPENOID_STEERING: 5-dim, κ(s,t) = A·sin(k·s - ω·t + φ) + κ_turn - CAN steer

Example:
    >>> from snake_hrl.cpg import DirectSerpenoidSteeringTransform
    >>>
    >>> transform = DirectSerpenoidSteeringTransform(num_joints=19)
    >>>
    >>> # RL action: [amplitude, frequency, wave_number, phase, turn_bias]
    >>> rl_action = np.array([0.5, 0.3, 0.7, 0.0, 0.2])  # turn_bias=0.2 curves left
    >>> curvatures = transform.step(rl_action, dt=0.01)  # shape: (19,)
"""

from .oscillators import (
    MatsuokaOscillator,
    HopfOscillator,
    CPGNetwork,
    AdaptiveCPGNetwork,
)
from .action_wrapper import (
    CPGActionTransform,
    CPGEnvWrapper,
    DirectSerpenoidTransform,
    DirectSerpenoidSteeringTransform,
)

__all__ = [
    # Oscillators
    "MatsuokaOscillator",
    "HopfOscillator",
    "CPGNetwork",
    "AdaptiveCPGNetwork",
    # Action transforms
    "CPGActionTransform",
    "CPGEnvWrapper",
    "DirectSerpenoidTransform",
    "DirectSerpenoidSteeringTransform",
]
