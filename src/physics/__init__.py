"""Physics simulation module for snake dynamics.

Supports multiple physics frameworks:
- DisMech: Discrete elastic rod with implicit Euler integration
- PyElastica: Cosserat rod theory with symplectic integration
- dismech-rods: C++ discrete elastic rod via py_dismech (SimulationManager API)
"""

from configs.physics import SolverFramework, PhysicsConfig
from physics.geometry import (
    SnakeGeometry,
    PreyGeometry,
    create_snake_geometry,
    create_prey_geometry,
    compute_contact_points,
    compute_wrap_angle,
)


def create_snake_robot(
    config: PhysicsConfig,
    initial_snake_position=None,
    initial_prey_position=None,
):
    """Factory function to create snake robot with selected physics framework.

    Args:
        config: Physics configuration with solver_framework selection
        initial_snake_position: Starting position of snake head
        initial_prey_position: Starting position of prey

    Returns:
        SnakeRobot, ElasticaSnakeRobot, or DismechRodsSnakeRobot based on config.solver_framework
    """
    if config.solver_framework == SolverFramework.ELASTICA:
        from physics.elastica_snake_robot import ElasticaSnakeRobot
        return ElasticaSnakeRobot(config, initial_snake_position, initial_prey_position)
    elif config.solver_framework == SolverFramework.DISMECH_RODS:
        from physics.dismech_rods_snake_robot import DismechRodsSnakeRobot
        return DismechRodsSnakeRobot(config, initial_snake_position, initial_prey_position)
    elif config.solver_framework == SolverFramework.MUJOCO:
        from physics.mujoco_snake_robot import MujocoSnakeRobot
        return MujocoSnakeRobot(config, initial_snake_position, initial_prey_position)
    else:
        from physics.snake_robot import SnakeRobot
        return SnakeRobot(config, initial_snake_position, initial_prey_position)


# Keep existing exports for backwards compatibility
from physics.snake_robot import SnakeRobot
from physics.cpg import (
    MatsuokaOscillator,
    HopfOscillator,
    CPGNetwork,
    AdaptiveCPGNetwork,
    CPGActionTransform,
    CPGEnvWrapper,
    DirectSerpenoidTransform,
    DirectSerpenoidSteeringTransform,
)

__all__ = [
    "create_snake_robot",
    "SnakeRobot",
    "SnakeGeometry",
    "PreyGeometry",
    "create_snake_geometry",
    "create_prey_geometry",
    "compute_contact_points",
    "compute_wrap_angle",
    # CPG / Actuators
    "MatsuokaOscillator",
    "HopfOscillator",
    "CPGNetwork",
    "AdaptiveCPGNetwork",
    "CPGActionTransform",
    "CPGEnvWrapper",
    "DirectSerpenoidTransform",
    "DirectSerpenoidSteeringTransform",
]
