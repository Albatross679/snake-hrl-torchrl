"""Physics simulation module for snake dynamics.

Supports multiple physics frameworks:
- DisMech: Discrete elastic rod with implicit Euler integration
- PyElastica: Cosserat rod theory with symplectic integration
"""

from snake_hrl.configs.env import SolverFramework, PhysicsConfig
from snake_hrl.physics.geometry import (
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
        SnakeRobot or ElasticaSnakeRobot based on config.solver_framework
    """
    if config.solver_framework == SolverFramework.ELASTICA:
        from snake_hrl.physics.elastica_snake_robot import ElasticaSnakeRobot
        return ElasticaSnakeRobot(config, initial_snake_position, initial_prey_position)
    else:
        from snake_hrl.physics.snake_robot import SnakeRobot
        return SnakeRobot(config, initial_snake_position, initial_prey_position)


# Keep existing exports for backwards compatibility
from snake_hrl.physics.snake_robot import SnakeRobot

__all__ = [
    "create_snake_robot",
    "SnakeRobot",
    "SnakeGeometry",
    "PreyGeometry",
    "create_snake_geometry",
    "create_prey_geometry",
    "compute_contact_points",
    "compute_wrap_angle",
]
