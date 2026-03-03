"""Configuration dataclasses for COBRA navigation (Jiang et al., 2024).

Hierarchy:
    MujocoPhysicsConfig → CobraPhysicsConfig (COBRA-specific physics)
    (standalone)        → CobraCPGConfig (CPG timing and action bounds)
    (standalone)        → CobraEnvConfig (physics + cpg + arena + episode)
    CobraEnvConfig      → CobraMazeEnvConfig (+ maze params)
    NetworkConfig       → CobraNetworkConfig (Actor [512,256,128], Critic [512,256,256])
    DDPGConfig          → CobraNavigationConfig (top-level: all sub-configs)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from configs.base import Checkpointing, TensorBoard
from configs.network import ActorConfig, CriticConfig, NetworkConfig
from configs.physics import MujocoPhysicsConfig
from configs.training import DDPGConfig


@dataclass
class CobraPhysicsConfig(MujocoPhysicsConfig):
    """MuJoCo physics for COBRA snake robot.

    11 alternating pitch/yaw joints, timestep=0.001, Euler integrator.
    """

    mujoco_timestep: float = 0.001
    mujoco_substeps: int = 20  # 20 MuJoCo steps per CPG step
    num_joints: int = 11

    @property
    def timestep(self) -> float:
        return self.mujoco_timestep

    @property
    def frame_skip(self) -> int:
        return self.mujoco_substeps


@dataclass
class CobraCPGConfig:
    """CPG timing and action bounds for COBRA."""

    # CPG integration
    cpg_dt: float = 0.01  # CPG timestep (seconds)
    num_cpg_steps: int = 100  # CPG steps per RL step (= 2s RL period at 0.5Hz)

    # Action bounds [R1, R2, omega, theta1, theta2, delta1, delta2]
    R_min: float = 0.0
    R_max: float = 1.5
    omega_min: float = -0.1
    omega_max: float = 0.1
    theta_min: float = -math.pi
    theta_max: float = math.pi
    delta_min: float = -0.1
    delta_max: float = 0.1


@dataclass
class CobraEnvConfig:
    """Environment configuration for COBRA navigation."""

    physics: CobraPhysicsConfig = field(default_factory=CobraPhysicsConfig)
    cpg: CobraCPGConfig = field(default_factory=CobraCPGConfig)

    # Episode settings
    max_episode_steps: int = 80  # 80 RL steps * 2s = 160s

    # Arena (for waypoint task without maze)
    arena_size: float = 10.0  # Half-size of square arena
    min_goal_dist: float = 3.0  # Minimum initial distance to goal
    max_goal_dist: float = 8.0  # Maximum initial distance to goal

    # Waypoint reaching threshold
    waypoint_threshold: float = 0.5  # meters — waypoint is "reached" below this

    # Reward weights (Eq. 12)
    reward_w1: float = 1.0  # Proximity
    reward_w2: float = 5.0  # Velocity
    reward_w3: float = 0.1  # Smoothness

    # Device
    device: str = "cpu"


@dataclass
class CobraMazeEnvConfig(CobraEnvConfig):
    """Configuration for COBRA maze navigation (extends waypoint navigation)."""

    # Maze parameters
    maze_rows: int = 5
    maze_cols: int = 5
    maze_cell_size: float = 2.0
    maze_wall_height: float = 0.3
    maze_wall_thickness: float = 0.1

    # A* planner
    planner_resolution: float = 0.25  # Grid cell size for A*
    planner_inflation: float = 0.3  # Safety margin around walls

    # Override episode length for maze (harder task)
    max_episode_steps: int = 200


@dataclass
class CobraNetworkConfig(NetworkConfig):
    """Network config for COBRA navigation (Jiang et al., 2024).

    Actor: [512, 256, 128] with ReLU
    Critic: [512, 256, 256] with ReLU
    """

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[512, 256, 128],
            activation="relu",
            ortho_init=True,
            init_gain=0.01,
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[512, 256, 256],
            activation="relu",
            ortho_init=True,
            init_gain=1.0,
        )
    )


@dataclass
class CobraNavigationConfig(DDPGConfig):
    """Top-level config for COBRA navigation training (Jiang et al., 2024).

    Inherits DDPG hyperparameters. Composes env, network, and logging configs.
    """

    name: str = "cobra_navigation"

    # Override DDPG defaults for this task
    total_frames: int = 3_000_000
    warmup_steps: int = 10000
    noise_type: str = "ou"
    noise_sigma: float = 0.2
    noise_theta: float = 0.15
    tau: float = 0.001
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch_size: int = 256
    experiment_name: str = "cobra_navigation"

    # Compose sub-configs
    env: CobraEnvConfig = field(default_factory=CobraEnvConfig)
    network: CobraNetworkConfig = field(default_factory=CobraNetworkConfig)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
