"""Configuration dataclasses for DQN gait learning (Shi, Dear & Kelly, 2020).

Paper: Deep Reinforcement Learning for Snake Robot Locomotion
(IFAC PapersOnLine 53-2, 2020, 9688–9695)

Two kinematic robots:
    1. Wheeled 3-link snake (nonholonomic, terrestrial)
    2. Swimming 3-link snake (low Reynolds number, aquatic)

Both use DQN with discrete joint-velocity actions and exploit geometric
symmetries (SE(2) fiber bundle structure) to reduce the state space
from 5D (x, y, α₁, α₂, θ) to 3D (α₁, α₂, θ).

Hierarchy:
    MLBaseConfig -> Shi2020Config (top-level, DQN training)

Composable pieces:
    KinematicRobotConfig    -- 3-link robot parameters
    Shi2020EnvConfig        -- composes robot + action discretization
    DQNConfig               -- DQN hyperparameters
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import math

from configs.base import MLBaseConfig, TensorBoard


class RobotType(str, Enum):
    """Two kinematic snake robot variants from the paper."""
    WHEELED = "wheeled"   # Nonholonomic wheeled 3-link
    SWIMMING = "swimming"  # Low Reynolds number swimmer


class GaitTask(str, Enum):
    """Locomotion primitive tasks."""
    FORWARD = "forward"
    BACKWARD = "backward"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"


# ---------------------------------------------------------------------------
# Robot parameters
# ---------------------------------------------------------------------------


@dataclass
class KinematicRobotConfig:
    """3-link kinematic snake robot parameters.

    Both wheeled and swimming robots share the same link geometry.
    The kinematic model is ξ = -A(α)·α̇ where A is the local connection
    form derived from geometric mechanics.
    """

    robot_type: RobotType = RobotType.WHEELED
    num_links: int = 3
    link_length: float = 1.0  # R: identical link lengths

    # Joint limits (avoid singularity at α₁ = α₂)
    joint_limit: float = math.pi  # ±π radians
    # For wheeled: constrain α₁ > 0, α₂ < 0 to avoid singularity
    avoid_singularity: bool = True

    # Swimming-specific
    fluid_drag_coefficient: float = 1.0  # k: characterizes fluid


# ---------------------------------------------------------------------------
# Action space (discrete joint velocities)
# ---------------------------------------------------------------------------


@dataclass
class ActionConfig:
    """Discrete action space for joint velocity control.

    Action space A = A₁ × A₂ where
    A_i = {a_max, a_max - a_interval, ..., 0, ..., -a_max + a_interval, -a_max}
    Action (0, 0) is removed to prevent stasis.
    """

    a_max: float = math.pi / 8       # Maximum joint velocity magnitude
    a_interval: float = math.pi / 8  # Discretization interval
    t_interval: float = 4.0          # Seconds per action

    @property
    def num_actions_per_joint(self) -> int:
        """Number of discrete velocity levels per joint."""
        return int(2 * self.a_max / self.a_interval) + 1

    @property
    def total_actions(self) -> int:
        """Total number of joint actions (excluding (0,0))."""
        return self.num_actions_per_joint ** 2 - 1


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class Shi2020EnvConfig:
    """Environment configuration for kinematic snake robot."""

    robot: KinematicRobotConfig = field(default_factory=KinematicRobotConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    # Task
    task: GaitTask = GaitTask.FORWARD

    # Episode settings
    max_episode_steps: int = 500  # T = 500 iterations per episode

    # Device
    device: str = "cpu"


# ---------------------------------------------------------------------------
# DQN hyperparameters (from Table 1 in the paper)
# ---------------------------------------------------------------------------


@dataclass
class DQNConfig:
    """DQN training hyperparameters from Shi et al. (2020) Table 1."""

    # Replay buffer
    memory_size: int = 250           # N_memory
    replay_start: int = 50           # C_memory: start learning after this many transitions
    minibatch_size: int = 8          # N_minibatch

    # Training
    num_episodes: int = 10           # M: total episodes (increased for convergence)
    iterations_per_episode: int = 500  # T

    # Exploration (ε-greedy)
    epsilon_init: float = 1.0        # ε₀
    epsilon_decay: float = 0.99954   # τ_decay
    epsilon_min: float = 0.01

    # Network updates
    target_update_freq: int = 20     # C_update: copy Q → Q̄ every N steps
    gamma: float = 0.99              # Discount factor
    learning_rate: float = 0.0002    # RMSProp lr

    # Reward weights
    c1: float = 10.0    # Weight for displacement (Δx or Δθ)
    c2: float = 10.0    # Weight for zero-displacement penalty P₀
    c3: float = 1.0     # Weight for orientation reward R_θ (forward task only)


# ---------------------------------------------------------------------------
# Network config (small Q-network)
# ---------------------------------------------------------------------------


@dataclass
class Shi2020NetworkConfig:
    """Q-network architecture: Input(5) → 50 ReLU → 10 ReLU → 1 linear.

    Input: concatenation of state (α₁, α₂, θ) and action (α̇₁, α̇₂) = 5 dims.
    Output: scalar Q-value.
    """

    input_dim: int = 5   # state(3) + action(2)
    hidden_dims: list = field(default_factory=lambda: [50, 10])
    activation: str = "relu"
    output_dim: int = 1  # Q-value


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class Shi2020Config(MLBaseConfig):
    """Top-level config for Shi et al. (2020) DQN gait learning."""

    name: str = "shi2020"

    env: Shi2020EnvConfig = field(default_factory=Shi2020EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    network: Shi2020NetworkConfig = field(default_factory=Shi2020NetworkConfig)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)

    def __post_init__(self):
        """Set name from robot type and task."""
        robot = self.env.robot.robot_type.value if isinstance(self.env.robot.robot_type, RobotType) else self.env.robot.robot_type
        task = self.env.task.value if isinstance(self.env.task, GaitTask) else self.env.task
        self.name = f"shi2020_{robot}_{task}"
