"""Configuration dataclasses for contact-aware CPG soft snake (Liu, Onal & Fu, 2021).

Paper: Learning Contact-aware CPG-based Locomotion in a Soft Snake Robot
(arXiv:2105.04608)

Architecture:
    C1 (RL controller): Steering + velocity via tonic CPG inputs + frequency
    R2 (RL regulator): Event-triggered contact-aware tonic input modulation
    Matsuoka CPG: Generates smooth rhythmic actuation from tonic inputs

Trained via fictitious cooperative game (smooth fictitious play → Nash eq.)
using PPO for both C1 and R2.

Hierarchy:
    PPOConfig -> Liu2021Config (top-level)

Composable pieces:
    SoftSnakeConfig       -- 4-link soft snake body parameters
    MatsuokaCPGConfig     -- CPG oscillator parameters
    ContactSensorConfig   -- contact force sensing
    ObstacleMazeConfig    -- obstacle layout for training/testing
    Liu2021EnvConfig      -- composes all above
    Liu2021NetworkConfig  -- 4-layer 128×128 MLP
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from src.configs.base import TensorBoard
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.training import PPOConfig


# ---------------------------------------------------------------------------
# Soft snake robot body
# ---------------------------------------------------------------------------


@dataclass
class SoftSnakeConfig:
    """4-link soft snake robot parameters.

    EcoFlex 00-30 silicone body with pneumatic actuators.
    Rigid body components between soft links house electronics.
    One-direction wheels model anisotropic friction.
    """

    num_links: int = 4
    num_rigid_parts: int = 5  # including head
    link_length: float = 0.05  # meters per soft link
    body_width: float = 0.03  # meters

    # Pneumatic actuation (one chamber per link)
    max_pressure: float = 1.0  # normalized
    actuation_delay: float = 0.05  # seconds (pneumatic delay)


# ---------------------------------------------------------------------------
# Matsuoka CPG
# ---------------------------------------------------------------------------


@dataclass
class MatsuokaCPGConfig:
    """Matsuoka neural oscillator CPG network parameters.

    Each primitive oscillator has extensor/flexor neuron pairs.
    Coupled oscillators produce coordinated rhythmic patterns.
    """

    num_oscillators: int = 4  # One per link
    tau_r: float = 0.5       # Discharge rate time constant
    tau_a: float = 2.0       # Adaptation rate time constant
    a: float = 2.5           # Mutual inhibition weight (flexor-extensor)
    b: float = 2.5           # Self-inhibition weight
    w_coupling: float = 1.0  # Inter-oscillator coupling weight

    # Frequency ratio (controls oscillation speed)
    K_f_default: float = 1.0


# ---------------------------------------------------------------------------
# Contact sensing
# ---------------------------------------------------------------------------


@dataclass
class ContactSensorConfig:
    """Contact force sensor configuration.

    4 point contact sensors on two sides of each rigid body.
    Force representation uses diagonal sensor pairs:
        f_{2i-1} = f_{iB} - f_{iC}
        f_{2i}   = f_{iD} - f_{iA}
    Total: 10 force values for 5 rigid parts.
    """

    num_sensors_per_body: int = 4  # A, B, C, D
    num_bodies: int = 5  # 5 rigid parts
    force_dim: int = 10  # 2 per body

    # Detection threshold for event-triggering
    detection_distance: float = 0.05  # D: sensing range (meters)
    force_threshold: float = 0.01  # Minimum force to detect contact


# ---------------------------------------------------------------------------
# Obstacle maze
# ---------------------------------------------------------------------------


@dataclass
class ObstacleMazeConfig:
    """Obstacle maze layout for training and testing.

    Training: 3×3 grid, obstacle spacing 0.08m
    Testing: 5×6 grid, more obstacles, longer distance
    """

    # Training environment
    train_grid_rows: int = 3
    train_grid_cols: int = 3
    train_obstacle_spacing: float = 0.08  # meters
    train_goal_distance: float = 1.5  # meters

    # Testing environment
    test_grid_rows: int = 5
    test_grid_cols: int = 6
    test_goal_distance: float = 2.0

    # Obstacle properties
    obstacle_radius: float = 0.02  # meters
    obstacle_noise: float = 0.01  # Gaussian noise on positions

    # Goal
    goal_radius: float = 0.05  # Accepting radius
    deviation_angle_range: float = 60.0  # degrees (initial deviation)

    # Failure conditions
    starvation_time: float = 0.9  # seconds (jammed timeout)
    wrong_direction_steps: int = 60  # steps heading wrong way


# ---------------------------------------------------------------------------
# Artificial Potential Field reward
# ---------------------------------------------------------------------------


@dataclass
class APFRewardConfig:
    """Artificial Potential Field reward weights.

    R = ω₁·R_goal + ω₂·R_att + ω₂·R_rep

    R_goal: termination reward for reaching goal
    R_att: attractive force alignment with velocity
    R_rep: repulsive force alignment (encourages beneficial contact)
    """

    omega_1: float = 1.0   # Goal reward weight
    omega_2: float = 1.0   # Attractive field weight
    omega_3: float = 1.0   # Repulsive field weight

    k_att: float = 1.0     # Attractive force constant
    k_rep: float = 0.5     # Repulsive force constant
    rho_0: float = 0.05    # Repulsive field activation radius

    # Multi-level goal radius
    goal_levels: List[float] = field(
        default_factory=lambda: [0.15, 0.10, 0.05]
    )


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class Liu2021EnvConfig:
    """Environment configuration for contact-aware soft snake."""

    snake: SoftSnakeConfig = field(default_factory=SoftSnakeConfig)
    cpg: MatsuokaCPGConfig = field(default_factory=MatsuokaCPGConfig)
    contact: ContactSensorConfig = field(default_factory=ContactSensorConfig)
    maze: ObstacleMazeConfig = field(default_factory=ObstacleMazeConfig)
    reward: APFRewardConfig = field(default_factory=APFRewardConfig)

    # Observation dimensions
    # ζ₁:₄ (dynamic state) + ζ₅:₈ (curvatures) + ζ₉:₁₄ (prev actions + option)
    # + ζ₁₅:₂₄ (contact forces) + ζ₂₅:₂₆ (obstacle distance/angle)
    obs_dim: int = 26

    # C1 observes ζ₁:₁₄ (14 dims), R2 observes all 26
    c1_obs_dim: int = 14
    r2_obs_dim: int = 26

    # Action dimensions
    c1_action_dim: int = 6  # 4 tonic inputs + option(2) = [o, β]
    r2_action_dim: int = 4  # 4 tonic input additions

    # Controller composition weights
    w1: float = 0.5  # C1 weight
    w2: float = 0.5  # R2 weight

    # Episode
    max_episode_steps: int = 500

    device: str = "cpu"


# ---------------------------------------------------------------------------
# Network config (4-layer 128×128)
# ---------------------------------------------------------------------------


@dataclass
class Liu2021NetworkConfig(NetworkConfig):
    """Network config for C1 and R2: 4-layer 128×128 hidden neurons."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[128, 128, 128, 128],
            activation="tanh",
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[128, 128, 128, 128],
            activation="tanh",
        )
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class Liu2021Config(PPOConfig):
    """Top-level config for contact-aware CPG soft snake PPO training.

    Uses fictitious cooperative game: C1 and R2 alternate training
    until Nash equilibrium (value function convergence).
    """

    name: str = "liu2021"
    experiment_name: str = "liu2021_contact_aware_cpg"

    # PPO hyperparameters
    total_frames: int = 2_000_000
    frames_per_batch: int = 2048
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    num_epochs: int = 10

    # Fictitious play
    max_macro_iterations: int = 20  # N: max alternating rounds
    convergence_threshold: float = 10.0  # ε: value function convergence

    # Composed configs
    env: Liu2021EnvConfig = field(default_factory=Liu2021EnvConfig)
    network: Liu2021NetworkConfig = field(default_factory=Liu2021NetworkConfig)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)
