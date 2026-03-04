"""Configuration dataclasses for biohybrid lattice worm (Schaffer et al., 2024).

Paper: Hitting the Gym: Reinforcement Learning Control of Exercise-Strengthened
Biohybrid Robots in Simulation (arXiv:2408.16069)

Hierarchy:
    ElasticaConfig        -> Schaffer2024PhysicsConfig (3D lattice worm)
    PPOConfig             -> Schaffer2024Config (top-level project config)

Composable pieces:
    MuscleConfig              -- muscle rod geometry and adaptation
    Schaffer2024EnvConfig     -- composes physics + muscles + target
    Schaffer2024NetworkConfig -- 2x64 tanh MLP (SB3 default)
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from src.configs.base import TensorBoard
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.configs.physics import ElasticaConfig, FrictionConfig, FrictionModel, GeometryConfig
from src.configs.training import PPOConfig


# ---------------------------------------------------------------------------
# Muscle configuration
# ---------------------------------------------------------------------------


@dataclass
class MuscleConfig:
    """Configuration for biohybrid muscle rods and adaptation model.

    The worm has 42 muscle rods attached to a lattice of 40 structural rods.
    Each muscle has an adaptive force ceiling that increases with exercise.
    """

    num_muscles: int = 42
    num_structural_rods: int = 40

    # Muscle rod geometry
    muscle_rod_radius: float = 0.005  # 5 mm
    structure_rod_radius: float = 0.01  # 10 mm

    # Initial force ceiling (mN)
    initial_force_ceiling: float = 2000.0
    max_force_ceiling_multiplier: float = 2.0  # cap at 2x initial

    # Adaptation coefficients (exercise-dependent strengthening)
    strain_coefficient: float = 1e-6  # beta
    force_coefficient: float = 4e-8  # gamma


# ---------------------------------------------------------------------------
# Physics config (inherits ElasticaConfig for Cosserat rod simulation)
# ---------------------------------------------------------------------------


@dataclass
class Schaffer2024PhysicsConfig(ElasticaConfig):
    """PyElastica physics for 3D lattice worm.

    Overrides ElasticaConfig defaults for the Schaffer et al. (2024) setup:
    - 3D simulation (worm body is 100mm tall, 75mm diameter lattice)
    - 40 structural rods + 42 muscle rods modeled as Cosserat rods
    - Gravity enabled (worm on ground plane)
    """

    # Worm body dimensions
    worm_height: float = 0.1  # 100 mm
    worm_diameter: float = 0.075  # 75 mm

    # Override rod defaults for lattice elements
    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(
            snake_length=0.1,  # individual rod length ~ worm height
            snake_radius=0.01,  # structural rod radius
            num_segments=10,  # segments per rod element
        )
    )
    dt: float = 2.5e-5  # Small dt for 3D Cosserat stability
    youngs_modulus: float = 1e5  # Softer material (biohybrid)
    poisson_ratio: float = 0.5
    density: float = 1060.0  # Close to biological tissue

    # Gravity
    enable_gravity: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Ground friction
    friction: FrictionConfig = field(
        default_factory=lambda: FrictionConfig(
            model=FrictionModel.COULOMB,
            mu_kinetic=0.3,
        )
    )

    # Elastica-specific
    elastica_substeps: int = 200  # Many substeps due to small dt
    elastica_damping: float = 0.05


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class TargetConfig:
    """Target position sampling for navigation task."""

    # 8 distinct target positions from the paper (relative to worm center)
    num_targets: int = 8
    target_distance: float = 0.1  # 100 mm from start
    target_threshold: float = 0.001  # 1 mm success radius


@dataclass
class Schaffer2024EnvConfig:
    """Environment configuration for biohybrid lattice worm."""

    physics: Schaffer2024PhysicsConfig = field(
        default_factory=Schaffer2024PhysicsConfig
    )
    muscles: MuscleConfig = field(default_factory=MuscleConfig)
    target: TargetConfig = field(default_factory=TargetConfig)

    # Episode settings
    max_episode_steps: int = 200

    # Muscle adaptation (per-episode update)
    enable_adaptation: bool = True

    # Device
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Network config (SB3 default: 2x64 tanh MLP)
# ---------------------------------------------------------------------------


@dataclass
class Schaffer2024NetworkConfig(NetworkConfig):
    """Network config matching Stable Baselines3 defaults: 2x64 tanh."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[64, 64],
            activation="tanh",
            distribution="normal",
            min_std=0.01,
            max_std=1.0,
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[64, 64],
            activation="tanh",
        )
    )


# ---------------------------------------------------------------------------
# Top-level config (PPO)
# ---------------------------------------------------------------------------


@dataclass
class Schaffer2024Config(PPOConfig):
    """Top-level config for biohybrid worm PPO training.

    Uses PPO with SB3-style defaults. The paper trains for up to 8500 episodes
    with n_steps=2 (very short rollouts).
    """

    name: str = "schaffer2024"
    experiment_name: str = "schaffer2024_biohybrid_worm"

    # PPO hyperparameters (SB3 defaults)
    total_frames: int = 500_000
    frames_per_batch: int = 2  # n_steps=2 from paper
    mini_batch_size: int = 2
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    num_epochs: int = 10
    max_grad_norm: float = 0.5

    # Composed configs
    env: Schaffer2024EnvConfig = field(default_factory=Schaffer2024EnvConfig)
    network: Schaffer2024NetworkConfig = field(
        default_factory=Schaffer2024NetworkConfig
    )
    tensorboard: TensorBoard = field(default_factory=TensorBoard)

    # Parallelism
    num_envs: int = 1
