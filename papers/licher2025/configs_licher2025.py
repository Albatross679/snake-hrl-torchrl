"""Configuration dataclasses for Cosserat rod DD-PINN + MPC (Licher et al., 2025).

Paper: Adaptive Model-Predictive Control of a Soft Continuum Robot Using a
Physics-Informed Neural Network Based on Cosserat Rod Theory
(arXiv:2508.12681, submitted to IEEE T-RO)

This is NOT an RL system — it uses a Domain-Decoupled PINN as a fast
dynamics surrogate (44,000x speedup) inside a nonlinear evolutionary MPC.

Hierarchy:
    CosseratConfig       -> Licher2025PhysicsConfig (3-chamber soft actuator)
    MLBaseConfig          -> Licher2025Config (top-level config)

Composable pieces:
    PINNConfig                -- DD-PINN architecture and training
    MPCConfig                 -- NEMPC horizon, population, constraints
    UKFConfig                 -- Unscented Kalman Filter for state estimation
    Licher2025EnvConfig       -- composes physics + PINN + MPC + UKF
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from src.configs.base import Checkpointing, MLBaseConfig, TensorBoard
from src.configs.physics import CosseratConfig, FrictionConfig, FrictionModel, GeometryConfig


# ---------------------------------------------------------------------------
# PINN configuration
# ---------------------------------------------------------------------------


@dataclass
class AnsatzConfig:
    """Configuration for the Ansatz function in DD-PINN.

    The Ansatz decouples time from the neural network:
        x_hat_t = g(f_NN(x_0, u_0, theta), t) + x_0

    g_j = sum_i(alpha_ij * (sin(beta_ij * t + gamma_ij) - sin(gamma_ij)))
    with optional exponential damping.
    """

    num_ansatz_terms: int = 4  # Number of sinusoidal terms per state
    use_exponential_damping: bool = True


@dataclass
class PINNConfig:
    """Configuration for the Domain-Decoupled PINN surrogate.

    The DD-PINN predicts Ansatz parameters from (x_0, u_0, theta),
    then evaluates the closed-form Ansatz at time t.
    """

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "gelu"
    num_hidden_layers: int = 2

    # Ansatz
    ansatz: AnsatzConfig = field(default_factory=AnsatzConfig)

    # State dimension (12 per spatial node * 6 nodes = 72)
    state_dim: int = 72
    num_spatial_nodes: int = 6

    # Input dimension: state + control + parameters
    control_dim: int = 3  # 3 pressure inputs
    param_dim: int = 3  # Bending compliance parameters

    # Training
    learning_rate: float = 1e-3
    lr_scheduler: str = "plateau"  # plateau, cosine, constant
    lr_patience: int = 50
    lr_factor: float = 0.5
    batch_size: int = 256
    num_epochs: int = 500

    # Loss weights
    physics_loss_weight: float = 1.0  # lambda_2
    data_loss_weight: float = 1.0  # lambda_3

    # Gradient computation
    analytic_gradients: bool = True  # Closed-form (not autodiff)


# ---------------------------------------------------------------------------
# MPC configuration
# ---------------------------------------------------------------------------


@dataclass
class MPCConfig:
    """Configuration for nonlinear evolutionary MPC (NEMPC).

    Uses CMA-ES or similar evolutionary strategy to optimize control
    sequences over a finite horizon using the DD-PINN as the forward model.
    """

    # Horizon
    prediction_horizon: int = 10  # steps
    control_horizon: int = 5  # steps (controls repeat after this)
    control_dt: float = 1.0 / 70.0  # 70 Hz control rate

    # Evolutionary optimizer
    population_size: int = 64
    num_generations: int = 20
    elite_fraction: float = 0.25
    mutation_sigma: float = 0.1

    # Constraints
    max_pressure: float = 1.5e5  # Pa (1.5 bar)
    min_pressure: float = 0.0  # Pa
    max_pressure_rate: float = 5e4  # Pa/step

    # Cost weights
    position_weight: float = 1.0
    orientation_weight: float = 0.5
    control_effort_weight: float = 0.01
    smoothness_weight: float = 0.1


# ---------------------------------------------------------------------------
# UKF configuration
# ---------------------------------------------------------------------------


@dataclass
class UKFConfig:
    """Unscented Kalman Filter for state and parameter estimation.

    Estimates the full 72-dim Cosserat rod state plus bending compliance
    from tip position/orientation measurements (motion capture).
    """

    # UKF tuning
    alpha: float = 1e-3  # Spread of sigma points
    beta: float = 2.0  # Prior knowledge (Gaussian optimal = 2)
    kappa: float = 0.0  # Secondary scaling

    # Process noise
    process_noise_state: float = 1e-4
    process_noise_param: float = 1e-6

    # Measurement noise
    measurement_noise_pos: float = 1e-3  # 1 mm position noise
    measurement_noise_orient: float = 0.01  # orientation noise (rad)

    # Estimated parameters
    num_estimated_params: int = 3  # Bending compliance per chamber


# ---------------------------------------------------------------------------
# Physics config (3-chamber soft pneumatic actuator)
# ---------------------------------------------------------------------------


@dataclass
class Licher2025PhysicsConfig(CosseratConfig):
    """Cosserat rod physics for 3-chamber soft pneumatic actuator.

    The actuator has three fiber-reinforced pneumatic chambers arranged
    symmetrically around a central axis. Pressurizing chambers produces
    bending in the corresponding direction.
    """

    # Actuator geometry
    actuator_length: float = 0.125  # 125 mm
    outer_diameter: float = 0.0424  # 42.4 mm
    inner_diameter: float = 0.007  # 7 mm
    chamber_radius: float = 0.0212  # 21.2 mm (from center)
    chamber_area: float = 235.6e-6  # 235.6 mm^2 cross-section
    num_chambers: int = 3

    # Material (EcoFlex 00-50 silicone body, Dragon Skin 30 caps)
    density: float = 1070.0  # kg/m^3
    youngs_modulus: float = 8.3e4  # Pa (EcoFlex 00-50)
    poisson_ratio: float = 0.49  # Nearly incompressible silicone

    # Override rod geometry
    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(
            snake_length=0.125,
            snake_radius=0.0212,
            num_segments=5,  # 6 spatial nodes
        )
    )
    dt: float = 1e-4  # Fine time step for dynamic Cosserat

    # First node clamped (base of actuator)
    clamp_base: bool = True

    # No gravity for horizontal actuator
    enable_gravity: bool = False

    # No ground friction
    friction: FrictionConfig = field(
        default_factory=lambda: FrictionConfig(model=FrictionModel.NONE)
    )


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@dataclass
class Licher2025EnvConfig:
    """Environment configuration for soft pneumatic actuator with MPC."""

    physics: Licher2025PhysicsConfig = field(
        default_factory=Licher2025PhysicsConfig
    )
    pinn: PINNConfig = field(default_factory=PINNConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    ukf: UKFConfig = field(default_factory=UKFConfig)

    # Episode settings
    max_episode_steps: int = 500  # ~7s at 70 Hz

    # Target tracking
    target_type: str = "circular"  # circular, figure_eight, step

    # Device
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Top-level config (PINN training, not RL)
# ---------------------------------------------------------------------------


@dataclass
class Licher2025Config(MLBaseConfig):
    """Top-level config for Licher et al. (2025) PINN training + MPC.

    This is a supervised learning config for the DD-PINN, not an RL config.
    The MPC controller uses the trained PINN at inference time.
    """

    name: str = "licher2025"

    # PINN training
    pinn_training_data: str = "data/licher2025_cosserat_trajectories.npz"
    pinn_validation_split: float = 0.1
    pinn_num_trajectories: int = 10000

    # Composed configs
    env: Licher2025EnvConfig = field(default_factory=Licher2025EnvConfig)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)
