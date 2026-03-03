"""Configuration dataclasses for Zheng, Li & Hayashibe (2022) reproduction.

Paper: "An Optimization-Based Approach to Locomotion of Underwater Snake Robots
with Series Elastic Actuators"

Hierarchy:
    MujocoPhysicsConfig -> Zheng2022PhysicsConfig
    Zheng2022CurriculumConfig  (standalone)
    Zheng2022EnvConfig         (composition of physics + curriculum + episode settings)
    NetworkConfig -> Zheng2022NetworkConfig
    PPOConfig -> Zheng2022Config (top-level)
"""

from dataclasses import dataclass, field
from typing import Optional

from configs.base import Checkpointing, TensorBoard
from configs.network import ActorConfig, CriticConfig, NetworkConfig
from configs.physics import MujocoPhysicsConfig
from configs.training import PPOConfig


# Pre-defined fluid configurations from the paper
WATER = {"fluid_density": 1000.0, "fluid_viscosity": 0.0009}
PROPYLENE_GLYCOL = {"fluid_density": 1036.0, "fluid_viscosity": 0.04}
ETHYLENE_GLYCOL = {"fluid_density": 1113.0, "fluid_viscosity": 0.016}


# ---------------------------------------------------------------------------
# Physics config
# ---------------------------------------------------------------------------


@dataclass
class Zheng2022PhysicsConfig(MujocoPhysicsConfig):
    """MuJoCo physics for 7-link underwater snake (Zheng et al. 2022).

    Rigid capsule chain in fluid with zero gravity (neutral buoyancy).
    """

    # Snake body parameters
    num_links: int = 7
    num_joints: int = 6  # num_links - 1
    link_length: float = 0.1  # meters
    link_radius: float = 0.01  # meters (diameter 0.02 m)
    link_density: float = 1000.0  # kg/m^3 (matches water for neutral buoyancy)
    joint_range_deg: float = 90.0  # +/- degrees
    motor_force_range: float = 1.0  # N (ctrl range [-1, 1])
    motor_gear: float = 0.1  # gear ratio -> effective torque [-0.1, 0.1] Nm
    joint_stiffness: float = 0.0  # Nm/rad (varied in stiffness experiments)
    joint_damping: float = 0.01  # Nm*s/rad (small damping for stability)
    armature: float = 0.001  # kg*m^2 (rotor inertia for stability)

    # Fluid parameters
    fluid_density: float = 1000.0  # kg/m^3 (water)
    fluid_viscosity: float = 0.0009  # Pa*s (water)

    # Simulation timing
    sim_timestep: float = 0.01  # seconds (100 Hz sim)
    sim_steps_per_control: int = 4  # 100 Hz / 25 Hz = 4
    control_dt: float = 0.04  # seconds (25 Hz control, derived)

    # Observation/action dimensions
    obs_dim: int = 16
    action_dim: int = 6

    def __post_init__(self):
        self.num_joints = self.num_links - 1
        self.action_dim = self.num_joints
        self.control_dt = self.sim_timestep * self.sim_steps_per_control


# ---------------------------------------------------------------------------
# Curriculum config
# ---------------------------------------------------------------------------


@dataclass
class Zheng2022CurriculumConfig:
    """Two-phase curriculum reward parameters from the paper.

    Phase 1 (epochs 0-2000): r2 = c * v_h - P_hat (maximize velocity)
    Phase 2 (epochs 2000+): r1 = r_v * r_P (match decreasing target velocity)
    """

    reward_c: float = 200.0  # Phase 1 velocity coefficient
    phase_transition_epoch: int = 2000  # Switch from Phase 1 to Phase 2
    velocity_decrease_interval: int = 1000  # epochs between target velocity decreases
    velocity_decrease_amount: float = 0.02  # m/s decrease per interval
    initial_target_velocity: float = 0.10  # m/s (found during Phase 1 training)
    min_target_velocity: float = 0.02  # m/s (minimum target)


# ---------------------------------------------------------------------------
# Environment config (composition)
# ---------------------------------------------------------------------------


@dataclass
class Zheng2022EnvConfig:
    """Environment configuration for underwater snake locomotion."""

    physics: Zheng2022PhysicsConfig = field(default_factory=Zheng2022PhysicsConfig)
    curriculum: Zheng2022CurriculumConfig = field(default_factory=Zheng2022CurriculumConfig)

    # Episode limits
    max_episode_steps: int = 1000  # 40 seconds at 25 Hz

    device: str = "cpu"


# ---------------------------------------------------------------------------
# Network config (2x256 ReLU MLP)
# ---------------------------------------------------------------------------


@dataclass
class Zheng2022NetworkConfig(NetworkConfig):
    """Network config: 2-layer 256-wide MLP with ReLU, matching the paper."""

    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(
            hidden_dims=[256, 256],
            activation="relu",
        )
    )
    critic: CriticConfig = field(
        default_factory=lambda: CriticConfig(
            hidden_dims=[256, 256],
            activation="relu",
        )
    )


# ---------------------------------------------------------------------------
# Top-level training config
# ---------------------------------------------------------------------------


@dataclass
class Zheng2022Config(PPOConfig):
    """Top-level config for Zheng 2022 replication.

    Inherits PPO hyperparameters from PPOConfig.
    The paper uses separate actor/critic LRs (0.003 / 0.001); we use the
    actor LR as the single optimizer LR for PPOTrainer compatibility.
    """

    name: str = "zheng2022"

    # PPO overrides from the paper
    total_frames: int = 10000 * 2048  # total_epochs * steps_per_epoch
    frames_per_batch: int = 2048
    mini_batch_size: int = 64
    learning_rate: float = 0.003  # paper's actor LR
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.97
    num_epochs: int = 10
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    experiment_name: str = "zheng2022_underwater"

    # Composed configs
    env: Zheng2022EnvConfig = field(default_factory=Zheng2022EnvConfig)
    network: Zheng2022NetworkConfig = field(default_factory=Zheng2022NetworkConfig)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)
