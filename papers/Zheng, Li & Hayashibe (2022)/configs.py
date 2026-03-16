"""Configuration for Zheng, Li & Hayashibe (2022) reproduction.

Paper: "An Optimization-Based Approach to Locomotion of Underwater Snake Robots
with Series Elastic Actuators"

All hyperparameters from the paper in a single dataclass.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ZhengConfig:
    """Configuration for Zheng, Li & Hayashibe (2022) reproduction."""

    # --- Snake physical parameters ---
    num_links: int = 7
    num_joints: int = 6  # num_links - 1
    link_length: float = 0.1  # meters
    link_radius: float = 0.01  # meters (diameter 0.02 m)
    total_mass: float = 0.25  # kg
    link_density: float = 1000.0  # kg/m^3 (matches water for neutral buoyancy)
    joint_range_deg: float = 90.0  # +/- degrees
    motor_force_range: float = 1.0  # N (ctrl range [-1, 1])
    motor_gear: float = 0.1  # gear ratio -> effective torque [-0.1, 0.1] Nm
    joint_stiffness: float = 0.0  # Nm/rad (varied in stiffness experiments)
    joint_damping: float = 0.01  # Nm·s/rad (small damping for stability)
    armature: float = 0.001  # kg·m^2 (rotor inertia for stability)

    # --- Fluid parameters ---
    fluid_density: float = 1000.0  # kg/m^3 (water)
    fluid_viscosity: float = 0.0009  # Pa·s (water)

    # --- Simulation timing ---
    sim_timestep: float = 0.01  # seconds (100 Hz sim)
    sim_steps_per_control: int = 4  # 100 Hz / 25 Hz = 4
    control_dt: float = 0.04  # seconds (25 Hz control, derived)

    # --- Environment ---
    max_episode_steps: int = 1000  # 40 seconds at 25 Hz
    # Gravity is set to 0 in MJCF (neutral buoyancy: snake density = fluid density)

    # --- Observation space (16D) ---
    obs_dim: int = 16
    action_dim: int = 6

    # --- Curriculum reward parameters ---
    reward_c: float = 200.0  # Phase 1 velocity coefficient
    phase_transition_epoch: int = 2000  # Switch from Phase 1 to Phase 2
    velocity_decrease_interval: int = 1000  # epochs between target velocity decreases
    velocity_decrease_amount: float = 0.02  # m/s decrease per interval
    initial_target_velocity: float = 0.10  # m/s (found during Phase 1 training)
    min_target_velocity: float = 0.02  # m/s (minimum target)

    # --- PPO hyperparameters ---
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    policy_lr: float = 0.003
    value_lr: float = 0.001
    gae_lambda: float = 0.97
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_ppo_epochs: int = 10
    mini_batch_size: int = 64

    # --- Network architecture ---
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"

    # --- Training ---
    total_epochs: int = 10000
    steps_per_epoch: int = 2048  # frames collected per epoch

    # --- Logging/saving ---
    log_interval: int = 10  # epochs between log prints
    save_interval: int = 100  # epochs between checkpoint saves
    eval_interval: int = 50  # epochs between evaluations
    eval_episodes: int = 5

    # --- Paths ---
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    experiment_name: str = "zheng2022"

    # --- Device ---
    device: str = "cpu"

    def __post_init__(self):
        self.num_joints = self.num_links - 1
        self.action_dim = self.num_joints
        self.control_dt = self.sim_timestep * self.sim_steps_per_control


# Pre-defined fluid configurations from the paper
WATER = {"fluid_density": 1000.0, "fluid_viscosity": 0.0009}
PROPYLENE_GLYCOL = {"fluid_density": 1036.0, "fluid_viscosity": 0.04}
ETHYLENE_GLYCOL = {"fluid_density": 1113.0, "fluid_viscosity": 0.016}
