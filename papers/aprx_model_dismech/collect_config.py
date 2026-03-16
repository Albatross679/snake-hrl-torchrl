"""Configuration for surrogate data collection from DisMech."""

from dataclasses import dataclass, field

from src.configs.physics import DismechConfig


@dataclass
class DataCollectionConfig:
    """Config for collecting training data from real DisMech env."""

    num_transitions: int = 1_000_000
    num_workers: int = 16            # parallel worker processes (1 env each)
    random_action_fraction: float = 0.5
    policy_checkpoint: str = ""      # path to trained policy .pt, empty = no policy
    save_dir: str = "data/surrogate_dismech_rl_step"
    episodes_per_save: int = 100
    seed: int = 42
    collect_forces: bool = False     # DisMech does not expose force arrays
    save_format: str = "pt"          # "pt" (torch tensor files)

    # DisMech: 1 implicit step per RL action (dt=0.05s)
    steps_per_run: int = 1

    # Flat output format (states/next_states)
    flat_output: bool = True

    # Exploration: Sobol quasi-random actions (better 5D coverage than uniform)
    use_sobol_actions: bool = True   # Sobol quasi-random vs uniform random

    # Exploration: perturb rod state after reset to reach unusual configurations
    perturbation_fraction: float = 0.3    # fraction of episodes with perturbed init
    perturb_position_std: float = 0.002   # meters (rod is 0.5m long)
    perturb_velocity_std: float = 0.01    # m/s
    perturb_omega_std: float = 1.5        # rad/s -- operational omega_z is ~1-10 rad/s
    perturb_curvature_max: float = 3.0    # max amplitude for initial curvature (rad/m)

    # Health monitoring
    poll_interval: float = 30.0          # seconds between health checks
    stall_intervals: int = 6             # consecutive zero-progress polls before stall
    nan_rate_threshold: float = 0.1      # alert if NaN discard rate exceeds 10%
    alert_wait_duration: int = 300       # W&B alert rate limit (seconds)

    # Physics config (DisMech)
    physics: DismechConfig = field(default_factory=DismechConfig)
