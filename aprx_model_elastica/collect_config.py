"""Configuration for surrogate data collection from PyElastica."""

from dataclasses import dataclass, field

from locomotion_elastica.config import LocomotionElasticaEnvConfig


@dataclass
class DataCollectionConfig:
    """Config for collecting training data from real PyElastica env."""

    num_transitions: int = 1_000_000
    num_workers: int = 16            # parallel worker processes (1 env each)
    random_action_fraction: float = 0.5
    policy_checkpoint: str = ""      # path to trained policy .pt, empty = no policy
    save_dir: str = "data/surrogate"
    episodes_per_save: int = 100
    seed: int = 42
    collect_forces: bool = False     # Phase 2.1+: forces not used in surrogate training
    save_format: str = "pt"          # "pt" (torch tensor files) or "parquet"

    # Phase 2.1+: checkpoint-format collection
    steps_per_run: int = 4           # env.step() calls per collection run (4 × 500 substeps = 2000 substeps)

    # Exploration: Sobol quasi-random actions (better 5D coverage than uniform)
    use_sobol_actions: bool = True   # Sobol quasi-random vs uniform random

    # Exploration: perturb rod state after reset to reach unusual configurations
    perturbation_fraction: float = 0.3    # fraction of episodes with perturbed init
    perturb_position_std: float = 0.002   # meters (rod is 0.5m long)
    perturb_velocity_std: float = 0.01    # m/s
    perturb_omega_std: float = 1.5        # rad/s — operational omega_z is ~1-10 rad/s (was 0.05)
    perturb_curvature_max: float = 3.0    # max amplitude for initial curvature (rad/m)
                                          # serpenoid range is 0-5 rad/m; 3.0 is moderate

    # Health monitoring
    poll_interval: float = 30.0          # seconds between health checks
    stall_intervals: int = 6             # consecutive zero-progress polls before stall (3 min at 30s poll)
    nan_rate_threshold: float = 0.1      # alert if NaN discard rate exceeds 10%
    alert_wait_duration: int = 300       # W&B alert rate limit (seconds)

    # Physics config (same as locomotion_elastica)
    env: LocomotionElasticaEnvConfig = field(
        default_factory=LocomotionElasticaEnvConfig
    )
