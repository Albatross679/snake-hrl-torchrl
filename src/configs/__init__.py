"""Configuration dataclasses for environments, networks, and training."""

# Base ML config and utilities
from configs.base import (
    MLBaseConfig,
    Checkpointing,
    MetricGroups,
    TensorBoard,
    Output,
    Console,
    resolve_device,
    save_config,
    load_config,
)

# Run directory and console logging
from configs.run_dir import setup_run_dir
from configs.console import ConsoleLogger

# Geometry config
from configs.geometry import GeometryConfig

# Physics config hierarchy
from configs.physics import (
    SolverFramework,
    ElasticaGroundContact,
    FrictionModel,
    FrictionConfig,
    PhysicsConfig,
    RodConfig,
    DERConfig,
    DismechConfig,
    DismechRodsConfig,
    CosseratConfig,
    ElasticaConfig,
    MujocoPhysicsConfig,
)

# Environment configs
from configs.env import (
    ControlMethod,
    StateRepresentation,
    EnvConfig,
    ApproachEnvConfig,
    CoilEnvConfig,
    HRLEnvConfig,
    GaitConfig,
    CPGConfig,
)

# Network configs
from configs.network import NetworkConfig, ActorConfig, CriticConfig, HRLNetworkConfig

# Training configs
from configs.training import (
    RLConfig,
    TrainingConfig,
    PPOConfig,
    SACConfig,
    DDPGConfig,
    HRLConfig,
    EvaluationConfig,
)

# Project-level configs
from configs.project import (
    SnakeApproachConfig,
    SnakeCoilConfig,
    SnakeHRLConfig,
)

# Locomotion configs (optional — module may not exist in all checkouts)
try:
    from bing2019.configs_bing2019 import (
        LocomotionPhysicsConfig,
        LocomotionEnvConfig,
        LocomotionConfig,
        LocomotionPPOConfig,  # backward-compat alias for LocomotionConfig
        LocomotionNetworkConfig,
    )
except ImportError:
    pass

# Liu 2023 configs (optional — liu2023 may not exist in all checkouts)
try:
    from liu2023.configs_liu2023 import (
        Liu2023PhysicsConfig,
        Liu2023CPGConfig,
        Liu2023CurriculumConfig,
        Liu2023EnvConfig,
        Liu2023NetworkConfig,
        Liu2023Config,
    )
except ImportError:
    pass

# Navigation configs (optional — jiang2024 may not exist in all checkouts)
try:
    from jiang2024.configs_jiang2024 import (
        CobraPhysicsConfig,
        CobraCPGConfig,
        CobraEnvConfig,
        CobraMazeEnvConfig,
        CobraNetworkConfig,
        CobraNavigationConfig,
    )
except ImportError:
    pass

# Zheng 2022 configs (optional — zheng2022 may not exist in all checkouts)
try:
    from zheng2022.configs_zheng2022 import (
        Zheng2022PhysicsConfig,
        Zheng2022CurriculumConfig,
        Zheng2022EnvConfig,
        Zheng2022NetworkConfig,
        Zheng2022Config,
    )
except ImportError:
    pass

__all__ = [
    # Base
    "MLBaseConfig",
    "Checkpointing",
    "MetricGroups",
    "TensorBoard",
    "Output",
    "Console",
    "resolve_device",
    "save_config",
    "load_config",
    # Run directory and console logging
    "setup_run_dir",
    "ConsoleLogger",
    # Geometry
    "GeometryConfig",
    # Physics hierarchy
    "SolverFramework",
    "ElasticaGroundContact",
    "FrictionModel",
    "FrictionConfig",
    "PhysicsConfig",
    "RodConfig",
    "DERConfig",
    "DismechConfig",
    "DismechRodsConfig",
    "CosseratConfig",
    "ElasticaConfig",
    "MujocoPhysicsConfig",
    # Environment configs
    "ControlMethod",
    "StateRepresentation",
    "EnvConfig",
    "ApproachEnvConfig",
    "CoilEnvConfig",
    "HRLEnvConfig",
    "GaitConfig",
    "CPGConfig",
    # Network configs
    "NetworkConfig",
    "ActorConfig",
    "CriticConfig",
    "HRLNetworkConfig",
    # Training configs
    "RLConfig",
    "TrainingConfig",
    "PPOConfig",
    "SACConfig",
    "DDPGConfig",
    "HRLConfig",
    "EvaluationConfig",
    # Project configs
    "SnakeApproachConfig",
    "SnakeCoilConfig",
    "SnakeHRLConfig",
]

# Extend __all__ with locomotion configs if available
if "LocomotionPhysicsConfig" in dir():
    __all__.extend([
        "LocomotionPhysicsConfig",
        "LocomotionEnvConfig",
        "LocomotionConfig",
        "LocomotionPPOConfig",
        "LocomotionNetworkConfig",
    ])

# Extend __all__ with Liu 2023 configs if available
if "Liu2023Config" in dir():
    __all__.extend([
        "Liu2023PhysicsConfig",
        "Liu2023CPGConfig",
        "Liu2023CurriculumConfig",
        "Liu2023EnvConfig",
        "Liu2023NetworkConfig",
        "Liu2023Config",
    ])

# Extend __all__ with navigation configs if available
if "CobraPhysicsConfig" in dir():
    __all__.extend([
        "CobraPhysicsConfig",
        "CobraCPGConfig",
        "CobraEnvConfig",
        "CobraMazeEnvConfig",
        "CobraNetworkConfig",
        "CobraNavigationConfig",
    ])

# Extend __all__ with Zheng 2022 configs if available
if "Zheng2022Config" in dir():
    __all__.extend([
        "Zheng2022PhysicsConfig",
        "Zheng2022CurriculumConfig",
        "Zheng2022EnvConfig",
        "Zheng2022NetworkConfig",
        "Zheng2022Config",
    ])
