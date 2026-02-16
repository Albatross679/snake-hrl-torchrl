"""Configuration dataclasses for environments, networks, and training."""

# Base ML config and utilities
from configs.base import (
    MLBaseConfig,
    Checkpointing,
    TensorBoard,
    save_config,
    load_config,
)

# Geometry config
from configs.geometry import GeometryConfig

# Physics config hierarchy
from configs.physics import (
    SolverFramework,
    ElasticaGroundContact,
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
    from configs.locomotion import (
        LocomotionPhysicsConfig,
        LocomotionEnvConfig,
        LocomotionPPOConfig,
        LocomotionNetworkConfig,
    )
except ImportError:
    pass

__all__ = [
    # Base
    "MLBaseConfig",
    "Checkpointing",
    "TensorBoard",
    "save_config",
    "load_config",
    # Geometry
    "GeometryConfig",
    # Physics hierarchy
    "SolverFramework",
    "ElasticaGroundContact",
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
        "LocomotionPPOConfig",
        "LocomotionNetworkConfig",
    ])
