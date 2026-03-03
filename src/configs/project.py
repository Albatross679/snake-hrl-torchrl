"""Project-level configs composing env + physics + network + training."""

from dataclasses import dataclass, field

from configs.base import MLBaseConfig, Checkpointing
from configs.training import PPOConfig
from configs.env import ApproachEnvConfig, CoilEnvConfig
from configs.network import NetworkConfig, HRLNetworkConfig


@dataclass
class SnakeApproachConfig(PPOConfig):
    """Top-level config for approach training."""

    name: str = "snake_approach"
    env: ApproachEnvConfig = field(default_factory=ApproachEnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)


@dataclass
class SnakeCoilConfig(PPOConfig):
    """Top-level config for coil training."""

    name: str = "snake_coil"
    env: CoilEnvConfig = field(default_factory=CoilEnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)


@dataclass
class SnakeHRLConfig(MLBaseConfig):
    """Top-level config for full HRL pipeline."""

    name: str = "snake_hrl"
    approach: SnakeApproachConfig = field(default_factory=SnakeApproachConfig)
    coil: SnakeCoilConfig = field(default_factory=SnakeCoilConfig)
    manager: PPOConfig = field(default_factory=PPOConfig)
    manager_network: HRLNetworkConfig = field(default_factory=HRLNetworkConfig)
    training_strategy: str = "sequential"
    use_curriculum: bool = True
