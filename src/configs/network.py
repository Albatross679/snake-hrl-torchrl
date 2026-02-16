"""Neural network architecture configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ActorConfig:
    """Configuration for actor (policy) network."""

    # Architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "tanh"  # tanh, relu, elu, silu

    # Output distribution
    distribution: str = "tanh_normal"  # tanh_normal, normal, beta
    min_std: float = 0.1
    max_std: float = 1.0
    init_std: float = 0.5

    # Initialization
    ortho_init: bool = True
    init_gain: float = 0.01  # Gain for final layer

    # Normalization
    use_layer_norm: bool = False
    use_spectral_norm: bool = False


@dataclass
class CriticConfig:
    """Configuration for critic (value) network."""

    # Architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "tanh"

    # Output
    num_outputs: int = 1  # 1 for value function, action_dim for Q-function

    # Initialization
    ortho_init: bool = True
    init_gain: float = 1.0

    # Normalization
    use_layer_norm: bool = False
    use_spectral_norm: bool = False


@dataclass
class NetworkConfig:
    """Combined network configuration."""

    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)

    # Shared settings
    device: str = "cpu"
    dtype: str = "float32"

    # Feature extraction (optional CNN for image observations)
    use_cnn: bool = False
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [8, 4, 3])
    cnn_strides: List[int] = field(default_factory=lambda: [4, 2, 1])

    # Recurrent (optional LSTM/GRU for partial observability)
    use_recurrent: bool = False
    recurrent_type: str = "lstm"  # lstm, gru
    recurrent_hidden_size: int = 256
    recurrent_num_layers: int = 1


@dataclass
class HRLNetworkConfig:
    """Network configuration for hierarchical RL."""

    # Manager (high-level) policy
    manager: NetworkConfig = field(default_factory=NetworkConfig)

    # Worker (low-level) policies - one per skill
    worker_approach: NetworkConfig = field(default_factory=NetworkConfig)
    worker_coil: NetworkConfig = field(default_factory=NetworkConfig)

    # Shared encoder (optional)
    share_encoder: bool = False
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    def __post_init__(self):
        """Adjust manager network for discrete skill selection."""
        # Manager outputs discrete skill selection
        self.manager.actor.distribution = "categorical"
