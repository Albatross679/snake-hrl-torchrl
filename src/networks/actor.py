"""Actor (policy) networks for TorchRL."""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn

from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    NormalParamExtractor,
)
try:
    from torchrl.data import BoundedTensorSpec, CompositeSpec
except ImportError:
    from torchrl.data import Bounded as BoundedTensorSpec, Composite as CompositeSpec
from tensordict.nn import TensorDictModule, TensorDictSequential

from src.configs.network import ActorConfig, NetworkConfig


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name.lower()]()


class ActorNetwork(nn.Module):
    """MLP network for actor (policy).

    Outputs mean and standard deviation for Gaussian policy.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        min_std: float = 0.1,
        max_std: float = 1.0,
        init_std: float = 0.5,
        ortho_init: bool = True,
        init_gain: float = 0.01,
        use_layer_norm: bool = False,
    ):
        """Initialize actor network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            min_std: Minimum standard deviation
            max_std: Maximum standard deviation
            init_std: Initial standard deviation
            ortho_init: Whether to use orthogonal initialization
            init_gain: Gain for final layer initialization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        # Build MLP layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

        # Initialize log_std bias to achieve init_std
        init_log_std = torch.log(torch.tensor(init_std))
        nn.init.constant_(self.log_std_head.bias, init_log_std.item())

        # Apply orthogonal initialization
        if ortho_init:
            self._apply_ortho_init(init_gain)

    def _apply_ortho_init(self, final_gain: float) -> None:
        """Apply orthogonal initialization to layers."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        # Final layers with smaller gain
        nn.init.orthogonal_(self.mean_head.weight, gain=final_gain)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.log_std_head.weight, gain=final_gain)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Observation tensor

        Returns:
            Tuple of (mean, std) for Gaussian policy
        """
        features = self.mlp(obs)
        mean = self.mean_head(features)

        # Clamp pre-tanh mean to prevent TanhNormal saturation.
        # Without entropy regularization (alpha=0), the mean drifts unboundedly,
        # causing tanh to saturate and log_prob → -inf. tanh(5)=0.9999 preserves
        # full action range while preventing catastrophic saturation.
        mean = torch.clamp(mean, -5.0, 5.0)

        log_std = self.log_std_head(features)

        # Clamp log_std and convert to std
        log_std = torch.clamp(
            log_std,
            min=torch.log(torch.tensor(self.min_std, device=log_std.device)),
            max=torch.log(torch.tensor(self.max_std, device=log_std.device)),
        )
        std = torch.exp(log_std)

        return mean, std


class CategoricalActorNetwork(nn.Module):
    """MLP network for categorical (discrete) actor.

    Used for manager policy in hierarchical RL.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        ortho_init: bool = True,
        init_gain: float = 0.01,
        use_layer_norm: bool = False,
    ):
        """Initialize categorical actor network.

        Args:
            obs_dim: Observation dimension
            num_actions: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            ortho_init: Whether to use orthogonal initialization
            init_gain: Gain for final layer initialization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Build MLP layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output logits
        self.logits_head = nn.Linear(prev_dim, num_actions)

        # Apply orthogonal initialization
        if ortho_init:
            self._apply_ortho_init(init_gain)

    def _apply_ortho_init(self, final_gain: float) -> None:
        """Apply orthogonal initialization to layers."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.logits_head.weight, gain=final_gain)
        nn.init.constant_(self.logits_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Observation tensor

        Returns:
            Logits for categorical distribution
        """
        features = self.mlp(obs)
        logits = self.logits_head(features)
        return logits


def create_actor(
    obs_dim: int,
    action_spec: BoundedTensorSpec,
    config: Optional[ActorConfig] = None,
    device: str = "cpu",
) -> ProbabilisticActor:
    """Create TorchRL ProbabilisticActor.

    Args:
        obs_dim: Observation dimension
        action_spec: Action specification
        config: Actor configuration
        device: Device for network

    Returns:
        ProbabilisticActor module
    """
    config = config or ActorConfig()
    action_dim = action_spec.shape[-1]

    # Create base network
    net = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        min_std=config.min_std,
        max_std=config.max_std,
        init_std=config.init_std,
        ortho_init=config.ortho_init,
        init_gain=config.init_gain,
        use_layer_norm=config.use_layer_norm,
    ).to(device)

    # Wrap in TensorDictModule
    actor_module = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    # Create probabilistic actor with TanhNormal distribution
    actor = ProbabilisticActor(
        module=actor_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,
        default_interaction_type="random",
    )

    return actor


def create_deterministic_actor(
    obs_dim: int,
    action_spec: BoundedTensorSpec,
    config: Optional[ActorConfig] = None,
    device: str = "cpu",
) -> TensorDictModule:
    """Create deterministic actor for evaluation.

    Args:
        obs_dim: Observation dimension
        action_spec: Action specification
        config: Actor configuration
        device: Device for network

    Returns:
        TensorDictModule that outputs deterministic actions
    """
    config = config or ActorConfig()
    action_dim = action_spec.shape[-1]

    # Create network that only outputs mean
    class DeterministicActorNet(nn.Module):
        def __init__(self, base_net: ActorNetwork):
            super().__init__()
            self.base = base_net

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            mean, _ = self.base(obs)
            # Apply tanh to bound output
            return torch.tanh(mean)

    base_net = ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        ortho_init=config.ortho_init,
        init_gain=config.init_gain,
        use_layer_norm=config.use_layer_norm,
    ).to(device)

    det_net = DeterministicActorNet(base_net)

    return TensorDictModule(
        det_net,
        in_keys=["observation"],
        out_keys=["action"],
    )
