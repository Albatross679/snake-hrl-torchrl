"""Critic (value) networks for TorchRL."""

from typing import Optional, List
import torch
import torch.nn as nn

from torchrl.modules import ValueOperator
from tensordict.nn import TensorDictModule

from snake_hrl.configs.network import CriticConfig
from snake_hrl.networks.actor import get_activation


class CriticNetwork(nn.Module):
    """MLP network for critic (value function).

    Can be used for both state-value V(s) and action-value Q(s, a).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_outputs: int = 1,
        activation: str = "tanh",
        ortho_init: bool = True,
        init_gain: float = 1.0,
        use_layer_norm: bool = False,
    ):
        """Initialize critic network.

        Args:
            input_dim: Input dimension (obs_dim for V, obs_dim + action_dim for Q)
            hidden_dims: Hidden layer dimensions
            num_outputs: Number of outputs (1 for V, action_dim for Q)
            activation: Activation function name
            ortho_init: Whether to use orthogonal initialization
            init_gain: Gain for final layer initialization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_outputs = num_outputs

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output head
        self.value_head = nn.Linear(prev_dim, num_outputs)

        # Apply orthogonal initialization
        if ortho_init:
            self._apply_ortho_init(init_gain)

    def _apply_ortho_init(self, final_gain: float) -> None:
        """Apply orthogonal initialization to layers."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=final_gain)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (observation or observation + action)

        Returns:
            Value estimate
        """
        features = self.mlp(x)
        value = self.value_head(features)
        return value


class QNetwork(nn.Module):
    """Q-network that takes observation and action as separate inputs."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        ortho_init: bool = True,
        init_gain: float = 1.0,
        use_layer_norm: bool = False,
    ):
        """Initialize Q-network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            ortho_init: Whether to use orthogonal initialization
            init_gain: Gain for final layer initialization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Observation encoder
        obs_layers = []
        prev_dim = obs_dim
        first_hidden = hidden_dims[0] if hidden_dims else 256

        obs_layers.append(nn.Linear(prev_dim, first_hidden))
        if use_layer_norm:
            obs_layers.append(nn.LayerNorm(first_hidden))
        obs_layers.append(get_activation(activation))

        self.obs_encoder = nn.Sequential(*obs_layers)

        # Combined layers (after concatenating encoded obs + action)
        combined_layers = []
        prev_dim = first_hidden + action_dim

        for hidden_dim in hidden_dims[1:]:
            combined_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                combined_layers.append(nn.LayerNorm(hidden_dim))
            combined_layers.append(get_activation(activation))
            prev_dim = hidden_dim

        self.combined = nn.Sequential(*combined_layers)

        # Output head
        self.q_head = nn.Linear(prev_dim, 1)

        if ortho_init:
            self._apply_ortho_init(init_gain)

    def _apply_ortho_init(self, final_gain: float) -> None:
        """Apply orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.q_head.weight, gain=final_gain)
        nn.init.constant_(self.q_head.bias, 0.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Observation tensor
            action: Action tensor

        Returns:
            Q-value estimate
        """
        obs_features = self.obs_encoder(obs)
        combined = torch.cat([obs_features, action], dim=-1)
        features = self.combined(combined)
        q_value = self.q_head(features)
        return q_value


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC (min of two Q-functions)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        ortho_init: bool = True,
        use_layer_norm: bool = False,
    ):
        """Initialize twin Q-networks.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            ortho_init: Whether to use orthogonal initialization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.q1 = QNetwork(
            obs_dim, action_dim, hidden_dims, activation, ortho_init, 1.0, use_layer_norm
        )
        self.q2 = QNetwork(
            obs_dim, action_dim, hidden_dims, activation, ortho_init, 1.0, use_layer_norm
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q-networks.

        Args:
            obs: Observation tensor
            action: Action tensor

        Returns:
            Tuple of (Q1, Q2) values
        """
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return q1, q2

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get minimum of both Q-values.

        Args:
            obs: Observation tensor
            action: Action tensor

        Returns:
            Minimum Q-value
        """
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


def create_critic(
    obs_dim: int,
    config: Optional[CriticConfig] = None,
    device: str = "cpu",
) -> ValueOperator:
    """Create TorchRL ValueOperator for state-value function V(s).

    Args:
        obs_dim: Observation dimension
        config: Critic configuration
        device: Device for network

    Returns:
        ValueOperator module
    """
    config = config or CriticConfig()

    # Create base network
    net = CriticNetwork(
        input_dim=obs_dim,
        hidden_dims=config.hidden_dims,
        num_outputs=1,
        activation=config.activation,
        ortho_init=config.ortho_init,
        init_gain=config.init_gain,
        use_layer_norm=config.use_layer_norm,
    ).to(device)

    # Wrap in ValueOperator
    critic = ValueOperator(
        module=net,
        in_keys=["observation"],
    )

    return critic


def create_q_critic(
    obs_dim: int,
    action_dim: int,
    config: Optional[CriticConfig] = None,
    device: str = "cpu",
) -> TensorDictModule:
    """Create Q-function critic for SAC.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        config: Critic configuration
        device: Device for network

    Returns:
        TensorDictModule for Q-function
    """
    config = config or CriticConfig()

    # Create twin Q-networks
    net = TwinQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        ortho_init=config.ortho_init,
        use_layer_norm=config.use_layer_norm,
    ).to(device)

    # Custom module to handle TensorDict interface
    class QCriticModule(nn.Module):
        def __init__(self, twin_q: TwinQNetwork):
            super().__init__()
            self.twin_q = twin_q

        def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return self.twin_q.min_q(obs, action)

    module = QCriticModule(net)

    return TensorDictModule(
        module,
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )
