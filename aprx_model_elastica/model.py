"""Surrogate MLP model for Cosserat rod dynamics.

Architecture:
    Input:  [state_normalized(124) | action(5) | sin_cos_time(2)] = 131
    Hidden: 3 × 512 with LayerNorm + SiLU
    Output: state delta (124), no activation
    Prediction: next_state = current_state + denormalize(model(input))
"""

import torch
import torch.nn as nn

from aprx_model_elastica.train_config import SurrogateModelConfig


class SurrogateModel(nn.Module):
    """MLP surrogate for one RL step of PyElastica Cosserat rod dynamics.

    Predicts the state delta (next_state - current_state) from the
    normalized current state, action, and serpenoid time encoding.
    """

    def __init__(self, config: SurrogateModelConfig = None):
        super().__init__()
        if config is None:
            config = SurrogateModelConfig()
        self.config = config

        # Build MLP layers
        layers = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        # Output layer (no activation — deltas can be positive or negative)
        layers.append(nn.Linear(in_dim, config.output_dim))
        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]()

    def _init_weights(self):
        """Xavier init for hidden layers, zero init for output layer."""
        for i, module in enumerate(self.mlp):
            if isinstance(module, nn.Linear):
                if module is self.mlp[-1]:
                    # Output layer: zero init → initial prediction is zero delta
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict state delta.

        Args:
            state: (B, 124) normalized rod state.
            action: (B, 5) raw action in [-1, 1].
            time_encoding: (B, 2) [sin(t), cos(t)].

        Returns:
            (B, 124) predicted state delta (in normalized space if trained
            with normalized targets).
        """
        x = torch.cat([state, action, time_encoding], dim=-1)  # (B, 131)
        return self.mlp(x)  # (B, 124)

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
        normalizer=None,
    ) -> torch.Tensor:
        """Predict next state with optional normalization handling.

        Args:
            state: (B, 124) raw (unnormalized) rod state.
            action: (B, 5) raw action in [-1, 1].
            time_encoding: (B, 2) [sin(t), cos(t)].
            normalizer: StateNormalizer instance (or None to skip normalization).

        Returns:
            (B, 124) predicted next rod state (unnormalized).
        """
        if normalizer is not None:
            state_norm = normalizer.normalize_state(state)
        else:
            state_norm = state

        delta_norm = self.forward(state_norm, action, time_encoding)

        if normalizer is not None:
            delta = normalizer.denormalize_delta(delta_norm)
        else:
            delta = delta_norm

        return state + delta


class ResidualBlock(nn.Module):
    """Two linear layers with LayerNorm, SiLU, and a skip connection.

    Requires input_dim == output_dim (no projection layer).
    skip: output = act(norm2(linear2(act(norm1(linear1(x))))) + x)
    """

    def __init__(self, dim: int, use_layer_norm: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.act = nn.SiLU()
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.linear1(x)))
        out = self.norm2(self.linear2(out))
        return self.act(out + x)


class ResidualSurrogateModel(nn.Module):
    """Surrogate model using residual MLP blocks.

    Architecture:
        Input projection: Linear(131, hidden_dim) + LayerNorm + SiLU
        Residual blocks:  floor(n_hidden/2) x ResidualBlock(hidden_dim)
        Extra layer:      (if n_hidden is odd) Linear + LayerNorm + SiLU
        Output:           Linear(hidden_dim, 124), zero-initialized

    Requires uniform hidden dims (all elements in hidden_dims must be equal).
    """

    def __init__(self, config: "SurrogateModelConfig" = None):
        super().__init__()
        if config is None:
            from aprx_model_elastica.train_config import SurrogateModelConfig
            config = SurrogateModelConfig()
        self.config = config

        assert len(set(config.hidden_dims)) == 1, (
            f"ResidualSurrogateModel requires uniform hidden dims, "
            f"got {config.hidden_dims}"
        )
        n_hidden = len(config.hidden_dims)
        hidden_dim = config.hidden_dims[0]

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity()
        self.input_act = nn.SiLU()
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Residual blocks (pair up hidden layers)
        n_blocks = n_hidden // 2
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, use_layer_norm=config.use_layer_norm)
            for _ in range(n_blocks)
        ])

        # Extra plain layer if n_hidden is odd
        self.extra_layer = None
        if n_hidden % 2 == 1:
            self.extra_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity(),
                nn.SiLU(),
            )
            nn.init.xavier_uniform_(self.extra_layer[0].weight)
            nn.init.zeros_(self.extra_layer[0].bias)

        # Output layer: zero-initialized so initial prediction is zero delta
        self.output_layer = nn.Linear(hidden_dim, config.output_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict state delta.

        Args:
            state: (B, 124) normalized rod state.
            action: (B, 5) raw action in [-1, 1].
            time_encoding: (B, 2) [sin(t), cos(t)].

        Returns:
            (B, 124) predicted state delta.
        """
        x = torch.cat([state, action, time_encoding], dim=-1)  # (B, 131)
        x = self.input_act(self.input_norm(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)
        if self.extra_layer is not None:
            x = self.extra_layer(x)
        return self.output_layer(x)

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
        normalizer=None,
    ) -> torch.Tensor:
        """Predict next state with optional normalization handling.

        Args:
            state: (B, 124) raw (unnormalized) rod state.
            action: (B, 5) raw action in [-1, 1].
            time_encoding: (B, 2) [sin(t), cos(t)].
            normalizer: StateNormalizer instance (or None to skip normalization).

        Returns:
            (B, 124) predicted next rod state (unnormalized).
        """
        if normalizer is not None:
            state_norm = normalizer.normalize_state(state)
        else:
            state_norm = state

        delta_norm = self.forward(state_norm, action, time_encoding)

        if normalizer is not None:
            delta = normalizer.denormalize_delta(delta_norm)
        else:
            delta = delta_norm

        return state + delta


class HistorySurrogateModel(nn.Module):
    """Surrogate model with a history window of K prior transitions.

    Architecture:
        Input: cat([state, action, time_enc, history_flat])
               = 131 + K * (state_dim + action_dim) = 131 + K*129
        MLP:   same hidden_dims/activation/layernorm as SurrogateModel
        Output: (B, 124) state delta, zero-initialized output layer

    The forward() method takes explicit history tensors rather than relying
    on the caller to pre-concatenate them.
    """

    def __init__(self, config: "SurrogateModelConfig" = None, history_k: int = 2):
        super().__init__()
        if config is None:
            from aprx_model_elastica.train_config import SurrogateModelConfig
            config = SurrogateModelConfig()
        self.config = config
        self.history_k = history_k

        # Compute extended input dim: 131 + K*(124+5)
        history_dim = history_k * (config.state_dim + config.action_dim)
        extended_input_dim = config.input_dim + history_dim  # 131 + K*129

        # Build plain MLP with the extended input dim
        layers = []
        in_dim = extended_input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        # Output layer: zero-initialized
        output_layer = nn.Linear(in_dim, config.output_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.mlp = nn.Sequential(*layers)

        # Xavier init for hidden layers
        for module in self.mlp:
            if isinstance(module, nn.Linear) and module is not self.mlp[-1]:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict state delta with history context.

        Args:
            state: (B, 124) normalized current state.
            action: (B, 5) current action.
            time_encoding: (B, 2) [sin(t), cos(t)].
            history_states: (B, K, 124) K prior states (oldest first).
            history_actions: (B, K, 5) K prior actions (oldest first).

        Returns:
            (B, 124) predicted state delta.
        """
        # Flatten history: (B, K*124) and (B, K*5)
        history_states_flat = history_states.flatten(-2, -1)    # (B, K*124)
        history_actions_flat = history_actions.flatten(-2, -1)  # (B, K*5)
        history_flat = torch.cat([history_states_flat, history_actions_flat], dim=-1)  # (B, K*129)

        x = torch.cat([state, action, time_encoding, history_flat], dim=-1)
        return self.mlp(x)

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        normalizer=None,
    ) -> torch.Tensor:
        """Predict next state with optional normalization handling.

        Args:
            state: (B, 124) raw (unnormalized) rod state.
            action: (B, 5) raw action in [-1, 1].
            time_encoding: (B, 2) [sin(t), cos(t)].
            history_states: (B, K, 124) K prior raw states.
            history_actions: (B, K, 5) K prior actions.
            normalizer: StateNormalizer instance (or None).

        Returns:
            (B, 124) predicted next rod state (unnormalized).
        """
        if normalizer is not None:
            state_norm = normalizer.normalize_state(state)
        else:
            state_norm = state

        delta_norm = self.forward(state_norm, action, time_encoding, history_states, history_actions)

        if normalizer is not None:
            delta = normalizer.denormalize_delta(delta_norm)
        else:
            delta = delta_norm

        return state + delta
