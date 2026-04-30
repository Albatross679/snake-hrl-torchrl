"""DD-PINN model and Fourier feature embedding.

DDPINNModel wraps a neural network with the DampedSinusoidalAnsatz to provide
the same forward(state, action, time_encoding) -> delta interface as existing
surrogates (SurrogateModel, ResidualSurrogateModel).

FourierFeatureEmbedding mitigates spectral bias by mapping inputs through
random Fourier features before the MLP.

Reference: Krauss et al., arXiv:2408.14951 (DD-PINN)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.pinn.ansatz import DampedSinusoidalAnsatz


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier feature embedding for spectral bias mitigation.

    Maps input x to [sin(2*pi*x@B), cos(2*pi*x@B)] where B is a fixed
    random matrix. Output dimension is 2 * n_features.

    Reference: Tancik et al., "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains" (NeurIPS 2020).
    """

    def __init__(self, input_dim: int, n_features: int = 256, sigma: float = 10.0):
        super().__init__()
        self.output_dim = 2 * n_features
        # Fixed random projection (not learned)
        self.register_buffer(
            "B", torch.randn(input_dim, n_features) * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, input_dim) -> (B, 2 * n_features)."""
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DDPINNModel(nn.Module):
    """DD-PINN model: neural network + damped sinusoidal ansatz.

    The NN maps (state, action, time_encoding) to ansatz parameters.
    The ansatz evaluates at t=dt to produce the state delta.

    Provides the same forward() interface as SurrogateModel for
    drop-in compatibility with the existing pipeline.

    Args:
        state_dim: State vector dimension (130 for relative state).
        action_dim: Action dimension (5).
        time_encoding_dim: Time encoding dimension (4).
        n_basis: Number of ansatz basis functions per state dim.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
        n_fourier: Number of Fourier features (output dim = 2 * n_fourier).
        fourier_sigma: Fourier feature frequency scale.
        dt: Simulation timestep (0.5s for this project).
    """

    def __init__(
        self,
        state_dim: int = 130,
        action_dim: int = 5,
        time_encoding_dim: int = 4,
        n_basis: int = 5,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_fourier: int = 128,
        fourier_sigma: float = 10.0,
        dt: float = 0.5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.config = None  # Placeholder for SurrogateModelConfig compatibility

        # Ansatz
        self.ansatz = DampedSinusoidalAnsatz(state_dim, n_basis)

        # Fourier feature embedding
        raw_input_dim = state_dim + action_dim + time_encoding_dim
        self.fourier = FourierFeatureEmbedding(raw_input_dim, n_fourier, fourier_sigma)

        # MLP with residual connections
        self.input_proj = nn.Linear(self.fourier.output_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        # Output layer: small random init to avoid gradient deadlock.
        # The ansatz guarantees g(a,0)=0 for ANY params, so zero-init is not
        # needed for IC satisfaction. Zero-init causes all ansatz params to be
        # zero, making alpha=0 which blocks all gradient flow (∂g/∂params = 0).
        self.output = nn.Linear(hidden_dim, self.ansatz.param_dim)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

        # Init hidden layers
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _encode(self, state: torch.Tensor, action: torch.Tensor,
                time_encoding: torch.Tensor) -> torch.Tensor:
        """Encode inputs to ansatz parameters via Fourier features + MLP.

        Returns:
            (B, ansatz.param_dim) parameter vector.
        """
        x = torch.cat([state, action, time_encoding], dim=-1)  # (B, 139)
        h = torch.tanh(self.input_proj(self.fourier(x)))  # (B, hidden_dim)
        for layer in self.layers:
            h = h + torch.tanh(layer(h))  # Residual
        return self.output(h)  # (B, ansatz.param_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict state delta at t=dt.

        Same signature as SurrogateModel.forward().

        Args:
            state: (B, state_dim) normalized rod state.
            action: (B, action_dim) raw action.
            time_encoding: (B, time_encoding_dim) time features.

        Returns:
            (B, state_dim) predicted state delta.
        """
        params = self._encode(state, action, time_encoding)
        t_dt = torch.tensor([self.dt], device=state.device, dtype=state.dtype)
        delta = self.ansatz(params, t_dt)  # (B, 1, state_dim)
        return delta.squeeze(1)  # (B, state_dim)

    def forward_trajectory(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
        t_colloc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate ansatz at multiple collocation points.

        Used during DD-PINN training for physics residual computation.

        Args:
            state: (B, state_dim) normalized rod state.
            action: (B, action_dim) raw action.
            time_encoding: (B, time_encoding_dim) time features.
            t_colloc: (N_c,) collocation time points.

        Returns:
            g: (B, N_c, state_dim) state deviations at collocation times.
            g_dot: (B, N_c, state_dim) time derivatives (closed-form).
        """
        params = self._encode(state, action, time_encoding)
        g = self.ansatz(params, t_colloc)
        g_dot = self.ansatz.time_derivative(params, t_colloc)
        return g, g_dot

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        time_encoding: torch.Tensor,
        normalizer=None,
    ) -> torch.Tensor:
        """Predict next state with optional normalization handling.

        Same pattern as SurrogateModel.predict_next_state().

        Args:
            state: (B, state_dim) raw (unnormalized) rod state.
            action: (B, action_dim) raw action.
            time_encoding: (B, time_encoding_dim) time features.
            normalizer: StateNormalizer instance (or None to skip).

        Returns:
            (B, state_dim) predicted next rod state (unnormalized).
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
