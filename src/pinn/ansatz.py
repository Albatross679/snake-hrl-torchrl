"""Damped sinusoidal ansatz for DD-PINN.

The ansatz g(a, t) is a linear combination of damped sinusoidal basis
functions that guarantees exact initial condition satisfaction:
g(a, 0) = 0 for any parameter values.

Each basis function:
    g_i(t) = alpha * exp(-delta * t) * [sin(beta * t + gamma) - sin(gamma)]

At t=0: sin(gamma) - sin(gamma) = 0, so g(a, 0) = 0 exactly.

The closed-form time derivative avoids expensive autodiff:
    dg_i/dt = alpha * exp(-delta * t) *
              [beta * cos(beta * t + gamma) - delta * (sin(beta * t + gamma) - sin(gamma))]

Reference: Krauss et al., arXiv:2408.14951 (DD-PINN)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DampedSinusoidalAnsatz(nn.Module):
    """Damped sinusoidal ansatz with exact IC satisfaction.

    Parameters:
        state_dim: Dimension of the state vector (130 for relative state).
        n_basis: Number of basis functions per state dimension.

    The neural network outputs a parameter vector of shape
    (B, 4 * state_dim * n_basis) which is parsed into
    (alpha, delta_damp, beta, gamma) each of shape (B, state_dim, n_basis).
    """

    def __init__(self, state_dim: int = 130, n_basis: int = 5):
        super().__init__()
        self.state_dim = state_dim
        self.n_basis = n_basis
        self.param_dim = 4 * state_dim * n_basis

    def _parse_params(self, params: torch.Tensor):
        """Parse flat parameter vector into (alpha, delta_damp, beta, gamma).

        Args:
            params: (B, 4 * m * n_g) flat parameter vector.

        Returns:
            Tuple of (alpha, delta_damp, beta, gamma), each (B, m, n_g).
        """
        B = params.shape[0]
        m, n_g = self.state_dim, self.n_basis
        # Reshape to (B, 4, m, n_g) then split
        p = params.view(B, 4, m, n_g)
        alpha = p[:, 0]
        delta_damp = F.softplus(p[:, 1])  # Ensure positive damping
        beta = p[:, 2]
        gamma = p[:, 3]
        return alpha, delta_damp, beta, gamma

    def forward(self, params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate ansatz at given time points.

        Args:
            params: (B, 4 * m * n_g) from neural network.
            t: (N_c,) collocation times, or scalar tensor.

        Returns:
            (B, N_c, m) if t has multiple points, or (B, m) if t is scalar.
        """
        alpha, delta_damp, beta, gamma = self._parse_params(params)
        # alpha, delta_damp, beta, gamma: (B, m, n_g)

        scalar_t = t.ndim == 0
        if scalar_t:
            t = t.unsqueeze(0)

        # t_exp: (1, N_c, 1, 1)
        t_exp = t[None, :, None, None]

        # Broadcast: (B, N_c, m, n_g)
        phase = beta[:, None, :, :] * t_exp + gamma[:, None, :, :]
        damping = torch.exp(-delta_damp[:, None, :, :] * t_exp)
        sin_gamma = torch.sin(gamma[:, None, :, :])

        # g = sum over n_g: alpha * exp(-delta*t) * (sin(phase) - sin(gamma))
        g = (alpha[:, None, :, :] * damping * (torch.sin(phase) - sin_gamma)).sum(-1)
        # g: (B, N_c, m)

        if scalar_t:
            return g.squeeze(1)  # (B, m)
        return g

    def time_derivative(self, params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Closed-form time derivative of the ansatz.

        dg/dt = alpha * exp(-delta*t) * [beta*cos(beta*t + gamma)
                - delta * (sin(beta*t + gamma) - sin(gamma))]

        Args:
            params: (B, 4 * m * n_g) from neural network.
            t: (N_c,) collocation times, or scalar tensor.

        Returns:
            (B, N_c, m) if t has multiple points, or (B, m) if t is scalar.
        """
        alpha, delta_damp, beta, gamma = self._parse_params(params)

        scalar_t = t.ndim == 0
        if scalar_t:
            t = t.unsqueeze(0)

        t_exp = t[None, :, None, None]

        phase = beta[:, None, :, :] * t_exp + gamma[:, None, :, :]
        damping = torch.exp(-delta_damp[:, None, :, :] * t_exp)
        sin_gamma = torch.sin(gamma[:, None, :, :])

        # dg/dt = alpha * exp(-delta*t) * [beta*cos(phase) - delta*(sin(phase) - sin(gamma))]
        g_dot = (
            alpha[:, None, :, :]
            * damping
            * (
                beta[:, None, :, :] * torch.cos(phase)
                - delta_damp[:, None, :, :] * (torch.sin(phase) - sin_gamma)
            )
        ).sum(-1)

        if scalar_t:
            return g_dot.squeeze(1)
        return g_dot
