"""Domain-Decoupled PINN surrogate for Cosserat rod dynamics (Licher et al., 2025).

The DD-PINN (Krauss & Habich, arXiv:2408.14951) decouples time from the
feedforward neural network:

    x_hat_t = g(f_NN(x_0, u_0, theta), t) + x_0

where f_NN predicts parameters (alpha, beta, gamma) for the Ansatz function:

    g_j(t) = sum_i alpha_ij * (sin(beta_ij * t + gamma_ij) - sin(gamma_ij))

This allows closed-form analytic gradient computation w.r.t. time, achieving
44,000x speedup over the numerical Cosserat rod solver.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from licher2025.configs_licher2025 import PINNConfig


class AnsatzLayer(nn.Module):
    """Ansatz function that evaluates predicted sinusoidal parameters at time t.

    For each state dimension j, computes:
        g_j(t) = sum_i alpha_ij * (sin(beta_ij * t + gamma_ij) - sin(gamma_ij))

    Optionally with exponential damping:
        g_j(t) = sum_i alpha_ij * exp(-delta_ij * t) * (sin(...) - sin(...))
    """

    def __init__(self, state_dim: int, num_terms: int, use_damping: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.num_terms = num_terms
        self.use_damping = use_damping

        # Parameters per state dim: 3 (alpha, beta, gamma) or 4 (+delta) per term
        self.params_per_term = 4 if use_damping else 3

    @property
    def total_params(self) -> int:
        """Number of Ansatz parameters the NN must predict."""
        return self.state_dim * self.num_terms * self.params_per_term

    def forward(
        self, ansatz_params: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the Ansatz at time t.

        Args:
            ansatz_params: Predicted parameters, shape (batch, total_params).
            t: Time values, shape (batch, 1) or (batch,).

        Returns:
            State prediction x_hat - x_0, shape (batch, state_dim).
        """
        batch = ansatz_params.shape[0]
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch, 1)

        # Reshape to (batch, state_dim, num_terms, params_per_term)
        params = ansatz_params.view(
            batch, self.state_dim, self.num_terms, self.params_per_term
        )

        alpha = params[..., 0]  # (batch, state_dim, num_terms)
        beta = params[..., 1]
        gamma = params[..., 2]

        # t broadcast: (batch, 1, 1)
        t_expanded = t.unsqueeze(-1)

        # g_j(t) = sum_i alpha_ij * (sin(beta_ij * t + gamma_ij) - sin(gamma_ij))
        phase = beta * t_expanded + gamma
        g = alpha * (torch.sin(phase) - torch.sin(gamma))

        if self.use_damping:
            delta = torch.abs(params[..., 3])  # Ensure positive damping
            g = g * torch.exp(-delta * t_expanded)

        # Sum over terms: (batch, state_dim)
        return g.sum(dim=-1)


class DomainDecoupledPINN(nn.Module):
    """DD-PINN: predicts Cosserat rod state at arbitrary future times.

    Architecture:
        1. f_NN: MLP mapping (x_0, u_0, theta) -> Ansatz parameters
        2. g: Closed-form Ansatz evaluating parameters at time t
        3. Output: x_hat_t = g(f_NN(x_0, u_0, theta), t) + x_0

    The time decoupling allows:
    - Analytic time derivatives (no autodiff needed for physics loss)
    - Evaluation at arbitrary t without re-running the network
    - 44,000x speedup over numerical Cosserat solver
    """

    def __init__(self, config: PINNConfig = None):
        super().__init__()
        self.config = config or PINNConfig()

        state_dim = self.config.state_dim
        control_dim = self.config.control_dim
        param_dim = self.config.param_dim

        # Input: x_0 (state) + u_0 (control) + theta (parameters)
        input_dim = state_dim + control_dim + param_dim

        # Ansatz layer
        self.ansatz = AnsatzLayer(
            state_dim=state_dim,
            num_terms=self.config.ansatz.num_ansatz_terms,
            use_damping=self.config.ansatz.use_exponential_damping,
        )

        # Build MLP: input -> hidden layers -> ansatz params
        layers = []
        dims = [input_dim] + self.config.hidden_dims
        activation = nn.GELU()

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)

        layers.append(nn.Linear(dims[-1], self.ansatz.total_params))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor,
        theta: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict state at time t given initial conditions.

        Args:
            x0: Initial state, shape (batch, state_dim).
            u0: Control input, shape (batch, control_dim).
            theta: Physical parameters, shape (batch, param_dim).
            t: Prediction time, shape (batch,) or (batch, 1).

        Returns:
            Predicted state x_hat_t, shape (batch, state_dim).
        """
        # Concatenate inputs for MLP
        mlp_input = torch.cat([x0, u0, theta], dim=-1)

        # Predict Ansatz parameters
        ansatz_params = self.mlp(mlp_input)

        # Evaluate Ansatz at time t and add initial state
        delta_x = self.ansatz(ansatz_params, t)
        return x0 + delta_x

    def predict_trajectory(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor,
        theta: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Predict states at multiple times (single MLP forward pass).

        Args:
            x0: Initial state, shape (batch, state_dim).
            u0: Control input, shape (batch, control_dim).
            theta: Physical parameters, shape (batch, param_dim).
            times: Time points, shape (num_times,).

        Returns:
            Predicted trajectory, shape (batch, num_times, state_dim).
        """
        batch = x0.shape[0]
        num_times = times.shape[0]

        # Single MLP pass
        mlp_input = torch.cat([x0, u0, theta], dim=-1)
        ansatz_params = self.mlp(mlp_input)

        # Evaluate at each time
        results = []
        for i in range(num_times):
            t_i = times[i].expand(batch)
            delta_x = self.ansatz(ansatz_params, t_i)
            results.append(x0 + delta_x)

        return torch.stack(results, dim=1)

    def time_derivative(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor,
        theta: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute analytic time derivative dx/dt (for physics loss).

        Uses closed-form derivative of the Ansatz:
            dg_j/dt = sum_i alpha_ij * beta_ij * cos(beta_ij * t + gamma_ij)
            (plus damping terms if enabled)

        Args:
            x0: Initial state, shape (batch, state_dim).
            u0: Control input, shape (batch, control_dim).
            theta: Physical parameters, shape (batch, param_dim).
            t: Time, shape (batch,) or (batch, 1).

        Returns:
            Time derivative, shape (batch, state_dim).
        """
        mlp_input = torch.cat([x0, u0, theta], dim=-1)
        ansatz_params = self.mlp(mlp_input)

        batch = ansatz_params.shape[0]
        cfg = self.config
        state_dim = cfg.state_dim
        num_terms = cfg.ansatz.num_ansatz_terms
        ppt = self.ansatz.params_per_term

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        params = ansatz_params.view(batch, state_dim, num_terms, ppt)
        alpha = params[..., 0]
        beta = params[..., 1]
        gamma = params[..., 2]

        t_expanded = t.unsqueeze(-1)
        phase = beta * t_expanded + gamma

        # dg/dt = alpha * beta * cos(phase)
        dg_dt = alpha * beta * torch.cos(phase)

        if self.ansatz.use_damping:
            delta = torch.abs(params[..., 3])
            exp_term = torch.exp(-delta * t_expanded)
            sin_diff = torch.sin(phase) - torch.sin(gamma)
            # Product rule: d/dt [exp(-dt) * sin_diff]
            dg_dt = exp_term * dg_dt - delta * exp_term * alpha * sin_diff

        return dg_dt.sum(dim=-1)
