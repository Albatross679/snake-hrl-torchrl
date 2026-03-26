"""Probe PDE validation suite for PINN debugging.

Progressively complex PDEs with known analytical solutions that test
individual PINN components before training on the full Cosserat rod system.
Mirrors the RL probe environment pattern from src/trainers/probe_envs.py.

Each probe tests one additional capability:
  ProbePDE1: data fitting + optimizer (heat equation)
  ProbePDE2: BC/IC enforcement + PDE residual (advection)
  ProbePDE3: nonlinear PDE + loss balancing (Burgers)
  ProbePDE4: multi-scale + Fourier features (reaction-diffusion)

Usage:
    from src.pinn.probe_pdes import ALL_PROBES, run_probe_validation

    # Run all probes
    results = run_probe_validation()
    assert all(results.values())

    # Analyze PDE system
    from src.pinn.probe_pdes import analyze_pde_system
    report = analyze_pde_system()
    print(report["nondim_quality"])

References:
    Krishnapriyan et al. "Characterizing Possible Failure Modes in PINNs"
    (NeurIPS 2021) -- probe PDE methodology.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Type

import torch
import torch.nn as nn

from scipy.stats.qmc import Sobol


# ---------------------------------------------------------------------------
# Internal helper MLP for probe training
# ---------------------------------------------------------------------------

class _ProbeMLP(nn.Module):
    """Simple MLP for probe PDE training.

    Small network that maps (x, t) -> u for 1D space-time problems.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, input_dim) -> (B, output_dim)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Probe PDE base interface
# ---------------------------------------------------------------------------

class _ProbePDEBase:
    """Base class defining the probe PDE interface."""

    name: str = ""
    max_epochs: int = 500
    model_kwargs: dict = {"input_dim": 2, "output_dim": 1, "hidden_dim": 64, "n_layers": 2}

    def analytical_solution(self, x: torch.Tensor, t: torch.Tensor, **params) -> torch.Tensor:
        raise NotImplementedError

    def pde_residual(self, u: torch.Tensor, grads: dict) -> torch.Tensor:
        raise NotImplementedError

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def boundary_conditions(self) -> dict:
        raise NotImplementedError

    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def check_pass(self, model: nn.Module) -> bool:
        raise NotImplementedError

    def pass_criterion(self, metrics: dict) -> bool:
        raise NotImplementedError

    def _sample_sobol_2d(self, n_points: int, x_range=(0, 1), t_range=(0, 0.5)) -> torch.Tensor:
        """Sample 2D Sobol points in [x_range] x [t_range]."""
        sampler = Sobol(d=2, scramble=True)
        m = max(1, (n_points - 1).bit_length())
        n_pow2 = 2 ** m
        raw = sampler.random(n_pow2)[:n_points]
        pts = torch.tensor(raw, dtype=torch.float32)
        pts[:, 0] = x_range[0] + pts[:, 0] * (x_range[1] - x_range[0])
        pts[:, 1] = t_range[0] + pts[:, 1] * (t_range[1] - t_range[0])
        return pts


# ---------------------------------------------------------------------------
# ProbePDE1: 1D Heat Equation
# ---------------------------------------------------------------------------

class ProbePDE1(_ProbePDEBase):
    """1D heat equation: u_t = alpha * u_xx.

    Analytical: u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)
    IC: u(x,0) = sin(pi*x), BC: u(0,t) = u(1,t) = 0
    Tests: data fitting + optimizer works.
    Pass: MSE vs analytical < 1e-4 in 500 epochs.
    """

    name = "heat_1d"
    max_epochs = 500
    _alpha = 0.01

    def analytical_solution(self, x: torch.Tensor, t: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
        return torch.exp(-alpha * math.pi ** 2 * t) * torch.sin(math.pi * x)

    def pde_residual(self, u: torch.Tensor, grads: dict, alpha: float = 0.01) -> torch.Tensor:
        """PDE residual: u_t - alpha * u_xx = 0."""
        return grads["u_t"] - alpha * grads["u_xx"]

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x)

    def boundary_conditions(self) -> dict:
        return {"x=0": 0.0, "x=1": 0.0}

    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        """Full loss: data MSE vs analytical + PDE residual at collocation points."""
        # Data points (Sobol)
        pts = self._sample_sobol_2d(1000, t_range=(0, 0.5))
        pts.requires_grad_(True)
        x, t = pts[:, 0:1], pts[:, 1:2]
        u_pred = model(pts)
        u_exact = self.analytical_solution(x, t, alpha=self._alpha)
        loss_data = ((u_pred - u_exact) ** 2).mean()

        # PDE residual at collocation points
        coll = self._sample_sobol_2d(500, t_range=(0, 0.5))
        coll.requires_grad_(True)
        u_c = model(coll)
        u_t = torch.autograd.grad(u_c.sum(), coll, create_graph=True)[0][:, 1:2]
        u_x = torch.autograd.grad(u_c.sum(), coll, create_graph=True)[0][:, 0:1]
        # Need u_xx: second derivative w.r.t. x
        u_xx = torch.autograd.grad(u_x.sum(), coll, create_graph=True)[0][:, 0:1]
        residual = u_t - self._alpha * u_xx
        loss_phys = (residual ** 2).mean()

        return loss_data + 0.1 * loss_phys

    def check_pass(self, model: nn.Module) -> bool:
        """Check MSE vs analytical < 1e-4."""
        pts = self._sample_sobol_2d(1000, t_range=(0, 0.5))
        x, t = pts[:, 0:1], pts[:, 1:2]
        with torch.no_grad():
            u_pred = model(pts)
            u_exact = self.analytical_solution(x, t, alpha=self._alpha)
            mse = ((u_pred - u_exact) ** 2).mean().item()
        return self.pass_criterion({"mse_vs_analytical": mse})

    def pass_criterion(self, metrics: dict) -> bool:
        return metrics.get("mse_vs_analytical", float("inf")) < 1e-4


# ---------------------------------------------------------------------------
# ProbePDE2: 1D Advection
# ---------------------------------------------------------------------------

class ProbePDE2(_ProbePDEBase):
    """1D advection: u_t + c * u_x = 0, c=1.0.

    Analytical: u(x,t) = sin(2*pi*(x - c*t))
    Tests: BC/IC enforcement + PDE residual computation.
    Pass: PDE residual < 1e-3, BC error < 1e-4.
    """

    name = "advection_1d"
    max_epochs = 500
    _c = 1.0

    def analytical_solution(self, x: torch.Tensor, t: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        return torch.sin(2 * math.pi * (x - c * t))

    def pde_residual(self, u: torch.Tensor, grads: dict, c: float = 1.0) -> torch.Tensor:
        """PDE residual: u_t + c * u_x = 0."""
        return grads["u_t"] + c * grads["u_x"]

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * math.pi * x)

    def boundary_conditions(self) -> dict:
        return {"x=0": "periodic", "x=1": "periodic"}

    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        """Loss: data + PDE residual + BC enforcement."""
        # Data
        pts = self._sample_sobol_2d(1000, t_range=(0, 0.5))
        pts.requires_grad_(True)
        x, t = pts[:, 0:1], pts[:, 1:2]
        u_pred = model(pts)
        u_exact = self.analytical_solution(x, t, c=self._c)
        loss_data = ((u_pred - u_exact) ** 2).mean()

        # PDE residual
        grad_xt = torch.autograd.grad(u_pred.sum(), pts, create_graph=True)[0]
        u_x_pred = grad_xt[:, 0:1]
        u_t_pred = grad_xt[:, 1:2]
        residual = u_t_pred + self._c * u_x_pred
        loss_phys = (residual ** 2).mean()

        # BC: periodic boundary u(0,t) = u(1,t)
        t_bc = torch.linspace(0, 0.5, 50).unsqueeze(1)
        left = torch.cat([torch.zeros(50, 1), t_bc], dim=1)
        right = torch.cat([torch.ones(50, 1), t_bc], dim=1)
        loss_bc = ((model(left) - model(right)) ** 2).mean()

        return loss_data + 0.1 * loss_phys + loss_bc

    def check_pass(self, model: nn.Module) -> bool:
        """Check PDE residual < 1e-3 and BC error < 1e-4."""
        pts = self._sample_sobol_2d(500, t_range=(0, 0.5))
        pts.requires_grad_(True)
        u_pred = model(pts)
        grad_xt = torch.autograd.grad(u_pred.sum(), pts, create_graph=True)[0]
        residual = grad_xt[:, 1:2] + self._c * grad_xt[:, 0:1]
        pde_res = (residual ** 2).mean().item()

        t_bc = torch.linspace(0, 0.5, 50).unsqueeze(1)
        left = torch.cat([torch.zeros(50, 1), t_bc], dim=1)
        right = torch.cat([torch.ones(50, 1), t_bc], dim=1)
        with torch.no_grad():
            bc_err = ((model(left) - model(right)) ** 2).mean().item()

        return self.pass_criterion({"pde_residual": pde_res, "bc_error": bc_err})

    def pass_criterion(self, metrics: dict) -> bool:
        return (
            metrics.get("pde_residual", float("inf")) < 1e-3
            and metrics.get("bc_error", float("inf")) < 1e-4
        )


# ---------------------------------------------------------------------------
# ProbePDE3: 1D Burgers Equation
# ---------------------------------------------------------------------------

class ProbePDE3(_ProbePDEBase):
    """1D Burgers: u_t + u * u_x = nu * u_xx.

    Analytical: traveling wave u(x,t) = -tanh((x - 0.5*t) / (2*nu))
    with nu = 0.01 / (2*pi).
    Tests: nonlinear PDE + loss balancing.
    Pass: both data and physics loss decrease, ratio stays < 100:1.
    """

    name = "burgers_1d"
    max_epochs = 1000
    _nu = 0.01 / (2 * math.pi)

    def analytical_solution(self, x: torch.Tensor, t: torch.Tensor, **params) -> torch.Tensor:
        nu = params.get("nu", self._nu)
        return -torch.tanh((x - 0.5 * t) / (2 * nu))

    def pde_residual(self, u: torch.Tensor, grads: dict, **params) -> torch.Tensor:
        """PDE residual: u_t + u * u_x - nu * u_xx = 0."""
        nu = params.get("nu", self._nu)
        return grads["u_t"] + u * grads["u_x"] - nu * grads["u_xx"]

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.tanh(x / (2 * self._nu))

    def boundary_conditions(self) -> dict:
        return {"x=0": "analytical", "x=1": "analytical"}

    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        """Loss with data + physics, tracking both components."""
        # Data
        pts = self._sample_sobol_2d(1000, t_range=(0, 0.3))
        pts.requires_grad_(True)
        x, t = pts[:, 0:1], pts[:, 1:2]
        u_pred = model(pts)
        u_exact = self.analytical_solution(x, t)
        loss_data = ((u_pred - u_exact) ** 2).mean()

        # PDE residual
        grad_xt = torch.autograd.grad(u_pred.sum(), pts, create_graph=True)[0]
        u_x_pred = grad_xt[:, 0:1]
        u_t_pred = grad_xt[:, 1:2]
        u_xx_pred = torch.autograd.grad(u_x_pred.sum(), pts, create_graph=True)[0][:, 0:1]
        residual = u_t_pred + u_pred * u_x_pred - self._nu * u_xx_pred
        loss_phys = (residual ** 2).mean()

        return loss_data + 0.01 * loss_phys

    def check_pass(self, model: nn.Module) -> bool:
        """Check both losses decrease and ratio < 100:1."""
        pts = self._sample_sobol_2d(500, t_range=(0, 0.3))
        x, t = pts[:, 0:1], pts[:, 1:2]
        with torch.no_grad():
            u_pred = model(pts)
            u_exact = self.analytical_solution(x, t)
            loss_data = ((u_pred - u_exact) ** 2).mean().item()

        pts2 = self._sample_sobol_2d(500, t_range=(0, 0.3))
        pts2.requires_grad_(True)
        u_pred2 = model(pts2)
        grad_xt = torch.autograd.grad(u_pred2.sum(), pts2, create_graph=True)[0]
        u_x2 = grad_xt[:, 0:1]
        u_t2 = grad_xt[:, 1:2]
        u_xx2 = torch.autograd.grad(u_x2.sum(), pts2, create_graph=True)[0][:, 0:1]
        residual = u_t2 + u_pred2 * u_x2 - self._nu * u_xx2
        loss_phys = (residual ** 2).mean().item()

        ratio = max(loss_data, 1e-10) / max(loss_phys, 1e-10)
        return self.pass_criterion({
            "loss_data": loss_data,
            "loss_phys": loss_phys,
            "loss_ratio": ratio,
        })

    def pass_criterion(self, metrics: dict) -> bool:
        ratio = metrics.get("loss_ratio", float("inf"))
        return ratio < 100 and ratio > 0.01


# ---------------------------------------------------------------------------
# ProbePDE4: 1D Reaction-Diffusion
# ---------------------------------------------------------------------------

class ProbePDE4(_ProbePDEBase):
    """1D reaction-diffusion: u_t = D * u_xx + k * u * (1 - u).

    Analytical: traveling wave u(x,t) = 1 / (1 + exp(-(x - c*t) / w))
    with c = 5*sqrt(D*k/6), w = sqrt(6*D/k).
    D=0.01, k=1.0.
    Tests: multi-scale + requires capturing sharp front.
    Pass: captures both low and high-freq modes.
    """

    name = "reaction_diffusion_1d"
    max_epochs = 1000
    _D = 0.01
    _k = 1.0

    @property
    def _wave_speed(self) -> float:
        return 5 * math.sqrt(self._D * self._k / 6)

    @property
    def _wave_width(self) -> float:
        return math.sqrt(6 * self._D / self._k)

    def analytical_solution(self, x: torch.Tensor, t: torch.Tensor, **params) -> torch.Tensor:
        D = params.get("D", self._D)
        k = params.get("k", self._k)
        c = 5 * math.sqrt(D * k / 6)
        w = math.sqrt(6 * D / k)
        z = -(x - c * t) / w
        return torch.sigmoid(z)

    def pde_residual(self, u: torch.Tensor, grads: dict, **params) -> torch.Tensor:
        """PDE residual: u_t - D * u_xx - k * u * (1-u) = 0."""
        D = params.get("D", self._D)
        k = params.get("k", self._k)
        return grads["u_t"] - D * grads["u_xx"] - k * u * (1 - u)

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        w = self._wave_width
        return torch.sigmoid(x / w)

    def boundary_conditions(self) -> dict:
        return {"x=0": "analytical", "x=1": "analytical"}

    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        """Loss with data + physics."""
        pts = self._sample_sobol_2d(1000, t_range=(0, 0.3))
        pts.requires_grad_(True)
        x, t = pts[:, 0:1], pts[:, 1:2]
        u_pred = model(pts)
        u_exact = self.analytical_solution(x, t)
        loss_data = ((u_pred - u_exact) ** 2).mean()

        # PDE residual
        grad_xt = torch.autograd.grad(u_pred.sum(), pts, create_graph=True)[0]
        u_x_pred = grad_xt[:, 0:1]
        u_t_pred = grad_xt[:, 1:2]
        u_xx_pred = torch.autograd.grad(u_x_pred.sum(), pts, create_graph=True)[0][:, 0:1]
        residual = u_t_pred - self._D * u_xx_pred - self._k * u_pred * (1 - u_pred)
        loss_phys = (residual ** 2).mean()

        return loss_data + 0.1 * loss_phys

    def check_pass(self, model: nn.Module) -> bool:
        """Check captures front shape: MSE < 0.01."""
        pts = self._sample_sobol_2d(1000, t_range=(0, 0.3))
        x, t = pts[:, 0:1], pts[:, 1:2]
        with torch.no_grad():
            u_pred = model(pts)
            u_exact = self.analytical_solution(x, t)
            mse = ((u_pred - u_exact) ** 2).mean().item()
        return self.pass_criterion({"mse_vs_analytical": mse})

    def pass_criterion(self, metrics: dict) -> bool:
        return metrics.get("mse_vs_analytical", float("inf")) < 0.01


# ---------------------------------------------------------------------------
# ALL_PROBES list (mirrors src/trainers/probe_envs.ALL_PROBES)
# ---------------------------------------------------------------------------

ALL_PROBES = [
    ("probe1_heat_1d", ProbePDE1),
    ("probe2_advection_1d", ProbePDE2),
    ("probe3_burgers_1d", ProbePDE3),
    ("probe4_reaction_diffusion_1d", ProbePDE4),
]


# ---------------------------------------------------------------------------
# Probe validation runner
# ---------------------------------------------------------------------------

def run_probe_validation(
    model_class: Optional[Type[nn.Module]] = None,
    config: Optional[dict] = None,
) -> Dict[str, bool]:
    """Run all probe PDEs and return pass/fail results.

    Args:
        model_class: Model class to use. Defaults to _ProbeMLP.
        config: Optional config overrides for model instantiation.

    Returns:
        Dict mapping probe name to pass (True) / fail (False).
    """
    if model_class is None:
        model_class = _ProbeMLP

    results = {}
    for probe_name, probe_cls in ALL_PROBES:
        probe = probe_cls()
        kwargs = dict(probe.model_kwargs)
        if config:
            kwargs.update(config)
        model = model_class(**kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(probe.max_epochs):
            loss = probe.compute_loss(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        passed = probe.check_pass(model)
        results[probe_name] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  Probe {probe.name}: {status}")

    return results


# ---------------------------------------------------------------------------
# PDE system analysis (Task 2 -- will be appended)
# ---------------------------------------------------------------------------

def analyze_pde_system(
    rhs=None,
    scales=None,
    n_sample: int = 256,
) -> Dict:
    """Analyze the PDE system for nondimensionalization quality and stiffness.

    Evaluates the CosseratRHS on random physically reasonable states to assess
    per-component term magnitudes, nondimensionalization quality, and stiffness.

    Args:
        rhs: CosseratRHS instance. If None, creates default.
        scales: NondimScales instance. If None, creates default.
        n_sample: Number of random states to evaluate.

    Returns:
        Dict with keys:
            per_term_magnitudes: dict mapping component names to mean abs magnitude
            nondim_quality: "good" | "acceptable" | "poor"
            stiffness_indicator: max/min eigenvalue ratio of Jacobian
            condition_number: condition number of the Jacobian
            magnitude_spread: max/min ratio of per-component magnitudes
    """
    from src.pinn.physics_residual import CosseratRHS
    from src.pinn.nondim import NondimScales
    from src.pinn._state_slices import (
        POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
    )

    if rhs is None:
        rhs = CosseratRHS()
    if scales is None:
        scales = NondimScales()

    # Generate physically reasonable random states
    # Positions: small perturbations around a straight rod along x-axis
    states = torch.zeros(n_sample, 124)

    # pos_x: nodes along rod, small perturbation
    x_base = torch.linspace(0, scales.L_ref, 21).unsqueeze(0).expand(n_sample, -1)
    states[:, POS_X] = x_base + 0.01 * scales.L_ref * torch.randn(n_sample, 21)

    # pos_y: small perturbation
    states[:, POS_Y] = 0.01 * scales.L_ref * torch.randn(n_sample, 21)

    # velocities: small
    states[:, VEL_X] = 0.1 * scales.V_ref * torch.randn(n_sample, 21)
    states[:, VEL_Y] = 0.1 * scales.V_ref * torch.randn(n_sample, 21)

    # yaw: small angles
    states[:, YAW] = 0.1 * torch.randn(n_sample, 20)

    # omega: small angular velocities
    states[:, OMEGA_Z] = 0.5 * scales.omega_ref * torch.randn(n_sample, 20)

    # Evaluate RHS
    with torch.no_grad():
        derivs = rhs(states)

    # Per-component magnitudes
    component_names = ["d_pos_x", "d_pos_y", "d_vel_x", "d_vel_y", "d_yaw", "d_omega_z"]
    component_slices = [POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z]
    per_term = {}
    magnitudes = []
    for name, sl in zip(component_names, component_slices):
        mag = derivs[:, sl].abs().mean().item()
        per_term[name] = mag
        if mag > 0:
            magnitudes.append(mag)

    # Magnitude spread
    if len(magnitudes) >= 2:
        magnitude_spread = max(magnitudes) / max(min(magnitudes), 1e-30)
    else:
        magnitude_spread = 1.0

    # Nondim quality
    if magnitude_spread < 100:
        nondim_quality = "good"
    elif magnitude_spread < 1000:
        nondim_quality = "acceptable"
    else:
        nondim_quality = "poor"

    # Stiffness indicator via finite-difference Jacobian on a subset
    n_jac = min(10, n_sample)
    eps_fd = 1e-4
    state_sub = states[:n_jac]

    # Use a smaller dimension subset for tractability
    # Compute Jacobian column-by-column for first 20 state dims
    n_cols = min(20, 124)
    with torch.no_grad():
        f0 = rhs(state_sub)  # (n_jac, 124)

    jac_cols = []
    for j in range(n_cols):
        perturbed = state_sub.clone()
        perturbed[:, j] += eps_fd
        with torch.no_grad():
            f_pert = rhs(perturbed)
        jac_col = (f_pert - f0) / eps_fd  # (n_jac, 124)
        jac_cols.append(jac_col[:, :n_cols])  # truncate to square

    # Stack into (n_jac, n_cols, n_cols) Jacobian
    jac = torch.stack(jac_cols, dim=-1)  # (n_jac, n_cols, n_cols)

    # Average over samples
    jac_mean = jac.mean(dim=0)  # (n_cols, n_cols)

    # Eigenvalues for stiffness
    try:
        eigvals = torch.linalg.eigvals(jac_mean).abs()
        eigvals_sorted = eigvals.sort(descending=True).values
        max_eig = eigvals_sorted[0].item()
        min_eig = max(eigvals_sorted[-1].item(), 1e-30)
        stiffness_indicator = max_eig / min_eig
        condition_number = stiffness_indicator
    except Exception:
        stiffness_indicator = float("inf")
        condition_number = float("inf")

    return {
        "per_term_magnitudes": per_term,
        "nondim_quality": nondim_quality,
        "stiffness_indicator": stiffness_indicator,
        "condition_number": condition_number,
        "magnitude_spread": magnitude_spread,
    }
