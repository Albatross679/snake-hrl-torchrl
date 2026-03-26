"""Tests for PINN diagnostics middleware.

Tests PINNDiagnostics class methods: loss ratio, per-loss gradients,
residual statistics, ReLoBRaLo health, per-component violations,
NTK eigenvalue computation, and the log_step aggregation method.
"""

import torch
import torch.nn as nn

import pytest


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _SimpleMLP(nn.Module):
    """2-layer MLP for testing diagnostics."""

    def __init__(self, in_dim=4, hidden=16, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# PINNDiagnostics.__init__
# ---------------------------------------------------------------------------

def test_init_stores_deque_history():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    assert hasattr(diag, "_history")
    assert "loss_ratio" in diag._history
    assert diag._history["loss_ratio"].maxlen == 100


def test_init_accepts_none_wandb():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    assert diag.wandb_run is None


# ---------------------------------------------------------------------------
# compute_loss_ratio
# ---------------------------------------------------------------------------

def test_compute_loss_ratio_basic():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    ratio = diag.compute_loss_ratio(
        loss_data=torch.tensor(0.01),
        loss_phys=torch.tensor(1.0),
    )
    assert abs(ratio - 100.0) < 1e-6


def test_compute_loss_ratio_zero_data():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    ratio = diag.compute_loss_ratio(
        loss_data=torch.tensor(0.0),
        loss_phys=torch.tensor(1.0),
    )
    assert ratio > 0  # large finite, not inf or NaN
    assert ratio == ratio  # not NaN


# ---------------------------------------------------------------------------
# compute_per_loss_gradients
# ---------------------------------------------------------------------------

def test_per_loss_gradients_returns_correct_keys():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    model = _SimpleMLP(in_dim=4, hidden=16, out_dim=1)
    x = torch.randn(8, 4)
    out = model(x)
    loss_data = out.mean()
    loss_phys = (out ** 2).mean()

    result = diag.compute_per_loss_gradients(model, loss_data, loss_phys)
    assert "diagnostics/grad_norm_data" in result
    assert "diagnostics/grad_norm_phys" in result
    assert "diagnostics/grad_norm_ratio" in result


def test_per_loss_gradients_returns_finite_positive():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    model = _SimpleMLP(in_dim=4, hidden=16, out_dim=1)
    x = torch.randn(8, 4)
    out = model(x)
    loss_data = out.mean()
    loss_phys = (out ** 2).mean()

    result = diag.compute_per_loss_gradients(model, loss_data, loss_phys)
    for key in ("diagnostics/grad_norm_data", "diagnostics/grad_norm_phys"):
        assert result[key] > 0, f"{key} should be positive"
        assert result[key] == result[key], f"{key} should not be NaN"


# ---------------------------------------------------------------------------
# compute_residual_statistics
# ---------------------------------------------------------------------------

def test_residual_statistics():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    residuals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = diag.compute_residual_statistics(residuals)

    assert abs(result["diagnostics/residual_mean"] - 3.0) < 1e-6
    assert abs(result["diagnostics/residual_max"] - 5.0) < 1e-6
    assert "diagnostics/residual_std" in result
    assert "diagnostics/residual_p95" in result


# ---------------------------------------------------------------------------
# compute_relobralo_health
# ---------------------------------------------------------------------------

def test_relobralo_health():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    weights = torch.tensor([0.3, 0.7])
    result = diag.compute_relobralo_health(weights)

    assert "diagnostics/relobralo_w_data" in result
    assert "diagnostics/relobralo_w_phys" in result
    assert "diagnostics/relobralo_ratio" in result
    expected_ratio = 0.7 / 0.3
    assert abs(result["diagnostics/relobralo_ratio"] - expected_ratio) < 0.01


# ---------------------------------------------------------------------------
# compute_ntk_eigenvalues
# ---------------------------------------------------------------------------

def test_ntk_eigenvalues_returns_correct_keys():
    from src.pinn.diagnostics import compute_ntk_eigenvalues

    model = _SimpleMLP(in_dim=4, hidden=16, out_dim=1)
    inputs = torch.randn(10, 4)
    result = compute_ntk_eigenvalues(model, inputs, n_params_sample=50)

    for key in ("ntk/eigenvalue_max", "ntk/eigenvalue_min",
                "ntk/condition_number", "ntk/spectral_decay_rate"):
        assert key in result, f"Missing key: {key}"


def test_ntk_eigenvalues_positive_and_finite():
    from src.pinn.diagnostics import compute_ntk_eigenvalues

    model = _SimpleMLP(in_dim=4, hidden=16, out_dim=1)
    inputs = torch.randn(10, 4)
    result = compute_ntk_eigenvalues(model, inputs, n_params_sample=50)

    assert result["ntk/eigenvalue_max"] > 0
    assert result["ntk/condition_number"] == result["ntk/condition_number"]  # not NaN
    assert result["ntk/condition_number"] > 0


# ---------------------------------------------------------------------------
# log_step
# ---------------------------------------------------------------------------

def test_log_step_aggregates_metrics():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None, ntk_interval=50)
    model = _SimpleMLP(in_dim=4, hidden=16, out_dim=1)
    loss_data = torch.tensor(0.5)
    loss_phys = torch.tensor(1.0)

    metrics = diag.log_step(
        epoch=1,
        model=model,
        loss_data=loss_data,
        loss_phys=loss_phys,
    )
    assert isinstance(metrics, dict)
    assert "diagnostics/loss_ratio" in metrics


def test_log_step_with_none_wandb_no_raise():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    model = _SimpleMLP(in_dim=4, hidden=16, out_dim=1)
    # Should not raise even with wandb_run=None
    metrics = diag.log_step(
        epoch=0,
        model=model,
        loss_data=torch.tensor(1.0),
        loss_phys=torch.tensor(1.0),
    )
    assert metrics is not None


# ---------------------------------------------------------------------------
# History deque capacity
# ---------------------------------------------------------------------------

def test_history_deque_stores_last_100():
    from src.pinn.diagnostics import PINNDiagnostics

    diag = PINNDiagnostics(wandb_run=None)
    for i in range(150):
        diag.compute_loss_ratio(
            loss_data=torch.tensor(1.0),
            loss_phys=torch.tensor(float(i)),
        )
    assert len(diag._history["loss_ratio"]) == 100
    # Should contain the last 100 values (50-149)
    assert diag._history["loss_ratio"][0] == 50.0
