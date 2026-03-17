"""Unit tests for src/pinn/ package.

Tests cover:
- PINN-01: PhysicsRegularizer (4 constraint types)
- PINN-02: ReLoBRaLo adaptive loss balancing
- PINN-05: Package structure and imports
- PINN-07: NondimScales physics-based nondimensionalization
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.pinn import PhysicsRegularizer, ReLoBRaLo, NondimScales


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seed():
    torch.manual_seed(42)


@pytest.fixture
def random_state(seed):
    """Random batch of raw 124-dim states."""
    return torch.randn(32, 124)


@pytest.fixture
def random_delta(seed):
    """Random batch of 124-dim state deltas."""
    # Use a different seed offset so delta != state
    torch.manual_seed(43)
    return torch.randn(32, 124)


# ---------------------------------------------------------------------------
# PhysicsRegularizer tests (PINN-01)
# ---------------------------------------------------------------------------

class TestPhysicsRegularizer:
    """Tests for PhysicsRegularizer constraint computation."""

    def test_physics_regularizer(self, random_state, random_delta):
        """Regularizer returns positive scalar with gradients for random inputs."""
        reg = PhysicsRegularizer(dt=0.5)
        # delta_pred typically comes from a model, so it has requires_grad
        delta = random_delta.requires_grad_(True)
        loss = reg(random_state, delta)

        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss.requires_grad, "Loss must have requires_grad=True"
        assert loss.item() > 0, "Loss should be positive for random inputs"

    def test_regularizer_gradients(self):
        """Gradients flow through regularizer back to model parameters."""
        torch.manual_seed(42)
        model = nn.Linear(124, 124)
        reg = PhysicsRegularizer(dt=0.5)

        state = torch.randn(16, 124)
        delta_pred = model(state)
        loss = reg(state, delta_pred)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.all(param.grad == 0), f"Gradient is all zeros for {name}"

    def test_regularizer_zero_delta(self):
        """Zero delta should produce near-zero loss (kinematic consistency satisfied)."""
        torch.manual_seed(42)
        # Use a state with zero velocities and zero omega so that
        # expected position change = avg_vel * dt = 0 = delta_pos
        state = torch.zeros(8, 124)
        delta = torch.zeros(8, 124)

        reg = PhysicsRegularizer(dt=0.5)
        loss = reg(state, delta)

        assert loss.item() < 1e-6, (
            f"Zero delta with zero-velocity state should give near-zero loss, "
            f"got {loss.item()}"
        )

    def test_regularizer_output_shape(self, random_state, random_delta):
        """Regularizer returns a scalar (0-dim tensor)."""
        reg = PhysicsRegularizer(dt=0.5)
        loss = reg(random_state, random_delta)
        assert loss.shape == torch.Size([])

    def test_regularizer_batch_independent(self):
        """Loss should scale with batch size but remain finite."""
        torch.manual_seed(42)
        state_small = torch.randn(4, 124)
        delta_small = torch.randn(4, 124)
        state_large = torch.randn(64, 124)
        delta_large = torch.randn(64, 124)

        reg = PhysicsRegularizer(dt=0.5)
        loss_small = reg(state_small, delta_small)
        loss_large = reg(state_large, delta_large)

        assert torch.isfinite(loss_small), "Loss must be finite"
        assert torch.isfinite(loss_large), "Loss must be finite"


# ---------------------------------------------------------------------------
# ReLoBRaLo tests (PINN-02)
# ---------------------------------------------------------------------------

class TestReLoBRaLo:
    """Tests for ReLoBRaLo adaptive loss balancing."""

    def test_relobralo_init(self):
        """Initial weights should all be 1.0."""
        balancer = ReLoBRaLo(n_losses=3)
        assert balancer.weights.shape == (3,)
        assert torch.allclose(balancer.weights, torch.ones(3))

    def test_relobralo_update(self):
        """Weights sum to n_losses and respond to imbalanced inputs."""
        torch.manual_seed(42)
        balancer = ReLoBRaLo(n_losses=3, alpha=0.999, temperature=1.0, rho=0.999)

        # First call initializes
        losses_init = [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)]
        w0 = balancer.update(losses_init)
        assert w0.shape == (3,)

        # Run 10 updates with highly unbalanced losses
        for _ in range(10):
            losses = [torch.tensor(100.0), torch.tensor(0.01), torch.tensor(1.0)]
            weights = balancer.update(losses)

        # Weights should sum to 3.0
        assert abs(weights.sum().item() - 3.0) < 1e-4, (
            f"Weights should sum to 3.0, got {weights.sum().item()}"
        )

        # After unbalanced updates, weights should NOT all be equal
        assert not torch.allclose(weights, torch.ones(3)), (
            "Weights should diverge from uniform after unbalanced updates"
        )

    def test_relobralo_first_call_returns_ones(self):
        """First update call returns ones (no previous reference)."""
        balancer = ReLoBRaLo(n_losses=4)
        losses = [torch.tensor(2.0), torch.tensor(0.5), torch.tensor(1.0), torch.tensor(3.0)]
        w = balancer.update(losses)
        assert torch.allclose(w, torch.ones(4))

    def test_relobralo_weight_sum(self):
        """Weights always sum to n_losses across multiple updates."""
        torch.manual_seed(42)
        n = 5
        balancer = ReLoBRaLo(n_losses=n)

        for i in range(20):
            losses = [torch.tensor(float(j + 1) * (i + 1)) for j in range(n)]
            w = balancer.update(losses)
            assert abs(w.sum().item() - float(n)) < 1e-3, (
                f"Step {i}: weights sum to {w.sum().item()}, expected {n}"
            )


# ---------------------------------------------------------------------------
# NondimScales tests (PINN-07)
# ---------------------------------------------------------------------------

class TestNondimScales:
    """Tests for physics-based nondimensionalization."""

    def test_nondim_scales(self):
        """Check reference scales are computed correctly."""
        scales = NondimScales()

        assert scales.L_ref == 1.0
        assert scales.t_ref == 0.5
        assert scales.F_ref > 0

        # F_ref = E * I / L^2 where I = pi * r^4 / 4
        I_expected = math.pi * 0.001 ** 4 / 4.0
        F_ref_expected = 2e6 * I_expected / 1.0 ** 2
        assert abs(scales.F_ref - F_ref_expected) < 1e-12, (
            f"F_ref mismatch: {scales.F_ref} vs {F_ref_expected}"
        )

    def test_nondim_scales_velocity(self):
        """V_ref and omega_ref are correct."""
        scales = NondimScales()
        assert abs(scales.V_ref - 2.0) < 1e-10
        assert abs(scales.omega_ref - 2.0) < 1e-10

    def test_nondim_roundtrip(self):
        """nondim_state followed by redim_state recovers original state."""
        torch.manual_seed(42)
        scales = NondimScales()
        state = torch.randn(16, 124)

        nondim = scales.nondim_state(state)
        recovered = scales.redim_state(nondim)

        assert torch.allclose(state, recovered, atol=1e-6), (
            f"Roundtrip error: max diff = {(state - recovered).abs().max().item()}"
        )

    def test_nondim_delta_roundtrip(self):
        """nondim_delta followed by redim_delta recovers original delta."""
        torch.manual_seed(42)
        scales = NondimScales()
        delta = torch.randn(16, 124)

        nondim = scales.nondim_delta(delta)
        recovered = scales.redim_delta(nondim)

        assert torch.allclose(delta, recovered, atol=1e-6)

    def test_nondim_yaw_unchanged(self):
        """Yaw (radians) should not be scaled."""
        torch.manual_seed(42)
        scales = NondimScales()
        state = torch.randn(8, 124)

        nondim = scales.nondim_state(state)

        # YAW slice: 84:104
        assert torch.allclose(state[..., 84:104], nondim[..., 84:104]), (
            "Yaw should be unchanged by nondimensionalization"
        )

    def test_nondim_positions_scaled(self):
        """Positions should be divided by L_ref."""
        scales = NondimScales(snake_length=2.0)
        state = torch.ones(4, 124)

        nondim = scales.nondim_state(state)

        # POS_X: 0:21, POS_Y: 21:42 should be divided by 2.0
        assert torch.allclose(nondim[..., 0:21], torch.full((4, 21), 0.5))
        assert torch.allclose(nondim[..., 21:42], torch.full((4, 21), 0.5))


# ---------------------------------------------------------------------------
# Package import tests (PINN-05)
# ---------------------------------------------------------------------------

class TestPackageImports:
    """Tests for package structure and public API."""

    def test_import_physics_regularizer(self):
        from src.pinn import PhysicsRegularizer
        assert PhysicsRegularizer is not None

    def test_import_relobralo(self):
        from src.pinn import ReLoBRaLo
        assert ReLoBRaLo is not None

    def test_import_nondim_scales(self):
        from src.pinn import NondimScales
        assert NondimScales is not None

    def test_regularizer_uses_state_slices(self):
        """Regularizer imports named slices from state module."""
        import src.pinn.regularizer as reg_mod
        source = open(reg_mod.__file__).read()
        assert "from papers.aprx_model_elastica.state import" in source
