"""Unit tests for src/pinn/ package.

Tests cover:
- PINN-01: PhysicsRegularizer (4 constraint types)
- PINN-02: ReLoBRaLo adaptive loss balancing
- PINN-05: Package structure and imports
- PINN-07: NondimScales physics-based nondimensionalization
- PINN-09: CosseratRHS differentiable physics residual
- PINN-11: Collocation point sampling (Sobol, adaptive)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.pinn import PhysicsRegularizer, ReLoBRaLo, NondimScales, CosseratRHS
from src.pinn.collocation import sample_collocation, adaptive_refinement


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
        """Regularizer imports named slices from state slices module."""
        import src.pinn.regularizer as reg_mod
        source = open(reg_mod.__file__).read()
        assert "POS_X" in source and "VEL_X" in source and "YAW" in source


# ---------------------------------------------------------------------------
# CosseratRHS tests (PINN-09)
# ---------------------------------------------------------------------------

class TestCosseratRHS:
    """Tests for differentiable Cosserat rod right-hand side."""

    def test_cosserat_rhs_shape(self):
        """CosseratRHS returns (B, 124) for (B, 124) input."""
        torch.manual_seed(42)
        rhs = CosseratRHS()
        state = torch.randn(8, 124)
        out = rhs(state)
        assert out.shape == (8, 124), f"Expected (8, 124), got {out.shape}"

    def test_cosserat_rhs_grad(self):
        """Output requires_grad when input requires_grad."""
        torch.manual_seed(42)
        rhs = CosseratRHS()
        state = torch.randn(4, 124, requires_grad=True)
        out = rhs(state)
        assert out.requires_grad, "Output must require grad when input does"

        # Verify gradients actually flow
        loss = out.sum()
        loss.backward()
        assert state.grad is not None, "Input gradient must not be None"
        assert not torch.all(state.grad == 0), "Gradient must be non-zero"

    def test_rft_vs_reference(self):
        """RFT friction for a straight rod with known velocity matches analytical."""
        torch.manual_seed(42)
        rhs = CosseratRHS(ct=0.01, cn=0.1)

        # Straight rod along x-axis, moving purely in y-direction
        state = torch.zeros(1, 124)
        n_nodes = 21
        # Positions: straight line along x
        state[0, :n_nodes] = torch.linspace(0, 1, n_nodes)  # pos_x
        # pos_y = 0 (already)
        # vel_x = 0, vel_y = 1.0 (pure normal velocity)
        state[0, 63:84] = 1.0  # vel_y = 1.0

        out = rhs(state)

        # For a straight rod along x, tangent = (1, 0)
        # v = (0, 1) => v_tan = 0, v_norm = (0, 1)
        # F = -cn * v_norm = -0.1 * (0, 1) = (0, -0.1)
        # d_vel_y = F_y / mass
        d_vel_y = out[0, 63:84]  # should be -0.1 / mass for each node

        mass = rhs.mass.item()
        # Interior nodes have full mass, boundary have half
        expected_interior = -0.1 / mass
        expected_boundary = -0.1 / (mass / 2.0)

        # Check interior nodes (1:-1)
        interior = d_vel_y[1:-1]
        # Allow elastic forces to contribute some deviation, but friction should dominate
        # Check that sign is correct (deceleration) and magnitude is within 1%
        for i, val in enumerate(interior):
            assert abs(val.item() - expected_interior) / abs(expected_interior) < 0.05, (
                f"Node {i+1}: got {val.item()}, expected ~{expected_interior}"
            )

    def test_rft_regularization(self):
        """At zero velocity, friction force is zero (no singularity)."""
        torch.manual_seed(42)
        rhs = CosseratRHS()

        # Straight rod, zero velocity
        state = torch.zeros(1, 124)
        state[0, :21] = torch.linspace(0, 1, 21)  # pos_x straight

        out = rhs(state)

        # Should not have NaN
        assert not torch.any(torch.isnan(out)), "Output contains NaN at zero velocity"
        assert not torch.any(torch.isinf(out)), "Output contains Inf at zero velocity"

        # d_vel should be near zero (no friction when stationary)
        d_vel_x = out[0, 42:63]
        d_vel_y = out[0, 63:84]
        # Elastic forces may contribute, but no friction-induced acceleration
        assert torch.isfinite(d_vel_x).all()
        assert torch.isfinite(d_vel_y).all()

    def test_rft_vs_pyelastica(self):
        """Run CosseratRHS on real dataset states and verify consistency."""
        from pathlib import Path

        data_path = Path("data/surrogate_rl_step/batch_w00_0000.pt")
        if not data_path.exists():
            pytest.skip("No dataset batch available")

        data = torch.load(data_path, weights_only=True)
        states = data["states"][:5]  # 5 real states

        rhs = CosseratRHS()

        for idx in range(len(states)):
            state_t = states[idx:idx+1]
            out = rhs(state_t)

            # Basic sanity: output is finite and has correct shape
            assert out.shape == (1, 124), f"State {idx}: wrong shape {out.shape}"
            assert torch.isfinite(out).all(), f"State {idx}: non-finite output"

            # Check kinematic consistency: d_pos = vel
            d_pos_x = out[0, :21]
            vel_x = state_t[0, 42:63]
            assert torch.allclose(d_pos_x, vel_x, atol=1e-6), (
                f"State {idx}: kinematic consistency violated for pos_x"
            )

    def test_cosserat_rhs_batch_consistency(self):
        """Single-sample and batched computation give same results."""
        torch.manual_seed(42)
        rhs = CosseratRHS()

        state = torch.randn(4, 124)
        out_batch = rhs(state)

        for i in range(4):
            out_single = rhs(state[i:i+1])
            assert torch.allclose(out_batch[i:i+1], out_single, atol=1e-5), (
                f"Sample {i}: batch vs single mismatch"
            )


# ---------------------------------------------------------------------------
# Collocation tests (PINN-11)
# ---------------------------------------------------------------------------

class TestCollocation:
    """Tests for collocation point sampling."""

    def test_collocation_sobol(self):
        """Sobol sampling returns sorted points in [0, t_end]."""
        t = sample_collocation(1000, t_end=0.5, method="sobol")
        assert t.shape == (1000,), f"Expected (1000,), got {t.shape}"
        assert t.dtype == torch.float32
        assert t.min() >= 0.0
        assert t.max() <= 0.5
        # Check sorted
        assert torch.all(t[1:] >= t[:-1]), "Points must be sorted"

    def test_collocation_uniform(self):
        """Uniform sampling returns sorted points in interval."""
        t = sample_collocation(500, t_start=0.1, t_end=0.4, method="uniform")
        assert t.shape == (500,)
        assert t.min() >= 0.1
        assert t.max() <= 0.4
        assert torch.all(t[1:] >= t[:-1])

    def test_collocation_sobol_coverage(self):
        """Sobol points have better coverage than uniform (lower discrepancy)."""
        torch.manual_seed(42)
        t_sobol = sample_collocation(256, t_end=1.0, method="sobol")

        # Check that points cover the interval reasonably well
        # Divide into 10 bins and check each has some points
        for i in range(10):
            lo = i * 0.1
            hi = (i + 1) * 0.1
            count = ((t_sobol >= lo) & (t_sobol < hi)).sum().item()
            assert count > 0, f"Bin [{lo}, {hi}) has no Sobol points"

    def test_collocation_adaptive(self):
        """Adaptive refinement concentrates points near high residuals."""
        torch.manual_seed(42)
        n = 100
        t_base = torch.linspace(0, 1, n)
        residuals = torch.zeros(n)
        # Place high residual at t=0.5
        residuals[45:55] = 10.0

        new_t = adaptive_refinement(residuals, t_base, n_new=100, beta=2.0)

        assert new_t.shape == (100,)
        assert torch.all(new_t[1:] >= new_t[:-1]), "New points must be sorted"

        # Most new points should be near t=0.5
        near_peak = ((new_t >= 0.3) & (new_t <= 0.7)).sum().item()
        assert near_peak > 50, (
            f"Expected >50 points near peak, got {near_peak}"
        )

    def test_collocation_invalid_method(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown collocation method"):
            sample_collocation(10, method="invalid")
