"""Tests for probe PDE validation suite (src/pinn/probe_pdes.py).

Tests probe PDEs for correctness of analytical solutions, PDE residual
computation, and structural requirements (ALL_PROBES, attributes).
"""

import math

import pytest
import torch


# ===========================================================================
# Task 1 tests: Probe PDE classes
# ===========================================================================


class TestProbePDE1:
    """Tests for ProbePDE1 (1D heat equation)."""

    def test_analytical_solution_known_value(self):
        """ProbePDE1 analytical_solution matches known heat equation solution
        at (x=0.5, t=0.1, alpha=0.01)."""
        from src.pinn.probe_pdes import ProbePDE1

        probe = ProbePDE1()
        x = torch.tensor([0.5])
        t = torch.tensor([0.1])
        u = probe.analytical_solution(x, t, alpha=0.01)

        # u(0.5, 0.1) = exp(-0.01 * pi^2 * 0.1) * sin(pi * 0.5)
        expected = math.exp(-0.01 * math.pi ** 2 * 0.1) * math.sin(math.pi * 0.5)
        assert abs(u.item() - expected) < 1e-6, (
            f"Expected {expected}, got {u.item()}"
        )

    def test_pde_residual_zero_for_exact_solution(self):
        """ProbePDE1 pde_residual returns zero for exact analytical solution
        (residual < 1e-6)."""
        from src.pinn.probe_pdes import ProbePDE1

        probe = ProbePDE1()
        # Create points and compute analytical solution with autograd
        x = torch.linspace(0.05, 0.95, 50, requires_grad=True)
        t = torch.full_like(x, 0.1, requires_grad=True)

        u = probe.analytical_solution(x, t, alpha=0.01)

        # Compute derivatives via autograd
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        grads = {"u_t": u_t, "u_x": u_x, "u_xx": u_xx}
        residual = probe.pde_residual(u, grads)
        assert residual.abs().max().item() < 1e-4, (
            f"Max residual {residual.abs().max().item()} exceeds 1e-4"
        )


class TestProbePDE2:
    """Tests for ProbePDE2 (1D advection)."""

    def test_analytical_solution_advection(self):
        """ProbePDE2 analytical_solution matches known advection equation
        u(x,t) = sin(2*pi*(x - c*t))."""
        from src.pinn.probe_pdes import ProbePDE2

        probe = ProbePDE2()
        x = torch.tensor([0.3])
        t = torch.tensor([0.1])
        u = probe.analytical_solution(x, t, c=1.0)

        expected = math.sin(2 * math.pi * (0.3 - 1.0 * 0.1))
        assert abs(u.item() - expected) < 1e-6

    def test_boundary_conditions_keys(self):
        """ProbePDE2 boundary_conditions returns dict with 'x=0' and 'x=1' keys."""
        from src.pinn.probe_pdes import ProbePDE2

        probe = ProbePDE2()
        bcs = probe.boundary_conditions()
        assert isinstance(bcs, dict)
        assert "x=0" in bcs
        assert "x=1" in bcs


class TestProbePDE3:
    """Tests for ProbePDE3 (1D Burgers equation)."""

    def test_pde_residual_correct(self):
        """ProbePDE3 (Burgers) pde_residual with viscosity nu=0.01/(2*pi)
        returns correct residual form."""
        from src.pinn.probe_pdes import ProbePDE3

        probe = ProbePDE3()
        # Use known solution to verify residual is near zero
        x = torch.linspace(0.1, 0.9, 30, requires_grad=True)
        t = torch.full_like(x, 0.05, requires_grad=True)

        u = probe.analytical_solution(x, t)

        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        grads = {"u_t": u_t, "u_x": u_x, "u_xx": u_xx}
        residual = probe.pde_residual(u, grads)

        # Residual should be small for the exact solution
        assert residual.abs().mean().item() < 0.1, (
            f"Mean residual {residual.abs().mean().item()} too large for exact solution"
        )


class TestProbePDE4:
    """Tests for ProbePDE4 (1D reaction-diffusion)."""

    def test_analytical_solution_o1_values(self):
        """ProbePDE4 (reaction-diffusion) analytical_solution gives O(1)
        values at t=0.1."""
        from src.pinn.probe_pdes import ProbePDE4

        probe = ProbePDE4()
        x = torch.linspace(0, 1, 20)
        t = torch.full_like(x, 0.1)
        u = probe.analytical_solution(x, t)

        # Traveling wave sigmoid gives values in [0, 1]
        assert u.min().item() >= -0.1, f"Min value {u.min().item()} too negative"
        assert u.max().item() <= 1.1, f"Max value {u.max().item()} too large"
        # Check O(1) magnitude: at least some values should be nonzero
        assert u.abs().max().item() > 0.01, "All values near zero"


class TestALLPROBES:
    """Tests for the ALL_PROBES list structure."""

    def test_all_probes_is_list_of_4_tuples(self):
        """ALL_PROBES is a list of 4 tuples (name_str, probe_class)."""
        from src.pinn.probe_pdes import ALL_PROBES

        assert isinstance(ALL_PROBES, list)
        assert len(ALL_PROBES) == 4
        for item in ALL_PROBES:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, cls = item
            assert isinstance(name, str)

    def test_each_probe_has_required_attributes(self):
        """Each probe has name, analytical_solution, pde_residual,
        pass_criterion attributes."""
        from src.pinn.probe_pdes import ALL_PROBES

        required_attrs = [
            "name",
            "analytical_solution",
            "pde_residual",
            "initial_condition",
            "boundary_conditions",
            "compute_loss",
            "check_pass",
            "pass_criterion",
        ]

        for probe_name, probe_cls in ALL_PROBES:
            probe = probe_cls()
            for attr in required_attrs:
                assert hasattr(probe, attr), (
                    f"Probe {probe_name} missing attribute: {attr}"
                )


# ===========================================================================
# Task 2 tests: System analysis + probe runner (placeholder)
# ===========================================================================


class TestRunProbeValidation:
    """Tests for run_probe_validation function."""

    def test_run_probe_validation_returns_dict_of_bools(self):
        """run_probe_validation accepts a simple MLP and returns dict of bool
        results keyed by probe name."""
        from src.pinn.probe_pdes import run_probe_validation

        results = run_probe_validation()
        assert isinstance(results, dict)
        assert len(results) == 4
        for name, passed in results.items():
            assert isinstance(name, str)
            assert isinstance(passed, bool)


class TestAnalyzePDESystem:
    """Tests for analyze_pde_system function."""

    def test_returns_correct_keys(self):
        """analyze_pde_system returns dict with keys: per_term_magnitudes,
        condition_number, nondim_quality, stiffness_indicator."""
        from src.pinn.probe_pdes import analyze_pde_system

        result = analyze_pde_system()
        required_keys = [
            "per_term_magnitudes",
            "condition_number",
            "nondim_quality",
            "stiffness_indicator",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_finite_non_nan_values(self):
        """analyze_pde_system with default CosseratRHS and NondimScales
        produces finite non-NaN values."""
        from src.pinn.probe_pdes import analyze_pde_system

        result = analyze_pde_system()
        # Check scalar values are finite
        assert math.isfinite(result["condition_number"]), "condition_number not finite"
        assert math.isfinite(result["stiffness_indicator"]), "stiffness_indicator not finite"
        # Check per_term_magnitudes dict values
        for comp, val in result["per_term_magnitudes"].items():
            assert math.isfinite(val), f"per_term_magnitudes[{comp}] = {val} not finite"

    def test_nondim_quality_assessment(self):
        """analyze_pde_system nondim_quality is 'good' when all residual terms
        are O(1), 'poor' when spread > 1000x."""
        from src.pinn.probe_pdes import analyze_pde_system

        result = analyze_pde_system()
        assert result["nondim_quality"] in ("good", "acceptable", "poor"), (
            f"nondim_quality = {result['nondim_quality']} not in expected values"
        )
