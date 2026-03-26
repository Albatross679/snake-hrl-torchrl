---
name: PINN Probe PDE Validation Suite
description: Created 4 generic probe PDEs with analytical solutions for PINN debugging pre-flight validation
type: log
status: complete
subtype: feature
created: 2026-03-26
updated: 2026-03-26
tags: [pinn, diagnostics, probe-pdes, debugging]
aliases: [probe-pdes]
---

# PINN Probe PDE Validation Suite

Created `src/pinn/probe_pdes.py` with 4 progressively complex probe PDEs that validate PINN implementation before expensive real training. Mirrors the RL probe environment pattern from `src/trainers/probe_envs.py`.

## Probes

| Probe | PDE | Tests | Pass Criterion |
|-------|-----|-------|----------------|
| ProbePDE1 | 1D heat: u_t = alpha * u_xx | Data fitting + optimizer | MSE < 1e-4 in 500 epochs |
| ProbePDE2 | 1D advection: u_t + c*u_x = 0 | BC/IC enforcement | Residual < 1e-3, BC error < 1e-4 |
| ProbePDE3 | 1D Burgers: u_t + u*u_x = nu*u_xx | Nonlinear PDE + loss balance | Loss ratio stays < 100:1 |
| ProbePDE4 | 1D reaction-diffusion | Multi-scale front capture | MSE < 0.01 |

## Key Components

- `_ProbeMLP`: 2-layer 64-dim Tanh MLP for probe training
- `_ProbePDEBase`: Base class with Sobol sampling and interface definition
- `ALL_PROBES`: List of (name, class) tuples mirroring RL probe pattern
- `run_probe_validation()`: Runner that trains and evaluates all probes
- `analyze_pde_system()`: CosseratRHS nondimensionalization quality checker

## Files

- `src/pinn/probe_pdes.py` (new)
- `tests/test_pinn_probes.py` (new)
