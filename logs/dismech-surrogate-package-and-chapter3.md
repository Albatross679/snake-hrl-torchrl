---
name: DisMech Surrogate Package and Chapter 3
description: Created papers/aprx_model_dismech/ package with all surrogate modules adapted for DisMech backend, and wrote Chapter 3 (DisMech Backend: Discrete Elastic Rods) in report/report.tex
type: log
status: complete
subtype: feature
created: 2026-03-16
updated: 2026-03-16
tags: [dismech, surrogate, report, physics, DER]
aliases: [dismech-surrogate, chapter3]
---

## Summary

Two changes made in a single quick task:

1. **Created `papers/aprx_model_dismech/` package** (14 Python files)
2. **Wrote Chapter 3 in `report/report.tex`** (DisMech Backend: Discrete Elastic Rods)

## Code: papers/aprx_model_dismech/

### state.py (full rewrite for DisMech)

The key difference from the Elastica version (`pack_from_rod`) is `pack_from_dismech(snake_robot)`:

- **Positions**: Extracted from DisMech's 3D state vector `q = snake_robot._dismech_robot.state.q`, reshaped to (21, 3), taking only (x, y) columns.
- **Velocities**: Same extraction from `u = snake_robot._dismech_robot.state.u`.
- **Yaw (20 elements)**: Computed from segment tangent vectors `arctan2(dy, dx)` instead of reading directly from `rod.tangents` (which DisMech does not expose).
- **Omega_z (20 elements)**: Computed via cross product formula `(dx * dv_y - dy * dv_x) / (dx^2 + dy^2 + eps)` instead of reading from `rod.omega_collection`.
- **Forces**: Returns zeros (DisMech does not expose per-node force arrays in the same API).

### DT_CTRL change

- **Elastica**: `DT_CTRL = 0.5` seconds (500 substeps at 0.001s each)
- **DisMech**: `DT_CTRL = 0.05` seconds (1 implicit step at 0.05s)

This affects the n_cycles encoding: `n_cycles = frequency * DT_CTRL`.

### Files copied vs adapted

**Copied with import path changes only** (physics-agnostic):
- `model.py` -- MLP architecture (input_dim=137, output_dim=128)
- `dataset.py` -- FlatStepDataset for .pt files
- `train_surrogate.py` -- Training loop with MSE loss
- `health.py` -- JSONL event logging and NaN validation
- `monitor.py` -- Live per-worker status display
- `preprocess_relative.py` -- 124-dim to 128-dim conversion

**Adapted for DisMech**:
- `state.py` -- Full rewrite (see above)
- `collect_data.py` -- Uses SnakeRobot instead of LocomotionElasticaEnv, generates serpenoid curvatures and applies via `set_curvature_control()`
- `collect_config.py` -- DisMech defaults (dt=0.05s, steps_per_run=1, flat_output=True)
- `train_config.py` -- Points to DismechConfig, data dirs `data/surrogate_dismech_*`, wandb project `snake-hrl-surrogate-dismech`
- `env.py` -- SurrogateLocomotionEnv using DismechConfig physics parameters
- `validate.py` -- Ground truth from SnakeRobot instead of LocomotionElasticaEnv
- `__init__.py` -- Updated docstring
- `__main__.py` -- Updated import paths

## Report: Chapter 3 (DisMech Backend)

Structure written in `report/report.tex`:

1. **Opening paragraph** -- Introduces DisMech, cites Bergou 2008/2010, states key difference (implicit vs explicit)
2. **Subsection: System Formulation** -- References same Cosserat PDEs from Ch 2, describes DER primary unknowns (vertex positions + Bishop frame), discrete bending energy equation, state vector extraction from 3D
3. **Subsection: Discrete Elastic Rod Discretization** -- Vertex-edge-hinge structure, references staggered grid figure from Ch 2, same action vector, curvature control via bend spring natural strain
4. **Subsection: Implicit Time Integration** -- Backward Euler equations, nonlinear residual, Newton solver with convergence criteria, explicit vs implicit comparison table
5. **Algorithm block** -- DisMech transition operator (two-column format matching Algorithm 1)
6. **Surrogate approximation target** -- Chain equation with DER notation, cost comparison (4.6ms vs 12ms)
7. **Closing paragraph** -- Backend-agnostic surrogate architecture

Added BibTeX entries: `Bergou2008`, `Bergou2010` to `report/references.bib`.
