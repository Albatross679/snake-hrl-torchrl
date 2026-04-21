---
phase: 13-implement-pinn-and-dd-pinn-surrogate-models
plan: 02
subsystem: physics
tags: [pinn, cosserat-rod, rft-friction, collocation, sobol, differentiable-physics]

# Dependency graph
requires:
  - phase: 13-implement-pinn-and-dd-pinn-surrogate-models (Plan 01)
    provides: "src/pinn package with _state_slices, regularizer, loss_balancing, nondim"
provides:
  - "CosseratRHS: differentiable Cosserat rod RHS (elastic bending + anisotropic RFT friction)"
  - "sample_collocation: Sobol quasi-random temporal collocation point sampling"
  - "adaptive_refinement: Residual-based Adaptive Refinement (RAR) for collocation"
affects: [13-03, 13-05, 13-06]

# Tech tracking
tech-stack:
  added: [scipy.stats.qmc.Sobol]
  patterns: [regularized-tangent-normalization, node-interpolated-rft, inextensible-rod-approximation]

key-files:
  created:
    - src/pinn/physics_residual.py
    - src/pinn/collocation.py
  modified:
    - tests/test_pinn.py
    - src/pinn/__init__.py

key-decisions:
  - "Inextensible rod approximation: omit stretching stiffness, keep only bending + friction + kinematics for 2D planar snake"
  - "Regularized tangent: use sqrt(norm^2 + eps^2) instead of (norm + eps) for gradient stability near zero element length"
  - "Mass lumping: boundary nodes get half element mass, interior nodes get full element mass"
  - "Elastic forces via moment/dl^2 at joints, distributed to nodes 1..19"

patterns-established:
  - "Regularized normalization: tangent / sqrt(norm^2 + eps^2) for differentiable tangent computation"
  - "Node-to-element interpolation: average neighboring element tangents for node quantities"
  - "Sobol-then-truncate: draw power-of-2 Sobol samples and truncate to requested n_points"

requirements-completed: [PINN-08, PINN-09, PINN-11]

# Metrics
duration: 3min
completed: 2026-03-18
---

# Phase 13 Plan 02: Differentiable CosseratRHS with RFT Friction and Sobol Collocation Summary

**Differentiable Cosserat rod RHS (bending + anisotropic RFT friction) with Sobol quasi-random collocation and residual-adaptive refinement**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-18T11:59:49Z
- **Completed:** 2026-03-18T12:02:49Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments
- CosseratRHS nn.Module computes dx/dt = f(x) for 124-dim rod state with elastic bending moments and full anisotropic RFT friction
- Gradient-safe regularized tangent normalization avoids singularity at zero velocity/length
- Sobol collocation sampling with adaptive refinement (RAR) for PINN temporal discretization
- All 44 tests pass including shape, gradient flow, analytical RFT verification, and real-data consistency

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement differentiable CosseratRHS and collocation sampling** - `f38155e` (feat)

## Files Created/Modified
- `src/pinn/physics_residual.py` - CosseratRHS nn.Module: differentiable Cosserat rod RHS with elastic bending + RFT friction
- `src/pinn/collocation.py` - Sobol/uniform collocation sampling + residual-adaptive refinement (RAR)
- `tests/test_pinn.py` - 12 new tests for CosseratRHS (shape, grad, RFT reference, regularization, PyElastica consistency) and collocation (Sobol, uniform, coverage, adaptive, invalid method)
- `src/pinn/__init__.py` - Exports CosseratRHS, sample_collocation, adaptive_refinement

## Decisions Made
- Inextensible rod approximation: stretching stiffness computed but unused -- bending dominates for 2D planar snake locomotion
- Regularized tangent normalization uses sqrt(norm^2 + eps^2) for smooth gradients near zero
- Boundary nodes get half element mass (mass lumping scheme matching standard FEM practice)
- Elastic force distribution: moment forces at joints (nodes 1-19), boundary nodes (0, 20) receive zero elastic force

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CosseratRHS ready for use in DD-PINN training (Plan 05) as physics residual evaluator
- Collocation utilities ready for temporal discretization in PINN loss computation
- All tests passing, no blockers for downstream plans

## Self-Check: PASSED

- src/pinn/physics_residual.py: FOUND
- src/pinn/collocation.py: FOUND
- tests/test_pinn.py: FOUND
- Commit f38155e: FOUND
- All 44 tests: PASSED

---
*Phase: 13-implement-pinn-and-dd-pinn-surrogate-models*
*Completed: 2026-03-18*
