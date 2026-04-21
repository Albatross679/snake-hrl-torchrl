---
phase: 13-implement-pinn-and-dd-pinn-surrogate-models
plan: 01
subsystem: ml
tags: [pinn, physics-regularizer, loss-balancing, nondimensionalization, cosserat-rod]

# Dependency graph
requires:
  - phase: 03-train-surrogate-model
    provides: "SurrogateModel architecture and state.py named slices"
provides:
  - "PhysicsRegularizer with 4 constraint types (kinematic, angular, curvature-moment, energy)"
  - "ReLoBRaLo adaptive multi-term loss balancing"
  - "NondimScales physics-based nondimensionalization"
  - "19 unit tests for src/pinn/ package"
affects: [13-02, 13-03, 13-04, 13-05, 13-06]

# Tech tracking
tech-stack:
  added: []
  patterns: [physics-regularizer-as-nn-module, algebraic-constraint-loss, relobralo-adaptive-weighting]

key-files:
  created:
    - src/pinn/__init__.py
    - src/pinn/regularizer.py
    - src/pinn/loss_balancing.py
    - src/pinn/nondim.py
    - tests/test_pinn.py
  modified: []

key-decisions:
  - "PhysicsRegularizer uses trapezoidal integration (avg velocity) for kinematic constraints"
  - "Curvature-moment constraint weighted 0.1x relative to kinematic constraints (approximate)"
  - "Energy constraint uses ReLU threshold (10.0) to only penalize unreasonable KE changes"
  - "NondimScales uses L_ref=1.0m, t_ref=0.5s, F_ref=E*I/L^2 (characteristic elastic force)"

patterns-established:
  - "Physics constraints as nn.Module with register_buffer for dt"
  - "Nondim uses clone() to avoid in-place modification of input tensors"
  - "ReLoBRaLo weights always sum to n_losses via softmax * n_losses"

requirements-completed: [PINN-01, PINN-02, PINN-07]

# Metrics
duration: 3min
completed: 2026-03-17
---

# Phase 13 Plan 01: PINN Foundation Summary

**Physics regularizer with 4 Cosserat rod constraints, ReLoBRaLo adaptive loss balancing, and physics-based nondimensionalization in src/pinn/ package**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-17T19:27:03Z
- **Completed:** 2026-03-17T19:30:02Z
- **Tasks:** 1
- **Files created:** 5

## Accomplishments
- Created src/pinn/ package with PhysicsRegularizer, ReLoBRaLo, and NondimScales
- PhysicsRegularizer computes 4 algebraic constraint losses: kinematic (x,y), angular, curvature-moment, energy
- ReLoBRaLo balances multi-term losses adaptively with random lookback and EMA smoothing
- NondimScales provides roundtrip-exact physics-based scaling for 124-dim rod states
- 19 passing unit tests covering all components, gradient flow, and package imports

## Task Commits

Each task was committed atomically:

1. **Task 1: Create src/pinn/ package with regularizer, loss balancing, and nondim** - `f40496f` (feat)

## Files Created/Modified
- `src/pinn/__init__.py` - Package root with public exports
- `src/pinn/regularizer.py` - PhysicsRegularizer with kinematic + angular + curvature-moment + energy constraints
- `src/pinn/loss_balancing.py` - ReLoBRaLo adaptive loss balancing
- `src/pinn/nondim.py` - Physics-based nondimensionalization scales
- `tests/test_pinn.py` - 19 unit tests for PINN-01, PINN-02, PINN-05, PINN-07

## Decisions Made
- PhysicsRegularizer uses trapezoidal integration (average of current and next velocities) for kinematic consistency
- Curvature-moment constraint weighted 0.1x since it is approximate (uses yaw differences as curvature proxy)
- Energy threshold of 10.0 (generous) to only penalize physically unreasonable KE changes
- NondimScales: F_ref = E*I/L^2 = 1.5708e-6 N (characteristic elastic force for r=0.001m rod)

## Deviations from Plan

None - plan executed exactly as written. Source files were already present as untracked files; tests were written and verified against existing implementation.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- src/pinn/ package ready for use in Plan 02 (DD-PINN ansatz implementation)
- PhysicsRegularizer can be added to any existing surrogate training loop via `loss += lambda_phys * regularizer(state, delta_pred)`
- ReLoBRaLo ready for multi-term PINN loss balancing in subsequent plans

---
*Phase: 13-implement-pinn-and-dd-pinn-surrogate-models*
*Completed: 2026-03-17*
