---
phase: 03-train-surrogate-model-using-supervised-learning
plan: "04"
subsystem: ml-surrogate
tags: [state-representation, com-relative-velocity, surrogate-model, pytorch]

# Dependency graph
requires:
  - phase: 03-train-surrogate-model-using-supervised-learning
    provides: "128-dim relative state representation (raw_to_relative/relative_to_raw in state.py)"
provides:
  - "130-dim CoM-relative velocity state representation (REL_STATE_DIM=130)"
  - "REL_COM_VEL_X, REL_COM_VEL_Y named slice constants"
  - "Invertible raw_to_relative() with CoM velocity subtraction from per-node velocities"
  - "Updated SurrogateModelConfig defaults (state_dim=130, input_dim=139, output_dim=130)"
  - "Named-constant-only evaluate() and compute_single_step_loss() with com_vel component"
affects: [surrogate-training, surrogate-validation, rl-with-surrogate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CoM velocity subtraction for velocity invariance (deformation signal isolation)"
    - "Named REL_* slice constants for all dimension indexing (no hardcoded indices)"

key-files:
  created: []
  modified:
    - papers/aprx_model_elastica/state.py
    - papers/aprx_model_elastica/train_config.py
    - papers/aprx_model_elastica/train_surrogate.py
    - papers/aprx_model_elastica/validate.py
    - papers/aprx_model_elastica/model.py

key-decisions:
  - "CoM velocity at indices 4-5 in relative state (after heading, before positions)"
  - "All REL_* slice constants shifted +2 to accommodate new CoM velocity dims"
  - "data_dir default changed to data/surrogate_rl_step for on-the-fly 124->130 conversion"
  - "model.py docstrings use state_dim/output_dim/input_dim placeholders instead of hardcoded numbers"

patterns-established:
  - "CoM-relative velocity: subtract mean node velocity from per-node velocities"
  - "Named slice constants for all R2/loss component computation (no hardcoded indices)"

requirements-completed: [SURR-02, SURR-04]

# Metrics
duration: 10min
completed: 2026-03-17
---

# Phase 03 Plan 04: Switch to CoM-Relative Velocities Summary

**130-dim CoM-relative velocity state representation with invertible transform and named-constant-only evaluate/loss code**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-17T15:24:37Z
- **Completed:** 2026-03-17T15:34:58Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Extended raw_to_relative() to subtract CoM velocity from per-node velocities, adding 2 dims (CoM vel x/y) for a total of 130-dim relative state
- Verified perfect round-trip reconstruction: raw(124) -> relative(130) -> raw(124) within 1e-5 tolerance for single and batched inputs
- Replaced all hardcoded dimension indices in evaluate() and compute_single_step_loss() with named REL_* slice constants
- Updated SurrogateModelConfig defaults to state_dim=130, input_dim=139, output_dim=130
- Eliminated all stale "128-dim" references across train_surrogate.py, validate.py, and model.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Update state.py -- CoM-relative velocity transform and constants** - `bfefe78` (feat)
2. **Task 2: Update train_config.py, train_surrogate.py, validate.py, model.py** - `3d02787` (feat)

## Files Created/Modified
- `papers/aprx_model_elastica/state.py` - Added CoM velocity subtraction to raw_to_relative(), reconstruction to relative_to_raw(), REL_COM_VEL_X/Y constants, shifted all REL_* slices by +2
- `papers/aprx_model_elastica/train_config.py` - Updated SurrogateModelConfig defaults (130/139/130), data_dir to data/surrogate_rl_step
- `papers/aprx_model_elastica/train_surrogate.py` - Added REL_COM_VEL imports, named constants in evaluate() and compute_single_step_loss(), added com_vel R2/loss component
- `papers/aprx_model_elastica/validate.py` - Updated dimension comments from 128 to 130
- `papers/aprx_model_elastica/model.py` - Updated all docstrings to use state_dim/output_dim/input_dim instead of hardcoded 124/131

## Decisions Made
- CoM velocity placed at indices 4-5 (after heading sin/cos, before relative positions) to maintain logical grouping of global quantities at the front of the state vector
- Used named constants everywhere instead of hardcoded indices, even for com (0:2) and heading (2:4) which are unchanged, to ensure grep-based verification catches stale references
- data_dir default changed from pre-processed data/surrogate_rl_step_rel128 to raw data/surrogate_rl_step with on-the-fly conversion via raw_to_relative()
- model.py docstrings use parametric references (state_dim, output_dim) rather than hardcoded numbers since dimensions now change between 128 and 130 representations

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 130-dim representation is ready for surrogate training
- Raw 124-dim data in data/surrogate_rl_step will be automatically converted on-the-fly
- All downstream code (training loop, validation, model) is updated to handle 130-dim
- Next step: retrain surrogate model with new representation to verify velocity R2 improvement

## Self-Check: PASSED

All 5 modified files exist. Both task commits (bfefe78, 3d02787) verified in git log. Summary file exists.

---
*Phase: 03-train-surrogate-model-using-supervised-learning*
*Completed: 2026-03-17*
