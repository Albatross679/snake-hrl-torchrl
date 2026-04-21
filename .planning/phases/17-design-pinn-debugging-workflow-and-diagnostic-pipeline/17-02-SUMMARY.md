---
phase: 17-design-pinn-debugging-workflow-and-diagnostic-pipeline
plan: 02
subsystem: testing
tags: [pinn, diagnostics, ntk, middleware, wandb, gradient-analysis, relobralo]

requires:
  - phase: 13-implement-pinn-and-dd-pinn-surrogate-models
    provides: CosseratRHS, NondimScales, DDPINNModel, ReLoBRaLo, train_pinn.py
  - phase: 17-01
    provides: probe PDE suite (run_probe_validation, ALL_PROBES)

provides:
  - PINNDiagnostics middleware class with 6 diagnostic metric categories
  - NTK eigenvalue computation with parameter subsampling
  - train_pinn.py integration with diagnostics hooks and probe pre-flight
  - Per-loss gradient norm monitoring every 10 epochs
  - --skip-probes CLI flag for probe pre-flight opt-out

affects: [17-03, pinn-debug-skill, future-pinn-training]

tech-stack:
  added: []
  patterns: [pinn-diagnostics-middleware, ntk-parameter-subsampling, probe-preflight-validation]

key-files:
  created:
    - src/pinn/diagnostics.py
    - tests/test_pinn_diagnostics.py
    - logs/pinn-diagnostics-middleware.md
  modified:
    - src/pinn/train_pinn.py

key-decisions:
  - "Log-only diagnostics per D-07: no wandb.alert(), no auto-stopping"
  - "NTK computed via per-sample Jacobian with parameter subsampling (n_params_sample=500)"
  - "Per-loss gradient norms computed every 10 epochs on first batch only to minimize overhead"
  - "Probe pre-flight runs by default before training, skippable with --skip-probes (D-04)"

patterns-established:
  - "PINNDiagnostics mirrors src/trainers/diagnostics.py: deque-based history, log_step() per epoch"
  - "compute_ntk_eigenvalues is standalone function for reuse outside middleware"
  - "diagnostics/ W&B prefix for all PINN diagnostic metrics"
  - "Probe pre-flight integrated into train_single_config() before training loop"

requirements-completed: [PDIAG-03, PDIAG-04, PDIAG-05]

duration: 7min
completed: 2026-03-26
---

# Phase 17 Plan 02: PINNDiagnostics Middleware Summary

**PINNDiagnostics middleware with loss ratios, gradient norms, residual stats, ReLoBRaLo health, NTK eigenvalues, and probe pre-flight integration into train_pinn.py**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-26T13:31:13Z
- **Completed:** 2026-03-26T13:38:23Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- PINNDiagnostics class with 6 diagnostic metric categories mirroring RL diagnostics pattern
- NTK eigenvalue computation using per-sample Jacobian with parameter subsampling for tractability
- Non-invasive train_pinn.py integration: probe pre-flight, epoch-level log_step, per-loss gradient norms
- 13 unit tests all passing on CPU

## Task Commits

Each task was committed atomically:

1. **Task 1: Create PINNDiagnostics middleware with NTK computation** - `7b93a9a` (test RED) + `3d633c2` (feat GREEN)
2. **Task 2: Integrate diagnostics and probe pre-flight into train_pinn.py** - `2467109` (feat)

_TDD task: test commit precedes implementation commit._

## Files Created/Modified
- `src/pinn/diagnostics.py` - PINNDiagnostics class, compute_ntk_eigenvalues, loss ratio, gradient norms, residual stats, ReLoBRaLo health, per-component violations
- `tests/test_pinn_diagnostics.py` - 13 tests for all diagnostic methods and NTK computation
- `src/pinn/train_pinn.py` - Added imports, --skip-probes flag, probe pre-flight, PINNDiagnostics instantiation, log_step call, per-loss gradient norms
- `logs/pinn-diagnostics-middleware.md` - Documentation log entry

## Decisions Made
- Log-only to W&B per D-07: no wandb.alert() or check_pinn_alerts() implemented. The research document contained alert code examples that were explicitly excluded per the locked decision.
- NTK computed via per-sample backward passes building a (N x n_params_sample) Jacobian, then K = J @ J.T with eigvalsh. More robust than torch.autograd.functional.jacobian for variable-output models.
- Per-loss gradient norms run every 10 epochs on batch_idx==0 only, to avoid doubling compute cost each epoch.
- Floating-point tolerance in test_compute_loss_ratio_basic relaxed from 1e-6 to 1e-3 due to torch.tensor(0.01).item() precision.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Relaxed floating-point test tolerance**
- **Found during:** Task 1 (TDD GREEN phase)
- **Issue:** test_compute_loss_ratio_basic asserted abs(ratio - 100.0) < 1e-6 but torch.tensor(0.01).item() introduces ~2.2e-6 error
- **Fix:** Relaxed tolerance to 1e-3 (still precise enough to validate correctness)
- **Files modified:** tests/test_pinn_diagnostics.py
- **Verification:** Test passes with relaxed tolerance
- **Committed in:** 3d633c2 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Trivial tolerance adjustment, no scope change.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functions fully implemented with real logic.

## Next Phase Readiness
- PINNDiagnostics middleware ready for use in PINN training runs
- Probe pre-flight validates pipeline mechanics before real training
- pinn-debug skill (Plan 03, already complete) references these diagnostics
- Full W&B dashboard monitoring available under diagnostics/ prefix

## Self-Check: PASSED

- FOUND: src/pinn/diagnostics.py
- FOUND: tests/test_pinn_diagnostics.py
- FOUND: src/pinn/train_pinn.py
- FOUND: commit 7b93a9a (test RED)
- FOUND: commit 3d633c2 (feat GREEN)
- FOUND: commit 2467109 (feat Task 2)

---
*Phase: 17-design-pinn-debugging-workflow-and-diagnostic-pipeline*
*Completed: 2026-03-26*
