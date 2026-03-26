---
phase: 17-design-pinn-debugging-workflow-and-diagnostic-pipeline
plan: 01
subsystem: testing
tags: [pinn, diagnostics, probe-pdes, analytical-solutions, nondimensionalization]

requires:
  - phase: 13-implement-pinn-and-dd-pinn-surrogate-models
    provides: CosseratRHS, NondimScales, collocation sampling, DD-PINN model

provides:
  - Probe PDE suite with 4 generic probes (heat, advection, Burgers, reaction-diffusion)
  - PDE system analysis function for CosseratRHS nondimensionalization validation
  - Probe validation runner for pre-flight PINN checks
  - ALL_PROBES registry mirroring RL probe_envs.py pattern

affects: [17-02, 17-03, pinn-debug-skill]

tech-stack:
  added: []
  patterns: [probe-pde-validation, pde-system-analysis, tdd-for-pinn-diagnostics]

key-files:
  created:
    - src/pinn/probe_pdes.py
    - tests/test_pinn_probes.py
    - logs/pinn-probe-pde-suite.md
  modified: []

key-decisions:
  - "Generic probe PDEs test fundamental PINN capabilities without Cosserat coupling"
  - "analyze_pde_system reports nondim_quality as good/acceptable/poor based on magnitude spread thresholds"
  - "Finite-difference Jacobian on 20-dim subspace for tractable stiffness estimation"
  - "Default CosseratRHS reports 'poor' nondim quality, confirming the diagnostic detects real issues"

patterns-established:
  - "Probe PDE pattern: _ProbePDEBase with analytical_solution, pde_residual, compute_loss, check_pass, pass_criterion"
  - "ALL_PROBES list of (name, class) tuples mirroring src/trainers/probe_envs.py"
  - "System analysis returns structured dict with per_term_magnitudes, nondim_quality, stiffness_indicator"

requirements-completed: [PDIAG-01, PDIAG-02]

duration: 7min
completed: 2026-03-26
---

# Phase 17 Plan 01: Probe PDE Validation Suite Summary

**4 generic probe PDEs with analytical solutions testing heat/advection/Burgers/reaction-diffusion, plus CosseratRHS nondimensionalization analysis reporting magnitude spread and stiffness**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-26T13:19:59Z
- **Completed:** 2026-03-26T13:27:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- 4 probe PDEs with known analytical solutions, each testing one progressive PINN capability
- PDE system analysis validates CosseratRHS term balance and nondimensionalization quality
- Probe runner trains and evaluates all probes with configurable model class
- 12 passing tests covering analytical correctness, residual computation, structural requirements

## Task Commits

Each task was committed atomically:

1. **Task 1: Create 4 probe PDE classes with analytical solutions and tests** - `7c0c543` (test) + `7a17107` (feat)
2. **Task 2: Create PDE system analysis, probe runner, and remaining tests** - `5120ce1` (feat)

_TDD tasks: test commits precede implementation commits._

## Files Created/Modified
- `src/pinn/probe_pdes.py` - 4 probe PDEs, _ProbeMLP helper, ALL_PROBES registry, run_probe_validation, analyze_pde_system
- `tests/test_pinn_probes.py` - 12 tests for probes, system analysis, and runner
- `logs/pinn-probe-pde-suite.md` - Documentation log entry

## Decisions Made
- Generic probes (not project-specific) per D-01: heat, advection, Burgers, reaction-diffusion cover fundamental PINN capabilities
- Nondim quality thresholds: good (<100x spread), acceptable (<1000x), poor (>=1000x) -- default CosseratRHS reports "poor" which correctly identifies the magnitude imbalance
- Finite-difference Jacobian computed on 20-dim subspace (first 20 state components) averaged over 10 samples for tractable stiffness estimation
- Sobol quasi-random sampling reused from existing collocation.py pattern (scipy.stats.qmc.Sobol)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functions fully implemented with real logic.

## Next Phase Readiness
- Probe PDEs ready for integration into train_pinn.py pre-flight checks (Plan 02)
- analyze_pde_system ready for PINNDiagnostics middleware integration (Plan 02)
- ALL_PROBES pattern established for pinn-debug Claude Code skill (Plan 03)

## Self-Check: PASSED

- FOUND: src/pinn/probe_pdes.py
- FOUND: tests/test_pinn_probes.py
- FOUND: logs/pinn-probe-pde-suite.md
- FOUND: commit 7c0c543 (test RED)
- FOUND: commit 7a17107 (feat GREEN)
- FOUND: commit 5120ce1 (feat Task 2)

---
*Phase: 17-design-pinn-debugging-workflow-and-diagnostic-pipeline*
*Completed: 2026-03-26*
