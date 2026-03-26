---
phase: 17-design-pinn-debugging-workflow-and-diagnostic-pipeline
plan: 03
subsystem: diagnostics
tags: [pinn, debugging, decision-tree, claude-skill, ntk, relobralo, spectral-bias, fourier-features]

# Dependency graph
requires:
  - phase: 13-implement-pinn-and-dd-pinn-surrogate-models
    provides: DD-PINN implementation, ReLoBRaLo, CosseratRHS, Fourier features, ansatz
provides:
  - pinn-debug Claude Code skill with 4-phase diagnostic workflow
  - PINN failure mode reference document with 7 documented failure modes
  - Decision trees for PINN training failure diagnosis
affects: [17-01, 17-02, pinn-training, dd-pinn-debugging]

# Tech tracking
tech-stack:
  added: []
  patterns: [4-phase diagnostic skill structure for PINN debugging, probe PDE validation before training]

key-files:
  created:
    - .claude/skills/pinn-debug/SKILL.md
    - .claude/skills/pinn-debug/references/failure-modes.md

key-decisions:
  - "Mirrored rl-debug skill structure exactly: 4 phases, decision tree format, quick symptom lookup, key principle"
  - "Full decision tree inline in SKILL.md so Claude can diagnose without reading source code"
  - "7 failure modes in reference doc covering optimization, architecture, physics, and DD-PINN-specific issues"

patterns-established:
  - "PINN skill structure: probe validation, dashboard metrics, loss decision tree, physics sub-tree"
  - "Failure mode documentation format: Signature, Root Cause, Diagnostic Metrics, Remediation, Code Example"

requirements-completed: [PDIAG-06]

# Metrics
duration: 4min
completed: 2026-03-26
---

# Phase 17 Plan 03: PINN Debug Skill Summary

**Claude Code skill for PINN debugging with 4-phase decision tree, 9-branch fault isolation, and 7-failure-mode reference backed by NeurIPS/ICML literature**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-26T13:19:59Z
- **Completed:** 2026-03-26T13:24:09Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Created pinn-debug SKILL.md (270 lines) with complete 4-phase diagnostic workflow mirroring rl-debug structure
- Created failure-modes.md reference (276 lines) covering 7 PINN failure modes with literature-backed remediation
- Decision tree covers 9 diagnostic branches for loss not decreasing, plus 3-tier physics accuracy sub-tree
- Skill references all diagnostic metric names from src/pinn/diagnostics.py and probe PDE names from src/pinn/probe_pdes.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Create pinn-debug SKILL.md with full 4-phase decision tree** - `d99717b` (feat)
2. **Task 2: Create failure-modes.md reference document** - `aa0a975` (feat)

## Files Created/Modified
- `.claude/skills/pinn-debug/SKILL.md` - Complete PINN training debugger skill with 4 phases, decision trees, metric references, symptom lookup
- `.claude/skills/pinn-debug/references/failure-modes.md` - 7 PINN failure modes with signatures, root causes, diagnostic metrics, remediation, literature citations

## Decisions Made
- Mirrored rl-debug skill structure exactly (4 phases, same section order, same formatting conventions)
- Included full decision tree inline in SKILL.md rather than referencing external documents -- Claude can diagnose from the skill alone
- Used 4 probe PDEs (not 5) per plan spec -- ProbePDE5 (simplified Cosserat) deferred to Plan 01 implementation
- Included 6 literature citations covering all referenced PINN failure analysis papers

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - both files are complete documentation artifacts with no code stubs.

## Next Phase Readiness
- pinn-debug skill ready for use by Claude Code in PINN training sessions
- Skill references src/pinn/diagnostics.py and src/pinn/probe_pdes.py which are created by Plans 01 and 02
- All 3 plans in Phase 17 can execute in parallel (wave 1) since they are independent documentation/code artifacts

## Self-Check: PASSED

All files found, all commits verified.

---
*Phase: 17-design-pinn-debugging-workflow-and-diagnostic-pipeline*
*Completed: 2026-03-26*
