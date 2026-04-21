---
phase: 17
slug: design-pinn-debugging-workflow-and-diagnostic-pipeline
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-26
---

# Phase 17 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/ -x -q --tb=short` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q --tb=short`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 17-01-T1 | 17-01 | 1 | PDIAG-01 | unit | `pytest tests/test_pinn_probes.py -x -q --tb=short -k "not analyze_pde and not run_probe"` | Wave 0 | pending |
| 17-01-T2 | 17-01 | 1 | PDIAG-02 | unit | `pytest tests/test_pinn_probes.py -x -q --tb=short` | Wave 0 | pending |
| 17-02-T1 | 17-02 | 2 | PDIAG-03, PDIAG-05 | unit | `pytest tests/test_pinn_diagnostics.py -x -q --tb=short` | Wave 0 | pending |
| 17-02-T2 | 17-02 | 2 | PDIAG-04 | integration | `python3 -c "from src.pinn.train_pinn import DDPINNTrainConfig; print('ok')"` | exists | pending |
| 17-03-T1 | 17-03 | 1 | PDIAG-06 | file check | `test -f .claude/skills/pinn-debug/SKILL.md && wc -l .claude/skills/pinn-debug/SKILL.md` | Wave 0 | pending |
| 17-03-T2 | 17-03 | 1 | PDIAG-06 | file check | `test -f .claude/skills/pinn-debug/references/failure-modes.md && wc -l .claude/skills/pinn-debug/references/failure-modes.md` | Wave 0 | pending |

*Status: pending | green | red | flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_pinn_probes.py` — created by Plan 17-01 Task 1, covers PDIAG-01 (probe PDEs) and PDIAG-02 (system analysis)
- [ ] `tests/test_pinn_diagnostics.py` — created by Plan 17-02 Task 1, covers PDIAG-03 (diagnostics middleware) and PDIAG-05 (NTK)
- [ ] `src/pinn/probe_pdes.py` — created by Plan 17-01, provides probe PDE suite
- [ ] `src/pinn/diagnostics.py` — created by Plan 17-02 Task 1, provides PINNDiagnostics middleware
- [ ] `.claude/skills/pinn-debug/SKILL.md` — created by Plan 17-03 Task 1, provides Claude Code skill
- [ ] `.claude/skills/pinn-debug/references/failure-modes.md` — created by Plan 17-03 Task 2, provides failure mode reference

*All Wave 0 test files are created by the plans themselves (TDD tasks create tests before implementation).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| SKILL.md is readable and self-contained for Claude Code PINN debugging | PDIAG-06 | Requires human judgment on clarity, completeness, and diagnostic usefulness of decision tree | Open `.claude/skills/pinn-debug/SKILL.md`, verify: (1) all 4 phases present and logically ordered, (2) decision tree has clear branching with actionable remediation, (3) metric names match `src/pinn/diagnostics.py` exports, (4) probe names match `src/pinn/probe_pdes.py` ALL_PROBES |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved
