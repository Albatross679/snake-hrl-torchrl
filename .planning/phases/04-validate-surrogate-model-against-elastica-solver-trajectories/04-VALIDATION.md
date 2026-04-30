---
phase: 4
slug: validate-surrogate-model-against-elastica-solver-trajectories
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-11
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/test_surrogate_phase3.py (existing) |
| **Quick run command** | `python -m pytest tests/test_surrogate_phase4.py -x -q` |
| **Full suite command** | `python -m pytest tests/test_surrogate_phase4.py -v` |
| **Estimated runtime** | ~30 seconds (unit tests only, no physics sim) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_surrogate_phase4.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_surrogate_phase4.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | Model discovery | unit | `pytest tests/test_surrogate_phase4.py::test_model_discovery` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | Architecture dispatch | unit | `pytest tests/test_surrogate_phase4.py::test_arch_dispatch` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | Scenario generation | unit | `pytest tests/test_surrogate_phase4.py::test_scenario_generation` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | Error metrics | unit | `pytest tests/test_surrogate_phase4.py::test_error_metrics` | ❌ W0 | ⬜ pending |
| 04-01-05 | 01 | 1 | Pass/fail grading | unit | `pytest tests/test_surrogate_phase4.py::test_grading` | ❌ W0 | ⬜ pending |
| 04-01-06 | 01 | 1 | Report generation | unit | `pytest tests/test_surrogate_phase4.py::test_report_gen` | ❌ W0 | ⬜ pending |
| 04-01-07 | 01 | 1 | Figure generation | integration | `pytest tests/test_surrogate_phase4.py::test_figures` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_surrogate_phase4.py` — test stubs for model discovery, arch dispatch, scenarios, metrics, grading, report, figures
- [ ] Shared fixtures for mock checkpoint directories and surrogate model configs

*Existing infrastructure: pytest installed, test_surrogate_phase3.py provides patterns to follow.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Figure visual quality | Trajectory overlays readable | Subjective visual assessment | Open figures in figures/surrogate_validation/, verify real vs surrogate trajectories are distinguishable |
| Report readability | Validation report coherent | Document structure review | Read output/surrogate/validation_report.md, verify all scenarios covered |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
