---
phase: 13
slug: implement-pinn-and-dd-pinn-surrogate-models
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-17
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing in project) |
| **Config file** | tests/ directory with existing test files |
| **Quick run command** | `python -m pytest tests/test_pinn.py -x -q --timeout=60` |
| **Full suite command** | `python -m pytest tests/ -v --timeout=120` |
| **Estimated runtime** | ~30 seconds (unit), ~120 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_pinn.py -x -q --timeout=60`
- **After every plan wave:** Run `python -m pytest tests/ -v --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | PINN-01 | unit | `pytest tests/test_pinn.py::test_physics_regularizer -x` | ❌ W0 | ⬜ pending |
| 13-01-02 | 01 | 1 | PINN-02 | unit | `pytest tests/test_pinn.py::test_regularizer_gradients -x` | ❌ W0 | ⬜ pending |
| 13-01-03 | 01 | 1 | PINN-03 | unit | `pytest tests/test_pinn.py::test_ansatz_ic -x` | ❌ W0 | ⬜ pending |
| 13-01-04 | 01 | 1 | PINN-04 | unit | `pytest tests/test_pinn.py::test_ansatz_derivative -x` | ❌ W0 | ⬜ pending |
| 13-01-05 | 01 | 1 | PINN-05 | unit | `pytest tests/test_pinn.py::test_fourier_features -x` | ❌ W0 | ⬜ pending |
| 13-01-06 | 01 | 1 | PINN-07 | integration | `pytest tests/test_pinn.py::test_loss_balancing -x` | ❌ W0 | ⬜ pending |
| 13-02-01 | 02 | 2 | PINN-08 | integration | `pytest tests/test_pinn.py::test_nondim_roundtrip -x` | ❌ W0 | ⬜ pending |
| 13-02-02 | 02 | 2 | PINN-09 | integration | `pytest tests/test_pinn.py::test_rft_vs_pyelastica -x` | ❌ W0 | ⬜ pending |
| 13-03-01 | 03 | 3 | PINN-10 | integration | `pytest tests/test_pinn.py::test_ddpinn_forward_interface -x` | ❌ W0 | ⬜ pending |
| 13-03-02 | 03 | 3 | PINN-11 | integration | `pytest tests/test_pinn.py::test_collocation_sobol -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_pinn.py` — stubs for PINN-01 through PINN-11
- [ ] `src/pinn/__init__.py` — package structure
- [ ] Verify pytest available in environment

*Existing infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Physics regularizer improves omega_z RMSE | PINN-EVAL-01 | Requires training run | Train with regularizer, compare per-component RMSE to Phase 3 baseline |
| DD-PINN forward() compatible with Phase 4 pipeline | PINN-COMPAT-01 | Requires integration test with full pipeline | Load DD-PINN checkpoint, run through validation pipeline |
| f_SSM matches PyElastica within tolerance | PINN-PHYSICS-01 | Requires PyElastica simulation comparison | Run test states through f_SSM and PyElastica, compare outputs |
| Comparison plots generated in figures/pinn/ | PINN-VIZ-01 | Visual inspection | Check bar charts, scatter plots, convergence curves |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
