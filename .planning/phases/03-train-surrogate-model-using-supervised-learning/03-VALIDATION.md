---
phase: "03"
slug: train-surrogate-model-using-supervised-learning
status: in_progress
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-10
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `python3 -m pytest tests/test_surrogate_arch.py -x -q` |
| **Full suite command** | `python3 -m pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds (unit tests) |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/test_surrogate_arch.py -x -q`
- **After every plan wave:** Run `python3 -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | SURR-01 | integration | `python3 -c "from aprx_model_elastica.train_surrogate import parse_args; ..."` | ✅ | ✅ green |
| 03-01-02 | 01 | 1 | SURR-02 | integration | `python3 -c "import json,os; runs=[d for d in os.listdir('output/surrogate') if d.startswith('sweep_')]; assert len(runs)>=5"` | ✅ | ✅ green |
| 03-02-01 | 02 | 2 | SURR-03 | integration | `python3 -c "import os; figs=[...]; ..."` | ✅ | ✅ green |
| 03-03-01 | 03 | 1 | ARCH-01 | unit | `pytest tests/test_surrogate_arch.py::test_residual_model_forward -x` | ✅ | ✅ green |
| 03-03-02 | 03 | 1 | ARCH-02 | unit | `pytest tests/test_surrogate_arch.py::test_history_model_forward -x` | ✅ | ✅ green |
| 03-03-03 | 03 | 1 | ARCH-03 | unit | `pytest tests/test_surrogate_arch.py::test_trajectory_dataset_windows -x` | ✅ | ✅ green |
| 03-03-04 | 03 | 1 | ARCH-04 | unit | `pytest tests/test_surrogate_arch.py::test_train_cli_args -x` | ✅ | ✅ green |
| 03-04-01 | 04 | 2 | ARCH-05 | smoke | `python3 -m aprx_model_elastica.arch_sweep --epochs 1 --dry-run` | ✅ | ✅ green |
| 03-05-01 | 05 | 3 | ARCH-01..05 | human+auto | See plan 05 Task 2 checkpoint gate | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

All Wave 0 requirements are met:

- [x] `tests/test_surrogate_arch.py` — ARCH-01 through ARCH-04 unit tests
- [x] `aprx_model_elastica/arch_sweep.py` — sweep runner with `--dry-run` (covers ARCH-05)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Sweep final metrics improvement over baseline | Architecture selection | Requires 5–7 hours of training | Compare arch_sweep_summary.json to baseline val_loss=0.2161 |
| Rollout drift reduction vs baseline | Rollout quality | Requires Phase 4 trajectory validation tooling | Run validate.py on best arch checkpoint vs baseline |
| Human approval of best checkpoint | Phase gate | By design | Plan 05 Task 2 checkpoint task |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** plans 01–04 complete; plan 05 pending sweep completion
