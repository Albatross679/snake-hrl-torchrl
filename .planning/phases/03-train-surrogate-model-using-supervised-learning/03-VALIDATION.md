---
phase: "03"
slug: train-surrogate-model-using-supervised-learning
status: in_progress
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-10
updated: 2026-03-11
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `python3 -m pytest tests/test_surrogate_phase3.py -x -q` |
| **Full suite command** | `python3 -m pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds (unit tests) |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/test_surrogate_phase3.py -x -q`
- **After every plan wave:** Run `python3 -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | SURR-01 | unit+tdd | `python3 -m pytest tests/test_surrogate_phase3.py -x -q` (TransformerSurrogateModel, config fields, RMSNorm) | pending | pending |
| 03-01-02 | 01 | 1 | SURR-01 | integration | `python3 -m pytest tests/test_surrogate_phase3.py -x -q` (FlatStepDataset wiring, --arch CLI, sweep configs) | pending | pending |
| 03-02-01 | 02 | 2 | SURR-02 | smoke | `python3 -c "import os,json; ..."` (smoke test outputs: model.pt, metrics.json with val_loss) | pending | pending |
| 03-02-02 | 02 | 2 | SURR-02 | runtime | `tmux has-session -t gsd-sweep` (sequential sweep running in tmux) | pending | pending |
| 03-03-01 | 03 | 3 | SURR-03, SURR-04 | runtime | `python3 -m aprx_model_elastica.analyze_sweep --output-base output/surrogate --figures-dir figures/surrogate_training` | pending | pending |
| 03-03-02 | 03 | 3 | SURR-05 | human | Checkpoint: human reviews plots and selects best model | pending | pending |
| 03-03-03 | 03 | 3 | SURR-05 | integration | `python3 -c "import json,os; s=json.load(open('output/surrogate/best/selection.json')); assert 'config_name' in s and os.path.exists('output/surrogate/best/model.pt')"` | pending | pending |

*Status: pending · green · red · flaky*

---

## Wave 0 Requirements

Wave 0 test file must be created as part of Plan 01 Task 1 (TDD):

- [ ] `tests/test_surrogate_phase3.py` — TransformerSurrogateModel shape/output tests, config field tests, sweep config count tests

---

## Key Files

| File | Plan | Purpose |
|------|------|---------|
| `tests/test_surrogate_phase3.py` | 01 | Unit + integration tests for all Phase 3 code changes |
| `aprx_model_elastica/sweep.py` | 01 (code), 02 (run) | 15-config sweep runner, sequential via subprocess.run() |
| `aprx_model_elastica/analyze_sweep.py` | 03 | Analysis script producing diagnostic plots and RMSE |
| `script/launch_sweep.sh` | 02 | tmux launch script for full sweep |

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Sweep final metrics across 15 configs | SURR-03 | Requires ~hours of training | Check sweep_summary.json after tmux session completes |
| Diagnostic plot quality and correctness | SURR-04 | Visual inspection | Review 4 plots in figures/surrogate_training/ |
| Human approval of best checkpoint | SURR-05 | By design (checkpoint gate) | Plan 03 Task 2 checkpoint task |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** Pending — plans 01-03 ready for execution
