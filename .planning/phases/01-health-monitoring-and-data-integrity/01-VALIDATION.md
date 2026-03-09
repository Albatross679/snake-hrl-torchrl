---
phase: 1
slug: health-monitoring-and-data-integrity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `python3 -m pytest tests/test_monitor.py -x -v` |
| **Full suite command** | `python3 -m pytest tests/ -x -v --timeout=60` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/test_monitor.py -x -v`
- **After every plan wave:** Run `python3 -m pytest tests/ -x -v --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | HLTH-01 | unit | `python3 -m pytest tests/test_monitor.py::test_per_worker_status -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | HLTH-02 | integration | `python3 -m pytest tests/test_monitor.py::test_worker_respawn -x` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | HLTH-03 | unit | `python3 -m pytest tests/test_monitor.py::test_stall_detection -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | HLTH-04 | unit | `python3 -m pytest tests/test_monitor.py::test_nan_filtering -x` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | HLTH-05 | integration | `python3 -m pytest tests/test_monitor.py::test_graceful_shutdown -x` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 1 | OBSV-01 | unit (mock) | `python3 -m pytest tests/test_monitor.py::test_wandb_alerts -x` | ❌ W0 | ⬜ pending |
| 01-03-02 | 03 | 1 | OBSV-04 | unit | `python3 -m pytest tests/test_monitor.py::test_event_log -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_monitor.py` — stubs for all 7 requirements (HLTH-01 through HLTH-05, OBSV-01, OBSV-04)
- [ ] `tests/conftest.py` — shared fixtures for mock W&B, temp save directories, fake worker processes

*Existing infrastructure covers framework install (pytest 9.0.2 already installed).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| W&B alerts appear in dashboard | OBSV-01 | Requires live W&B connection | Run collector + monitor, kill a worker, check W&B alerts panel |
| Overnight stability | All HLTH-* | Requires 8+ hour run | Run full overnight collection, review event log and W&B dashboard next morning |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
