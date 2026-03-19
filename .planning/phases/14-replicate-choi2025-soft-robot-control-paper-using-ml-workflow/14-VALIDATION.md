---
phase: 14
slug: replicate-choi2025-soft-robot-control-paper-using-ml-workflow
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-19
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_choi2025.py -x -q` |
| **Full suite command** | `pytest tests/test_choi2025.py -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_choi2025.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 14-01-01 | 01 | 1 | CHOI-01 | unit | `pytest tests/test_choi2025.py::TestChoi2025PPOConfig -v` | ✅ green |
| 14-01-02 | 01 | 1 | CHOI-02 | integration | `pytest tests/test_choi2025.py::TestTrainPPOWiring -v` | ✅ green |
| 14-01-03 | 01 | 1 | CHOI-03 | integration | `pytest tests/test_choi2025.py::TestRunExperimentWiring tests/test_choi2025.py::TestEvaluateWiring -v` | ✅ green |
| 14-02-01 | 02 | 2 | CHOI-04 | smoke | `pytest tests/test_choi2025.py::TestConfigToTrainerWiring -v` | ✅ green |
| 14-02-02 | 02 | 2 | CHOI-05 | N/A | Launch action — verified by tmux session existence | ✅ green |
| 14-03-01 | 03 | 3 | CHOI-06 | unit | `pytest tests/test_choi2025.py::TestRecordDualAlgo -v` | ✅ green |
| 14-03-02 | 03 | 3 | CHOI-07 | N/A | Documentation artifact — verified by file existence | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No new framework or fixtures needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| PPO learning signal with full 1M-frame training | CHOI-05 | Full training takes ~9 days; cannot run in CI | Check W&B `choi2025-replication` project for upward reward trends in PPO runs |
| Video rollout visual quality | CHOI-06 | Requires human visual assessment | Play MP4 files in `media/choi2025/`; verify SAC shows intentional manipulation, not random motion |
| Mock physics fidelity | CHOI-04 | Domain expertise needed for physics comparison | Compare mock vs real DisMech rod dynamics when DisMech is installed |
| W&B dashboard organization | CHOI-05 | Requires web UI access | Open W&B project `choi2025-replication`, verify runs are organized with correct naming and timing metrics |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 5s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-03-19

---

## Validation Audit 2026-03-19

| Metric | Count |
|--------|-------|
| Gaps found | 5 |
| Resolved | 5 |
| Escalated | 0 |

Test breakdown: 15 (CHOI-01) + 6 (CHOI-02) + 11 (CHOI-03) + 4 (CHOI-04) + 5 (CHOI-06) = 41 new tests.
All 60 tests passing (19 original + 41 new).
