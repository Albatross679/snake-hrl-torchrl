---
phase: 15
slug: implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none (pytest defaults) |
| **Quick run command** | `python -m pytest tests/test_otpg.py -x` |
| **Full suite command** | `python -m pytest tests/ -x` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_otpg.py -x`
- **After every plan wave:** Run `python -m pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | OTPG-01 | unit | `python -m pytest tests/test_otpg.py::test_config -x` | ❌ W0 | ⬜ pending |
| 15-01-02 | 01 | 1 | OTPG-02 | unit | `python -m pytest tests/test_otpg.py::test_trainer_init -x` | ❌ W0 | ⬜ pending |
| 15-01-03 | 01 | 1 | OTPG-03 | unit | `python -m pytest tests/test_otpg.py::test_mmd_penalty -x` | ❌ W0 | ⬜ pending |
| 15-01-04 | 01 | 1 | OTPG-04 | unit | `python -m pytest tests/test_otpg.py::test_update_step -x` | ❌ W0 | ⬜ pending |
| 15-02-01 | 02 | 2 | OTPG-05 | smoke | `python -m pytest tests/test_otpg.py::test_short_training -x` | ❌ W0 | ⬜ pending |
| 15-01-05 | 01 | 1 | OTPG-06 | unit | `python -m pytest tests/test_otpg.py::test_checkpoint -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_otpg.py` — stubs for OTPG-01 through OTPG-06
- [ ] Framework install: pytest already available

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 100K frame learning signal | OTPG-eval | GPU required, long runtime | Run `python papers/choi2025/train_otpg.py --task follow_target`, check W&B for reward increase |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
