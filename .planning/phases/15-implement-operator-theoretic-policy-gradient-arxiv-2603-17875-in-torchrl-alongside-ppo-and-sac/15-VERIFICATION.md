---
phase: 15-implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac
verified: 2026-03-19T22:34:39Z
status: passed
score: 5/5 must-haves verified
---

# Phase 15: Implement Operator-Theoretic Policy Gradient Verification Report

**Phase Goal:** Add OTPG (Operator-Theoretic Policy Gradient) as a third RL trainer alongside PPO and SAC, implementing the MM-RKHS algorithm from Gupta & Mahajan (2026) adapted for continuous action spaces with neural network function approximation. Benchmark on the Choi2025 4-task suite for direct comparison with Phase 14's PPO/SAC results. This is a learning-signal validation (100K frames), not a full training campaign.

**Verified:** 2026-03-19T22:34:39Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

Truths derived from ROADMAP success criteria.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | OTPGConfig dataclass with MM-RKHS hyperparameters (beta, eta, MMD bandwidth/samples) inheriting RLConfig | VERIFIED | `src/configs/training.py` lines 116-150: `class OTPGConfig(RLConfig)` with beta=1.0, eta=1.0, mmd_bandwidth=1.0, mmd_num_samples=16, gae_lambda=0.95, value_coef=0.5, no clip_epsilon, no entropy_coef. Import test passes. |
| 2 | OTPGTrainer class with MMD-based trust region loss, following PPOTrainer pattern | VERIFIED | `src/trainers/otpg.py` (841 lines): `class OTPGTrainer` with `__init__`/`train()`/`_update()`/`_compute_mmd_penalty()`/`save_checkpoint()`/`load_checkpoint()`/`evaluate()`. Loss formula at line 610: `-surr_advantage.mean() + self.config.beta * mmd + (1.0 / self.config.eta) * kl + self.config.value_coef * critic_loss`. Uses `action_log_prob` key (2 occurrences, 0 `sample_log_prob`). RBF kernel MMD with linear-time estimator. Log-ratio clamped to [-20, 20]. TanhNormal old distribution reconstruction from stored loc/scale. |
| 3 | Unit tests pass for config, trainer init, MMD computation, update step, checkpoint roundtrip | VERIFIED | `python3 -m pytest tests/test_otpg.py -x -v` -- 12 passed in 5.32s. Tests: test_simple_pendulum_specs, test_reset_shape, test_step_shape, test_config, test_config_inherits_rl, test_no_clip_or_entropy, test_trainer_init, test_mmd_penalty, test_mmd_penalty_identical_dists, test_update_step, test_short_training, test_checkpoint. No gymnasium imports. |
| 4 | Choi2025OTPGConfig + train_otpg.py entry point wired into benchmark suite | VERIFIED | `papers/choi2025/config.py` lines 325-366: `class Choi2025OTPGConfig(OTPGConfig)` with __post_init__ setting name containing 'otpg'. `papers/choi2025/train_otpg.py` (148 lines): imports OTPGTrainer, Choi2025OTPGConfig; has --task, --total-frames, --num-envs, --max-wall-time, --resume args; ParallelEnv, RewardSum, GpuLock all present. Import test passes. |
| 5 | 100K-frame quick validation on follow_target completes without crash, W&B logs OTPG metrics | VERIFIED | `experiments/otpg-100k-validation-follow-target.md` documents completed run: 106496 frames, 512 episodes, best reward 21.18, rolling-100 = 16.92. W&B run URL provided. Metrics logged: mmd_penalty (0.0015 -> 0.0011), kl_divergence (8.04 -> 5.85), policy_entropy. No NaN or instability. Commit `7539c71` confirmed in git log. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/configs/training.py` | OTPGConfig dataclass | VERIFIED | Lines 116-150: `class OTPGConfig(RLConfig)` with all MM-RKHS fields. 275 total lines. |
| `src/trainers/otpg.py` | OTPGTrainer class with MMD-based policy gradient | VERIFIED | 841 lines (min 250 required). Full trainer with _compute_mmd_penalty, _update, save/load checkpoint, evaluate, train loop. |
| `src/trainers/__init__.py` | OTPGTrainer export | VERIFIED | Line 6: `from .otpg import OTPGTrainer`. Line 10: `"OTPGTrainer"` in __all__. |
| `tests/test_otpg.py` | Unit tests with SimplePendulum | VERIFIED | 407 lines (min 120 required). SimplePendulum EnvBase, 12 passing tests, no gymnasium imports. |
| `papers/choi2025/config.py` | Choi2025OTPGConfig dataclass | VERIFIED | Lines 325-366: `class Choi2025OTPGConfig(OTPGConfig)` with __post_init__ setting name. |
| `papers/choi2025/train_otpg.py` | OTPG training entry point for Choi2025 tasks | VERIFIED | 148 lines (min 80 required). Full CLI with argparse, ParallelEnv, RewardSum, GpuLock. |

### Key Link Verification

**Plan 01 Key Links:**

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/trainers/otpg.py` | `src/configs/training.py` | OTPGConfig import | WIRED | Line 41: `from src.configs.training import OTPGConfig` |
| `src/trainers/otpg.py` | `src/networks/actor.py` | create_actor() call | WIRED | Line 45: import, Line 89: `self.actor = create_actor(...)` |
| `src/trainers/otpg.py` | `src/networks/critic.py` | create_critic() call | WIRED | Line 46: import, Line 96: `self.critic = create_critic(...)` |
| `src/trainers/otpg.py` | batch key access | action_log_prob key (NOT sample_log_prob) | WIRED | 2 occurrences of `action_log_prob`, 0 of `sample_log_prob` |
| `src/trainers/__init__.py` | `src/trainers/otpg.py` | re-export | WIRED | Line 6: `from .otpg import OTPGTrainer` |

**Plan 02 Key Links:**

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `papers/choi2025/config.py` | `src/configs/training.py` | OTPGConfig import | WIRED | Line 24: `from src.configs.training import OTPGConfig, PPOConfig, SACConfig` |
| `papers/choi2025/train_otpg.py` | `src/trainers/otpg.py` | OTPGTrainer import | WIRED | Line 24: `from src.trainers.otpg import OTPGTrainer` |
| `papers/choi2025/train_otpg.py` | `papers/choi2025/config.py` | Choi2025OTPGConfig import | WIRED | Line 19: `from choi2025.config import Choi2025OTPGConfig, Choi2025EnvConfig, TaskType` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| OTPG-01 | Plan 01 | OTPGConfig dataclass with beta, eta, mmd fields, inheriting RLConfig, no entropy_coef or clip_epsilon | SATISFIED | `class OTPGConfig(RLConfig)` at lines 116-150. Import test passes. Fields verified: beta=1.0, eta=1.0, mmd_bandwidth=1.0, mmd_num_samples=16. `not hasattr(c, 'clip_epsilon')` and `not hasattr(c, 'entropy_coef')` confirmed. |
| OTPG-02 | Plan 01 | OTPGTrainer class following PPOTrainer pattern with init/train/_update, actor via create_actor, critic via create_critic, Adam optimizer, GAE, SyncDataCollector | SATISFIED | 841-line trainer with all methods. create_actor (line 89), create_critic (line 96), Adam (line 106), GAE (line 113), SyncDataCollector (line 133). |
| OTPG-03 | Plan 01 | MMD penalty computation with RBF kernel, linear-time O(n) estimator, configurable bandwidth/samples, returns finite non-negative scalar | SATISFIED | `_compute_mmd_penalty()` at lines 447-516 uses RBF kernel, linear-time paired estimator, clamp(min=0.0). test_mmd_penalty and test_mmd_penalty_identical_dists both pass. |
| OTPG-04 | Plan 01 | MM-RKHS loss function: -E[ratio*A] + beta*MMD^2 + (1/eta)*KL + value_coef*critic_loss, log-ratio clamped [-20,20], advantage normalized, NaN guards | SATISFIED | Loss at lines 610-615. Log-ratio clamp at line 564. Advantage normalization at lines 569-572. NaN guards at lines 621 and 640. test_update_step passes with all finite metrics. |
| OTPG-05 | Plan 02 | Choi2025 benchmark integration: Choi2025OTPGConfig + train_otpg.py, 100K validation without crash | SATISFIED | Choi2025OTPGConfig at config.py lines 325-366. train_otpg.py (148 lines) with full CLI. 100K validation completed: 512 episodes, best reward 21.18, W&B metrics logged. |
| OTPG-06 | Plan 01 | Checkpoint save/load: atomic saves with backup, round-trip restores actor/critic/optimizer state | SATISFIED | save_checkpoint (lines 760-791) with atomic temp-file rename and backup. load_checkpoint (lines 794-803) restores all state dicts. test_checkpoint passes: saves, zeros params, reloads, verifies restoration. |

No orphaned requirements found. All 6 OTPG requirements declared in REQUIREMENTS.md for Phase 15 are covered by Plans 01 and 02.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/trainers/otpg.py` | 58 | `return nullcontext()` | Info | Not a stub -- legitimate AMP fallback returning a no-op context manager. No concern. |

No TODOs, FIXMEs, placeholders, or empty implementations found in any phase files.

### Human Verification Required

### 1. Visual W&B Metrics Review

**Test:** Open W&B run URL (https://wandb.ai/qifan_wen-ohio-state-university/choi2025-replication/runs/bswz0spf) and inspect training curves.
**Expected:** train/mmd_penalty, train/kl_divergence, train/policy_entropy plots present. Reward curve shows upward trend. No NaN or discontinuities.
**Why human:** W&B dashboard visualization requires browser access.

### 2. Full Training Campaign Readiness

**Test:** Run `python -m choi2025.train_otpg --task follow_target --total-frames 5000000 --num-envs 500` for a full-scale benchmark.
**Expected:** Training runs to completion with reward comparable to PPO/SAC on follow_target.
**Why human:** Full 5M-frame training takes hours and requires GPU time allocation decisions.

### Gaps Summary

No gaps found. All 5 success criteria from ROADMAP are verified. All 6 OTPG requirements are satisfied. All artifacts exist, are substantive (well above minimum line counts), and are fully wired. All 12 unit tests pass. The 100K-frame validation run completed successfully with documented results. All 4 commits confirmed in git history.

---

_Verified: 2026-03-19T22:34:39Z_
_Verifier: Claude (gsd-verifier)_
