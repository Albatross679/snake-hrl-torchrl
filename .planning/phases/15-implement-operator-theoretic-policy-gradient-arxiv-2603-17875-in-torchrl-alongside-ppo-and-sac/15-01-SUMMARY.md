---
phase: 15-implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac
plan: 01
subsystem: rl-training
tags: [otpg, mmd, rkhs, policy-gradient, torchrl, tensordict, tanhNormal]

# Dependency graph
requires:
  - phase: 14-replicate-choi2025-soft-robot-control-paper-using-ml-workflow
    provides: "PPOTrainer pattern, create_actor/create_critic factories, wandb_utils"
provides:
  - "OTPGConfig dataclass with MM-RKHS hyperparameters"
  - "OTPGTrainer class with MMD-based trust region policy gradient"
  - "SimplePendulum TorchRL-native test environment"
  - "12 passing unit tests for OTPG algorithm"
affects: [15-02, choi2025-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns: ["MMD-based trust region via RBF kernel", "Linear-time unbiased MMD^2 estimator", "TanhNormal old distribution reconstruction from stored loc/scale"]

key-files:
  created:
    - src/trainers/otpg.py
    - tests/test_otpg.py
    - logs/otpg-trainer-implementation.md
  modified:
    - src/configs/training.py
    - src/trainers/__init__.py

key-decisions:
  - "No entropy bonus in OTPG loss -- KL regularizer handles exploration per CONTEXT.md locked decision"
  - "No PPO-style clipping -- trust region enforced entirely by MMD penalty + KL regularizer"
  - "Single Adam optimizer over combined actor+critic parameters (PPOTrainer pattern)"
  - "Old distribution reconstructed from stored loc/scale via TanhNormal (not from saved policy weights)"
  - "Linear-time RBF MMD^2 estimator (O(n) not O(n^2)) using paired samples"

patterns-established:
  - "OTPG trainer extends PPOTrainer pattern with custom loss computation (no ClipPPOLoss)"
  - "SimplePendulum EnvBase for fast TorchRL-native unit testing without gymnasium"

requirements-completed: [OTPG-01, OTPG-02, OTPG-03, OTPG-04, OTPG-06]

# Metrics
duration: 6min
completed: 2026-03-19
---

# Phase 15 Plan 01: OTPG Core Implementation Summary

**OTPGTrainer with MM-RKHS loss (MMD^2 + KL trust region), OTPGConfig dataclass, and 12 passing unit tests on SimplePendulum**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-19T22:02:03Z
- **Completed:** 2026-03-19T22:08:30Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- OTPGConfig dataclass with all MM-RKHS hyperparameters (beta, eta, mmd_bandwidth, mmd_num_samples) extending RLConfig -- no entropy_coef, no clip_epsilon
- OTPGTrainer implementing full training pipeline: data collection, GAE, mini-batch updates, MMD-based trust region loss, checkpoint save/load, W&B logging, graceful shutdown
- Linear-time RBF kernel MMD^2 estimator using paired action samples from old/new policy distributions
- SimplePendulum TorchRL-native environment (3-dim obs, 1-dim action, 200-step episodes) verified via check_env_specs
- 12 passing unit tests covering: environment specs, config defaults/inheritance, trainer initialization, MMD computation (finite/non-negative, near-zero for identical dists), _update() finite metrics, 512-frame training completion, checkpoint roundtrip

## Task Commits

Each task was committed atomically:

1. **Task 1: Add OTPGConfig dataclass and create OTPGTrainer with MMD loss** - `f902c0f` (feat)
2. **Task 2: Write unit tests with SimplePendulum TorchRL-native environment** - `f9113fa` (test)

## Files Created/Modified
- `src/configs/training.py` - Added OTPGConfig dataclass after PPOConfig
- `src/trainers/otpg.py` - Full OTPGTrainer class (841 lines) with MM-RKHS loss computation
- `src/trainers/__init__.py` - Added OTPGTrainer export
- `tests/test_otpg.py` - SimplePendulum environment and 12 unit tests (407 lines)
- `logs/otpg-trainer-implementation.md` - Feature log entry

## Decisions Made
- Used action_log_prob key throughout (not sample_log_prob) -- empirically verified correct key from TorchRL ProbabilisticActor with return_log_prob=True
- Old policy distribution reconstructed from batch-stored loc/scale via TanhNormal rather than maintaining a copy of old policy weights
- Approximate KL divergence via (ratio - 1 - log_ratio) instead of exact KL between TanhNormal distributions
- Policy entropy computed from Gaussian base distribution (TanhNormal entropy is intractable) for diagnostic logging only (not included in loss)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_step_shape accessing wrong TensorDict level**
- **Found during:** Task 2 (unit tests)
- **Issue:** test_step_shape accessed `td_next["reward"]` directly, but env.step() wraps next-state data under `"next"` key
- **Fix:** Changed to `td_next["next", "reward"]` (TorchRL convention)
- **Files modified:** tests/test_otpg.py
- **Verification:** Test passes
- **Committed in:** f9113fa (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test code)
**Impact on plan:** Trivial fix, test was accessing TensorDict at wrong level. No scope creep.

## Issues Encountered
None -- implementation followed PPOTrainer pattern closely with targeted modifications for OTPG-specific loss computation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- OTPGTrainer ready for Choi2025 benchmark integration (Plan 02)
- SimplePendulum available as fast smoke-test environment for iterative development
- All 12 unit tests green -- safe foundation for Plan 02's train_otpg.py and benchmark runs

## Self-Check: PASSED

All files verified present:
- src/configs/training.py -- FOUND
- src/trainers/otpg.py -- FOUND
- src/trainers/__init__.py -- FOUND
- tests/test_otpg.py -- FOUND
- logs/otpg-trainer-implementation.md -- FOUND

All commits verified:
- f902c0f (Task 1: OTPGConfig + OTPGTrainer) -- FOUND
- f9113fa (Task 2: unit tests) -- FOUND

---
*Phase: 15-implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac*
*Completed: 2026-03-19*
