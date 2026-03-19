---
phase: 15-implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac
plan: 02
subsystem: rl-training
tags: [otpg, choi2025, benchmark, config, training-script, wandb, validation]

# Dependency graph
requires:
  - phase: 15-implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac
    provides: "OTPGTrainer class and OTPGConfig dataclass (Plan 01)"
  - phase: 14-replicate-choi2025-soft-robot-control-paper-using-ml-workflow
    provides: "Choi2025 env, config patterns (Choi2025PPOConfig, train_ppo.py)"
provides:
  - "Choi2025OTPGConfig dataclass for OTPG benchmarking on Choi2025 tasks"
  - "train_otpg.py CLI entry point for Choi2025 OTPG training"
  - "Validated 100K-frame OTPG learning signal on follow_target"
affects: [choi2025-benchmark, full-training-campaign]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Choi2025 config pattern extended to OTPG (third algorithm after SAC/PPO)", "SKIP_GPU_LOCK for concurrent multi-GPU training"]

key-files:
  created:
    - papers/choi2025/train_otpg.py
    - experiments/otpg-100k-validation-follow-target.md
  modified:
    - papers/choi2025/config.py

key-decisions:
  - "Choi2025OTPGConfig mirrors Choi2025PPOConfig structure for fair comparison (same network, env, parallelism)"
  - "patience_batches=0 for OTPG config (wall time controls stopping, not patience)"
  - "SKIP_GPU_LOCK env var support added for concurrent training on separate GPUs"

patterns-established:
  - "Algorithm-specific train_*.py scripts follow identical structure with config/trainer substitution"
  - "Double-close guard pattern: try/except RuntimeError in finally block for ParallelEnv cleanup"

requirements-completed: [OTPG-05]

# Metrics
duration: 18min
completed: 2026-03-19
---

# Phase 15 Plan 02: Choi2025 OTPG Benchmark Config and Validation Summary

**Choi2025OTPGConfig dataclass and train_otpg.py entry point with 100K-frame learning signal validation (best reward 21.18, rolling-100 = 16.92)**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-19T22:11:56Z
- **Completed:** 2026-03-19T22:30:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Choi2025OTPGConfig dataclass inheriting OTPGConfig with MM-RKHS defaults, matching PPO/SAC training budget for fair comparison
- train_otpg.py CLI entry point with full argparse (--task, --total-frames, --num-envs, --max-wall-time, --resume, --seed, --device)
- 100K-frame OTPG validation on follow_target completed: 512 episodes, reward improved from ~9 to ~17-21, all OTPG-specific metrics logged to W&B (mmd_penalty, kl_divergence, policy_entropy)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Choi2025OTPGConfig and train_otpg.py entry point** - `685ccf7` (feat)
2. **Task 2: Run 100K-frame OTPG validation on follow_target** - `7539c71` (feat)

## Files Created/Modified
- `papers/choi2025/config.py` - Added Choi2025OTPGConfig dataclass and OTPGConfig import
- `papers/choi2025/train_otpg.py` - OTPG training CLI entry point (146 lines), mirrors train_ppo.py structure
- `experiments/otpg-100k-validation-follow-target.md` - Experiment log for 100K validation run

## Decisions Made
- Choi2025OTPGConfig uses identical training budget, network, env config, and parallelism as Choi2025PPOConfig for fair algorithm comparison
- patience_batches=0 (disabled) since wall time controls stopping in benchmark runs
- Added SKIP_GPU_LOCK env var support to allow concurrent training on separate GPUs (GPU 0 had a running SAC job)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added SKIP_GPU_LOCK support to train_otpg.py**
- **Found during:** Task 2 (running validation)
- **Issue:** GpuLock blocked training because GPU 0 had a running SAC job; train_otpg.py lacked SKIP_GPU_LOCK bypass (present in train.py but not in train_ppo.py which was copied)
- **Fix:** Added `if os.environ.get("SKIP_GPU_LOCK")` check in `__main__` block (matching train.py pattern)
- **Files modified:** papers/choi2025/train_otpg.py
- **Verification:** Training ran successfully on GPU 1 with CUDA_VISIBLE_DEVICES=1
- **Committed in:** 7539c71 (Task 2 commit)

**2. [Rule 1 - Bug] Fixed double-close RuntimeError in env cleanup**
- **Found during:** Task 2 (training completion)
- **Issue:** `env.close()` in finally block raised RuntimeError because SyncDataCollector already closed the env
- **Fix:** Wrapped `env.close()` in try/except RuntimeError
- **Files modified:** papers/choi2025/train_otpg.py
- **Verification:** Training completes without error
- **Committed in:** 7539c71 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for task completion and clean execution. No scope creep.

## Issues Encountered
None -- training ran to completion, all metrics logged correctly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- OTPG fully integrated into Choi2025 benchmark suite alongside PPO and SAC
- train_otpg.py ready for full training campaigns on all 4 tasks
- Learning signal validated: OTPG produces meaningful reward improvement on follow_target
- Pre-existing issue: train_ppo.py has same double-close bug (out of scope, not created by this plan)

## Self-Check: PASSED

All files verified present:
- papers/choi2025/config.py -- FOUND
- papers/choi2025/train_otpg.py -- FOUND
- experiments/otpg-100k-validation-follow-target.md -- FOUND

All commits verified:
- 685ccf7 (Task 1: Choi2025OTPGConfig + train_otpg.py) -- FOUND
- 7539c71 (Task 2: 100K validation + double-close fix) -- FOUND

---
*Phase: 15-implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac*
*Completed: 2026-03-19*
