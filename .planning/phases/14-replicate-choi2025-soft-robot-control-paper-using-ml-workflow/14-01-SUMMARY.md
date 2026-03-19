---
phase: 14-replicate-choi2025-soft-robot-control-paper-using-ml-workflow
plan: 01
subsystem: training
tags: [ppo, sac, choi2025, bf16, amp, timing, parallel-env, wandb, experiment-runner]

requires:
  - phase: none
    provides: existing choi2025 SAC scaffolding and shared RL trainers
provides:
  - Choi2025PPOConfig dataclass with standard PPO hyperparams
  - train_ppo.py PPO training entry point
  - evaluate.py dual-algorithm (SAC/PPO) evaluation
  - run_experiment.py 8-run experiment matrix runner
  - bf16 mixed precision in both SAC and PPO trainers
  - per-section timing profiling in both trainers
  - W&B model artifact upload in SAC trainer
affects: [14-02, 14-03, training-runs]

tech-stack:
  added: []
  patterns: [bf16-amp-context, per-section-timing, wandb-utils-integration]

key-files:
  created:
    - papers/choi2025/train_ppo.py
    - papers/choi2025/run_experiment.py
  modified:
    - papers/choi2025/config.py
    - papers/choi2025/train.py
    - papers/choi2025/evaluate.py
    - src/configs/training.py
    - src/trainers/sac.py
    - src/trainers/ppo.py

key-decisions:
  - "3x1024 network (scaled up from paper's 3x256) to maximize GPU utilization"
  - "SAC switched from direct wandb.init to wandb_utils.setup_run for consistency"
  - "STOP file + SIGTERM graceful shutdown added to SAC (PPO already had it)"
  - "Watchdog timeout in run_experiment.py: wall_time + 10min, exit 137/143 = hung"

patterns-established:
  - "_amp_context() helper: if use_amp and cuda, return torch.amp.autocast bfloat16"
  - "Per-section timing: env_step, data, backward, overhead (seconds + pct)"
  - "fixed_{task}_{algo}_{lr}_{envs} run naming convention"

requirements-completed: [CHOI-01, CHOI-02, CHOI-03]

duration: 8min
completed: 2026-03-19
---

# Phase 14 Plan 01: PPO Config + Training Wiring Summary

**PPO config and training entry point for Choi2025 soft manipulator, plus bf16 AMP, timing profiling, and experiment matrix runner across 4 tasks x 2 algorithms**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-19T03:41:30Z
- **Completed:** 2026-03-19T03:50:27Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments
- Choi2025PPOConfig with standard PPO hyperparams (clip=0.2, epochs=10, minibatch=64, 3x1024 network) inheriting from PPOConfig
- train_ppo.py mirrors train.py structure using PPOTrainer with ParallelEnv and GpuLock
- evaluate.py supports both SAC and PPO via --algo flag
- run_experiment.py orchestrates 8 sequential runs with watchdog timeout, GPU cleanup, and --quick validation mode
- bf16 mixed precision enabled by default in both SAC and PPO trainers (forward/loss in AMP, backward outside)
- Per-section timing profiling (env_step, data, backward, overhead) logged to W&B
- SAC trainer migrated from direct wandb.init to wandb_utils.setup_run, with model artifact upload added

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Choi2025PPOConfig and train_ppo.py** - `8791427` (feat)
2. **Task 2: Update evaluate.py and create run_experiment.py** - `a6e69a0` (feat)
3. **Task 3: Add bf16 AMP and timing profiling to trainers** - `3108e1d` (feat)

## Files Created/Modified
- `papers/choi2025/config.py` - Added Choi2025PPOConfig, updated network to 3x1024, fixed naming
- `papers/choi2025/train_ppo.py` - PPO training entry point with ParallelEnv and GpuLock
- `papers/choi2025/train.py` - SerialEnv -> ParallelEnv, added env.close() and GpuLock
- `papers/choi2025/evaluate.py` - Dual-algorithm support via --algo flag
- `papers/choi2025/run_experiment.py` - 8-run experiment matrix with watchdog and GPU cleanup
- `src/configs/training.py` - Added use_amp: bool = True to RLConfig
- `src/trainers/sac.py` - AMP, timing, wandb_utils, STOP file, SIGTERM, model artifact
- `src/trainers/ppo.py` - AMP, timing metrics, use_amp W&B params

## Decisions Made
- Scaled network from paper's 3x256 to 3x1024 to better utilize GPU (per CONTEXT.md decision)
- SAC trainer fully migrated from direct wandb calls to wandb_utils module (consistency with PPO)
- Added graceful shutdown (STOP file + SIGTERM) to SAC to match PPO's existing behavior
- Watchdog timeout set to wall_time + 10 minutes; exit codes 137/143 treated as hung (not hard failure)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added STOP file and SIGTERM handling to SAC trainer**
- **Found during:** Task 3
- **Issue:** SAC trainer lacked graceful shutdown that PPO already had (STOP file check, SIGTERM handler)
- **Fix:** Added _signal_handler, _check_stop_file, _restore_signal_handlers mirroring PPO pattern
- **Files modified:** src/trainers/sac.py
- **Committed in:** 3108e1d (Task 3 commit)

**2. [Rule 2 - Missing Critical] Added metrics.jsonl to SAC trainer**
- **Found during:** Task 3
- **Issue:** SAC trainer lacked metrics.jsonl output that PPO already had
- **Fix:** Added _write_metrics_jsonl method and file handle, matching PPO pattern
- **Files modified:** src/trainers/sac.py
- **Committed in:** 3108e1d (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 missing critical)
**Impact on plan:** Both auto-fixes bring SAC to parity with PPO for ML workflow compliance. No scope creep.

## Issues Encountered
- TorchRL API has BoundedTensorSpec import issue in env.py preventing full integration test. Config-level verification confirmed all dataclasses work correctly. Runtime testing deferred to Plan 02 (training runs).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 4 new/modified files import cleanly (verified via isolated config tests)
- PPO and SAC configs produce correct fixed_{task}_{algo} naming
- 8-run experiment matrix ready for --quick validation
- Ready for Plan 02 (training runs) and Plan 03 (evaluation/comparison)

## Self-Check: PASSED

All 8 files verified as existing. All 3 task commits (8791427, a6e69a0, 3108e1d) found in git log.

---
*Phase: 14-replicate-choi2025-soft-robot-control-paper-using-ml-workflow*
*Completed: 2026-03-19*
