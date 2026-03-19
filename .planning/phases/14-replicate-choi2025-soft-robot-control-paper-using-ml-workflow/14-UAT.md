---
status: testing
phase: 14-replicate-choi2025-soft-robot-control-paper-using-ml-workflow
source: [14-01-SUMMARY.md, 14-02-SUMMARY.md, 14-03-SUMMARY.md]
started: 2026-03-19T06:00:00Z
updated: 2026-03-19T06:00:00Z
---

## Current Test

number: 1
name: PPO training runs without crashing
expected: |
  Run `python papers/choi2025/train_ppo.py --task follow_target --total_frames 5000` (or similar minimal invocation). Training starts, logs to stdout, and completes without errors. W&B run is created in choi2025-replication project.
awaiting: user response

## Tests

### 1. PPO training runs without crashing
expected: Run `python papers/choi2025/train_ppo.py --task follow_target --total_frames 5000` (or similar minimal invocation). Training starts, logs to stdout, and completes without errors. W&B run is created in choi2025-replication project.
result: [pending]

### 2. SAC training runs without crashing
expected: Run `python papers/choi2025/train.py --task follow_target --total_frames 5000` (or similar minimal invocation). Training starts, logs to stdout, and completes without errors. W&B run is created in choi2025-replication project.
result: [pending]

### 3. Evaluate.py supports both SAC and PPO
expected: Run `python papers/choi2025/evaluate.py --algo sac --task follow_target` and `--algo ppo`. Both load the correct checkpoint and run evaluation episodes without errors.
result: [pending]

### 4. Experiment matrix runner validates all 8 configs
expected: Run `python papers/choi2025/run_experiment.py --quick`. All 8 configs (4 tasks x 2 algos) are validated sequentially. Script completes with a summary showing pass/fail per config.
result: [pending]

### 5. Video rollouts recorded for all task-algo combinations
expected: 8 MP4 files exist in `media/choi2025/`: fixed_{follow_target,inverse_kinematics,tight_obstacles,random_obstacles}_{sac,ppo}.mp4. Each file is non-empty and playable.
result: [pending]

### 6. bf16 mixed precision and timing profiled in W&B
expected: Check any W&B run in choi2025-replication project. Logged metrics include per-section timing (env_step, data, backward, overhead) and config shows use_amp=True.
result: [pending]

### 7. Experiment report documents learning signal
expected: File `experiments/choi2025-full-results.md` exists and contains: SAC reward progression for all 4 tasks, PPO assessment, timing breakdown, and throughput comparison between algorithms.
result: [pending]

## Summary

total: 7
passed: 0
issues: 0
pending: 7
skipped: 0

## Gaps

[none yet]
