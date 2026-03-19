---
name: Choi2025 Full Training Launch
description: Launched full 1M-frame experiment matrix (8 runs) in tmux for Choi2025 soft manipulator replication
type: log
status: draft
subtype: training
created: 2026-03-19
updated: 2026-03-19
tags: [choi2025, replication, training, sac, ppo]
aliases: []
---

# Choi2025 Full Training Launch

## Summary

Launched the full 1M-frame experiment matrix (4 tasks x 2 algorithms = 8 sequential runs) in tmux session `choi2025-full`.

## Configuration

- **Total frames per run:** 1,000,000
- **Number of environments:** 1 (single env; ParallelEnv available but mock physics makes IPC overhead dominant)
- **Wall-clock limit per run:** 4 hours
- **Physics backend:** Mock (_MockRodState fallback -- DisMech not installed)
- **Network:** 3x1024 ReLU MLP (scaled up from paper's 3x256)
- **GPU:** NVIDIA RTX A4000 (16 GB), bf16 enabled

## Experiment Matrix

| Run | Task               | Algorithm | Status    |
|-----|--------------------|-----------|-----------|
| 1   | follow_target      | SAC       | Running   |
| 2   | follow_target      | PPO       | Queued    |
| 3   | inverse_kinematics | SAC       | Queued    |
| 4   | inverse_kinematics | PPO       | Queued    |
| 5   | tight_obstacles    | SAC       | Queued    |
| 6   | tight_obstacles    | PPO       | Queued    |
| 7   | random_obstacles   | SAC       | Queued    |
| 8   | random_obstacles   | PPO       | Queued    |

## Estimated Duration

Based on quick validation throughput:
- SAC runs: ~10 it/s -> 1M frames takes ~28 hours per run
- PPO runs: ~150 it/s -> 1M frames takes ~2 hours per run
- Total estimated: ~(4 x 28h) + (4 x 2h) = ~120 hours
- Wall-clock limit (4h) will stop SAC runs at ~144K frames
- PPO runs should complete within wall-clock limit

## How to Monitor

```bash
# Check tmux session
tmux attach -t choi2025-full

# Check log file
tail -f output/choi2025_full_run.log

# Check W&B dashboard
# Project: choi2025-replication

# Check GPU utilization
nvidia-smi

# Graceful stop (current run finishes, then stops)
touch STOP
```

## tmux Session

- **Name:** `choi2025-full`
- **Log file:** `output/choi2025_full_run.log`
- **W&B project:** `choi2025-replication`
- **Watchdog timeout:** wall_time + 10 minutes per run
