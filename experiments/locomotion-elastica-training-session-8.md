---
type: experiment
created: 2026-03-06T00:00:00
updated: 2026-03-06T00:00:00
tags: [experiment, training, locomotion, elastica, ppo, parallelism]
status: superseded
---

# Locomotion Elastica Training Session 8

## Context

Session 8 tested two improvements over Session 7:
1. `log_interval=1` for per-batch logging (was 10 in Session 7)
2. Two parallel runs with different seeds to better utilize 48 CPU cores

Also tested 40 envs with 32768 frames_per_batch but abandoned — see below.

**Session 8 was killed early** when a KL early stopping bug was discovered in the PPO trainer. See `doc/logs/ppo-kl-early-stopping-fix.md` for details. Training continued in Session 9.

## 40-Env Attempt (abandoned)

W&B run `00k3vcee` — killed after ~10 minutes, first batch never completed.

| Config | FPS | CPU Usage | Notes |
|--------|-----|-----------|-------|
| 40 envs, 32768 fpb | <3.5 (estimated) | 7.3% | Slower than 16 envs due to IPC overhead |
| 16 envs, 8192 fpb | 60 | 11.3% | Proven configuration |

Despite each env running 820 steps (good for GAE), the SyncDataCollector synchronization overhead with 40 processes made it ~4× slower. The PyElastica physics simulation is too heavy per step for 40-process parallelism to help.

## Final Configuration

Two parallel runs, each with 16 envs:

| Parameter | Value |
|-----------|-------|
| Run 1 W&B | `xq3stapz` (seed 42) |
| Run 2 W&B | (seed 123) |
| Run dir 1 | `output/locomotion_elastica_forward_20260306_215924` |
| Run dir 2 | `output/locomotion_elastica_forward_20260306_221059` |
| Parallel envs | 16 per run (32 total) |
| Total frames | 2M per run |
| FPS | ~40 per run (~63 combined) |
| CPU usage | ~17% (32 workers on 48 cores) |
| log_interval | 1 (every batch) |

## Results (in progress)

### Run 1 (seed 42)

| Step | Reward | Critic Loss | Actor Loss |
|------|--------|-------------|------------|
| 8,192 | 89.89 | 0.0607 | 0.0001 |
| 16,384 | 89.62 | 0.0750 | 0.0002 |
| 24,576 | 101.88 | 0.0835 | 0.0001 |
| 32,768 | 107.72 | 0.0874 | 0.0001 |
| 40,960 | 131.79 | 0.1171 | 0.0000 |
| 49,152 | 120.41 | 0.0918 | -0.0000 |
| 57,344 | 107.17 | 0.0691 | -0.0000 |
| 65,536 | 94.51 | 0.0500 | -0.0000 |

### Run 2 (seed 123)

| Step | Reward | Critic Loss | Actor Loss |
|------|--------|-------------|------------|
| 8,192 | 101.91 | 0.0807 | 0.0002 |
| 16,384 | 89.10 | 0.0740 | 0.0001 |
| 24,576 | 89.77 | 0.0645 | 0.0001 |

## Analysis

### Reward oscillation (89-132)
- Oscillation is batch-to-batch variance, not instability
- Each batch has different random initial headings — some easier than others
- Distance-based reward means episodes starting near-aligned get ~130, misaligned get ~90
- This is expected and healthy — the policy needs to learn to handle all headings

### Parallel training strategy
- Two runs at 16 envs each (32 total) uses 17% of 48 cores
- Combined FPS (~63) matches single-run FPS (~60)
- Provides seed diversity at no throughput cost
- Could add a 3rd run to reach ~25% utilization
