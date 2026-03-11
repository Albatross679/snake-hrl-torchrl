---
name: cpu-utilization-analysis
description: Analysis of CPU utilization and parallel env scaling for PyElastica training
type: knowledge
created: 2026-03-06T00:00:00
updated: 2026-03-06T00:00:00
tags: [knowledge, performance, cpu, parallelism, torchrl]
aliases: []
---

# CPU Utilization Analysis

## Current State

- **Machine**: 48 CPU cores, 251 GB RAM, Tesla V100-PCIE-16GB GPU
- **Training**: 16 parallel envs (ParallelEnv) using `SyncDataCollector`
- **CPU usage**: ~528% of 4800% = **11% utilization**
- **Each worker**: ~33% of one core

## Why Only 16 Envs?

### Tested configurations

| Envs | FPS | Steps/Env/Batch | Full Episodes/Batch | Stability |
|------|-----|-----------------|---------------------|-----------|
| 8 | 33 | 1024 | ~2 | Good GAE |
| 16 | 57 | 512 | ~1 | Good GAE |
| 40 | 14 | 205 | ~0.4 | **Bad GAE** |

### Key findings

1. **40 envs was SLOWER (14 FPS) than 16 envs (57 FPS)** despite using more workers
2. With 40 envs, each env only runs 205 steps per batch — less than half an episode (500 steps)
3. GAE (Generalized Advantage Estimation) computes returns by bootstrapping at trajectory boundaries — short fragments give poor estimates
4. The goal-reaching episode terminates at ~160 steps, so 205 steps barely covers one episode

## Bottleneck Analysis

### Why adding more envs doesn't help

1. **SyncDataCollector is synchronous**: Waits for ALL envs to complete their portion before proceeding. The slowest env dictates throughput.

2. **PyElastica physics is expensive per step**: Each RL step = 10 physics steps × 50 substeps = **500 PyElastica timesteps**. Each timestep runs the full Cosserat rod solver + RFT force computation.

3. **IPC overhead scales with num_envs**: ParallelEnv uses multiprocessing. With 40 processes, the overhead of spawning, serializing TensorDicts, and synchronizing dominates.

4. **Fixed batch size creates tradeoff**: `frames_per_batch=8192` is split across envs. More envs = shorter trajectory fragments per env.

### Why each worker only uses ~33% CPU

Each worker process:
- Runs PyElastica physics (CPU-intensive, single-threaded NumPy)
- BUT spends time waiting for IPC (receive action, send observation)
- AND waiting for batch synchronization (all envs must finish together)
- OpenBLAS/MKL threads are pinned to 1 (required to avoid thread contention)

## Solutions to Improve Utilization

### 1. ~~Increase frames_per_batch~~ (TESTED — does NOT help)

Tested 40 envs with `frames_per_batch=32768` in Session 8:
- First batch never completed after 10 minutes
- CPU usage dropped to 7.3% (worse than 16 envs at 11.3%)
- IPC overhead with 40 processes dominated despite larger batch
- **Conclusion**: PyElastica physics is too expensive per step for 40-process parallelism

### 2. Multiple independent runs (CONFIRMED — best approach)

Tested in Session 8: two runs (seeds 42, 123) each with 16 envs:
- Combined FPS: ~63 (40 + 23), matching single-run FPS of ~60
- CPU usage: 17% (32 workers on 48 cores)
- Provides seed diversity at zero throughput cost
- Could add 3rd run to reach ~25% utilization

### 3. AsyncDataCollector (TorchRL feature)

Collect data asynchronously while PPO update runs on GPU:
- Overlaps CPU (data collection) and GPU (policy update)
- Requires TorchRL v0.12+ (currently v0.11.1)
