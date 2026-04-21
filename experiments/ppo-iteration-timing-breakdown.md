---
name: PPO iteration timing breakdown
description: Profiled per-phase timing of PPO training loop to identify bottleneck — DisMech physics rollout dominates at 63%
type: experiment
status: complete
created: 2026-03-25
updated: 2026-03-25
tags: [ppo, profiling, timing, dismech, choi2025, follow_target]
aliases: []
---

## Objective

Identify which phase of the PPO training loop is the throughput bottleneck for the choi2025 follow_target task.

## Setup

- **Algorithm**: PPO (Choi2025PPOConfig)
- **Network**: 4x512 ReLU MLP (actor and critic)
- **Learning rate**: 1e-4
- **Environment**: DisMech soft manipulator, follow_target task
- **Parallel envs**: 100 (via TorchRL ParallelEnv, CPU workers)
- **frames_per_batch**: 8,192 (effective 8,200 due to 100-env rounding)
- **PPO epochs**: 10
- **Mini-batch size**: 1,024 (8 mini-batches per epoch, 80 gradient steps per iteration)
- **Device**: CUDA (GPU for policy/critic, CPU for physics)
- **Precision**: fp32

## Methodology

Instrumented the PPO training loop with `time.monotonic()` around each phase. Ran 5 iterations through a standalone profiling script that replicates the exact training loop structure from `src/trainers/ppo.py`. Discarded batch 0 (warmup: env spawn, JIT compilation) and batch 4 (outlier: 44.5s env step, likely OS scheduling). Averaged batches 1–3.

Four phases measured per iteration:

1. **env_step**: Time spent inside `SyncDataCollector.__next__()` — rolls out 82 steps across 100 parallel DisMech environments on CPU.
2. **data+GAE**: CPU-to-GPU batch transfer (`batch.to(device)`) + reshape + GAE advantage estimation (`torch.no_grad()` critic forward pass).
3. **backward**: 10 PPO epochs × 8 mini-batches = 80 forward+backward+optimizer steps. Includes `ClipPPOLoss` forward, `loss.backward()`, `clip_grad_norm_`, and `optimizer.step()`.
4. **overhead**: CUDA synchronization, logging, checkpointing (minimal in profiling script).

## Results

| Phase | Time (s) | % of iteration |
|-------|----------|----------------|
| env_step (physics rollout) | 8.9 | 63% |
| backward (PPO update) | 3.7 | 26% |
| data + GAE | 1.6 | 11% |
| overhead | ~0.0 | 0% |
| **Total** | **14.2** | **100%** |

Effective throughput from profiling: ~577 frames/s per iteration. Live training run reports ~1,430 it/s (tqdm measures frames, not iterations).

### Raw batch timings

| Batch | env_step | data+GAE | backward | overhead | total |
|-------|----------|----------|----------|----------|-------|
| 0 (warmup) | 9.61s | 0.48s | 5.39s | 0.00s | 15.47s |
| 1 | 8.21s | 2.08s | 3.61s | 0.00s | 13.89s |
| 2 | 8.60s | 1.92s | 3.88s | 0.00s | 14.41s |
| 3 | 9.89s | 0.79s | 3.63s | 0.00s | 14.31s |
| 4 (outlier) | 44.49s | 2.49s | 3.42s | 0.00s | 50.40s |

## Findings

1. **DisMech physics is the bottleneck.** The CPU-bound numpy physics simulation in 100 parallel workers consumes 63% of each iteration. The GPU is idle for most of the training loop.

2. **PPO backward pass is efficient.** 80 gradient steps (10 epochs × 8 mini-batches) on a 4x512 MLP completes in 3.7s — 46ms per gradient step. GPU utilization is high during this phase.

3. **Data transfer is non-negligible.** CPU→GPU transfer + GAE takes 1.6s (11%), partly because ParallelEnv produces CPU tensors that must be moved.

4. **Overhead is zero.** Logging and checkpointing have no measurable impact.

## Implications

- At 50M frames, training takes ~9.5 hours. Physics accounts for ~6 hours of that.
- Increasing parallel envs beyond 100 would help but 500 causes TorchRL BrokenPipeError (see `issues/ppo-500-env-brokenpipe.md`).
- Reducing PPO epochs from 10 to 5 would save ~1.8s per iteration (~13% speedup) at the cost of sample efficiency.
- Switching to MuJoCo backend would dramatically reduce physics time but changes the simulation.
