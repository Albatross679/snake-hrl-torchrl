---
name: Choi2025 Quick Validation
description: Pipeline validation of all 8 experiment configs (4 tasks x 2 algorithms x 10K frames) with mock physics backend
type: experiment
status: complete
created: 2026-03-19
updated: 2026-03-19
tags: [choi2025, replication, validation, sac, ppo]
aliases: []
---

# Choi2025 Quick Validation

## Objective

Validate that the full training pipeline works end-to-end for all 8 experiment configurations (4 tasks x 2 algorithms) before committing to long training runs. Catches import errors, shape mismatches, reward function bugs, and env/trainer integration issues.

## Setup

- **Frames per run:** 10,000 (reduced from 100K due to SAC UTD=4 throughput with single env)
- **Environments:** 1 (single env; ParallelEnv tested separately with 4-env and 8-env configurations)
- **Physics backend:** Mock (DisMech not installed; uses _MockRodState fallback with simplified rod dynamics)
- **GPU:** NVIDIA RTX A4000 (16 GB), CUDA 12.8
- **PyTorch:** 2.10.0+cu128, TorchRL 0.11.1

## Results

| Task               | Algo | Status  | Duration | Frames | W&B Run |
|--------------------|------|---------|----------|--------|---------|
| follow_target      | SAC  | SUCCESS | 10.3m    | 10,000 | logged  |
| follow_target      | PPO  | SUCCESS | 1.3m     | 10,000 | logged  |
| inverse_kinematics | SAC  | SUCCESS | 10.3m    | 10,000 | logged  |
| inverse_kinematics | PPO  | SUCCESS | 1.3m     | 10,000 | logged  |
| tight_obstacles    | SAC  | SUCCESS | 10.3m    | 10,000 | logged  |
| tight_obstacles    | PPO  | SUCCESS | 1.3m     | 10,000 | logged  |
| random_obstacles   | SAC  | SUCCESS | 10.3m    | 10,000 | logged  |
| random_obstacles   | PPO  | SUCCESS | 1.4m     | 10,000 | logged  |

**Total: 8/8 success, 0 hung, 0 errors**

## Issues Found and Fixed

### 1. TorchRL Spec API Renamed (import error)

TorchRL 0.11 renamed spec classes: `BoundedTensorSpec` -> `Bounded`, `CompositeSpec` -> `Composite`, `UnboundedContinuousTensorSpec` -> `Unbounded`. Updated all imports in `env.py`.

### 2. DisMech Not Installed (blocking)

DisMech C++ physics backend not installed. Created `_MockRodState` fallback in `env.py` that provides the same observation/action/reward interface using simplified curvature-driven rod dynamics. Training runs the same code paths (networks, optimizers, W&B logging) just with faster/simpler physics.

### 3. ParallelEnv CUDA Error with 32 Workers

32 ParallelEnv workers each trying to create CUDA contexts exhausted device resources. Fixed by running env workers on CPU (standard practice -- physics is CPU-bound numpy).

### 4. SAC Replay Buffer Missing reward Key (vectorized path)

TorchRL's `env.step()` nests reward under `"next"` key, but `_update()` expected `batch["reward"]` at top level. Fixed by lifting reward and done from `next` to top level before storing in replay buffer.

### 5. SAC Vectorized Auto-reset MDP State

After vectorized auto-reset, the `td = next_td` assignment carried stale observations. Fixed with `step_mdp(next_td)` to properly advance MDP state.

### 6. Config Name Bug (num_envs not reflected)

`__post_init__` ran before `num_envs` CLI override was applied, giving run names like `1envs` when 32 were used. Fixed by re-running `__post_init__()` after CLI overrides.

## Throughput Notes

- SAC with UTD=4 and single env: ~10 it/s (gradient updates dominate)
- PPO with frames_per_batch=4096: ~150 it/s (batch rollouts are efficient)
- ParallelEnv overhead with mock physics is high because IPC cost dominates the fast mock step
- For real DisMech physics, ParallelEnv should provide significant speedup (physics is the bottleneck)

## W&B Dashboard

- **Project:** `choi2025-replication`
- **Total runs:** 16 (8 validation + 8 from smoke tests/debugging)
- **All runs log:** timing metrics, gradient norms, episode rewards, system metrics
