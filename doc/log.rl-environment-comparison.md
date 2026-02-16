---
id: a1b2c3d4-e5f6-7890-abcd-ef1234567890
name: log.rl-environment-comparison
description: Comparison of Gymnasium, TorchRL EnvBase, and MuJoCo native as RL environment interfaces
type: log
created: 2026-02-09
updated: 2026-02-09
tags: [log, rl, environments, comparison]
aliases: []
---

# RL Environment Interface Comparison

**Date:** 2026-02-09

Comparison of three RL environment interfaces evaluated for the snake-hrl project.

## API Overview

| | **Gymnasium** | **TorchRL EnvBase** | **MuJoCo Native** |
|---|---|---|---|
| **Data format** | NumPy arrays, dicts | `TensorDict` (PyTorch tensors) | C structs (`MjData`, `MjModel`) |
| **Step call** | `obs, rew, term, trunc, info = env.step(a)` | `td = env.step(td)` | `mujoco.mj_step(model, data)` |
| **Spaces** | `Box`, `Discrete`, `Dict` | `CompositeSpec`, `BoundedTensorSpec` | None — user-defined |
| **Batching** | Wrapper (`VectorEnv`) | Native (`ParallelEnv`) | Manual |
| **Device** | CPU only | CPU or GPU | CPU only |

## Strengths and Weaknesses

### Gymnasium

- Largest ecosystem — near-universal RL library support
- Simple, well-documented API
- Weak batching (wrapper-based, not native)
- NumPy-only — CPU-to-GPU transfer overhead during training

### TorchRL EnvBase

- PyTorch tensors end-to-end — no conversion overhead
- Native batching and parallel environments
- Tight integration with TorchRL collectors, replay buffers, and losses
- Smaller ecosystem and steeper learning curve

### MuJoCo Native

- Not an RL interface — a physics engine API (rigid + limited soft body)
- Maximum control and performance (C backend)
- Requires building observations, rewards, resets, and termination manually
- Typically wrapped by Gymnasium or dm_env for RL use

## Relevance to snake-hrl

| Concern | Best Option |
|---|---|
| Integration with TorchRL training | TorchRL EnvBase |
| Community support / examples | Gymnasium |
| Rigid-body robot physics | MuJoCo |
| No NumPy-to-Tensor overhead | TorchRL EnvBase |
| Soft-body snake physics | None (use dismech-python) |

## Conclusion

**TorchRL EnvBase** is the most natural fit for this project since training already uses TorchRL. It eliminates the Gymnasium middle layer and avoids tensor conversion overhead. MuJoCo is unsuitable because the snake requires soft-body physics provided by dismech-python.
