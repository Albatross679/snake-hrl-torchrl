---
name: PPO observations not normalized
description: Raw observations with scales from 0 to 60 cause value function instability and poor learning
type: issue
status: resolved
severity: high
subtype: training
created: 2025-03-25
updated: 2025-03-25
tags: [ppo, observations, normalization, debugging]
aliases: []
---

## Problem

The PPO trainer received raw, unnormalized observations from the SoftManipulatorEnv. The 148-dim observation vector (positions, velocities, curvatures, target) has wildly heterogeneous scales:

| Component | Dims | Scale |
|-----------|------|-------|
| Node positions | 63 | 0–1m (some clamped at 0) |
| Node velocities | 63 | std up to 20.3 |
| Curvatures | 20 | varies |
| Target position | 3 | 0–1m |

Statistics from random rollout:
- **12 dead dims** (std = 0): clamped base node positions/velocities
- **12 extreme dims** (std > 10): velocity components
- Abs max per dim: up to 59.8

## Impact

- Value function couldn't learn reliably: explained_variance oscillated between -1.86 and 0.75
- Network gradients dominated by high-scale dimensions
- 13.7% of batches had negative explained variance (anti-correlated predictions)

## Fix

Added `ObservationNorm` transform with running mean/std in `papers/choi2025/train_ppo.py`:

```python
from torchrl.envs.transforms import ObservationNorm

obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
env = env.append_transform(obs_norm)
obs_norm.init_stats(num_iter=200, reduce_dim=[0, 1], cat_dim=0)
```

After fix: all dims have std < 2.3, no dims with std > 10.

## Verification

With obs normalization:
- Explained variance trending up (0.02 → 0.70)
- Rewards improved from 13 → 19 in early training
- Value function learning stabilized
