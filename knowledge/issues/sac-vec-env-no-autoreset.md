---
name: SAC vectorized env never auto-resets after episode done
description: ParallelEnv episodes stuck at length=1 because env.step() does not auto-reset done envs in TorchRL 0.11.x
type: issue
status: resolved
severity: critical
subtype: training
created: 2026-03-19
updated: 2026-03-19
tags: [sac, torchrl, parallelenv, auto-reset, step_and_maybe_reset]
aliases: [sac-autoreset-bug]
---

## Symptom

SAC training with 32 parallel envs: first 32 episodes complete correctly (length=200), then all subsequent episodes have length=1 with near-zero rewards. Training appears to run but learns nothing useful.

## Root Cause

Two issues combined:

1. **Missing auto-reset**: The SAC trainer's vectorized path called `env.step()` + manual `step_mdp()`, but in TorchRL 0.11.x, `ParallelEnv.step()` does NOT auto-reset done environments. The correct API is `env.step_and_maybe_reset()`, which calls step → step_mdp → maybe_reset.

2. **String device bug**: The `SoftManipulatorEnv` stored `self._device = "cpu"` (a string), but TorchRL's `BatchedEnvBase._reset()` (line 2369) calls `self.device.type` which fails on strings — it expects a `torch.device` object.

Without auto-reset, after the first batch of episodes completes, `_step_count` stays ≥200 in every worker env, causing every subsequent step to return `done=True`.

## Fix

**`src/trainers/sac.py`**: Replace `env.step()` + `step_mdp()` with `env.step_and_maybe_reset()` in the vectorized path:
```python
if is_vec:
    next_td, td_reset = self.env.step_and_maybe_reset(td_env)
else:
    next_td = self.env.step(td_env)
```

**`papers/choi2025/env.py`**: Convert device to `torch.device` in constructor:
```python
device = torch.device(device) if isinstance(device, str) else device
```

## Verification

After fix, all episodes consistently show length=200 through multiple reset cycles (verified through 96+ episodes / 3 resets).
