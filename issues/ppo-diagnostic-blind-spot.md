---
name: PPO diagnostic metrics not logged
description: Diagnostic metrics (explained_variance, action_stats, advantage_stats) were missing from training logs due to uncommitted code and averaging bug
type: issue
status: resolved
severity: high
subtype: training
created: 2025-03-25
updated: 2025-03-25
tags: [ppo, diagnostics, debugging]
aliases: []
---

## Problem

PPO training runs had no diagnostic metrics (explained_variance, action_std_min, entropy_proxy, advantage_stats) in metrics.jsonl or W&B. Only basic loss metrics were logged.

## Root Causes

### 1. Uncommitted diagnostics code
The diagnostics module (`src/trainers/diagnostics.py`) and the updated PPO trainer that imports it were modified in the working copy but never committed. The training process launched from the old committed code, so diagnostics were never computed.

### 2. Averaging bug in `_update()`
In `src/trainers/ppo.py`, the averaging loop at the end of `_update()` divided ALL metrics by `actual_updates`:

```python
for key in metrics:
    metrics[key] /= max(1, actual_updates)
```

This incorrectly divided one-shot diagnostic metrics (computed once before the update loop) by ~80 (10 epochs * 8 mini-batches). Only accumulated loss metrics should be averaged.

## Fix

1. Changed averaging to only apply to accumulated keys:
```python
_accumulated_keys = {
    "loss_actor", "loss_critic", "loss_entropy",
    "kl_divergence", "grad_norm", "clip_fraction",
}
for key in _accumulated_keys:
    metrics[key] /= max(1, actual_updates)
```

2. Committed and used the updated code for new training runs.

## Impact

Without diagnostics, the previous 50M-frame training run was "flying blind" — no way to detect value function health, entropy collapse, or advantage explosions. This delayed identification of the actual training problems by at least one full training cycle.
