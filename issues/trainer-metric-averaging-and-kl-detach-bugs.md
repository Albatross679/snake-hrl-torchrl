---
name: Trainer metric averaging bug + MM-RKHS KL detached from graph
description: All trainers (PPO, MM-RKHS) divided all metrics by update count instead of only accumulated losses, corrupting diagnostics; MM-RKHS also had KL wrapped in no_grad
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-19
updated: 2026-03-26
tags: [ppo, mmrkhs, metrics, kl-divergence, diagnostics, logging]
aliases: [ppo-zero-loss-bug, ppo-diagnostic-blind-spot, mmrkhs-kl-detached]
---

## Overview

A single averaging bug was independently present in two trainers (PPO and MM-RKHS), and the MM-RKHS trainer had an additional bug where KL divergence was detached from the computational graph. These were discovered across three debugging sessions but share a common root cause and fix pattern.

## Bug 1: Metric averaging (PPO + MM-RKHS)

### Symptom

- **PPO**: All loss metrics displayed as 0.0000 despite active learning
- **MM-RKHS**: Diagnostic metrics (explained_variance, action_std, advantage_stats) appeared ~8x smaller than reality

### Root cause

Both trainers' `_update()` methods divided ALL metrics by the update count, instead of only the accumulated loss metrics:

```python
# Bug: divides everything, including one-shot diagnostics
for key in metrics:
    metrics[key] /= max(1, actual_updates)
```

In PPO, KL early stopping made this catastrophic: when training stopped after 1-2 updates out of 40 theoretical max, values were divided by 40x, producing <0.00005 (displayed as 0.0000). In MM-RKHS with 4 mini-batches x 2 epochs = 8 updates, one-shot diagnostics like `explained_variance` (should be in [-1, 1]) were divided by 8.

A compounding factor in PPO was that the original code divided by the theoretical maximum (`num_epochs * num_batches`) rather than actual updates performed, making the KL early stopping case even worse.

### Fix

Both trainers now only average the accumulated keys:

```python
# PPO accumulated keys
_accumulated_keys = {
    "loss_actor", "loss_critic", "loss_entropy",
    "kl_divergence", "grad_norm", "clip_fraction",
}

# MM-RKHS accumulated keys
_accumulated_keys = {
    "loss_policy", "loss_critic", "mmd_penalty",
    "kl_divergence", "grad_norm", "policy_entropy",
}

for key in _accumulated_keys:
    if key in metrics:
        metrics[key] /= max(1, actual_updates)
```

### Impact

Without correct diagnostics, a 50M-frame PPO training run was "flying blind" — no way to detect value function health, entropy collapse, or advantage explosions. This delayed identification of training problems by at least one full training cycle.

## Bug 2: MM-RKHS KL detached from computational graph

### Location

`src/trainers/mmrkhs.py:632-633`

### Root cause

KL divergence was computed inside `torch.no_grad()`:

```python
# Bug: KL provides zero gradients to actor
with torch.no_grad():
    kl = (ratio - 1.0 - log_ratio).mean()
```

The `(1/eta) * kl` term was supposed to be a KL regularizer penalizing policy deviation from the old policy. With `no_grad()`, it contributed zero gradients. The algorithm effectively ran as:

`L = -surr_advantage + beta * MMD + value_coef * critic_loss`

...missing the entire KL component of the MM-RKHS trust region.

### Fix

Removed `torch.no_grad()` wrapper so KL flows gradients to the actor.

## Files modified

- `src/trainers/ppo.py` — track `actual_updates` counter, average only accumulated keys
- `src/trainers/mmrkhs.py` — average only accumulated keys, remove `no_grad()` from KL
- `src/trainers/diagnostics.py` — committed previously uncommitted diagnostics module

## Validation

- 56/56 tests pass (44 diagnostic + 12 MM-RKHS)
- KL now has `requires_grad=True` (verified)
- `explained_variance` stays in [-1, 1] range
- `action_std_mean` stays > 0.01
- PPO loss metrics display correct magnitudes
