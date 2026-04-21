---
name: MM-RKHS KL detached from graph + diagnostic metrics corrupted
description: Two bugs in MMRKHSTrainer — KL regularizer provided zero gradients, and diagnostic metrics were divided by update count
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-26
updated: 2026-03-26
tags: [mmrkhs, training-bug, kl-divergence, diagnostics]
aliases: []
---

## Bug 1: KL divergence detached from computational graph

**Location**: `src/trainers/mmrkhs.py:632-633`

```python
# BEFORE (bug): KL computed inside no_grad — provides zero gradients
with torch.no_grad():
    kl = (ratio - 1.0 - log_ratio).mean()
```

The `(1/eta) * kl` term in the MM-RKHS loss was supposed to be a KL regularizer penalizing policy deviation from the old policy. But `torch.no_grad()` meant the term contributed **zero gradients** to the actor parameters. The algorithm effectively ran as:

`L = -surr_advantage + beta * MMD + value_coef * critic_loss`

...missing the entire KL component of the MM-RKHS trust region.

**Fix**: Remove `torch.no_grad()` wrapper so KL flows gradients.

## Bug 2: Diagnostic metrics corrupted by averaging

**Location**: `src/trainers/mmrkhs.py:700-703`

```python
# BEFORE (bug): divides ALL metrics by update count
for key in metrics:
    metrics[key] /= max(1, actual_updates)
```

This divided ALL metric keys by the number of gradient updates per batch, including:
- `explained_variance` (should be in [-1, 1])
- `advantage_mean`, `advantage_std`, `advantage_abs_max`
- `diagnostics/action_std_mean`, `diagnostics/action_std_min`
- `diagnostics/log_prob_mean`, `diagnostics/entropy_proxy`

With 4 mini-batches × 2 epochs = 8 updates, these values were divided by 8, making them appear ~8x smaller than reality. This corrupted all W&B dashboard metrics and could suppress diagnostic alerts.

**Fix**: Only average the 6 accumulated metrics (`loss_policy`, `loss_critic`, `mmd_penalty`, `kl_divergence`, `grad_norm`, `policy_entropy`).

## Validation

- 56/56 tests pass (44 diagnostic + 12 MM-RKHS)
- KL now has `requires_grad=True` (verified)
- `explained_variance` stays in [-1, 1] range (not divided by update count)
- `action_std_mean` stays > 0.01 (not divided by update count)
