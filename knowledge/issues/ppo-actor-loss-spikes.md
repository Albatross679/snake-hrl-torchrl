---
name: PPO actor loss spikes to 1e11
description: bf16 AMP autocast on PPO loss module causes catastrophic log-prob precision loss with TanhNormal
type: issue
status: resolved
severity: critical
subtype: training
created: 2025-03-19
updated: 2025-03-19
tags: [ppo, tanhnormal, numerical-stability, choi2025, bf16, amp]
aliases: []
---

# PPO Actor Loss Spikes to 1e11

## Symptom

During `follow_target` PPO training with 32 envs, actor loss intermittently spikes:

- Step 471,040: actor_loss = 7,509
- Step 487,424: actor_loss = 199,405,980,877
- Step 507,904: actor_loss = 556

Critic loss remains stable (~0.01-0.04). Training continues but learning is corrupted.

## Root Cause

**bf16 mixed precision (AMP autocast) wrapping the PPO loss module.**

The `_update()` method ran `self.loss_module(mini_batch)` inside a `torch.amp.autocast('cuda', dtype=torch.bfloat16)` context. The loss module internally computes:
1. Actor forward pass (network inference — bf16 is fine here)
2. **TanhNormal log-prob computation** (bf16 is catastrophic here)
3. **Importance ratio `exp(lp_new - lp_old)`** (bf16 amplifies errors)

bf16 has only ~3 decimal digits of precision. With `min_std=0.01`, log-prob intermediate values are large (e.g., `(x - loc)^2 / (2 * std^2)` with std=0.01 gives values in the thousands). bf16 truncates these, producing garbage:

- **f32**: `lp_old = 62.73` (correct)
- **bf16**: `lp_old = -28160.00` (completely wrong)

The corrupted `lp_new - lp_old` difference then exponentiates to `inf`, producing the observed loss spikes.

Empirical validation: 0/2000 ratio explosions in f32, 35/2000 (1.75%) in bf16 under identical conditions.

## Investigation: What Didn't Cause It

- **TanhNormal Jacobian**: TorchRL already uses `SafeTanhTransform` with a numerically stable `log_abs_det_jacobian` (the softplus formula, equivalent to SB3). This was not the issue.
- **PPO ratio clipping**: `clip_epsilon=0.2` clips the surrogate objective but can't help when the log-prob values themselves are garbage due to bf16 precision loss.
- **num_epochs=10**: Not the root cause. The `target_kl=0.01` early stopping already handles excessive policy drift adaptively.

## Fix

1. **Remove AMP autocast from loss computation** (`src/trainers/ppo.py:_update`): The loss module must run in f32. Only the network forward passes benefit from bf16, but the log-prob and ratio math requires full precision.

2. **`min_std` 0.01 → 0.1** (`papers/choi2025/config.py`): Secondary fix — larger min_std reduces the magnitude of log-prob intermediate values, making bf16 errors less likely even if AMP were re-enabled. Also matches the base `ActorConfig` default.
