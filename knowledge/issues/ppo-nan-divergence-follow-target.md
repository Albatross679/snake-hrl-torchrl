---
name: PPO NaN divergence in follow_target training
description: Training diverged to NaN at step ~278k due to actor_loss spike without NaN guard
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-19
updated: 2026-03-19
tags: [ppo, nan, training-stability, choi2025]
aliases: []
---

# PPO NaN Divergence in follow_target Training

## Symptom

PPO training for `follow_target` task diverged to all-NaN metrics at step 278528.
All losses (actor, critic, entropy, kl) reported as `nan` from that point onward.

## Root Cause

KL divergence spiked progressively before NaN:

| Step    | actor_loss | kl     |
|---------|-----------|--------|
| 258048  | 6.6971    | 0.1313 |
| 262144  | 134.9434  | 0.2708 |
| 270336  | 0.3909    | 0.3886 |
| 274432  | 0.0225    | 0.1680 |
| 278528  | nan       | nan    |

The policy took too-large steps (KL >> 0.01 target). The massive actor_loss=134.94
produced a gradient that, even after clipping (max_grad_norm=0.5), corrupted weights
with NaN values. Once NaN entered the weights, all subsequent steps produced NaN.

Existing defenses were insufficient:
- **Gradient clipping** (0.5): couldn't prevent NaN from backward pass through inf loss
- **KL early stopping** (target_kl=0.01): only breaks the epoch loop, not within batches;
  also uses averaged KL which dilutes spikes

## Fix

Added NaN guard in `src/trainers/ppo.py` before `optimizer.step()`:
- Check `torch.isfinite(loss)` after backward pass
- If loss is NaN/inf, zero gradients and skip the optimizer step
- Print warning to log for monitoring

This prevents NaN from propagating into network weights while allowing training
to recover on the next batch.

## Actions Taken

### Attempt 1 (insufficient)
1. Killed diverged process (PID 941439)
2. Added NaN guard checking `torch.isfinite(loss)` after backward — skips optimizer.step()
3. Restarted (PID 948131) — NaN recurred at step ~258k

### Attempt 2 (robust fix)
NaN recurred because: (a) checking loss after backward still allows NaN gradients from
large-but-finite losses, (b) `clip_grad_norm_` with NaN gradients produces NaN clip coefficients,
(c) per-epoch KL stopping missed within-epoch spikes.

Three-layer fix applied to `src/trainers/ppo.py`:
1. **Layer 1**: Check `torch.isfinite(loss)` BEFORE `backward()` — skip entire batch via `continue`
2. **Layer 2**: Check `torch.isfinite(grad_norm)` after clipping — zero_grad and skip step
3. **Per-batch KL early stopping**: Break inner mini-batch loop if any batch's KL exceeds 1.5× target_kl

Also killed zombie multiprocessing workers (PIDs 930161, 930165) that caused thread exhaustion.

4. Restarted training — new run dir: `output/fixed_follow_target_ppo_lr3e4_32envs_20260319_122525`
