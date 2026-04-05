---
name: SAC instability without entropy — NaN divergence and gradient explosion
description: Paper's alpha=0 SAC causes two linked failures — acute NaN from 0*(-inf) in entropy term, and chronic gradient explosion from Q-landscape sharpening
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-19
updated: 2026-03-26
tags: [sac, choi2025, follow_target, nan, gradient-explosion, entropy, training-instability, bf16]
aliases: [sac-nan-divergence, sac-gradient-explosion]
---

## Overview

The paper's SAC configuration disables entropy regularization (`alpha=0`, `auto_alpha=False`), citing Yu et al. (2022). This creates two linked instabilities that share the same root cause chain: unbounded actor means due to no entropy penalty.

- **Acute failure (NaN)**: Training crashes at ~1.75M steps when `0.0 * (-inf) = NaN`
- **Chronic failure (gradient explosion)**: Actor gradient norm grows from 0.04 to 14.6B over 20M frames, causing reward collapse from 73.84 to 18.76

Both stem from `alpha=0 -> no entropy penalty -> actor means grow unbounded -> tanh saturation`.

## Shared Causal Chain

1. `alpha=0` -> no entropy penalty -> policy free to become deterministic
2. Actor pushes pre-tanh means beyond safe thresholds
3. `tanh` saturates to exactly +/-1.0 -> `log(1 - tanh^2) = log(0) = -inf`
4. **NaN path**: `0.0 * (-inf) = NaN` (IEEE 754) in target value computation -> everything NaN
5. **Gradient path**: Q-landscape sharpens without entropy smoothing -> actor gradients reflect Q-sharpness -> exponential growth -> oversized updates -> reward collapse

## Failure 1: NaN Divergence

### Observed trajectory

| Step | Reward | Actor Grad Norm | Notes |
|------|--------|-----------------|-------|
| 26K | 10-12 | 35 | Early training |
| 326K | **32** | 29K | First reward peak |
| 1.1M | **39** | 20M | Best reward |
| 1.75M | NaN | NaN | Full divergence |

### Implementation bug: multiply-by-zero vs if-guard

The paper's ALF implementation skips entropy computation entirely when disabled:

```python
# ALF (paper's code) -- skips when disabled
if self._use_entropy_reward:
    entropy_reward = -torch.exp(log_alpha) * log_pi
```

Our trainer always computed the entropy term, relying on `alpha * log_prob = 0 * x = 0`:

```python
# Our code (before fix) -- always evaluates
target_value = reward + gamma * (q_target - alpha * next_log_prob)
```

When `log_prob = -inf` (saturated tanh), `0.0 * (-inf) = NaN`.

### bf16 accelerates the failure

Our base config defaulted to `use_amp=True` (bf16). The paper runs fp32. bf16 has only 8 mantissa bits vs fp32's 23:

| Precision | tanh saturates at | log_prob = -inf at |
|-----------|-------------------|--------------------|
| **bf16** | **\|x\| >= 3.5** | **\|x\| >= 3.5** |
| fp32 | \|x\| >= 10.0 | \|x\| >= 10.0 |

## Failure 2: Gradient Explosion

### Observed trajectory (20M frame run, after NaN fix applied)

| Step | Reward | Actor Grad Norm | Critic Grad Norm |
|------|--------|-----------------|------------------|
| Early | ~10 | ~0.04 | ~0.35 |
| ~5M | **73.84** (best) | ~10K | ~0.5 |
| 10M | ~30 | ~1M | ~0.7 |
| 20M | 18.76 | **14.6B** | 0.92 |

Without entropy, the Q-landscape becomes increasingly sharp/peaky. The actor loss `(-Q).mean()` produces gradients that reflect Q-sharpness directly. The critic remains stable because MSE loss is self-normalizing.

### Config mismatches discovered during this investigation

1. **`max_grad_norm`: 0.5 -> None** -- inherited from `RLConfig` base class but paper doesn't use gradient clipping. The 0.5 clip was crushing 430M-norm gradients, giving effective learning rate ~1e-12.
2. **`actor_update_frequency`: 2 -> 1** -- default updated actor every 2 critic updates, but paper updates every critic update (standard SAC).

## Why the Paper Didn't Hit This

1. **ALF skips entropy when disabled** -- never computes `alpha * log_prob`
2. **fp32 precision** -- tanh doesn't saturate until `|mean| >= 10`
3. **Only 5M training steps** -- gradient explosion may not manifest severely at that scale
4. **Possible seed selection** -- 5 seeds reported; some may have diverged but weren't shown

## Fixes Applied

### Fix 1: If-guard on entropy terms (`src/trainers/sac.py`)

Added `self._use_entropy` flag (True when `auto_alpha=True` or `alpha > 0`). All entropy computations guarded:

```python
if self._use_entropy:
    q_target = q_target - self.alpha * next_log_prob.unsqueeze(-1)
target_value = reward + (1 - done) * gamma * q_target

if self._use_entropy:
    actor_loss = (self.alpha * log_prob.unsqueeze(-1) - q_new).mean()
else:
    actor_loss = (-q_new).mean()
```

### Fix 2: Switch to fp32 (`papers/choi2025/config.py`)

```python
use_amp: bool = False  # Paper uses fp32
```

### Fix 3: Safe log_alpha initialization (`src/trainers/sac.py`)

```python
safe_alpha = max(self.config.alpha, 1e-10)
self.log_alpha = torch.tensor(np.log(safe_alpha), device=self.device)
```

### Fix 4: Actor-only gradient clipping

Added `actor_max_grad_norm` field to `SACConfig` -- separate from `max_grad_norm` (critic clipping). `Choi2025Config` sets `actor_max_grad_norm=1.0`, `max_grad_norm=None` (critic stays unclipped since its gradients are stable).

### Fix 5: Pre-tanh mean clamping (`[-5, 5]`)

Added `mean = torch.clamp(mean, -5.0, 5.0)` in `ActorNetwork.forward()`. `tanh(5) = 0.9999` preserves full action range while preventing catastrophic saturation.

## Files Modified

- `src/trainers/sac.py` -- entropy if-guard, safe log_alpha init, actor-only gradient clipping
- `src/configs/training.py` -- added `actor_max_grad_norm` field to `SACConfig`
- `src/networks/actor.py` -- clamp pre-tanh mean to `[-5, 5]`
- `papers/choi2025/config.py` -- `use_amp=False`, `actor_max_grad_norm=1.0`, `actor_update_frequency=1`, `max_grad_norm=None`

## Validation

- All 44 diagnostic tests pass
- `clip_grad_norm_` bounds actor gradients (34438 -> 1.0)
- Mean clamping prevents tanh saturation (mean=100 -> clamped to 5, log_probs finite)
- Probe env (alpha=0, 1901 updates): `action_std > 0` and `log_prob finite` throughout
- Before fix: action_std collapsed to 0.0, log_prob reached -6.7e23
- After fix: action_std maintained at 0.056-0.43, log_prob stable at -0.63 to 1.16

## W&B Runs

- NaN divergence run: https://wandb.ai/qifan_wen-ohio-state-university/choi2025-replication/runs/13pqzhli
- Gradient explosion run: https://wandb.ai/qifan_wen-ohio-state-university/choi2025-replication/runs/o976ome9

## Checkpoints Preserved

- `best.pt` -- best reward (~39 reward at ~1.1M steps, pre-NaN fix run)
- Periodic checkpoints from both runs
