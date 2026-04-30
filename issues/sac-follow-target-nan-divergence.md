---
name: SAC follow_target NaN divergence at 1.75M steps
description: SAC training diverged to NaN due to 0*(-inf)=NaN in entropy term (alpha=0 + bf16 tanh saturation)
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-19
updated: 2026-03-20
tags: [sac, choi2025, follow_target, nan, gradient-explosion, training-instability, bf16]
aliases: []
---

## Symptom

SAC training for the Choi 2025 `follow_target` task diverged to NaN at ~1.75M/5M steps (35% through training). All metrics (reward, critic_loss, actor_loss, q-values) became NaN simultaneously.

## Observed Trajectory

| Step | Reward | Actor Grad Norm (pre-clip) | Notes |
|------|--------|---------------------------|-------|
| 26K | 10-12 | 35 | Early training |
| 163K | 13-15 | 2,130 | Grad norm starting to grow |
| 326K | **32** | 29K | First reward peak |
| 490K | 14 | 356K | Reward collapse |
| 954K | 12 | 9.4M | Prolonged trough |
| 1.1M | **39** | 20M | Second peak (best) |
| 1.27M | 14 | 37M | Second collapse |
| 1.43M | 23 | 62M | Partial recovery |
| 1.59M | 17 | 99M | Declining |
| 1.75M | NaN | NaN | Full divergence |

## Root Cause: `0.0 * (-inf) = NaN` (IEEE 754)

Two factors combined to produce the NaN:

### Factor 1: Multiply-by-zero instead of if-guard (implementation bug)

The paper uses **ALF (Agent Learning Framework)** which handles `use_entropy_reward=False` with an if-guard — the entropy computation is skipped entirely:

```python
# ALF (paper's code) — skips when disabled
if self._use_entropy_reward:
    entropy_reward = -torch.exp(log_alpha) * log_pi
    reward = reward + entropy_reward * discount
```

Our SAC trainer always computed the entropy term, relying on `alpha * log_prob = 0 * x = 0`:

```python
# Our code (before fix) — always evaluates
target_value = reward + gamma * (q_target - alpha * next_log_prob)
```

When `log_prob = -inf` (saturated tanh), this becomes `0.0 * (-inf) = NaN` in IEEE 754.

### Factor 2: bf16 causes tanh saturation 3x earlier

Our base config defaults to `use_amp=True` (bf16). The paper runs fp32. bf16 has only 8 mantissa bits vs fp32's 23, causing tanh to round to exactly `1.0` much sooner:

| Precision | `tanh(x) = 1.0` exactly at | `log(1 - tanh²) = -inf` at |
|-----------|---------------------------|---------------------------|
| **bf16** | **\|x\| >= 3.5** | **\|x\| >= 3.5** |
| fp32 | \|x\| >= 10.0 | \|x\| >= 10.0 |

Without entropy regularization (`alpha=0`), the actor has no incentive to keep actions stochastic. The actor mean grows freely, and at `|mean| > 3.5` (bf16) the TanhNormal `log_prob` hits `-inf`. This threshold is easily reached during normal training. In fp32, the safe range extends to `|mean| < 10`, which is much harder to reach.

### Full causal chain

1. `alpha=0` → no entropy penalty → policy free to become deterministic
2. Actor pushes means beyond `|3.5|` (bf16 threshold) → `tanh` saturates to exactly `1.0`
3. TanhNormal correction: `log(1 - tanh²) = log(0) = -inf` → `log_prob = -inf`
4. Target value: `0.0 * (-inf) = NaN`
5. NaN target → critic loss NaN → everything NaN simultaneously

The gradient norm growth (35 → 99M) was an early warning of mean saturation, not the cause itself.

## Fix Applied (2026-03-20)

### 1. If-guard on entropy terms in `src/trainers/sac.py`

Added `self._use_entropy` flag (True when `auto_alpha=True` or `alpha > 0`). All entropy computations in `_update()` are guarded:

```python
# Target value — entropy term only evaluated when enabled
if self._use_entropy:
    q_target = q_target - self.alpha * next_log_prob.unsqueeze(-1)
target_value = reward + (1 - done) * gamma * q_target

# Actor loss — entropy term only evaluated when enabled
if self._use_entropy:
    actor_loss = (self.alpha * log_prob.unsqueeze(-1) - q_new).mean()
else:
    actor_loss = (-q_new).mean()
```

This matches how ALF handles `use_entropy_reward=False`.

### 2. Switch Choi2025Config to fp32 in `papers/choi2025/config.py`

```python
use_amp: bool = False  # Paper uses fp32
```

bf16 narrows the safe numerical range for tanh-squashed distributions. Since the paper ran fp32, we match it.

### 3. Safe log_alpha initialization in `src/trainers/sac.py`

```python
safe_alpha = max(self.config.alpha, 1e-10)
self.log_alpha = torch.tensor(np.log(safe_alpha), device=self.device)
```

Prevents `log_alpha = np.log(0.0) = -inf` as a latent NaN source.

## Why the paper didn't hit this

1. **ALF skips entropy when disabled** — never computes `alpha * log_prob`
2. **fp32 precision** — tanh doesn't saturate until `|mean| >= 10`, a much harder threshold to reach
3. **Possible seed selection** — 5 seeds reported; some may have diverged but weren't shown

## Checkpoints Preserved

- `best.pt` — best reward checkpoint (~39 reward at ~1.1M steps)
- `step_100000.pt` through `step_1700000.pt` — periodic checkpoints
- `final.pt`, `interrupted.pt` — end-of-run checkpoints (NaN state, not useful)

## W&B Run

https://wandb.ai/qifan_wen-ohio-state-university/choi2025-replication/runs/13pqzhli
