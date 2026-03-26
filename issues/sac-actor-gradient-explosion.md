---
name: SAC actor gradient explosion without entropy regularization
description: Actor gradient norms grow from 0.04 to 14.6B over 20M frames, causing reward collapse after early peak — inherent to paper's no-entropy SAC design
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-25
updated: 2026-03-26
tags: [sac, choi2025, follow_target, gradient-explosion, entropy, training-instability]
aliases: []
---

## Symptom

SAC training for `follow_target` reaches a best reward of 73.84 early in training, then collapses to ~18.76 by the end of 20M frames. The actor gradient norm grows exponentially from ~0.04 to 14.6 billion, while the critic gradient norm remains stable at 0.35–1.1 throughout.

## Observed Trajectory

| Step | Reward | Actor Grad Norm | Critic Grad Norm |
|------|--------|-----------------|------------------|
| Early | ~10 | ~0.04 | ~0.35 |
| ~5M | **73.84** (best) | ~10K | ~0.5 |
| 10M | ~30 | ~1M | ~0.7 |
| 15M | ~20 | ~1B | ~0.9 |
| 20M | 18.76 | **14.6B** | 0.92 |

## Root Cause

The paper explicitly disables entropy regularization (`alpha=0.0`, `auto_alpha=False`), citing Yu et al. (2022) "Do you need the entropy reward (in practice)?". Table A.1 lists "Entropy reward: None".

Without entropy:
1. The Q-landscape becomes increasingly sharp/peaky over training — the critic learns narrow, high-value regions
2. The actor loss is simply `(-Q).mean()`, so actor gradients reflect the sharpness of the Q-landscape directly
3. As Q-peaks sharpen, actor gradients grow exponentially
4. Eventually gradients are so large that even a small learning rate (0.001) produces oversized updates
5. The actor overshoots Q-peaks → reward collapses

The critic remains stable because its MSE loss is self-normalizing — the target values don't create the same sharpening effect.

## Paper Context

The paper trains for only **5M env steps** (Figure 3), where Follow Target reaches ~95 discounted return. The gradient explosion may not manifest severely at that scale, or their custom SAC implementation may handle it differently. Our 20M-frame run exposed the long-horizon instability.

The paper's SAC implementation is at `https://github.com/QuantuMope/dismech-rl` — we have not verified whether it includes implicit gradient handling that differs from standard SAC.

## Related Config Fixes (Same Investigation)

Two config mismatches were also discovered and fixed during this investigation:

1. **`max_grad_norm`: 0.5 → None** — inherited from `RLConfig` base class, but the paper does not use gradient clipping (not in Table A.1). The 0.5 clip was crushing 430M-norm gradients, giving an effective learning rate of ~1e-12. Removing the clip allowed the actor to learn but also exposed the underlying explosion.

2. **`actor_update_frequency`: 2 → 1** — the default updated the actor every 2 critic updates, but the paper updates every critic update (standard SAC). Fixed in `Choi2025Config`.

Both changes are in [config.py](papers/choi2025/config.py).

## Fix Applied (2026-03-26)

**Actor-only gradient clipping** (`actor_max_grad_norm=1.0`):

1. Added `actor_max_grad_norm` field to `SACConfig` — separate from `max_grad_norm` (critic clipping)
2. SAC trainer uses `actor_max_grad_norm` for actor, `max_grad_norm` for critic
3. `Choi2025Config` sets `actor_max_grad_norm=1.0`, `max_grad_norm=None` (critic stays unclipped since its gradients are stable at ~1.0)

This directly bounds actor update magnitude without changing the algorithm's semantics. The paper doesn't use entropy regularization, and we preserve that design choice. The logged `actor_grad_norm` metric still shows the pre-clip value for monitoring.

### Files changed
- `src/configs/training.py` — added `actor_max_grad_norm` field to `SACConfig`
- `src/trainers/sac.py` — use `actor_max_grad_norm` for actor gradient clipping
- `papers/choi2025/config.py` — set `actor_max_grad_norm=1.0` in `Choi2025Config`

### Validation
- All 44 diagnostic tests pass
- Unit test confirms `clip_grad_norm_` bounds actor gradients (34438 → 1.0)
- Probe env validation: 4/5 pass (probe 2 marginal, probe 5 which subsumes it passes)

## Previously Considered Mitigations

1. **Re-enable entropy** (`auto_alpha=True`) — stabilizes Q-landscape but deviates from paper
2. ~~**Gradient clipping**~~ → **Applied**: actor-only clipping via `actor_max_grad_norm=1.0`
3. **Train only 5M frames** — match paper's training horizon where the issue is less severe
4. **Spectral normalization** on critic — prevents Q-sharpening at the source (not needed with actor clipping)
5. **Lower learning rate** — slows explosion but also slows learning

## Run Details

- Config: `Choi2025Config` (SAC, alpha=0.0, auto_alpha=False, max_grad_norm=None, fp32)
- Run: `output/fixed_follow_target_sac_lr1e3_100envs_20260323_022956/`
- W&B: `https://wandb.ai/qifan_wen-ohio-state-university/choi2025-replication/runs/o976ome9`
- Final metrics: reward=18.76, actor_grad_norm=14.6B, critic_loss=0.004, critic_grad_norm=0.92

## Related Issues

- [sac-follow-target-nan-divergence.md](sac-follow-target-nan-divergence.md) — the `0*(-inf)=NaN` bug (different issue, now fixed)
