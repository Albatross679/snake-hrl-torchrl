---
name: SAC critic divergence due to missing gradient clipping
description: Critic loss exploded to 10K+ with grad norms >100K because SAC trainer did not apply max_grad_norm from config
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-19
updated: 2026-03-19
tags: [sac, gradient-clipping, training-stability, critic-divergence]
aliases: [sac-grad-clip]
---

## Symptom

SAC training diverged at ~17% (864K/5M frames). Critic loss rose from 0.3 → 11,356 → 53,565. Critic gradient norms exceeded 460K. Q-values swung wildly (-3 to +204). Alpha collapsed to ~4e-7 (zero entropy). Rewards dropped from 40+ to 7-20.

## Root Cause

The SAC trainer's `_update()` method did not apply gradient clipping, even though `max_grad_norm=0.5` was defined in the parent `RLConfig` class and inherited by `SACConfig`. With lr=0.001 and UTD=4, a single large gradient update could destabilize the critic, leading to cascading divergence.

## Fix

Added gradient clipping to both critic and actor backward passes in `src/trainers/sac.py`:
```python
if self.config.max_grad_norm is not None:
    nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
```

Same for actor. Uses `max_grad_norm=0.5` from `RLConfig` (the default).

## Timeline

- Step 0-275K: Healthy training, rewards 28-44
- Step 275K-563K: Instability begins — Q-values turn negative, rewards dip to 11-19
- Step 563K-716K: Rewards briefly recover to 27-44 but critic loss rising
- Step 716K-864K: Full divergence — critic loss >10K, Q-values exploding
- Fix applied: gradient clipping + fresh restart
