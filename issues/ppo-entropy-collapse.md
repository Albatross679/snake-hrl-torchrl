---
name: PPO entropy collapse in follow_target task
description: Policy entropy collapses from 4.3 to -0.5 within 1.2M frames, causing reward decline
type: issue
status: resolved
severity: high
subtype: training
created: 2025-03-25
updated: 2025-03-25
tags: [ppo, entropy, exploration, debugging]
aliases: []
---

## Problem

After adding observation normalization, the PPO policy entropy collapsed rapidly:

| Phase | Frames | Entropy Proxy | Episode Reward |
|-------|--------|---------------|----------------|
| Early | 0–400K | 4.31 | 20.03 |
| Mid | 400K–800K | 1.78 | 15.60 |
| Late | 800K–1.2M | -0.52 | 16.67 |

Entropy proxy went from 4.3 to -0.5, meaning the policy became near-deterministic. Rewards initially improved then declined as exploration died.

## Diagnosis

Following the RL debug decision tree:
- `entropy collapsed (< 0.01)?` → YES
- → "Policy died. Increase entropy coef, reduce LR."

The `entropy_coef=0.01` (PPO default) was too weak to maintain exploration, especially after observation normalization made learning faster.

## Fix

Increased `entropy_coef` from 0.01 to 0.05 in `papers/choi2025/config.py` (Choi2025PPOConfig).

## Related

The actor config already has `min_std=0.1` to floor action standard deviations, but this alone wasn't sufficient to prevent entropy collapse at the policy level.
