---
name: PPO missing episode reward tracking
description: RewardSum transform not applied — episode_reward never populated, best_reward stuck at -Infinity
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-19
updated: 2026-03-19
tags: [ppo, choi2025, reward, monitoring]
aliases: []
---

## Symptom

`best_reward: -Infinity` and `total_episodes: 0` across the entire 1M-frame training run. No reward data logged to W&B or metrics.jsonl despite the environment correctly emitting per-step rewards.

## Root Cause

The PPO trainer expects an `episode_reward` key in the rollout TensorDict at done boundaries (ppo.py:303). TorchRL's collector only populates this key if the env is wrapped with a `RewardSum` transform, which accumulates per-step rewards into a cumulative episode total and resets on done.

The `train_ppo.py` script was missing this transform. The env emitted rewards, PPO used them for optimization (training itself worked), but the monitoring/logging path had no data to report.

## Fix Applied

Added `env = env.append_transform(RewardSum())` after env creation in `papers/choi2025/train_ppo.py`. This populates `episode_reward` in the rollout data at episode boundaries.

## Files Modified

- `papers/choi2025/train_ppo.py` — added `RewardSum` import and transform
