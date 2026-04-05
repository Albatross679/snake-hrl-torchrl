---
name: steepness-ablation-launch
description: Added reward_steepness parameter and launched 14-hour steepness + PPO tuning ablation
type: log
status: complete
subtype: feature
created: 2026-04-03
updated: 2026-04-03
tags: [reward-design, steepness, ablation, ppo-tuning]
aliases: []
---

# Steepness Ablation Launch

## Code Changes

| File | Change |
|------|--------|
| `papers/choi2025/config.py` | Added `reward_steepness: float = 5.0` to `Choi2025EnvConfig` |
| `papers/choi2025/rewards.py` | Added `reward_steepness` param; replaced hardcoded `-5.0` with `-reward_steepness` |
| `papers/choi2025/env.py` | Passes `reward_steepness=self.config.reward_steepness` to reward function |
| `papers/choi2025/train_ppo.py` | Added `--reward-steepness`, `--gae-lambda`, `--entropy-coef` CLI args |
| `script/run_steepness_ablation.sh` | 3-phase launcher: steepness sweep → PPO tuning → scale-up |

## Experiment

Running in tmux session `ablation`. 8 runs, ~14h total wall time, 100 envs each.

- **Phase 1** (6h): S1 (k=1), S2 (k=2), S3 (k=3), S4 (k=5 baseline)
- **Phase 2** (4.5h): T1 (GAE λ=0.99), T2 (entropy=0.02), T3 (combined)
- **Phase 3** (3.5h): L1 (best overall, long run)

Results will be in `output/steepness_20260403_234027/`.
