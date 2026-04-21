---
name: PPO Run 6 — clean reward (exp(-5*dist) only)
description: PPO follow_target with simplified reward function after removing improvement bonus. All diagnostics healthy but rewards plateau at 1.8x random.
type: experiment
status: complete
created: 2025-03-25
updated: 2025-03-25
tags: [ppo, reward, follow-target, phase4-plateau]
aliases: []
---

## Config

- **Reward**: `exp(-5.0 * dist)` — bounded [0, 1], no improvement bonus
- **entropy_coef**: 0.1
- **min_std**: 0.2
- **Network**: 4×512 ReLU MLP
- **Envs**: 100 parallel
- **Frames**: 5M (checked at 3M)
- **Seed**: 42

## Phase 2 Results (healthy)

| Metric | Early (0-600K) | Late (2.4-3M) | Healthy? |
|--------|----------------|---------------|----------|
| Explained Variance | 0.65 | 0.32 | Yes (>0) |
| EV negative % | 1.4% | ~11% | Acceptable |
| Entropy | 5.0 | 4.7 | Yes (stable) |
| Action std_min | 0.41 | 0.30 | Yes (>0.01) |
| Grad norm | 2.2 | 4.1 | Yes (0.01-10) |
| Advantage abs_max | 4.3 | 5.1 | Yes (<20) |
| Clip fraction | 0.07 | 0.07 | Yes (0.03-0.15) |
| **Reward SNR** | **1.02** | **1.06** | **Yes — 17x better than old reward (0.06)** |

## Phase 4 Plateau Diagnosis

| Check | Finding |
|-------|---------|
| Action dim stds | Vary 0.30-0.54 across 10 dims — not collapsed |
| Avg episode length | 199/200 — **all episodes hit max_steps** |
| Agent vs random | ~10.8 vs ~6.0 — **only 1.8x random** |

## Reward Trend (5 chunks across 3M frames)

| Chunk | Frames | EV | Reward | Entropy |
|-------|--------|-----|--------|---------|
| 1/5 | 0-600K | 0.65 | 12.0 | 4.25 |
| 2/5 | 600K-1.2M | 0.34 | 11.0 | 3.72 |
| 3/5 | 1.2M-1.8M | 0.28 | 10.4 | 5.17 |
| 4/5 | 1.8M-2.4M | 0.35 | 9.9 | 4.59 |
| 5/5 | 2.4M-3M | 0.32 | 13.2 | 4.69 |

## Conclusion

**Reward function fix worked** — SNR improved 17x (0.06→1.02), EV peaked at 0.83 (was 0.24), all diagnostics healthy. The trainer is no longer the bottleneck.

**New bottleneck: task difficulty.** The follow_target task with a moving target (0.05 m/s) in 3D under gravity is hard for flat PPO. The Phase 4 decision tree indicates:

> "All episodes hit max_steps → Agent never reaches goal. Task too hard at current shaping."

## Recommended Next Steps

1. **Reward shaping**: Add a smaller velocity-alignment bonus (reward moving toward target, but bounded to avoid SNR regression)
2. **Curriculum**: Start with stationary target, gradually increase `target_speed`
3. **Try SAC**: The paper used SAC which may simply be better suited (off-policy, replay buffer, continuous action space)
4. **Observation enrichment**: Add relative target-tip vector, tip velocity, target velocity to obs
