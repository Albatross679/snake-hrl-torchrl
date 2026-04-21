---
name: PPO heading reward for follow_target
description: Heading reward (w=0.3) with curriculum doubles avg reward to 40.8 — 2.2x over curriculum-only, 3.3x over baseline
type: experiment
status: complete
created: 2026-03-25
updated: 2026-03-25
tags: [ppo, heading-reward, curriculum, follow-target, reward-shaping]
aliases: []
---

## Objective

Test whether adding a heading reward component (bonus for pointing tip toward target) improves PPO training on `follow_target`, which plateaued at ~18.7 avg with curriculum-only.

## Setup

- **Heading reward**: `r = 0.7 * exp(-5*dist) + 0.3 * (1+cos_sim)/2`
  - cos_sim = dot(tip_tangent, direction_to_target)
  - Provides gradient signal even when far away (where exp(-5*dist) ≈ 0)
- **Curriculum**: target speed ramps 20% → 100% over 100 warmup episodes/worker
- **Config**: entropy_coef=0.1, min_std=0.2, ObservationNorm, 4x512 MLP
- **Run ID**: `691dzznj` (W&B: `choi2025-replication`)
- **Training**: 5M frames, 100 parallel envs, seed 42

## Results

### Three-Way Comparison

| Metric | No Curriculum | Curriculum Only | **Curriculum + Heading** |
|--------|--------------|-----------------|--------------------------|
| Avg reward (final chunk) | 10-13 | 18.7 | **40.8** |
| Best reward | 34.6 | 61.3 | **56.0** |
| vs random (6.0) | 1.8x | 3.1x | **6.8x** |
| EV (final chunk) | 0.55 | 0.55 | **0.63** |
| Entropy (final) | 4.6 | 5.49 | 4.85 |
| SNR | 1.02 | 1.30 | **1.91** |
| Still improving at 5M? | No | Slowly | **Yes (+3.4)** |

### Reward Trend (5 chunks)

| Chunk | Frames | EV | Entropy | Grad | Reward Avg | Max | SNR |
|-------|--------|-----|---------|------|-----------|-----|-----|
| 1/5 | 0-1M | 0.535 | 4.29 | 7.6 | 35.4 | 54.8 | 1.65 |
| 2/5 | 1-2M | 0.640 | 3.57 | 8.7 | 35.0 | 53.7 | 1.81 |
| 3/5 | 2-3M | 0.566 | 3.62 | 7.9 | 37.4 | 51.9 | 1.89 |
| 4/5 | 3-4M | 0.625 | 4.25 | 6.3 | 38.1 | 52.9 | 1.81 |
| 5/5 | 4-5M | 0.633 | 4.85 | 5.1 | **40.8** | 56.0 | 1.91 |

Reward still rising at 5M frames (first half 35.6 → second half 39.0).

### Mid-Training Instability (Resolved)

Around 0.7-1.6M frames (during curriculum warmup → full-speed transition):
- Entropy dipped to 3.14 (from 5.09)
- Grad norm spiked to 11.9 (from 3.2)
- Action std_min dropped to 0.291

All three metrics recovered naturally by 3M frames without intervention. Likely caused by the reward distribution shift when curriculum reaches full speed.

## Diagnostics

All Phase 2 metrics healthy at completion:
- EV: 0.633 (last chunk avg), 0.879 (final iteration) — best of all runs
- Entropy: 4.85 (healthy, not collapsing)
- Grad norm: 5.1 (declining, well within range)
- Clip fraction: 0.058 (healthy, declining)
- Advantage abs_max: ~4.0 (excellent)
- Action dim spread: 0.097 (no dimension collapse)
- EV negative: 7.5% (down from 11% in curriculum-only)

Phase 4: Episodes still hit max_steps (200/200). Agent tracks much better but task remains challenging.

## Conclusions

1. **Heading reward is the largest single improvement**: 2.2x over curriculum-only, 3.3x over baseline
2. **Better SNR**: 1.91 vs 1.30 — heading gives cleaner signal than distance-only
3. **Better value function**: EV 0.63 vs 0.55 — combined reward is more predictable
4. **Still improving**: reward curve not plateaued at 5M frames
5. **Transient curriculum transition instability**: mid-training entropy/grad spikes resolve naturally

## Recommended Next Steps

1. **Train longer** — 20-50M frames (reward still climbing)
2. **Try heading_weight sweep** — 0.1, 0.2, 0.3, 0.4, 0.5 to find optimal
3. **Add distance-to-target in observation** — scalar dist might help value function

## Files Modified

- `papers/choi2025/rewards.py`: Extended `compute_follow_target_reward` with heading
- `papers/choi2025/config.py`: Added `heading_weight` to `Choi2025EnvConfig`
- `papers/choi2025/env.py`: Pass tip_tangent + heading_weight to reward
- `papers/choi2025/train_ppo.py`: `--heading-weight` CLI flag
