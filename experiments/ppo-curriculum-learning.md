---
name: PPO curriculum learning for follow_target
description: Curriculum ramps target speed from 20% to 100% over 100 warmup episodes per worker — improves reward 50% over non-curriculum baseline
type: experiment
status: running
created: 2026-03-25
updated: 2026-03-25
tags: [ppo, curriculum, follow-target, reward-shaping]
aliases: []
---

## Objective

Test whether curriculum learning (starting with a slower target and ramping to full speed) improves PPO training on the `follow_target` task, which plateaued at ~10-13 reward with flat `exp(-5*dist)` reward.

## Setup

- **Curriculum**: target speed ramps from 20% (0.01 m/s) to 100% (0.05 m/s) over 100 warmup episodes per worker
- **Why 20% not 0%**: Starting at speed=0 (static target) caused reward scale shock — rewards jumped to ~200/episode (vs ~10 moving), causing critic loss explosion and NaN gradients. Starting at 20% keeps reward scale consistent.
- **Config**: Same as clean-reward run (entropy_coef=0.1, min_std=0.2, ObservationNorm)
- **Implementation**: `CurriculumConfig` in config.py, speed ramp in env `_reset()`, `--curriculum` CLI flag
- **Run ID**: `z1qgpx5s` (W&B: `choi2025-replication`)
- **Training**: 5M frames, 100 parallel envs, seed 42

## Results

### Comparison: Curriculum vs No-Curriculum

| Metric | No Curriculum (3M frames) | Curriculum (3.4M frames) |
|--------|--------------------------|-------------------------|
| Avg reward | 10-13 | **15-18** |
| Max reward | 34.6 | **61.3** |
| vs random (6.0) | 1.8x | **2.7x** |
| EV | 0.28-0.65 (volatile) | 0.39-0.56 (stable) |
| Entropy | 4.6 (flat) | **5.2-5.7 (rising)** |
| Clip fraction | 0.07 | 0.07 |
| Grad norm | 4-7 | 3.7-4.5 |

### Reward Trend (Curriculum, 5 chunks)

| Chunk | EV | Reward | Entropy |
|-------|-----|--------|---------|
| 1/5 (0-670K) | 0.46 | 14.9 | 5.22 |
| 2/5 (670K-1.3M) | 0.40 | 16.0 | 5.57 |
| 3/5 (1.3M-2M) | 0.47 | 16.1 | 5.64 |
| 4/5 (2M-2.7M) | 0.39 | 15.0 | 5.47 |
| 5/5 (2.7M-3.4M) | 0.56 | **17.6** | 5.70 |

Warmup completed at ~2M frames (100 eps/worker). Post-warmup performance continues to improve.

### Failed First Attempt: speed=0 Start

Starting curriculum at 0% speed (static target) caused immediate training collapse:
- Static target -> agent gets ~1.0 reward/step -> ~200/episode
- Critic loss exploded to 3714 (vs normal ~0.03)
- Entropy collapsed to -660K
- KL divergence hit 4.8M
- NaN gradients within 16K frames

Fix: Start at 20% speed (`initial_speed_frac=0.2`) to maintain consistent reward scale.

## Diagnostics

All Phase 2 metrics healthy throughout:
- EV: 0.39-0.56 (more stable than non-curriculum's 0.04-0.83)
- Entropy: 5.2-5.7 (rising — curriculum encouraged broader exploration)
- Grad norm: 3.7-4.5 (well within 0.01-10)
- Clip fraction: 0.067-0.075 (healthy PPO clipping)
- Advantage abs_max: <6 (no explosion)

Phase 4: Episodes still hit max_steps (200/200). Agent is tracking better but task remains hard.

## Conclusions

1. **Curriculum helps**: ~50% reward improvement (10-13 -> 15-18 avg), 77% max improvement (34.6 -> 61.3)
2. **Better exploration**: Entropy stays higher with curriculum (5.5 vs 4.6) — the agent learns a broader repertoire during the easy phase
3. **More stable EV**: Curriculum EV is less volatile (0.39-0.56 vs 0.04-0.83)
4. **Task still hard**: Even with curriculum, avg episode length is 200/200 (never solves early)
5. **Initial speed matters**: Can't start at 0% — reward scale mismatch causes training collapse

## Files Modified

- `papers/choi2025/config.py`: Added `CurriculumConfig` dataclass
- `papers/choi2025/env.py`: Curriculum speed ramp in `_reset()`
- `papers/choi2025/tasks.py`: `speed_override` parameter in `TargetGenerator.sample()`
- `papers/choi2025/train_ppo.py`: `--curriculum` and `--warmup-episodes` CLI flags
