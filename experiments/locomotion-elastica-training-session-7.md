---
type: experiment
created: 2026-03-06T00:00:00
updated: 2026-03-06T00:00:00
tags: [experiment, training, locomotion, elastica, ppo]
status: running
---

# Locomotion Elastica Training Session 7

## Context

Session 7 is the first training run with corrected physics parameters and reward function. Previous sessions 1-6 all failed due to:

1. Rod radius too small (0.001m → EI negligible → zero or chaotic motion)
2. Velocity-based reward always negative (snake overshoots/drifts laterally)
3. RFT friction in slow Python for-loop

See `doc/issues/locomotion-elastica-physics-diagnosis.md` for full diagnosis.

## Configuration

| Parameter | Value |
|-----------|-------|
| W&B run | `jowp6ov2` |
| Run dir | `output/locomotion_elastica_forward_20260306_190328` |
| Rod radius | **0.02m** (was 0.001m) |
| Young's modulus | **1e5 Pa** (was 2e6) |
| Amplitude range | **(0, 5.0)** (was (0, 0.15)) |
| Reward function | **Distance-based potential** (was velocity-based) |
| RFT friction | Vectorized NumPy (was Python for-loop) |
| Parallel envs | 16 |
| Total frames | 2M |
| FPS | ~57-60 |
| ETA | ~10 hours |

## Results (in progress)

| Step | Reward | Critic Loss | Actor Loss |
|------|--------|-------------|------------|
| 8,192 | 101.72 | 0.0804 | 0.0003 |
| 90,112 | 100.98 | 0.0567 | 0.0001 |
| 172,032 | 153.61 | 0.1391 | 0.0002 |
| 253,952 | 108.36 | 0.0971 | 0.0003 |
| 335,872 | 143.19 | 0.1831 | 0.0001 |

## Analysis

### Positive signs
- **Reward is consistently positive** from batch 1 — massive improvement over sessions 1-6 (which were always negative, -52 to -124)
- **Reward trending upward** with oscillation: 101 → 101 → 154 → 108 → 143
- **FPS 4x faster** than session 5-6 (~57 vs ~14), thanks to vectorized RFT
- **Critic loss stable** at 0.06-0.18 (learning the value function)
- **Actor loss minimal** (0.0001-0.0003) — policy updates are smooth

### Potential concerns
- Reward oscillation between 100-154 — may be due to random initial heading requiring different strategies
- Critic loss increase at step 172k and 336k — could indicate reward variance

### Comparison with previous sessions

| Metric | Sessions 1-6 | Session 7 |
|--------|-------------|-----------|
| Mean reward | -52 to -124 | +101 to +154 |
| Goal reached | Never | Yes (verified in diagnostic) |
| FPS | 14-33 | 57-60 |
| v_g (velocity toward goal) | Always negative | Positive (implied by distance reduction) |
