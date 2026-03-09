---
type: experiment
created: 2026-03-07T00:00:00
updated: 2026-03-07T00:00:00
tags: [experiment, training, locomotion, elastica, ppo]
status: superseded
---

# Locomotion Elastica Training Session 9

## Context

Session 9 includes a critical fix: KL early stopping bug in PPO trainer was comparing accumulated sum (not average) against `target_kl=0.01`, causing only 1 of 10 epochs to run per batch. Fix: compare average KL per update. See `doc/logs/ppo-kl-early-stopping-fix.md`.

Also added entropy and KL to console output for diagnostics.

## Configuration

| Parameter | Value |
|-----------|-------|
| W&B run | `4i3z97iz` |
| Run dir | `output/locomotion_elastica_forward_20260306_231732` |
| Parallel envs | 16 |
| Total frames | 2M |
| FPS | ~55-60 (single run) |
| KL early stopping | Fixed (average KL, not accumulated) |
| Best checkpoint | `best.pt` at step 155,648 (reward=148.65) |

## Results (in progress, ~450k/2M frames)

### 10-Batch Rolling Average Rewards

| Batch Group | Steps | Avg Reward | Trend |
|-------------|-------|------------|-------|
| 1-10 | 8-82k | 87.6 | Baseline |
| 11-20 | 90-164k | 109.2 | +25% |
| 21-30 | 172-246k | 119.5 | +9% (peak) |
| 31-40 | 254-328k | 114.1 | -4% |
| 41-50 | 336-410k | 97.3 | -15% |

### Key Metrics at Peak (step 155,648)

- Reward: **148.65**
- Entropy: -0.0022
- KL: 0.0011
- Critic loss: 0.1551

## Analysis

### KL Fix Impact
- Actor loss increased ~40× (from 0.0001 to 0.004) — more policy improvement per batch
- Critic loss increased ~10× (from 0.06 to 0.57 at first batch) — value function learning more
- All 10 PPO epochs now run (vs only 1 before)
- Clear reward improvement trend: 88 → 120 over first 30 batches

### Post-Peak Decline
- After peaking at batch 25-30 (~200-245k), reward declined from avg 120 to avg 97
- Possible causes:
  1. PPO policy oscillation — typical in noisy reward environments
  2. High batch-to-batch variance from random initial headings
  3. Learning rate still relatively high at 19% through training (2.4e-4)
- Best checkpoint saved at peak — decline doesn't affect the best saved model

### Reward Variance
- Batch rewards range from 60-149
- Caused by random initial heading: aligned episodes (~140) vs misaligned (~60)
- Future improvement: normalize initial heading or use curriculum
