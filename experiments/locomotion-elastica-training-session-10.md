---
type: experiment
created: 2026-03-07T00:00:00
updated: 2026-03-07T00:00:00
tags: [experiment, training, locomotion, elastica, ppo]
status: complete
---

# Locomotion Elastica Training Session 10

## Context

Session 10 uses conservative PPO hyperparameters after Session 9 showed reward collapse (peaked at 119.5 avg then declined to 85). Key changes: lr=1e-4 (was 3e-4), epochs=4 (was 10), entropy_coef=0.02 (was 0.01).

## Configuration

| Parameter | Value |
|-----------|-------|
| W&B run | `367x9rh3` |
| Run dir | `output/locomotion_elastica_forward_20260307_021959` |
| Parallel envs | 16 |
| Total frames | 2M (2,007,040 actual) |
| FPS | ~55-60 |
| Training time | ~9 hours |
| Total episodes | 6,041 |
| learning_rate | **1e-4** (was 3e-4) |
| num_epochs | **4** (was 10) |
| entropy_coef | **0.02** (was 0.01) |
| Best checkpoint | `best.pt` at step 155,648 (reward=**156.66**) |
| Final checkpoint | `final.pt` |

## Results

### 10-Batch Rolling Average Rewards (full training)

| Batch Group | Steps | Avg Reward | Phase |
|-------------|-------|------------|-------|
| 1-10 | ~82k | 115.4 | Learning |
| 11-20 | ~164k | **140.0** | **Peak** |
| 21-30 | ~246k | 133.1 | Decline |
| 31-40 | ~328k | 106.4 | Decline |
| 41-50 | ~410k | 98.5 | Decline |
| 51-60 | ~492k | 84.6 | Trough |
| 61-70 | ~574k | 106.2 | Recovery |
| 71-80 | ~656k | 95.1 | Oscillation |
| 81-90 | ~738k | 102.4 | Steady state |
| 91-100 | ~820k | 105.3 | Steady state |
| 101-110 | ~902k | 102.6 | Steady state |
| 111-120 | ~984k | 102.9 | Steady state |
| 121-130 | ~1.06M | 104.9 | Steady state |
| 131-140 | ~1.15M | 108.1 | Steady state |
| 141-150 | ~1.23M | 104.4 | Steady state |
| 151-160 | ~1.31M | 102.9 | Steady state |
| 161-170 | ~1.39M | 104.6 | Steady state |
| 171-180 | ~1.47M | 100.9 | Steady state |
| 181-190 | ~1.56M | 100.6 | Steady state |
| 191-200 | ~1.64M | 95.7 | Steady state |
| 201-210 | ~1.72M | 103.0 | Steady state |
| 211-220 | ~1.80M | 101.6 | Steady state |
| 221-230 | ~1.88M | 105.9 | Steady state |
| 231-240 | ~1.97M | 90.3 | Final |

**Overall average: 104.6 across 244 batches**

### Final Episode Statistics
- Starvation rate: 14.3% (episodes terminated for no progress)
- Truncation rate: 21.4% (episodes hit 500-step limit)
- Goal reach rate: 64.3% (implied: 100% - 14.3% - 21.4%)
- Min episode reward: -36.17 (some episodes with bad headings)

### Key Observations

1. **Peak 17% higher than Session 9** (140 vs 119.5 avg, 157 vs 149 single batch)
2. **Three training phases**: Learning (0-164k), Decline (164-574k), Steady state (574k-2M)
3. **Steady state converged at ~102** for 1.4M frames (70% of training)
4. **Best model saved early** — step 155,648 captured the peak before decline
5. **Entropy maintained** — never collapsed to near-zero like Session 9

## Analysis

### vs Session 9

| Metric | Session 9 | Session 10 |
|--------|----------|-----------|
| Peak 10-batch avg | 119.5 | **140.0** (+17%) |
| Best single batch | 148.7 | **156.7** (+5%) |
| Steady state avg | 85-100 | **95-108** |
| Entropy at 100k | -0.005 (collapsed) | -0.044 (healthy) |
| Overall avg | ~95 | **104.6** (+10%) |

### Potential Improvements for Future Sessions

1. **Early stopping** at batch 20-30 to capture peak without decline
2. **Increase frames_per_batch** (16384 with 16 envs = 1024 steps/env) for lower batch variance
3. **Reward normalization / PopArt** to stabilize value targets
4. **Heading curriculum** — start with fixed heading, randomize later
