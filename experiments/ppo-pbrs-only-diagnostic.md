---
name: PPO PBRS-Only Diagnostic
description: PBRS-only reward (no base distance) performs 28.7% worse than baseline — shaping signal alone is insufficient due to critically low SNR
type: experiment
status: complete
created: 2026-03-26
updated: 2026-03-26
tags: [ppo, pbrs, reward-shaping, curriculum, follow-target, rl-debug, ablation]
---

# PPO PBRS-Only Diagnostic (4-Phase RL Debug)

## Summary

PBRS-only reward (removing the base exp(-5*dist) distance reward, using only the shaping signal F(s,s') = prev_dist - gamma*dist) performs **28.7% worse** than the curriculum-only baseline on the fair metric. The shaping signal alone has critically low SNR (0.103) and is insufficient to drive learning. This confirms that PBRS works best as a **supplement** to the base reward, not a replacement.

## Setup

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Task | follow_target |
| PBRS gamma | 0.99 |
| PBRS only | **True** (no base distance reward) |
| Heading weight | 0.0 (disabled) |
| Curriculum | enabled (warmup=100 episodes, initial_frac=0.2) |
| Num envs | 100 parallel |
| Total frames | 5,000,000 (4,993,800 actual) |
| Seed | 42 |
| W&B run | `61aq9blv` |
| Run dir | `output/fixed_follow_target_ppo_lr1e4_100envs_20260325_231809` |

Command:
```bash
python3 papers/choi2025/train_ppo.py --task follow_target --num-envs 100 \
  --total-frames 5000000 --seed 42 --curriculum --warmup-episodes 100 \
  --pbrs-gamma 0.99 --pbrs-only
```

## Phase 1: Probe Validation

44/44 probes pass. No implementation bugs.

## Phase 2: Diagnostic Metrics (5 chunks)

| Metric | C1 | C2 | C3 | C4 | C5 | Verdict |
|--------|-----|-----|-----|-----|-----|---------|
| EV | 0.588 | 0.713 | 0.811 | 0.875 | 0.884 | Healthy, rising strongly |
| Entropy | 5.03 | 5.37 | 5.59 | 5.54 | 5.71 | Healthy, highest of all runs |
| Grad norm | 3.23 | 3.17 | 2.99 | 3.29 | 3.36 | Healthy, stable |
| Clip frac | 0.073 | 0.074 | 0.069 | 0.066 | 0.062 | Healthy |
| Std min | 0.389 | 0.420 | 0.439 | 0.449 | 0.489 | Healthy, rising |

Notable: EV is the **highest** of all four runs (0.884 at C5), meaning the value function predicts the PBRS-only returns very well. This makes sense — the PBRS reward is highly predictable (just the change in distance). Entropy is also highest (5.71), suggesting the weak reward signal fails to differentiate between good and bad actions, so the policy stays near-uniform.

## Phase 3: Decision Tree

All diagnostics healthy. Trainer working correctly. Problem is reward signal → Phase 4.

## Phase 4, Tier 1: Reward Component Analysis

### Tier 1.1 — Component Balance

With PBRS-only, the entire reward IS the PBRS component. The base distance reward contributes 0% by design.

| Chunk | reward_pbrs (= total) | For reference: reward_dist |
|-------|----------------------|---------------------------|
| C1 | 0.01209 | 0.0602 |
| C2 | 0.01399 | 0.0654 |
| C3 | 0.01247 | 0.0602 |
| C4 | 0.01314 | 0.0644 |
| C5 | 0.01251 | 0.0667 |

The PBRS reward per step is ~0.013 — about 5x smaller than what the base distance reward would provide.

### Tier 1.2 — Reward SNR (Critical Finding)

| Run | SNR | Assessment |
|-----|-----|-----------|
| Baseline | 1.21 | Healthy |
| Heading | 4.73 | Inflated |
| PBRS+Base | 1.40 | Healthy |
| **PBRS-Only** | **0.103** | **Critical — below 0.1 threshold** |

**The PBRS-only SNR is 12x lower than baseline.** The shaping signal F = prev_dist - gamma*dist is near-zero-mean by construction (getting closer yields positive, moving away yields negative, with gamma=0.99 making them nearly cancel). This means the signal-to-noise ratio is critically low — the agent cannot reliably distinguish which actions are beneficial.

### Fair Comparison: reward_dist = exp(-5*dist)

| Chunk | Baseline | Heading | PBRS+Base | **PBRS-Only** |
|-------|----------|---------|-----------|---------------|
| C1 (0-1M) | 0.0811 | 0.0724 | 0.0981 | 0.0602 |
| C2 (1-2M) | 0.0762 | 0.0577 | 0.0988 | 0.0654 |
| C3 (2-3M) | 0.0801 | 0.0689 | 0.0980 | 0.0602 |
| C4 (3-4M) | 0.0828 | 0.0664 | 0.1038 | 0.0644 |
| C5 (4-5M) | 0.0935 | 0.0775 | 0.1287 | **0.0667** |

**PBRS-Only is the worst performer** at -28.7% vs baseline. The improvement from C1→C5 is minimal (+10.8%), suggesting the agent barely learns from the shaping signal alone.

### Distance to Target

| Chunk | Baseline | Heading | PBRS+Base | PBRS-Only |
|-------|----------|---------|-----------|-----------|
| C1 | 0.503 | 0.664 | 0.619 | 0.751 |
| C5 | 0.474 | 0.640 | 0.548 | **0.727** |

PBRS-Only maintains the largest distance to target throughout training.

### Episode Rewards

| Run | C1 ep_rew | C5 ep_rew | Best |
|-----|-----------|-----------|------|
| Baseline | 16.1 | 18.7 | 61.3 |
| Heading | 36.5 | 39.1 | 84.9 |
| PBRS+Base | 21.3 | 27.4 | 92.6 |
| PBRS-Only | 1.89 | 1.80 | 3.3 |

PBRS-Only episodes are 10x shorter in total reward because the shaping signal is so small.

## Conclusions

1. **PBRS-only is insufficient**: -28.7% on the fair metric, worst of all four runs
2. **SNR is the root cause**: at 0.103, the reward signal is near the noise floor (Tier 1.2 criterion: SNR < 0.1 = "drowned in noise")
3. **Value function learns well but can't help**: EV=0.884 (highest) because the PBRS signal is predictable, but predicting near-zero returns doesn't help the policy improve
4. **Entropy stays high**: 5.71 (highest) because the weak signal can't differentiate actions
5. **PBRS is a supplement, not a replacement**: It needs the base distance reward to provide the learning gradient; PBRS then accelerates learning by +38%

## Run Ranking (5M frames, fair metric)

| Rank | Run | r_dist C5 | vs Baseline |
|------|-----|-----------|-------------|
| 1 | PBRS+Base | 0.1287 | **+37.6%** |
| 2 | Baseline | 0.0935 | -- |
| 3 | Heading | 0.0775 | -17.1% |
| 4 | PBRS-Only | 0.0667 | -28.7% |

## Key Takeaway

PBRS (Ng et al. 1999) is mathematically guaranteed to preserve the optimal policy — but this guarantee assumes the base MDP already has a reward signal that defines which states are good. Removing the base reward removes the signal that tells the agent WHERE to go; PBRS only tells it whether it's getting CLOSER or FARTHER, which at gamma=0.99 nearly cancels out over trajectories.
