---
name: PPO Heading Reward Component Analysis
description: Fair evaluation of heading reward via per-component logging — heading hurts distance tracking
type: experiment
status: complete
created: 2026-03-25
updated: 2026-03-25
tags: [ppo, reward-shaping, heading, component-logging, follow-target]
---

# PPO Heading Reward — Fair Component Analysis

## Motivation

Previous run added a heading reward (w=0.3) and showed 40.8 raw episode reward vs 18.7 baseline, claiming "2.2x improvement." This comparison was **invalid** — the heading component adds baseline reward regardless of policy quality. We added per-component logging to enable fair evaluation.

## Setup

### Run 1 (baseline): Curriculum Only

- W&B run: curriculum-only
- Output: `output/fixed_follow_target_ppo_lr1e4_100envs_20260325_163109`
- heading_weight: 0.0

### Run 2 (first heading run, no component logging)

- W&B run: `691dzznj`
- Output: `output/fixed_follow_target_ppo_lr1e4_100envs_20260325_185714`
- heading_weight: 0.3
- **No per-component logging** — cannot decompose reward retroactively

### Run 3 (heading with component logging)

- W&B run: `z3g57afm`
- Output: `output/fixed_follow_target_ppo_lr1e4_100envs_20260325_204111`
- heading_weight: 0.3, same seed/params as Run 2
- **Has per-component logging**: `reward_dist`, `reward_align`, `dist_to_goal`

Common config: 100 parallel envs, 5M frames, seed 42, curriculum (warmup=100, initial_frac=0.2).

## Code Changes for Component Logging

1. **`rewards.py`**: Added `return_components=True` parameter → returns `(reward, {dist_to_goal, reward_dist, reward_align})`
2. **`env.py`**: Registered component keys in `observation_spec` (required for ParallelEnv shared memory). Returns in both `_reset` (zeros) and `_step` (actual values).
3. **PPO trainer**: Already had plumbing at lines 363-366 and 651-658 — no changes needed.

## Phase 1: Probe Validation

44/44 probes pass. No implementation bugs.

## Phase 2: Diagnostic Metrics (5 chunks)

| Metric | C1 | C2 | C3 | C4 | C5 | Verdict |
|--------|-----|-----|-----|-----|-----|---------|
| EV | 0.608 | 0.615 | 0.551 | 0.644 | 0.570 | Healthy (>0.5) |
| Entropy | 4.25 | 3.65 | 4.33 | 4.26 | 4.32 | Healthy |
| Grad norm | 8.5 | 9.3 | 7.6 | 6.3 | 5.5 | Healthy, improving |
| Clip frac | 0.071 | 0.077 | 0.070 | 0.066 | 0.057 | Healthy (<0.3) |
| Std min | 0.365 | 0.307 | 0.346 | 0.309 | 0.320 | Healthy (>0.2) |

## Phase 3: Decision Tree

All diagnostics healthy → trainer is working. Problem (if any) is in reward/env/task design.

## Phase 4, Tier 1: Reward Component Analysis

### Tier 1.1 — Component Dominance

The heading component contributes ~72% of total reward:

| Chunk | Total reward/step | Dist contribution (0.7 × reward_dist) | Heading contribution (0.3 × reward_align) | Heading % |
|-------|------------------|---------------------------------------|------------------------------------------|-----------|
| 1 | 0.183 | 0.051 | 0.132 | 72% |
| 5 | 0.196 | 0.054 | 0.141 | 72% |

### Fair Comparison: Distance Component Only

The only valid cross-run metric is `reward_dist = exp(-5*dist)`:

| Chunk | Baseline | Heading | Delta | % Change |
|-------|----------|---------|-------|----------|
| 1 (0-1M) | 0.0811 | 0.0724 | -0.0087 | **-10.7%** |
| 2 (1-2M) | 0.0762 | 0.0577 | -0.0185 | **-24.3%** |
| 3 (2-3M) | 0.0801 | 0.0689 | -0.0112 | **-14.0%** |
| 4 (3-4M) | 0.0828 | 0.0664 | -0.0164 | **-19.8%** |
| 5 (4-5M) | 0.0935 | 0.0775 | -0.0160 | **-17.1%** |

**Heading reward makes distance tracking 10-24% worse across all chunks.**

### Distance to Target

| Chunk | Baseline (inferred) | Heading (measured) | Delta |
|-------|--------------------|--------------------|-------|
| 1 | 0.503 | 0.664 | +0.162 |
| 2 | 0.515 | 0.711 | +0.196 |
| 3 | 0.505 | 0.676 | +0.171 |
| 4 | 0.498 | 0.680 | +0.182 |
| 5 | 0.474 | 0.640 | +0.166 |

Agent with heading reward stays ~0.17m farther from target on average.

### Heading Alignment (heading run only)

| Chunk | reward_align |
|-------|-------------|
| 1 | 0.440 |
| 2 | 0.461 |
| 3 | 0.458 |
| 4 | 0.462 |
| 5 | 0.471 |

Alignment barely improves (0.44 → 0.47). Random alignment is ~0.5. The agent is not meaningfully learning to point toward the target.

### Tier 1.2 — Reward SNR

| Run | SNR |
|-----|-----|
| Baseline | 1.21 |
| Heading | 4.73 |

Higher SNR from heading run is misleading — the heading component inflates the mean without adding useful learning signal.

### Tier 1.5 — Reward Hacking

The heading objective partially conflicts with the distance objective. The agent can improve alignment (rotate tip toward target) without actually moving closer. This creates an optimization shortcut that degrades the primary tracking task.

## Conclusions

1. **Heading reward (w=0.3) is harmful** — it makes the agent 10-24% worse at the actual task (approaching the target)
2. **Per-component logging is essential** for any multi-component reward function — raw reward totals are not comparable across different reward shapes
3. **Curriculum-only** showed genuine improvement (reward_dist 0.081 → 0.094, still climbing at 5M frames) and remains the best configuration

## Recommendations

1. **Disable heading reward** (heading_weight=0.0) for future runs
2. **Keep per-component logging** — enables fair evaluation of any future reward changes
3. **Train curriculum-only longer** (10-20M frames) — it was still improving at 5M
4. If directional signal is still desired, try **potential-based reward shaping** (PBRS) for distance reduction rather than heading alignment — PBRS is guaranteed not to change the optimal policy
