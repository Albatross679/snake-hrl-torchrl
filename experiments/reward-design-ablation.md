---
name: reward-design-ablation
description: Systematic ablation study for follow_target reward design with PPO
type: experiment
status: planned
created: 2026-04-02
updated: 2026-04-02
tags: [reward-design, ablation, ppo, follow-target, curriculum, pbrs]
---

# Reward Design Ablation Study

## Motivation

PPO on `follow_target` plateaus at `dist_to_goal ≈ 0.73` with the current reward config
(`dist_weight=0.3, pbrs_gamma=0.99, smooth_weight=0.02`). The agent learns a basic "stay in
the area" behavior in the first few batches and never improves.

**Hypotheses to test:**
1. The agent barely beats random — the task is too hard without curriculum
2. PBRS adds noise without helping (SNR collapsed in PBRS-only run)
3. Curriculum is necessary — the agent needs easy targets to learn "approach" first
4. Heading reward provides useful directional signal
5. Smoothness penalty hurts more than it helps at this stage

## Experiment Matrix

### Group A: Baselines

| ID  | Config | Tests | Compare Against |
|-----|--------|-------|-----------------|
| A0  | Random policy (no training) | Is the task learnable at all? | All |
| A1  | dist=1.0 (vanilla paper default) | Does dense reward alone work? | A0 |

### Group B: Reward Component Ablation (no curriculum)

| ID  | dist_weight | pbrs_gamma | smooth | heading | Tests |
|-----|-------------|------------|--------|---------|-------|
| B1  | 0.0 | 0.99 | 0 | 0 | PBRS only (confirmed SNR collapse) |
| B2  | 1.0 | 0.99 | 0 | 0 | Does PBRS help full dense? |
| B3  | 0.3 | 0.99 | 0 | 0 | Reduced dense + PBRS (isolate smooth) |
| B4  | 0.3 | 0.99 | 0.02 | 0 | + smoothness (our current config) |
| B5  | 0.3 | 0.99 | 0 | 0.3 | + heading instead of smooth |
| B6  | 0.3 | 0.99 | 0.02 | 0.3 | Full stack, no curriculum |

### Group C: Curriculum Ablation

| ID  | dist_weight | pbrs_gamma | smooth | heading | curriculum | warmup |
|-----|-------------|------------|--------|---------|------------|--------|
| C1  | 1.0 | 0 | 0 | 0 | yes | 200 | Simplest: vanilla + curriculum |
| C2  | 1.0 | 0.99 | 0 | 0 | yes | 200 | Dense + PBRS + curriculum |
| C3  | 0.3 | 0.99 | 0.02 | 0 | yes | 200 | Current best + curriculum |
| C4  | 0.3 | 0.99 | 0.02 | 0.3 | yes | 200 | Full stack + curriculum |
| C5  | 0.3 | 0.99 | 0.02 | 0 | yes | 500 | Longer warmup |

## Comparison Plan

Each comparison answers one question. Primary metric: `mean_dist_to_goal` trend.

| Question | Compare | Expected if hypothesis holds |
|----------|---------|------------------------------|
| Does trained beat random? | A0 vs A1 | A1 dist < A0 dist |
| Does PBRS help dense? | A1 vs B2 | B2 dist < A1 dist |
| Does reducing dense weight help with PBRS? | B2 vs B3 | B3 ≤ B2 (might not help) |
| Does smoothness help? | B3 vs B4 | B4 dist < B3 dist |
| Does heading help? | B3 vs B5 | B5 dist < B3 dist |
| Is heading better than smoothness? | B4 vs B5 | Compare dist |
| Does full stack beat parts? | B6 vs B3,B4,B5 | B6 < all |
| Does curriculum help vanilla? | A1 vs C1 | C1 dist << A1 dist |
| Does curriculum help dense+PBRS? | B2 vs C2 | C2 dist << B2 dist |
| Does curriculum help current config? | B4 vs C3 | C3 dist << B4 dist |
| Best overall config? | C1 vs C2 vs C3 vs C4 | Lowest dist |
| Does longer warmup help? | C3 vs C5 | C5 dist < C3 dist |

## Run Parameters

All runs use:
- `--num-envs 32` (fast iteration)
- `--max-wall-time 30m` (screening budget)
- `--seed 42` (reproducible)
- `--task follow_target`

## Metrics to Track (per run)

### Primary
- `mean_dist_to_goal` (task performance — lower is better)
- `mean_episode_reward` / `rolling_mean_reward_100` (reward trend)

### Diagnostic (Phase 2 health)
- `explained_variance` (value function health)
- `diagnostics/action_std_min` (exploration)
- `grad_norm` (pre-clip gradient magnitude)
- Per-step reward SNR: `|reward_mean| / reward_std`

### Reward Components
- `mean_reward_dist` (raw distance component)
- `mean_reward_pbrs` (PBRS component)
- `mean_reward_smooth` (smoothness component)
- `mean_reward_align` (heading component)

## Success Criteria

1. **Baseline established**: A0 random baseline quantified
2. **Component contributions isolated**: Each B experiment compared to its ablation neighbor
3. **Curriculum impact measured**: C vs B comparisons show whether curriculum breaks plateau
4. **Best config identified**: Lowest dist_to_goal with healthy diagnostics
5. **No false positives**: Reward improvement without dist_to_goal improvement = reward hacking
