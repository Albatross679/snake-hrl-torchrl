---
name: steepness-and-tuning-ablation
description: 14-hour experiment studying reward steepness, PPO tuning, and scale-up for follow_target
type: experiment
status: planned
created: 2026-04-03
updated: 2026-04-03
tags: [reward-design, steepness, ppo-tuning, ablation, follow-target, curriculum]
---

# Reward Steepness & PPO Tuning Ablation

## Motivation

The previous ablation (12 configs, 32 envs, 30 min each) established:
- Best config C5: `dist_weight=0.3, pbrs_gamma=0.99, smooth=0.02, curriculum(500)` → d=0.543
- Curriculum is the dominant factor; reward shaping provides incremental gains
- The reward landscape diagnostics (Tab. reward-landscape) showed exp(-5d) is fundamentally sparse: 70% of steps yield r < 0.05

**What was never tested:** the steepness coefficient `k` in `exp(-kd)`. This is hardcoded at k=5. At the operating distance d≈0.6:

| k | exp(-k×0.6) | Signal ratio vs k=5 |
|---|-------------|---------------------|
| 5 | 0.050       | 1.0×                |
| 3 | 0.165       | 3.3×                |
| 2 | 0.301       | 6.0×                |
| 1 | 0.549       | 11.0×               |

Shallower steepness provides denser gradient signal exactly where the agent spends most of its time. This could break the distance plateau without any other changes.

## Questions

| # | Question | Why it matters |
|---|----------|----------------|
| Q1 | Does shallower steepness (k<5) break the d≈0.54 plateau? | Highest-potential unexplored variable; directly addresses the root cause |
| Q2 | What is the optimal steepness k? | Trade-off: too shallow loses discrimination near target, too steep loses signal at distance |
| Q3 | Does GAE λ=0.99 help PBRS (theory: longer telescoping horizon)? | The PBRS SNR analysis suggested higher λ smooths per-step noise |
| Q4 | Does lower entropy (0.02 vs 0.1) sharpen the policy once it's in the right region? | Current entropy is aggressive; may prevent fine-tuning near target |
| Q5 | How far can the best configuration push with a longer training budget? | Determines the performance ceiling |

## Prerequisites

1. **Add `--reward-steepness` CLI argument** to `papers/choi2025/train_ppo.py`
2. **Add `reward_steepness: float = 5.0` field** to `Choi2025EnvConfig`
3. **Pass steepness to reward function** — replace hardcoded `-5.0` in `rewards.py`
4. **Add `--gae-lambda` and `--entropy-coef` CLI overrides** to `train_ppo.py`

## Hardware

- 1× NVIDIA RTX A4000 16 GB, 12 CPU cores
- Measured throughput: ~2000 FPS at 32 envs, est. ~2500 FPS at 100 envs
- 14 hours wall time → ~126M total frames

## Experiment Plan

All runs use **100 envs** (3× previous ablation), **best reward stack** (dist_weight=0.3, pbrs_gamma=0.99, smooth=0.02, curriculum(500)), and **seed 42**.

### Phase 1: Steepness Sweep (4 runs × 1.5h = 6h)

| ID  | k | Expected signal at d=0.6 | Hypothesis |
|-----|---|--------------------------|------------|
| S1  | 1 | 0.549 (11×)              | Densest signal; may over-reward distant positions |
| S2  | 2 | 0.301 (6×)               | Sweet spot: dense gradient + still discriminates near/far |
| S3  | 3 | 0.165 (3.3×)             | Moderate improvement over k=5 |
| S4  | 5 | 0.050 (1×)               | C5 baseline at 100 envs (controls for env-count effect) |

**Common args:**
```bash
--task follow_target --num-envs 100 --max-wall-time 90m --seed 42 \
--dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 \
--curriculum --warmup-episodes 500
```

**Pairwise comparisons:**
- S1 vs S4: full steepness range effect
- S2 vs S3 vs S4: marginal steepness improvements
- S4 vs previous C5 (32 envs): env-count effect

### Phase 2: PPO Tuning (3 runs × 1.5h = 4.5h)

Uses the **best k from Phase 1**. These test whether PPO hyperparameters interact with steepness.

| ID  | Change from Phase 1 best | Rationale |
|-----|--------------------------|-----------|
| T1  | GAE λ: 0.95 → 0.99      | Higher λ lets telescoping smooth PBRS noise over longer horizons |
| T2  | Entropy: 0.1 → 0.02     | Current entropy may prevent fine policy sharpening near target |
| T3  | λ=0.99 + entropy=0.02   | Combined: test whether they interact positively |

### Phase 3: Scale-Up (1 run × 3.5h = 3.5h)

Best config from Phases 1+2. Determines **performance ceiling** with ~37M frames.

| ID  | Config | Budget |
|-----|--------|--------|
| L1  | Best overall | 3.5h (~37M frames) |

## Execution Plan

```
Phase 1 (sequential, 6h total):
  S1 (1.5h) → S2 (1.5h) → S3 (1.5h) → S4 (1.5h)
  ↓ analyze → pick best k

Phase 2 (sequential, 4.5h total):
  T1 (1.5h) → T2 (1.5h) → T3 (1.5h)
  ↓ analyze → pick best PPO config

Phase 3 (1 run, 3.5h):
  L1 (3.5h)
  ↓ final analysis
```

## Success Criteria

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Steepness breaks plateau | Any S1-S3 achieves d < 0.50 (vs S4 baseline) | Proves reward sparsity is the bottleneck |
| Best config beats C5 | d < 0.50 (vs C5's 0.543) | Net improvement over previous best |
| Scale-up finds ceiling | L1 performance converges (flattens in Q4) | Determines if more training helps |
| All runs healthy | EV > 0.3, SNR > 0.1, σ_min > 0.1 | Validates steepness doesn't break training |

## Metrics

Same as previous ablation:
- **Primary:** dist_to_goal (Q4 average, lower = better)
- **Diagnostic:** SNR, explained variance (EV), grad norm, action std
- **Secondary:** ΔDist (Q1→Q4 trend), episode reward, batches_since_improvement

## Analysis Plan

After all runs complete:
1. Steepness curve: plot d(Q4) vs k — expect U-shaped or monotone decreasing
2. Pairwise: S_best vs T_best vs L1 — quantify each intervention's contribution
3. Learning curves: overlay all 8 runs to see convergence speed differences
4. Reward hacking check: verify lower k doesn't inflate reward without improving distance
5. Update report Section 6.2 with new ablation results
