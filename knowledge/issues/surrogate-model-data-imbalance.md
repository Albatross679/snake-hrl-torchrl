---
name: Surrogate Model Data Imbalance
description: Training data from policy rollouts clusters around a few modes in state-action space, leaving large regions underrepresented for the neural surrogate model
type: issue
created: 2026-03-09
updated: 2026-03-10
status: open
severity: high
subtype: training
tags: [surrogate-model, data-imbalance, sampling, elastica]
aliases: []
---

# Surrogate Model Data Imbalance

## Problem

When collecting training data for the Elastica neural surrogate model using a trained policy's rollouts, the data distribution clusters around a few behavioral modes (e.g., the learned gait). Large regions of the state-action space are underrepresented or entirely unvisited. This causes the surrogate to be accurate only near the policy's trajectory distribution and unreliable elsewhere — a critical problem if the surrogate is later used for RL training, where exploration will push into unseen regions.

## Impact

- Surrogate model generalizes poorly outside the training distribution
- RL agents trained on the surrogate may exploit inaccurate predictions in sparse regions
- Wasted compute on redundant samples from dense regions

## Proposed Solutions

### Data Collection (Primary)

Replace policy-dependent rollouts with space-filling sampling:

1. **Sobol sequences** (recommended) — deterministic, low-discrepancy, incrementally extensible, good joint-space uniformity via `scipy.stats.qmc.Sobol`
2. **Latin Hypercube Sampling** — guarantees marginal uniformity, good for one-shot collection
3. **Grid sampling** — only viable for very low-dimensional subspaces (d ≤ 3)

### Loss Weighting (Complementary)

- Estimate sample density (KDE or histogram binning) and weight loss inversely by `1/density`
- Focal loss to upweight hard/rare examples

### Hybrid Collection Strategy

- Collect a base dataset with Sobol/LHS for uniform coverage
- Supplement with policy rollouts for on-distribution accuracy
- Add exploration noise to policy rollouts to broaden coverage

## Angular Velocity (Omega) Coverage Gap

Phase 2 validation identified a specific axis of imbalance: **angular velocities (omega) are severely underrepresented** at realistic mid-gait magnitudes.

### Root Cause

Before each collection run, the rod is reset and then perturbed via `perturb_rod_state()`. The omega perturbation draws noise from `N(0, omega_std)` rad/s. Phase 1 used the default `omega_std=0.05`, which is essentially zero noise — almost every run started with near-zero angular velocities. During actual locomotion, segments rotate at 1–3 rad/s as the wave propagates. The surrogate never saw these states during training.

### Fix: perturb_omega_std=1.5

Phase 02.1 and Phase 02.2 raise this to `perturb_omega_std=1.5` (30× larger), so runs start with angular velocities drawn from a range that overlaps with real locomotion dynamics. This is a **data coverage parameter**, not an ML hyperparameter — it controls what region of state space the collected transitions cover, not model training.

| Phase | perturb_omega_std | Effect |
|-------|-------------------|--------|
| Phase 1 | 0.05 rad/s | Near-zero omega at run start — high-omega states absent from dataset |
| Phase 02.1 / 02.2 | 1.5 rad/s | Realistic rotational velocities covered — dataset spans mid-gait states |

### Why This Matters for RL

The RL agent's policy operates in exactly the high-omega regime (the snake is actively undulating). A surrogate trained only on low-omega states will produce inaccurate predictions on every step of an RL rollout, making surrogate-based RL training unreliable.

## Collection Monitoring

Long-running data collection (16 workers, 50M transitions) runs in a tmux session and requires health monitoring to catch stalls, NaN floods, or disk issues early. Phase 02.2 uses a `/loop 15m` monitor that checks every 15 minutes:

1. tmux session alive (`gsd-collect-rl`)
2. Batch file count and recency in `data/surrogate_rl_step/`
3. Total transitions collected so far
4. Last 10 lines of `output/collect_rl_step.log` for errors

The loop is started immediately after collection launches and stopped when collection completes or is interrupted. This pattern should be reused for any future long-running collection phase.

## References

- [Non-Uniform Data Coverage in Supervised Learning](../knowledge/non-uniform-data-coverage-supervised-learning.md)
