---
name: PPO Follow-Target Diagnosis
description: Systematic diagnosis of PPO failure on follow_target task
type: experiment
status: running
created: 2026-04-02
updated: 2026-04-02
tags: [ppo, follow-target, diagnosis, reward-shaping]
aliases: []
---

# PPO Follow-Target Diagnosis

## Background

PPO shows no learning on follow_target (mean distance ~0.884m, CoV 23.4%) while SAC learns on the same task. This experiment systematically tests 6 hypotheses to isolate the root cause of PPO failure.

The diagnostic consists of 3 fast probe experiments (no training) and 5 training variant experiments (200K frames each).

## Hypotheses

### H1: Reward Sparsity

`exp(-5*d)` at the typical random-policy distance of ~0.78m gives reward ~0.02. This near-zero reward provides negligible gradient signal for PPO's on-policy updates. SAC succeeds because its off-policy replay buffer accumulates rare high-reward transitions.

**Tested by:** EXP1 (reward landscape probe), EXP5 (dense reward training)

### H2: Integrative Action Space

Delta curvature actions accumulate over time, making credit assignment difficult. The agent must learn that a sequence of small curvature changes produces the desired tip motion -- a temporal credit assignment problem that PPO's short rollouts may not capture.

**Tested by:** EXP2 (action impact probe), EXP4 (static target removes one dynamic)

### H3: Mock Physics Limitations

The simplified mock dynamics (damped by `0.1*dt`) may not produce physically meaningful responses to actions. If actions have no measurable effect on tip position, no algorithm can learn.

**Tested by:** EXP2 (action impact probe)

### H4: 3D Reachability

Upper hemisphere targets (z > 0) may be unreachable under gravity for a clamped rod. If a large fraction of targets are geometrically unreachable, the agent sees no reward signal regardless of actions.

**Tested by:** EXP3 (reachability analysis)

### H5: Observation Overload

148 observation dimensions (63 pos + 63 vel + 19 curvature + 3 target) overwhelm PPO's 8192-frame batches. The effective sample-to-parameter ratio is too low for on-policy learning with this observation dimensionality.

**Tested by:** EXP7 (reduced 9-dim observation)

### H6: Network Overparameterization

4x512 network (~1M parameters) with PPO's small effective batch sizes means insufficient gradient updates per parameter. A smaller network may learn more efficiently.

**Tested by:** EXP8 (2x128 network)

## Probe Results (EXP1-3)

### EXP1: Reward Landscape

Ran 50 episodes (10,000 steps) with random actions to characterize the reward distribution.

| Metric | Value |
|--------|-------|
| Mean reward (exp(-5d)) | 0.049304 |
| Std reward | 0.078497 |
| Median reward | 0.020485 |
| Mean distance | 0.7809m |
| Std distance | 0.2831m |
| Frac reward > 0.01 | 69.82% |
| Frac reward > 0.05 | 26.45% |
| Frac reward > 0.1 | 13.43% |

**Alternative reward scales (same distances):**

| Reward function | Mean | Std | Median |
|----------------|------|-----|--------|
| exp(-5d) (baseline) | 0.0493 | 0.0785 | 0.0205 |
| exp(-2d) | 0.2447 | -- | -- |
| exp(-1d) | 0.4764 | -- | -- |

**Interpretation:** The reward is not as sparse as originally hypothesized. At mean distance 0.78m (vs previously reported 0.884m), the mean reward is ~0.049 -- small but non-zero. However, 70% of steps have reward > 0.01, suggesting the signal exists but is weak. Using exp(-2d) would increase mean reward 5x to 0.24, providing substantially more gradient signal. **H1 is partially supported: reward is weak but not truly sparse.**

### EXP2: Action Impact

Compared 10 episodes with zero actions vs 10 episodes with random actions.

| Metric | Zero Actions | Random Actions |
|--------|-------------|----------------|
| Mean tip displacement/step | 0.000000 | 0.052417 |
| Std tip displacement/step | 0.000000 | 0.029581 |
| Mean state change/step | 0.005000 | 2.122621 |
| Displacement ratio | inf (zero baseline is exactly 0) |

**Interpretation:** Actions have a clear, measurable effect on tip motion and overall state. Zero actions produce zero tip displacement (rod stays straight), while random actions produce ~5cm tip displacement per step and 2.1 units of total state change. **H3 is refuted: mock physics responds meaningfully to actions.** The ratio is infinite because zero actions produce exactly zero displacement, confirming deterministic mock physics.

### EXP3: Reachability Analysis

Tested 20 targets with 10 random-action rollouts of 100 steps each.

| Region | N targets | Frac < 0.3m | Frac < 0.1m | Frac < 0.05m |
|--------|-----------|-------------|-------------|--------------|
| Overall | 20 | 90.0% | 25.0% | 10.0% |
| Upper (z > 0.15) | 7 | 85.7% | 14.3% | -- |
| Lower (z < -0.15) | 8 | 87.5% | 50.0% | -- |
| Equatorial (\|z\| <= 0.15) | 5 | 100.0% | 0.0% | -- |

Mean minimum distance achieved: 0.1534m.

**Interpretation:** 90% of targets are reachable within 0.3m with random actions, confirming the workspace is generally accessible. Lower hemisphere targets are easier to reach (50% within 0.1m) than upper hemisphere (14.3%), consistent with gravity assisting downward motion. **H4 is partially supported: upper hemisphere is harder but not unreachable.** The equatorial band shows 100% reachability at 0.3m but 0% at 0.1m, suggesting targets near the horizontal plane require more precise control.

## Training Experiments (EXP4-8)

Each experiment uses 200K frames, single environment, 30-minute wall time limit.

### EXP4: Static Target

**Modification:** `target_speed = 0.0` (target stays at initial sampled position)

**Expected outcome if target motion is the issue:** Learning signal appears since the agent only needs to solve a fixed-target reaching problem.

```
python -m choi2025.diagnose_ppo train --exp static_target
```

### EXP5: Dense Reward

**Modification:** Replace `exp(-5*d)` with `exp(-2*d)` via reward function patch.

**Expected outcome if H1 (reward sparsity) is correct:** Clear learning signal emerges. The 5x increase in mean reward from 0.049 to 0.245 should provide substantially better gradient signal for PPO.

```
python -m choi2025.diagnose_ppo train --exp dense_reward
```

### EXP6: PBRS

**Modification:** Enable potential-based reward shaping with `pbrs_gamma = 0.99`.

**Expected outcome:** PBRS adds a policy-invariant shaping signal `F = prev_dist - 0.99 * dist` that directly rewards reducing distance. This should provide step-wise gradient signal even when absolute reward is small.

```
python -m choi2025.diagnose_ppo train --exp pbrs
```

### EXP7: Reduced Observation

**Modification:** Replace 148-dim observation with 9 dims (tip_pos, tip_vel, target_pos). Use 2x64 network.

**Expected outcome if H5 (observation overload) is correct:** Learning signal with the minimal observation set. The 16x dimension reduction should make PPO's small batches sufficient for statistical estimation.

```
python -m choi2025.diagnose_ppo train --exp reduced_obs
```

### EXP8: Small Network

**Modification:** Replace 4x512 network with 2x128 (from ~1M to ~35K parameters).

**Expected outcome if H6 (overparameterization) is correct:** Faster learning due to better gradient-to-parameter ratio with PPO's batch sizes.

```
python -m choi2025.diagnose_ppo train --exp small_network
```

### Run All

```
python -m choi2025.diagnose_ppo train --exp all
```

## Analysis Framework

Decision tree for interpreting training experiment results:

1. **If EXP2 shows actions have no effect** -> H3 confirmed (mock physics broken), stop.
   - **Result: REFUTED.** Actions have clear effect (ratio = inf).

2. **If EXP3 shows poor reachability in upper hemisphere** -> H4 confirmed.
   - **Result: PARTIALLY SUPPORTED.** Upper hemisphere harder (14% vs 50% within 0.1m) but 86% reachable within 0.3m.

3. **If EXP5 (dense_reward) shows learning but baseline doesn't** -> H1 confirmed (reward sparsity is the bottleneck).

4. **If EXP7 (reduced_obs) shows learning** -> H5 confirmed (observation overload).

5. **If EXP8 (small_network) shows learning** -> H6 confirmed (network too large).

6. **If EXP4 (static_target) shows learning** -> Target motion is the issue (dynamic tracking too hard for PPO at this sample budget).

7. **If EXP6 (pbrs) shows learning** -> Shaping signal sufficient; base reward shape is the issue.

8. **If nothing helps** -> Multiple factors combine; may need curriculum + dense reward + reduced obs simultaneously.

## Output Files

- `output/diagnostics/exp1_reward_landscape.json` -- Reward landscape statistics
- `output/diagnostics/exp2_action_impact.json` -- Action impact measurements
- `output/diagnostics/exp3_reachability.json` -- Reachability by hemisphere
- `output/diagnostics/exp{4-8}_{name}/results.json` -- Training experiment results (after running)
