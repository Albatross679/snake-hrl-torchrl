---
name: surrogate-omega-z-poor-prediction
description: Surrogate MLP predicts angular velocity (omega_z) poorly (R²=0.23), with some rod elements going negative. Likely to cause compounding errors in multi-step rollouts.
type: issue
status: open
severity: medium
subtype: training
created: 2026-03-10
updated: 2026-03-10
tags: [surrogate, omega_z, rollout, phase-4, phase-3.1]
aliases: []
---

# Surrogate: Poor omega_z Prediction (R²=0.23)

## Observation

After training the best surrogate model (`sweep_lr1e3_h512x3`, 512x3 MLP, lr=1e-3), per-component R² on the validation set:

| Component | R² (mean) | R² (min) | R² (max) |
|-----------|-----------|----------|----------|
| pos_x | 0.896 | 0.733 | 0.964 |
| pos_y | 0.889 | 0.727 | 0.962 |
| vel_x | 0.688 | 0.580 | 0.755 |
| vel_y | 0.675 | 0.578 | 0.738 |
| yaw | 0.775 | 0.619 | 0.831 |
| **omega_z** | **0.232** | **-0.026** | **0.406** |

`omega_z` (angular velocity around z-axis, one per element, 20 values) has mean R²=0.23 with some elements below zero — meaning the model is worse than predicting the mean for those elements.

## Root Cause

`omega_z` is the **first derivative of yaw** (angular velocity) — analogous to linear velocity being the first derivative of position. It is not the second derivative of position; that would be linear acceleration.

The core problem is the **coarse transition window (dt = 0.5s)** combined with the CPG frequency range (0.5–3.0 Hz):

- At 3 Hz, dt=0.5s spans **1.5 full oscillation cycles** — omega_z reverses sign 2–3 times within a single transition
- The model must predict the *net* Δomega_z across those 1.5 cycles from only the start state
- This net change is highly sensitive to frequency: at f=2 Hz (dt = 1 full period), Δomega_z ≈ 0; at f=3 Hz (dt = 1.5 periods), Δomega_z ≈ −2·omega_z(t₀)

Concretely, two training samples with identical omega_z=+4 rad/s but different frequencies yield:
```
f=2 Hz: omega_z(t₀+0.5) = +4   →  Δomega_z =  0
f=3 Hz: omega_z(t₀+0.5) = -4   →  Δomega_z = -8
```

So the conditional distribution p(Δomega_z | inputs) is very wide. The model learns the conditional mean (near zero), which minimizes MSE but is useless for individual predictions — hence R²=0.23.

**Additional factor — missing per-element phase:** The model receives the global CPG phase as `[sin(ωt), cos(ωt)]`. But each element's omega_z is driven by its *local* phase in the traveling serpenoid wave:

```
kappa(s_i, t) = A · sin(k·s_i + ω·t + φ)
                         ↑         ↑
               element-specific  global phase
               phase offset      (what model gets)
```

Elements at opposite ends of the rod can be at completely opposite phases when the global phase is the same. The model has `wave_number` (k) in the action and implicit element ordering in the flat state vector, but must reconstruct per-element phase `k·s_i` implicitly through shared MLP weights — a significant implicit learning burden across all 20 elements simultaneously.

## Impact

- **Single-step prediction:** Low practical impact. `omega_z` errors don't directly corrupt position/velocity predictions in one step.
- **Multi-step rollout (Phase 4):** Medium risk. Errors in predicted `omega_z` feed back into the next state's `yaw` prediction (yaw += omega_z * dt), which in turn affects position. Rollout drift is expected to accumulate faster in heading than in translation.
- **RL training (Phase 5):** Low-medium risk. RL reward is position/velocity-based; `omega_z` is not directly rewarded. But heading errors can cause the surrogate trajectory to diverge from Elastica after many steps.

## Mitigation Options

### Architecture (Phase 3.1)
1. **History window** — provide the last k omega_z values per element as additional input. Each element's local phase can then be inferred directly from its own recent history, bypassing the need to reconstruct `k·s_i` from global phase + wave_number. Strong prior: this is the most principled fix.
2. **Explicit per-element phase encoding** — compute `k·s_i + ωt` for each element explicitly and append as 20 extra features. Gives the model the per-element phase directly without needing history.
3. **Increase rollout loss weight** — `rollout_loss_weight=0.1` currently; increasing to 0.3–0.5 forces consistency across steps without architectural change.
4. **Residual/separate omega_z head** — train a dedicated small branch on omega_z only, possibly with recurrent state.

### Data collection
5. **Finer dt** — reduce `substeps_per_action` from 500 to 100 (0.5s → 0.1s). At 3 Hz, this spans 0.3 cycles instead of 1.5 — Δomega_z becomes small and smooth. Trade-off: 5× more transitions needed for same coverage.
6. **Larger omega perturbation** — `perturb_omega_std=0.05 rad/s` is much smaller than operational values (~1–10 rad/s). Increase to 0.5–2.0 rad/s for better state space coverage.
7. **Save k-step sequence windows** — instead of independent (s, a, s') pairs, save sequences of k consecutive transitions. Enables history-conditioned training without architectural changes to the data format.

### Workaround
8. **Drop omega_z from state** — not needed for reward computation; reconstruct from yaw differences during rollout. Reduces state dim to 104 and eliminates the problem entirely.

## Recommended Next Action

Phase 3.1 (architecture experiments) is already planned. Priority order to try:
1. Explicit per-element phase encoding (cheap, no data change needed, directly addresses root cause)
2. History window k=3 (principled fix, requires dataset change to save sequences)
3. Higher rollout loss weight (free experiment, may help multi-step consistency)

If Phase 4 trajectory validation shows heading drift >~5° over 50 steps despite Phase 3.1 improvements, revisit finer-dt data collection (option 5).
