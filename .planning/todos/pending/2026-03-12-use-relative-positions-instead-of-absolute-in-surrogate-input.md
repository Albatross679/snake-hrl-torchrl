---
created: 2026-03-12T16:55:40.178Z
title: Use relative positions instead of absolute in surrogate input
area: general
files:
  - report/report.tex:404-416
  - aprx_model_elastica/model.py
  - aprx_model_elastica/train_surrogate.py
---

## Problem

The surrogate model's 124-dim state vector includes absolute node positions (x_{t,i}, y_{t,i}) for all 21 nodes. However, the underlying Cosserat rod physics is translation-invariant: the ODE right-hand side (accelerations, forces, moments) depends only on relative quantities — inter-node displacements, curvatures, local velocities, and angular velocities. Absolute position never enters any force computation (internal forces use spatial derivatives, RFT uses local velocity, CPG uses arc-length position).

This means the transition function T(s_t, a_t) - s_t is invariant to rigid translation of the rod, but the surrogate must learn this invariance implicitly from data. This wastes model capacity and likely hurts generalization to rod positions not well-covered in the training set.

## Solution

Transform the position components of the surrogate input from absolute to relative coordinates. Options to investigate:

1. **Center-of-mass relative** — subtract COM from all node positions: x̃_{t,i} = x_{t,i} - x̄_t. Preserves rod shape, removes global translation. Requires predicting COM delta separately or reconstructing from relative predictions.

2. **Inter-node displacements** — use Δx_{i} = x_{t,i+1} - x_{t,i} (20 values instead of 21). More compact, directly encodes rod shape, but loses one DOF (COM position). Since COM velocity is already captured in the velocity components, this may be sufficient.

3. **Hybrid** — keep COM position as 2 features + 20 inter-node displacements (42 → 22 features per axis, total input drops from 189 to ~149).

Key considerations:
- The output (state delta prediction) may also benefit from relative framing
- Normalization statistics (μ_Δ, σ_Δ) would need recomputation
- Need to verify that the RL policy can still reconstruct absolute positions for reward computation
- Re-training required; compare validation MSE and rollout RMSE against absolute-position baseline
