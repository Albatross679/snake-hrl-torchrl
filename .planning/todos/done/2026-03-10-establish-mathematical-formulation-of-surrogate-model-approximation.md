---
created: "2026-03-10T12:44:34.076Z"
title: Establish mathematical formulation of surrogate model approximation
area: general
files: []
---

## Problem

It is not yet formally stated what function the surrogate model is trying to approximate. Without a precise mathematical formulation, it is hard to evaluate the model, choose the right architecture, design the loss function, or communicate the approach clearly.

Concretely, we need to specify:
- The input space (e.g., current state, control action, time step)
- The output space (e.g., next state, state delta, per-node quantities)
- The ground-truth function being approximated (e.g., the Cosserat rod ODE/PDE solver step)
- The approximation class (e.g., neural network as a learned map f_θ: X → Y)
- The training objective (e.g., MSE on next-state prediction, rollout loss)
- Any physics-informed constraints or residual structure

## Solution

Write a formal mathematical definition:

1. Let the physics simulator define a transition operator T: (s_t, a_t) → s_{t+1} where s_t is the full rod state (positions, velocities, directors, curvatures) and a_t is the CPG action.
2. Define the surrogate as f_θ: (s_t, a_t) → ŝ_{t+1} trained to minimize E[||f_θ(s_t, a_t) - T(s_t, a_t)||²].
3. Specify exactly which components of s_t are inputs vs. outputs (e.g., per-node positions, angular velocities, omega_z).
4. State whether the model is autoregressive (rollout), single-step, or residual.
5. Document this in `knowledge/surrogate-mathematical-formulation.md` and reference it from the phase plan.
