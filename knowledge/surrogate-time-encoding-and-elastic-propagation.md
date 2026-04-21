---
name: Surrogate time encoding and elastic propagation analysis
description: Analysis of why the surrogate needs omega*t phase encoding (not raw t), and why elastic perturbation propagation does not affect surrogate design
type: knowledge
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, time-encoding, cosserat-rod, elastic-propagation, phase, serpenoid]
aliases: []
---

## Time Encoding Problem

### Current encoding

The surrogate model receives time as `[sin(t), cos(t)]` where `t` is the raw serpenoid time accumulator (elapsed simulation seconds). This is defined in `aprx_model_elastica/state.py:131-152`.

### Why time encoding is needed

The serpenoid curvature is recomputed at every substep within an RL step:

```python
for _ in range(500):
    curvatures = A * sin(k * s + omega * time + phi)
    rod.rest_kappa = curvatures
    # integrate one substep
    time += dt_substep
```

The rest curvature pattern shifts phase by `omega * dt_substep` each substep. Two identical rod states receiving identical actions but starting at different phases in the oscillation cycle receive different sequences of 500 bending torque profiles and end up in different final states. Without time information, the model sees identical `(state, action)` inputs mapping to different `next_state` targets, which it can only average over.

### The problem with `sin(t)` vs `sin(omega*t)`

The physically meaningful quantity is the oscillation phase `omega * t`, not raw time `t`:

- `omega = 2 * pi * frequency`, where `frequency` comes from the agent's action (action index 1)
- `frequency` is denormalized from [-1, 1] to the configured range (typically 0.5–3.0 Hz)
- So `omega` ranges from ~pi to ~6*pi rad/s
- `omega` is **not** a hyperparameter — it changes every RL step based on the policy output

The current `[sin(t), cos(t)]` encodes a single frequency cycle every ~6.28 seconds. The actual curvature oscillation at e.g. 2 Hz has `omega*t` cycling 2x per second — a completely different period.

To recover `sin(omega*t)` from `sin(t)`, `cos(t)`, and `omega`, the MLP would need to approximate the composition `sin(2*pi*f * t)` from `sin(t)`, `cos(t)`, and `f`. This is a deeply nonlinear relationship that is very difficult for an MLP to learn.

### Correct encoding

The time encoding should track **accumulated phase** rather than raw time. Since `omega` changes each RL step (it depends on the action), the accumulated phase is:

```
phi_accum = sum(omega_i * dt_rl)  over all past RL steps
```

not `omega_current * t_raw`. The encoding would then be `[sin(phi_accum), cos(phi_accum)]`.

This gives the model direct access to the oscillation phase without requiring it to internally compose trigonometric functions with varying frequency.

### Status

**Fixed.** All call sites now compute `omega*t` from the action's frequency and pass it to `encode_phase_batch`. See `logs/surrogate-phase-encoding-fix.md` for details. Previously trained checkpoints are incompatible and must be retrained (no data recollection needed).

## Elastic Perturbation Propagation

### Observation

PyElastica uses a 3-node stencil for internal forces/torques. Per substep, information propagates ±1 node. With 500 substeps and 21 nodes, a perturbation at node 0 reaches node 20 after just 20 substeps, then bounces back and forth ~25 times within a single RL step.

### Mechanism

When `rest_kappa[i]` is set, the mismatch between current and rest curvature creates an internal elastic bending torque (moment):

```
M[i] = EI * (kappa_current[i] - kappa_rest[i])
```

The propagation chain per substep:

1. Curvature mismatch at element `i` → elastic bending torque
2. Torque accelerates angular velocity `omega[i]` → changes orientation `theta[i]`
3. Changed orientation moves adjacent node positions
4. Moved nodes change strain/curvature at neighboring elements `i±1`
5. New mismatches at `i±1` → new torques
6. Repeat — elastic wave propagates ±1 element per substep

This is not an external force. It is the rod's own elastic restoring response propagating through internal coupling between adjacent elements.

### Impact on surrogate design: None

The rod equilibrates fully within ~20 substeps, and 500 are run. By the time the surrogate sees the output state, the elastic response has traversed the entire rod ~25 times. There is no residual propagation information lost between RL steps — the rod has settled into a quasi-steady state driven by the final curvature profile.

The surrogate only needs to predict the **endpoint** of those 500 substeps, not the intermediate dynamics. The full-rod coupling is already baked into the `(state, action, time) → next_state` mapping in the training data. The MLP learns the global input-output relationship directly — it does not need to simulate the propagation.

The perturbation propagation analysis explains *why* every node's final position depends on every other node (global coupling), which the MLP handles fine as a global function approximator. It does not create any additional feature requirements beyond what is already provided.
