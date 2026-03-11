---
name: control-and-actuation
description: How controllers and actuators work across all physics backends and control methods
type: knowledge
created: 2026-02-16T00:00:00
updated: 2026-02-16T00:00:00
tags: [knowledge, control, actuation, physics, mujoco, dismech, elastica, cpg]
aliases: []
---

# Control and Actuation

## Overview

The RL policy never outputs torques directly. Every control path produces a **19-dim target curvature vector** `κ[0..18]`, clipped to `[-10, 10] rad/m`. Each physics backend then internally converts that target into forces/torques that drive the rod toward the desired shape.

```
RL Policy (normalized action [-1, 1])
    │
    ├─ DIRECT ──────────── scale → 19 curvatures
    ├─ CPG ─────────────── 4 params → oscillator network → 19 curvatures
    ├─ SERPENOID ───────── 4 params → A·sin(k·s − ωt + φ) → 19 curvatures
    └─ SERPENOID_STEERING  5 params → A·sin(k·s − ωt + φ) + κ_turn → 19 curvatures
                │
                ▼
      set_curvature_control(κ)   ← unified interface, all backends
                │
                ▼
      Backend-specific force computation → rod deforms toward target
```

## Control Methods

Defined in `src/configs/env.py` as `ControlMethod` enum.

### DIRECT (19-dim)

Each action dimension maps to one joint's target curvature. Maximum flexibility, no coordination.

```python
curvatures = action * action_scale * 5.0   # src/envs/base_env.py
```

### CPG (4-dim)

Four gait parameters drive a network of coupled neural oscillators (Hopf or Matsuoka). The oscillator outputs are interpolated to 19 joints.

| Parameter    | Range        |
|--------------|--------------|
| amplitude    | [0.0, 2.0]   |
| frequency    | [0.5, 3.0] Hz |
| wave_number  | [0.5, 3.0]   |
| phase_offset | [0, 2π]      |

Implementation: `src/cpg/oscillators.py` (oscillator dynamics), `src/cpg/action_wrapper.py` (parameter mapping).

### SERPENOID (4-dim)

Analytical traveling-wave formula, same 4 parameters as CPG:

```
κ(s, t) = A · sin(k · s − ω · t + φ)
```

where `s ∈ [0, 1]` is normalized position along the body. No steering capability.

### SERPENOID_STEERING (5-dim)

Adds a constant curvature offset for turning:

```
κ(s, t) = A · sin(k · s − ω · t + φ) + κ_turn
```

| Parameter  | Range          | Effect                         |
|------------|----------------|--------------------------------|
| turn_bias  | [-2.0, 2.0]   | κ_turn > 0: left, < 0: right  |

Turn radius ≈ `1 / |κ_turn|`. This is the only control method that can steer.

## Actuation: How Backends Apply Target Curvature

All backends accept the same `set_curvature_control(κ)` call. None apply curvature instantaneously — all compute internal forces/torques that drive the rod toward the target over time.

### MuJoCo — PD Position Actuators

**Mechanism:** Explicit PD torque controller on rigid-body hinge joints.

```python
# mujoco_snake_robot.py, step()
target_angles = target_curvatures * segment_length      # κ → θ
self._data.ctrl[aid] = target_angles[i]                 # set actuator target
```

```xml
<!-- MJCF definition -->
<position name="a_{i}" joint="j_{i}" kp="50.0" ctrlrange="-3.14 3.14"/>
<joint name="j_{i}" type="hinge" axis="0 0 1" damping="0.1"/>
```

MuJoCo internally computes:

```
τ = kp · (θ_target − θ_current) − kd · θ̇
```

The policy sets a **target position**; a low-level PD controller converts it to **torque** every substep.

### DisMech (Python) — Elastic Energy Gradient

**Mechanism:** Sets the rod's natural (rest) curvature. Internal elastic forces arise from the energy gradient.

```python
# snake_robot.py, _apply_curvature_to_dismech()
bend_springs.nat_strain[i, 0] = target_curvatures[i]
```

The implicit Euler solver then computes:

1. Strain difference: `Δκ = κ_current − κ_rest`
2. Elastic energy: `E = ½ · EI · Δκ²`  (`EI` = bending stiffness)
3. Generalized force: `F = −∂E/∂q`  (gradient of energy w.r.t. nodal DOFs)
4. Stiffness matrix: `J = −∂²E/∂q²`  (Hessian for Newton solver)
5. Newton iteration: solve `J · Δq = F` until convergence

The force `F` is the elastic bending moment mapped to nodal forces — it IS a torque, computed from continuum mechanics rather than an explicit PD formula.

**Key source files:**
- `dismech-python/src/dismech/elastics/bend_energy.py` — bending strain, gradient, Hessian
- `dismech-python/src/dismech/elastics/elastic_energy.py:74-117` — energy → force → Jacobian
- `dismech-python/src/dismech/time_steppers/time_stepper.py:112-203` — Newton solve loop

### PyElastica — Cosserat Rod Elasticity

**Mechanism:** Same principle as DisMech, different mathematical formulation (Cosserat rod theory).

```python
# elastica_snake_robot.py, _apply_curvature_to_elastica()
self._rod.rest_kappa[0, i] = target_curvatures[i]    # κ1 (normal curvature)
self._rod.rest_kappa[1, i] = 0.0                      # κ2 (binormal, unused)
```

PyElastica uses a symplectic integrator (PositionVerlet or PEFRL) with explicit substeps (default 50 per RL step). The rod's internal elastic forces drive it toward `rest_kappa`.

### dismech-rods (C++) — Curvature Boundary Conditions

**Mechanism:** Same DER physics as DisMech, implemented in C++ for speed.

```python
# dismech_rods_snake_robot.py, step()
curvature_bc = np.zeros((n_joints, 4))
curvature_bc[:, 0] = 0                             # limb index
curvature_bc[:, 1] = np.arange(1, n_joints + 1)    # edge index (1-based)
curvature_bc[:, 2] = target_curvatures              # cx (planar)
curvature_bc[:, 3] = 0.0                            # cy (out-of-plane)

self._sim_manager.step_simulation({"curvature": curvature_bc})
```

## Comparison

| | MuJoCo | DisMech / dismech-rods | PyElastica |
|---|---|---|---|
| **Physics model** | Rigid-body chain + hinge joints | Discrete elastic rod (DER) | Cosserat rod |
| **What you set** | Target joint angle | Rest curvature (`nat_strain`) | Rest curvature (`rest_kappa`) |
| **Error signal** | `θ_target − θ_current` | `κ_current − κ_rest` | `κ_current − κ_rest` |
| **Force law** | `τ = kp·Δθ − kd·θ̇` (PD controller) | `F = −∂/∂q [½·EI·Δκ²]` (energy gradient) | `F = −∂/∂q [½·EI·Δκ²]` (energy gradient) |
| **Stiffness param** | `kp` (config: 50.0) | `EI` (Young's modulus × area moment) | `EI` (material property) |
| **Damping** | Joint damping `kd` (config: 0.1) | Implicit Euler numerical damping + optional `DampingForce` | Explicit damping term |
| **Solver** | Explicit substeps (`mj_step`) | Implicit Newton iteration | Explicit symplectic (PositionVerlet / PEFRL) |
| **Are there torques?** | Yes — explicit PD formula | Yes — from elastic energy gradient | Yes — from elastic energy gradient |

### Key Insight

**No backend applies curvature directly.** In all cases:

1. A target curvature creates a **force/torque** proportional to `(current − target)` × stiffness
2. The rod deforms dynamically under those forces, subject to inertia, damping, gravity, contact, and friction
3. The rod does NOT instantly snap to the target — it approaches it over time

The difference is only in *how* the torque is computed:
- **MuJoCo**: Explicit PD formula — simple, fast, easy to tune
- **Elastic rod backends**: Energy gradient from continuum mechanics — physically rigorous, captures large deformations and nonlinear elasticity
