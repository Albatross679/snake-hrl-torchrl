---
name: Control Methods Comparison
description: Comparison of absolute curvature (Elastica snake) vs delta curvature (Choi2025 manipulator) control methods
type: knowledge
created: 2026-03-19
updated: 2026-03-19
tags: [control, curvature, elastica, dismech, choi2025, cpg]
aliases: [curvature-control-comparison]
---

# Control Methods Comparison

This project implements two distinct curvature-based control methods for soft robots. Both operate on the principle of modifying a rod's natural/rest curvature so the physics engine generates elastic restoring forces that drive the rod toward the target shape. The methods differ in how actions map to curvature commands, whether state accumulates, and how spatial smoothing is handled.

## 1. Elastica Snake Robot — Absolute Curvature Control

**Domain:** Snake robot locomotion and predation (HRL with approach + coil policies)

**Physics backend:** PyElastica (Cosserat rod model)

**Source files:**
- `src/physics/elastica_snake_robot.py` — physics simulation
- `src/envs/base_env.py` — TorchRL environment
- `src/physics/cpg/action_wrapper.py` — optional CPG wrapper
- `src/physics/cpg/oscillators.py` — Matsuoka and Hopf oscillators
- `src/configs/env.py` — environment config (`action_scale`)
- `src/configs/geometry.py` — geometry config (`num_segments = 20`)

### Control Flow

1. The policy outputs a normalized action in [-1, 1] with dimension `num_segments - 1 = 19` (one value per internal joint for a 20-segment rod). See `base_env.py` line 63 and line 106.
2. The environment scales the action: `curvatures = action * action_scale * 5.0` (`base_env.py` line 198), where `action_scale` defaults to 1.0 (`env.py` line 104).
3. `ElasticaSnakeRobot.set_curvature_control()` clips to [-10, 10] rad/m (`elastica_snake_robot.py` line 364).
4. `_apply_curvature_to_elastica()` writes directly to `rest_kappa[0, i]` (the kappa1 component for planar bending) at each of the `n_elements - 1` bend locations (`elastica_snake_robot.py` lines 369-396).

### Key Properties

- **1:1 mapping** — Each action dimension maps directly to one joint's rest curvature. No spatial interpolation.
- **Stateless** — The policy outputs the desired shape at each timestep. There is no accumulation; the rest curvature is overwritten every step.
- **Action dim:** 19 (direct curvature) or 4-5 (with CPG wrapper).

### Optional CPG Wrapper

The CPG action wrapper (`action_wrapper.py`) reduces the action space from 19 to 4 dimensions (amplitude, frequency, wave_number, phase_offset) by generating a serpenoid curvature wave via coupled oscillators. A steering variant adds a 5th parameter (turn_bias) for directional control. The CPG outputs are interpolated from oscillator positions to the 19 joint positions (`oscillators.py` line 339).

## 2. Choi2025 Soft Manipulator — Delta Curvature Control

**Domain:** Soft manipulator reaching, tracking, and obstacle avoidance (single policy, SAC or PPO)

**Physics backend:** DisMech (Discrete Elastic Rods / DER), with a mock fallback

**Source files:**
- `papers/choi2025/control.py` — delta curvature controller with Voronoi smoothing
- `papers/choi2025/env.py` — TorchRL environment
- `papers/choi2025/config.py` — configuration (physics, control, training)

### Control Flow

1. The policy outputs a normalized action in [-1, 1] with dimension `num_control_points * 2 = 10` (5 control points x 2 curvature components for 3D bending). In 2D mode, action dim is 5. See `env.py` lines 185-187.
2. `DeltaCurvatureController.apply_delta()` clips each component to [-1, 1], then scales by `max_delta_curvature` (default 1.0 rad/m per step) (`control.py` lines 106, 116-117; `config.py` line 105).
3. The deltas at control points are interpolated to all bend springs via a Voronoi weight matrix: `delta_springs = W @ delta_cp` (`control.py` lines 112, 118-119).
4. The interpolated deltas are **accumulated** onto the internal `curvature_state` array (`control.py` lines 113, 118-119).
5. The accumulated curvature is written to the physics backend's `nat_strain` (`env.py` lines 401-407).

### Key Properties

- **Sparse control** — 5 control points map to 19 bend springs via Voronoi (piecewise linear) interpolation (`control.py` lines 41-92).
- **Stateful** — Curvature state accumulates deltas over time. The controller maintains internal memory that persists across steps and is only zeroed on `reset()` (`control.py` lines 123-125).
- **Action dim:** 10 (3D) or 5 (2D).
- **Temporal smoothing** — Delta control inherently limits the rate of curvature change per step, encouraging smooth trajectories.
- **Spatial smoothing** — Voronoi interpolation ensures a smooth curvature distribution along the rod even with sparse control inputs.

### Voronoi Weight Matrix

The weight matrix `W` of shape `(num_bend_springs, num_control_points)` is built by placing control points uniformly on [0, 1] and computing piecewise linear interpolation weights between each bend spring's parametric position and its two nearest control points (`control.py` lines 63-92).

## Key Differences

| Aspect | Elastica Snake | Choi2025 Manipulator |
|---|---|---|
| Physics | PyElastica (Cosserat rod) | DisMech (Discrete Elastic Rods) |
| Control type | Absolute curvature | Delta curvature (incremental) |
| Action dim | 19 (direct) or 4-5 (CPG) | 10 (3D) or 5 (2D) |
| Mapping | 1:1 action-to-joint | Voronoi interpolation (5 points to 19 springs) |
| State | Stateless (overwrite each step) | Stateful (accumulates deltas) |
| Curvature clipping | [-10, 10] rad/m absolute | +/- 1.0 rad/m per step (delta) |
| CPG option | Yes (Matsuoka / Hopf oscillators) | No |
| Spatial smoothing | None (direct) | Voronoi piecewise-linear interpolation |
| Temporal smoothing | None (instant setpoint) | Implicit from delta accumulation |
| Rod attachment | Free (locomotion on ground) | Clamped at one end |
| Tasks | Approach + coil (HRL hierarchy) | Reach, track, avoid obstacles (single policy) |
| RL algorithms | PPO | SAC or PPO |

## Why Different Methods?

**Elastica snake** needs full-body flexibility for diverse maneuvers (approaching prey along arbitrary paths, then coiling around it). Direct 19-dim absolute curvature commands give the HRL policies complete control over body shape at every instant, which is essential when the approach and coil sub-policies require qualitatively different postures.

**Choi2025 manipulator** uses sparse delta control for three reasons:
1. **Dimensionality reduction** — 5 control points instead of 19 joints improves sample efficiency for single-policy RL.
2. **Smooth trajectories** — Delta accumulation limits curvature rate-of-change per step, analogous to PD joint position control in rigid robots (the paper's stated motivation).
3. **Spatial coherence** — Voronoi interpolation ensures the rod profile is smooth between control points, preventing physically unrealistic high-frequency curvature patterns.

## Mathematical Formulation

### Shared Principle: Elastic Restoring Forces from Curvature

Both methods exploit the same physics: the bending energy of an elastic rod is

$$E_b = \frac{1}{2} K_b \sum_{i} \frac{(\kappa_i - \bar{\kappa}_i)^2}{V_i}$$

where $\kappa_i$ is the current geometric curvature at node $i$, $\bar{\kappa}_i$ is the natural (rest) curvature, $K_b$ is bending stiffness, and $V_i$ is the Voronoi length. The elastic force is the negative gradient $\mathbf{f}_i = -\nabla_{\mathbf{q}} E_b$, which drives the rod toward $\kappa_i = \bar{\kappa}_i$. Both control methods work by modifying $\bar{\kappa}_i$ — they never output forces directly.

### Elastica Snake: Absolute Curvature

At each timestep $t$, the policy outputs action $\mathbf{a}_t \in [-1, 1]^{19}$. The rest curvature is set directly:

$$\bar{\kappa}_i^{(t)} = \text{clip}\!\left(a_i^{(t)} \cdot s \cdot 5.0,\; -10,\; 10\right) \quad \text{for } i = 1, \ldots, 19$$

where $s$ is `action_scale` (default 1.0). The mapping is **memoryless** — $\bar{\kappa}^{(t)}$ depends only on $\mathbf{a}_t$, not on previous actions. Each action dimension maps 1:1 to a joint.

### Choi2025 Manipulator: Delta Curvature with Voronoi Interpolation

At each timestep $t$, the policy outputs action $\mathbf{a}_t \in [-1, 1]^{2C}$ (where $C = 5$ control points, factor of 2 for 3D bending components $\kappa_1, \kappa_2$). The control proceeds in three stages:

**Stage 1 — Scale to delta curvature at control points:**

$$\Delta \bar{\kappa}_{c}^{(t)} = \text{clip}(a_c^{(t)},\; -1,\; 1) \cdot \Delta_{\max} \quad \text{for } c = 1, \ldots, C$$

where $\Delta_{\max}$ is `max_delta_curvature` (default 1.0 rad/m per step).

**Stage 2 — Interpolate to all bend springs via Voronoi weight matrix:**

$$\Delta \bar{\kappa}_{\text{springs}}^{(t)} = \mathbf{W} \, \Delta \bar{\kappa}_{cp}^{(t)}$$

where $\mathbf{W} \in \mathbb{R}^{N_s \times C}$ is a piecewise-linear interpolation matrix. Each row of $\mathbf{W}$ has at most two nonzero entries that sum to 1, corresponding to the two nearest control points. For a bend spring at parametric position $u \in [0, 1]$ between control points $c_j$ (at $u_j$) and $c_{j+1}$ (at $u_{j+1}$):

$$W_{i,j} = \frac{u_{j+1} - u}{u_{j+1} - u_j}, \quad W_{i,j+1} = \frac{u - u_j}{u_{j+1} - u_j}$$

**Stage 3 — Accumulate into curvature state:**

$$\bar{\kappa}_i^{(t)} = \bar{\kappa}_i^{(t-1)} + \Delta \bar{\kappa}_{\text{springs},i}^{(t)}$$

This is the critical difference: the rest curvature is a **running sum** of all past deltas. The system has memory — $\bar{\kappa}^{(t)}$ depends on the entire action history $\{\mathbf{a}_0, \ldots, \mathbf{a}_t\}$. On episode reset, $\bar{\kappa}^{(0)} = \mathbf{0}$.

### Summary of Mathematical Differences

| | Elastica (absolute) | Choi2025 (delta) |
|---|---|---|
| **Update rule** | $\bar{\kappa}^{(t)} = f(\mathbf{a}_t)$ | $\bar{\kappa}^{(t)} = \bar{\kappa}^{(t-1)} + \mathbf{W} \cdot g(\mathbf{a}_t)$ |
| **Dependence** | Current action only | Full action history |
| **Rate limit** | None (can jump to any curvature) | $\|\Delta \bar{\kappa}\|_\infty \leq \Delta_{\max}$ per step |
| **Spatial resolution** | $N-1$ DOF (full) | $C$ DOF (sparse), interpolated to $N-1$ |
| **Dynamical analogy** | Position control | Velocity control (integrator) |

The last row captures the key insight: absolute curvature control is analogous to **position control** in rigid robots (set the joint angle directly), while delta curvature control is analogous to **velocity control** (command the joint velocity, let it integrate). The delta formulation naturally limits jerk and encourages smooth trajectories, at the cost of requiring more steps to reach a target configuration.
