---
name: Surrogate Model Input Representation Research
description: Research on optimal state representations for neural surrogate models of elastic rod dynamics — relative vs absolute, body-frame features, invariance, and feature engineering
type: knowledge
created: 2026-03-16
updated: 2026-03-16
tags:
  - surrogate-model
  - state-representation
  - invariance
  - feature-engineering
  - elastic-rod
  - neural-network
  - body-frame
aliases:
  - surrogate input features
  - body-frame representation
  - invariant dynamics representation
---

# Surrogate Model Input Representation Research

Research on optimal state representations for neural surrogate models of elastic rod dynamics. The central question: should the surrogate use absolute coordinates (current approach) or relative/body-frame features?

---

## Table of Contents

1. [Executive Summary and Recommendation](#1-executive-summary-and-recommendation)
2. [Current System Analysis](#2-current-system-analysis)
3. [What the Physics Tells Us](#3-what-the-physics-tells-us)
4. [Literature: Invariant Representations for Physics Surrogates](#4-literature-invariant-representations-for-physics-surrogates)
5. [Literature: Elastic Rod and Beam Representations](#5-literature-elastic-rod-and-beam-representations)
6. [Literature: Snake Robot RL Observations](#6-literature-snake-robot-rl-observations)
7. [Literature: Equivariant Approaches](#7-literature-equivariant-approaches)
8. [Proposed Body-Frame Representation](#8-proposed-body-frame-representation)
9. [Handling Velocities](#9-handling-velocities)
10. [Handling Angular Representation](#10-handling-angular-representation)
11. [Global Pose Tracking](#11-global-pose-tracking)
12. [Normalization Strategy](#12-normalization-strategy)
13. [Implementation Plan](#13-implementation-plan)
14. [Risk Assessment](#14-risk-assessment)
15. [Sources](#15-sources)

---

## 1. Executive Summary and Recommendation

**Primary recommendation:** Decompose the 124-dim absolute state into a **body-frame shape representation** (translation/rotation invariant) plus a small **global pose vector**, and train the surrogate on the body-frame features.

### Why

The current absolute representation has fundamental problems:

1. **Spurious correlations**: The same snake shape at position (0.5, 0.3) and (1.2, 0.8) produces different state vectors despite identical internal dynamics. The model must learn that dynamics are translation-invariant — this wastes capacity.

2. **Poor generalization**: Training data covers a limited region of absolute space. A snake that drifts to a new region of (x, y) space sees out-of-distribution inputs even though the body shape and dynamics are identical.

3. **Physics mismatch**: Internal forces depend on strains and curvatures (relative quantities), not absolute positions. External forces (gravity, RFT friction) depend on velocities and heading, not absolute position. Only the surrogate's *input* uses absolute coordinates — the physics never does.

4. **Confirmed by literature**: Every major physics surrogate framework (GNS, MeshGraphNets, beam constitutive models) uses relative/local features, not absolute coordinates.

### The Decomposition

```
Absolute state (124-dim)
    ├── Global pose (4-dim): CoM_x, CoM_y, heading, heading_rate
    ├── Body shape (82-dim): relative node positions in body frame
    │     ├── Δx_i, Δy_i relative to CoM, rotated into body frame (40-dim)
    │     ├── curvatures κ_i = Δψ_{i+1} - Δψ_i (19-dim)
    │     └── inter-element angles Δψ_i = ψ_i - ψ_head (19-dim + 1 redundant check)
    └── Body velocities (42-dim): velocities in body frame
          ├── v_x_i, v_y_i in body frame, relative to CoM velocity (40-dim)
          └── relative angular velocities (same as absolute, frame-independent in 2D)
```

The surrogate predicts **deltas in body-frame coordinates**. The global pose update (CoM translation, heading change) is predicted separately or derived from the body-frame prediction.

**Confidence: HIGH** — This recommendation is supported by multiple independent lines of evidence from the literature, from physics analysis, and from established RL practice.

---

## 2. Current System Analysis

### Current State Vector (124-dim)

| Slice | Variable | Count | Physical meaning | Frame |
|---|---|---|---|---|
| `[0:21]` | x_i | 21 | Node x-positions (m) | **World** |
| `[21:42]` | y_i | 21 | Node y-positions (m) | **World** |
| `[42:63]` | vx_i | 21 | Node x-velocities (m/s) | **World** |
| `[63:84]` | vy_i | 21 | Node y-velocities (m/s) | **World** |
| `[84:104]` | ψ_e | 20 | Element yaw angles (rad) | **World** |
| `[104:124]` | ω_z,e | 20 | Angular velocities (rad/s) | **Frame-independent** (2D) |

**Key observation**: 104 of 124 state dimensions are in the world frame. Only ω_z (20-dim) is already frame-independent. The model must implicitly learn translation and rotation invariance from data.

### Current Input (189-dim)

State (124) + action (5) + per-element phase encoding (60) = 189 total input dimensions.

### Current Prediction

Predicts normalized Δs (state delta) in absolute coordinates. Next state = current + denormalized delta.

---

## 3. What the Physics Tells Us

### Internal Forces Depend on Strains, NOT Positions

From the Cosserat rod constitutive relations (verified against `overview.tex` and `report/system-formulation.tex`):

```
Internal force:  n = S(ε - ε₀)     where ε = strain (relative)
Internal moment: m = B(κ - κ₀)     where κ = curvature (relative)
```

Curvature κ_i at element i depends only on the relative rotation between adjacent elements: `κ_i = (ψ_{i+1} - ψ_i) / ds`. This is a **local, relative quantity**. The absolute yaw ψ_i never appears in the constitutive law — only differences Δψ.

### External Forces

| Force | Depends on | Absolute position? |
|---|---|---|
| Gravity | Constant (2D, z-down) | **No** |
| RFT friction | Velocity direction relative to body tangent | **No** — depends on heading and velocity direction, not position |
| Viscous damping | Velocity magnitude | **No** |
| CPG muscle torque | κ - κ_target | **No** — depends on curvature difference |

**No physical force in this system depends on absolute (x, y) position.** The dynamics are exactly SE(2)-equivariant: if you translate and rotate the entire snake, the dynamics are identical.

### Formal Symmetry Analysis

The system is governed by:

```
ρA ẍ_i = F_internal(Δx, Δψ) + f_friction(v, tangent) + f_gravity
ρI ω̇_i = M_internal(Δψ) + m_cpg(κ - κ₀)
```

Under an SE(2) transformation g = (R, t):
- x → Rx + t
- v → Rv
- ψ → ψ + θ (where θ is the rotation angle)
- Δψ → Δψ (invariant — differences are preserved)
- κ → κ (invariant — curvature is a relative quantity)
- ω → ω (invariant in 2D — angular velocity about z is a scalar)

The equations of motion are exactly SE(2)-equivariant. **The surrogate should exploit this.**

---

## 4. Literature: Invariant Representations for Physics Surrogates

### 4.1 Graph Network Simulator (GNS) — Sanchez-Gonzalez et al. (ICML 2020)

**The foundational work on learned physics simulation.** Key representation choices (Confidence: HIGH — verified against paper):

**Node features** per particle i:
```
x_i = [p_i, ṗ_i^{t-C+1}, ..., ṗ_i^t, f_i]
```
where p_i is position, ṗ are C=5 previous velocities (from finite differences), and f_i is a material type one-hot encoding.

**Critical design choice — absolute position is MASKED OUT:**
> "The encoder masks out absolute position information of particles and instead encodes the relative displacement between particles into features, which empowers the model to have strong inductive biases for spatial invariance."

**Edge features** between particles i and j:
```
e_{i,j} = [(p_i - p_j), ||p_i - p_j||]
```
Only relative displacements and their magnitudes — no absolute coordinates.

**Output**: Per-particle acceleration (not position or velocity delta).

**Lesson for our surrogate**: Even though GNS uses a graph structure and we use an MLP, the **representation insight transfers**: use relative positions between nodes, not absolute coordinates. The MLP equivalent is to compute relative features as a preprocessing step.

### 4.2 MeshGraphNets — Pfaff et al. (ICLR 2021)

Extends GNS to mesh-based simulations (cloth, aerodynamics). Key representation: node features include velocity and node type, while edge features encode **relative mesh positions** and world-space displacements. Mesh connectivity defines the graph topology.

**Output**: Per-node acceleration, integrated via semi-implicit Euler.

**Lesson**: Even for fixed-topology meshes (like our rod), relative encodings are standard.

### 4.3 Neural Robot Dynamics (NeRD) — 2025

> "Enforces dynamics invariance under translation and rotation around the gravity axis, enhancing spatial generalizability and training efficiency."

This directly addresses our problem: a dynamics model that should be invariant to the robot's global position and heading. They achieve this through input canonicalization.

### 4.4 Residual Dynamics Learning (UZH RPG, 2025)

Body rates (ωx, ωy, ωz) are expressed in the **body frame**, and the model predicts **residual acceleration** — the difference between observed and nominal dynamics. Key insight: representing angular quantities in the body frame avoids the discontinuity and coordinate-dependence issues of world-frame angles.

---

## 5. Literature: Elastic Rod and Beam Representations

### 5.1 Physics-Augmented Neural Networks for Beams (Gebhardt et al., CMAME 2024)

**Directly relevant** — a neural network constitutive model for geometrically exact (Cosserat) beams. (Confidence: HIGH — verified against paper.)

**Input representation**: 6-dim strain vector p = (ε, κ) ∈ R^6
- ε₁, ε₂: shear strain components
- ε₃: axial stretch
- κ₁, κ₂: bending curvatures
- κ₃: twist

**All inputs are in the material (body) frame.** The paper states:
> "The strain measures ε and κ only depend on the arc-length parameter s [...] objectivity is guaranteed by the definition of the strain measures."

**Output**: Beam strain energy potential ψ, from which forces n = ∂ψ/∂ε and moments m = ∂ψ/∂κ are derived via automatic differentiation.

**Lesson**: For Cosserat rod constitutive modeling, the standard input is strain/curvature in the body frame, not absolute positions. Objectivity (frame-indifference) is achieved by construction through the choice of input variables.

### 5.2 DNN Surrogate for Thin-Walled Rod Members (Computations, 2025)

Uses rod degrees of freedom (generalized strains) as neural network inputs. The surrogate replaces the full 3D solid FE computation with a learned mapping from generalized strains to stress resultants. Again, **all in the material frame**.

### 5.3 GNN Surrogates for Contacting Deformable Bodies (2025)

Uses graph neural networks with node features including velocity and material properties, edge features including relative positions. Handles contact detection for soft body interactions.

---

## 6. Literature: Snake Robot RL Observations

### 6.1 Jiang et al. (2024) — Hierarchical RL for COBRA Snake Navigation

(Confidence: HIGH — verified against project's own implementation in `papers/jiang2024/env_jiang2024.py`)

**Observation space (21-dim):**
- Joint positions (11-dim): **proprioceptive, body-relative**
- IMU / gyro readings (3-dim): **body-frame accelerations**
- Displacement to waypoint (3-dim): **relative, in robot frame**
- Relative rotation to waypoint (4-dim): **axis-angle, relative**

**All observations are ego-centric.** No absolute world coordinates. The paper explicitly states this enables zero-shot transfer to new environments.

### 6.2 Liu et al. (2023) — CPG-Regulated Soft Snake Locomotion

(Confidence: HIGH — verified against `papers/liu2023/env_liu2023.py`)

**Observation space (8-dim):**
- Distance to goal: **relative scalar**
- Velocity toward goal: **relative scalar (body-frame projection)**
- Heading error: **relative angle**
- Angular velocity: **frame-independent**
- Joint curvatures (4-dim): **local body-shape quantities**

Again, **no absolute coordinates**. All features are either body-relative or intrinsic body-shape measurements.

### 6.3 MuJoCo Standard Environments (Gymnasium)

(Confidence: HIGH — verified against Gymnasium documentation)

Standard practice in MuJoCo locomotion environments:
> "By default, the observation does not include the x- and y-coordinates of the torso."

The Ant, Humanoid, Walker, HalfCheetah environments all **exclude global position** from observations. Joint angles (qpos excluding root) and joint velocities (qvel) form the core observation. Root angular velocities are in the **local body frame**.

### 6.4 DeepMind Control Suite

Proprioceptive observations for locomotion use body-centric representations. Global position is excluded. The design principle: policies should be position-invariant for locomotion.

### 6.5 Summary of RL Observation Practices

| System | Absolute position in obs? | Body-frame features? |
|---|---|---|
| Jiang 2024 (COBRA snake) | No | Yes — joint pos, IMU, relative goal |
| Liu 2023 (soft snake) | No | Yes — heading error, curvatures, relative distance |
| MuJoCo Ant/Humanoid | No (excluded by default) | Yes — joint angles, body-frame velocities |
| DeepMind Control Suite | No | Yes — proprioception |
| GNS (Sanchez-Gonzalez 2020) | Masked out | Yes — relative displacements |

**Universal pattern**: Absolute world-frame positions are either excluded or masked. Body-relative features are universal.

---

## 7. Literature: Equivariant Approaches

### 7.1 Canonicalization vs. Special Architecture

There are two paths to equivariance:

**Path A: Equivariant architecture** (SEGNN, E(n)-GNN, etc.) — constrain the network layers to preserve symmetry. Powerful but requires custom architecture changes.

**Path B: Input canonicalization** — transform inputs into a canonical frame before feeding to a standard network. Any MLP becomes equivariant.

**Kaba et al. (ICML 2023)** prove that an MLP with learned canonicalization is a **universal approximator** for equivariant functions:
> "A G-equivariant parameterized function written with a G-equivariant continuous canonicalization function and a multilayer perceptron as a prediction function is a universal approximator."

**For our case**: We don't need to learn canonicalization — we can **analytically canonicalize** by transforming to body frame. This is simpler, exact, and has zero overhead.

### 7.2 SE(2) Equivariance by Input Design

Kim et al. (CoRL 2023) decompose object pose into:
1. Pose projected to the surface (global pose)
2. Relative rigid-body transformation (shape)

This decomposition enables SE(2)-equivariant dynamics learning with a standard network. Our proposed decomposition follows the same principle.

### 7.3 Why We Don't Need Special Architecture

For our fixed-topology, 2D planar problem:
- The symmetry group is SE(2) = translations + rotations in the plane
- We can analytically compute the body frame (CoM + heading)
- Body-frame features are automatically SE(2)-invariant
- A standard MLP on body-frame features is therefore SE(2)-equivariant by construction

No E(n)-GNN, no SEGNN, no learned canonicalization needed. **The representation does the work.**

---

## 8. Proposed Body-Frame Representation

### 8.1 Body Frame Definition

Define the body frame from the current absolute state:

```python
# Center of mass
com_x = mean(x_nodes)          # scalar
com_y = mean(y_nodes)          # scalar

# Heading: average tangent direction of first few elements
# (or equivalently, the angle of the line from tail CoM to head CoM)
heading = mean(psi_elements)   # scalar, in radians

# Rotation matrix for body frame
R = [[cos(heading), sin(heading)],
     [-sin(heading), cos(heading)]]
```

Alternative heading definitions (in order of robustness):
1. **Mean element yaw** `mean(ψ_e)` — smooth, well-defined, uses all elements
2. **Head-to-tail direction** `atan2(y_head - y_tail, x_head - x_tail)` — intuitive but noisy
3. **First element yaw** `ψ_0` — simple but sensitive to head deformation
4. **CoM velocity direction** `atan2(vy_com, vx_com)` — undefined at zero velocity

**Use option 1: mean element yaw.** It is smooth, uses all elements, and aligns with the "average body direction."

### 8.2 Body-Frame Shape Features

**Relative node positions (40-dim):**
```python
# Relative positions in body frame
dx_i = x_i - com_x    # 21 values
dy_i = y_i - com_y    # 21 values

# Rotate into body frame
body_x_i = R[0,0] * dx_i + R[0,1] * dy_i    # 21 values
body_y_i = R[1,0] * dx_i + R[1,1] * dy_i    # 21 values
```

But: 21 + 21 = 42 positions, minus 2 for CoM constraint (body_x and body_y of CoM are zero by construction). Effectively 40 independent dimensions.

In practice, include all 42 and let the network handle the redundancy — removing 2 dims saves negligible computation but complicates the code.

**Relative angles (20-dim):**
```python
# Element angles relative to heading
dpsi_e = psi_e - heading    # 20 values, in [-pi, pi]
```

These are the element orientations in the body frame. They encode the body shape completely — a straight snake has all dpsi_e = 0, a curved snake has nonzero values.

**Alternative: curvatures (19-dim):**
```python
# Discrete curvatures (angle differences between adjacent elements)
kappa_i = (psi_{i+1} - psi_i) / ds    # 19 values
```

Curvatures are **more invariant** than relative angles (they don't depend on heading at all), but they lose one dimension. Use both: relative angles for full shape reconstruction, curvatures as supplementary features that directly appear in the constitutive law.

### 8.3 Body-Frame Velocity Features

**Node velocities in body frame (42-dim):**
```python
# Subtract CoM velocity
vx_rel_i = vx_i - vx_com    # 21 values
vy_rel_i = vy_i - vy_com    # 21 values

# Rotate into body frame
body_vx_i = R[0,0] * vx_rel_i + R[0,1] * vy_rel_i
body_vy_i = R[1,0] * vx_rel_i + R[1,1] * vy_rel_i
```

Again 42 dims, but body_vx_com = body_vy_com = 0 by construction.

**Angular velocities (20-dim):**
```python
omega_z_e    # 20 values — already frame-independent in 2D
```

Angular velocity about z in 2D is a scalar that does not change under SE(2) transformation. Use as-is.

### 8.4 Global Pose Features (4-dim)

These are needed to reconstruct absolute state but do NOT enter the shape dynamics:

```python
global_pose = [com_x, com_y, heading, heading_rate]
```

where `heading_rate = mean(omega_z_e)` is the average angular velocity.

### 8.5 Complete Feature Vector

**Surrogate input for shape dynamics:**

| Group | Features | Count | Notes |
|---|---|---|---|
| Body-frame positions | body_x_i, body_y_i | 42 | Relative to CoM, rotated |
| Body-frame angles | dpsi_e = ψ_e - heading | 20 | Element orientation in body frame |
| Body-frame velocities | body_vx_i, body_vy_i | 42 | Relative to CoM vel, rotated |
| Angular velocities | ω_z,e | 20 | Already frame-independent |
| **Body shape total** | | **124** | Same dim as original |
| Action | a | 5 | CPG parameters |
| Phase encoding | per-element phase | 60 | sin/cos/kappa per element |
| **Total input** | | **189** | Same as current |

**Key change**: The 124 state dimensions now encode **body shape** rather than **world-frame configuration**. The total input dimension is unchanged at 189.

**Global pose (separate, 4-dim):**

| Feature | Count |
|---|---|
| CoM_x, CoM_y | 2 |
| heading (mean ψ) | 1 |
| heading_rate (mean ω_z) | 1 |
| **Total** | **4** |

### 8.6 Prediction Targets

**Option A (recommended): Predict body-frame deltas**

The surrogate predicts Δ(body-frame state), then the absolute next state is reconstructed:
1. Predict body-frame shape delta → get next body-frame shape
2. Predict global pose delta (Δcom_x, Δcom_y, Δheading) → update global pose
3. Un-rotate and un-translate to get absolute next state

This keeps the main prediction (shape dynamics) fully translation/rotation invariant.

**Option B: Predict body-frame next state directly**

Same as Option A but predict next body-frame state instead of delta. Less common, and delta prediction has shown advantages (see current system's residual prediction approach).

---

## 9. Handling Velocities

### Should Velocities Be in Body Frame or World Frame?

**Use body frame.** Reasoning:

1. **RFT friction** depends on the velocity direction relative to the body tangent. Body-frame velocities encode this directly — a forward-moving node has positive body_vx regardless of heading.

2. **Damping** depends on velocity magnitude, which is frame-independent.

3. **Generalization**: A snake moving north at 0.1 m/s and a snake moving east at 0.1 m/s should produce identical body-frame dynamics. In world-frame, these are different velocity vectors. In body-frame, they are identical.

4. **GNS precedent**: Sanchez-Gonzalez et al. use velocity histories (relative quantities) as node features, not absolute velocities.

### CoM Velocity

The CoM velocity is part of the global pose, not the shape dynamics. Subtracting it before rotation ensures body-frame velocities encode only deformation velocities — how the body is changing shape, not how it is translating.

The CoM velocity affects the RFT friction (via absolute velocity magnitude and direction). Include CoM speed `||v_com||` and CoM velocity angle relative to heading `atan2(vy_com_body, vx_com_body)` as additional context features (2-dim) if needed. In practice, these can be derived from the global pose prediction.

---

## 10. Handling Angular Representation

### Absolute Yaw (ψ) vs Relative Angles (Δψ)

The current representation uses absolute yaw ψ_e per element. Problems:

1. **Wrapping**: ψ is periodic with period 2π. A snake at heading π-ε and heading π+ε has nearly identical configurations but very different ψ values. The z-score normalization does not account for this circularity.

2. **Redundancy with position**: Given node positions, the tangent directions (and hence yaw) can be computed. ψ_e = atan2(y_{e+1} - y_e, x_{e+1} - x_e).

3. **Not invariant**: ψ changes under rotation.

**Proposed**: Use relative angles Δψ_e = ψ_e - heading. These:
- Are rotation-invariant (Δψ doesn't change when heading changes)
- Have smaller magnitude (centered around 0 for nearly-straight configurations)
- Avoid the wrapping problem for body-frame angles (if the snake doesn't curl more than ±π from the heading, which is typical for locomotion)

**Additionally**: Include discrete curvatures κ_i = (Δψ_{i+1} - Δψ_i) / ds as supplementary features. These are **doubly invariant** (independent of both heading and overall body curvature) and directly appear in the constitutive law.

### Encoding Angles

For any angular quantity θ, consider encoding as (sin θ, cos θ) to avoid wrapping issues. For Δψ_e values that stay within [-π/2, π/2] (typical for locomotion), raw values are fine. For curvatures, raw values are always fine (no periodicity issue).

---

## 11. Global Pose Tracking

### The Global Pose Update Problem

The body-frame decomposition creates a secondary prediction task: predicting how the global pose changes.

**Approach: Predict global pose deltas**

```python
# Predict from body-frame features:
delta_com_body = model_global(body_shape, action, phase)  # (2,) in body frame
delta_heading = model_heading(body_shape, action, phase)  # scalar

# Convert to world frame:
delta_com_world = R_inv @ delta_com_body
```

**Why this works**: The CoM displacement in the body frame depends on the gait pattern (body shape + action), not on absolute position. A snake performing lateral undulation moves forward (in its body frame) by a predictable amount given its curvature wave.

**Implementation options**:

1. **Shared model**: The same surrogate predicts both body-frame shape deltas and global pose deltas as additional output dimensions. Simplest approach.

2. **Separate head**: A small separate network predicts global pose deltas from the body-frame shape + action. Allows different learning dynamics.

3. **Derived from shape**: For small dt, CoM displacement can be estimated from the velocity field. If the surrogate accurately predicts body-frame velocity changes, the global pose update can be derived analytically.

**Recommendation**: Use option 1 (shared model) initially. Add global pose deltas as 3 extra output dimensions (Δcom_body_x, Δcom_body_y, Δheading). Total output: 124 (shape delta) + 3 (pose delta) = 127.

---

## 12. Normalization Strategy

### Per-Feature Z-Score (Current)

The current StateNormalizer applies per-feature z-score: `(x - μ) / σ`. This is appropriate and should be retained for the body-frame features.

**Key difference**: Body-frame features will have **tighter, more consistent distributions** than absolute features. Body-frame node positions are bounded by the rod length (~0.5m), whereas absolute positions grow without bound as the snake moves. This alone should improve normalization quality.

### Feature Group Scaling

| Feature group | Expected scale | Normalization |
|---|---|---|
| Body-frame positions | ±0.25 m (half rod length) | z-score |
| Body-frame angles Δψ | ±1.5 rad | z-score |
| Curvatures κ | ±10 rad/m | z-score |
| Body-frame velocities | ±0.2 m/s | z-score |
| Angular velocities ω_z | ±10 rad/s | z-score |
| Action | [-1, 1] | None (already normalized) |
| Phase encoding | [-1, 1] (sin/cos), [-5, 5] (kappa) | z-score |

### Delta Normalization

Deltas in body-frame will also have tighter distributions. The body shape changes slowly relative to the body frame — shape deltas are small. This is favorable for residual prediction (the model predicts a small perturbation around zero).

---

## 13. Implementation Plan

### Step 1: Feature Engineering Functions

```python
def absolute_to_body_frame(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 124-dim absolute state to body-frame features + global pose.

    Args:
        state: (..., 124) absolute state tensor

    Returns:
        body_state: (..., 124) body-frame features
        global_pose: (..., 4) [com_x, com_y, heading, heading_rate]
    """
    # Extract components
    pos_x = state[..., 0:21]       # (21,)
    pos_y = state[..., 21:42]      # (21,)
    vel_x = state[..., 42:63]      # (21,)
    vel_y = state[..., 63:84]      # (21,)
    yaw = state[..., 84:104]       # (20,)
    omega_z = state[..., 104:124]  # (20,)

    # Global pose
    com_x = pos_x.mean(dim=-1, keepdim=True)
    com_y = pos_y.mean(dim=-1, keepdim=True)
    heading = yaw.mean(dim=-1, keepdim=True)
    heading_rate = omega_z.mean(dim=-1, keepdim=True)

    # Body-frame rotation
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    # Relative positions
    dx = pos_x - com_x
    dy = pos_y - com_y
    body_x = cos_h * dx + sin_h * dy
    body_y = -sin_h * dx + cos_h * dy

    # Relative velocities
    vx_com = vel_x.mean(dim=-1, keepdim=True)
    vy_com = vel_y.mean(dim=-1, keepdim=True)
    dvx = vel_x - vx_com
    dvy = vel_y - vy_com
    body_vx = cos_h * dvx + sin_h * dvy
    body_vy = -sin_h * dvx + cos_h * dvy

    # Relative angles
    dpsi = yaw - heading

    # Pack body-frame state (same layout as absolute for compatibility)
    body_state = torch.cat([body_x, body_y, body_vx, body_vy, dpsi, omega_z], dim=-1)
    global_pose = torch.cat([com_x, com_y, heading, heading_rate], dim=-1)

    return body_state, global_pose


def body_frame_to_absolute(
    body_state: torch.Tensor,
    global_pose: torch.Tensor
) -> torch.Tensor:
    """Reconstruct absolute state from body-frame + global pose.

    Inverse of absolute_to_body_frame.
    """
    com_x = global_pose[..., 0:1]
    com_y = global_pose[..., 1:2]
    heading = global_pose[..., 2:3]
    # heading_rate = global_pose[..., 3:4]  # not needed for reconstruction

    body_x = body_state[..., 0:21]
    body_y = body_state[..., 21:42]
    body_vx = body_state[..., 42:63]
    body_vy = body_state[..., 63:84]
    dpsi = body_state[..., 84:104]
    omega_z = body_state[..., 104:124]

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    # Inverse rotation + translation
    pos_x = cos_h * body_x - sin_h * body_y + com_x
    pos_y = sin_h * body_x + cos_h * body_y + com_y

    # Need CoM velocity for reconstruction (derive from body_vx, body_vy means)
    # body_vx.mean ≈ 0 by construction, but store vx_com in global_pose if needed
    vel_x = cos_h * body_vx - sin_h * body_vy  # + vx_com
    vel_y = sin_h * body_vx + cos_h * body_vy  # + vy_com

    yaw = dpsi + heading

    return torch.cat([pos_x, pos_y, vel_x, vel_y, yaw, omega_z], dim=-1)
```

### Step 2: Dataset Preprocessing

Convert existing dataset on-the-fly during training:
```python
# In dataset __getitem__ or collate_fn:
body_state, global_pose = absolute_to_body_frame(state)
body_next, global_next = absolute_to_body_frame(next_state)

body_delta = body_next - body_state
pose_delta_body = compute_pose_delta_in_body_frame(global_pose, global_next)

input_features = [normalize(body_state), action, phase_encoding]
target = normalize_delta(body_delta)
```

**No data recollection needed** — the transformation is applied at training time from existing absolute-state data.

### Step 3: Model Modification

Minimal changes to the MLP:
- Input dim: 189 (unchanged if we keep same body-frame state layout)
- Output dim: 124 + 3 = 127 (add global pose delta)
- Or keep 124 output and derive pose delta from body-frame velocity prediction

### Step 4: Inference-Time Pipeline

```
1. absolute_to_body_frame(s_t) → body_state_t, global_pose_t
2. model(body_state_t, action, phase) → body_delta, pose_delta_body
3. body_state_{t+1} = body_state_t + denormalize(body_delta)
4. global_pose_{t+1} = update_global_pose(global_pose_t, pose_delta_body)
5. body_frame_to_absolute(body_state_{t+1}, global_pose_{t+1}) → s_{t+1}
```

---

## 14. Risk Assessment

### Risks of Switching to Body-Frame

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Body-frame deltas are harder to predict than absolute deltas | Low | Low | Physics analysis shows body-frame dynamics are simpler (invariant), not harder |
| Heading computation is noisy for highly curved configurations | Medium | Low | Mean element yaw is robust; only fails for >π total curvature which is rare in locomotion |
| Round-trip conversion introduces numerical error | Low | Medium | Use float64 for conversion, float32 for model. Error is ~1e-7, negligible |
| Global pose prediction is inaccurate | Medium | Medium | Global pose evolves smoothly; even a simple model should work. Fall back to deriving from body-frame velocity if needed |
| CoM velocity reconstruction for absolute state | Medium | Low | Store CoM velocity in extended global pose (6-dim instead of 4-dim) |

### Expected Benefits

1. **Better generalization**: Training data from any region of space generalizes everywhere
2. **Smaller effective state space**: Body shapes occupy a much smaller volume than absolute configurations
3. **Tighter normalization**: Body-frame positions bounded by rod length, not workspace size
4. **Physics-aligned features**: Curvatures and strains are what the constitutive law sees
5. **Potentially better omega_z prediction**: Body-frame velocities may be easier to predict since they reflect intrinsic deformation rather than mixed translation+deformation

### What Could Go Wrong

The main risk is that the body-frame representation makes some aspect of the dynamics harder to capture. Specifically, the **friction force** depends on the absolute velocity direction (not just body-frame). However, the friction depends on the velocity relative to the body tangent direction, which is exactly what body-frame velocities encode. So this should actually be *easier*, not harder, in body frame.

---

## 15. Sources

### Primary (HIGH confidence)

- **Sanchez-Gonzalez, Godwin, Pfaff et al. (2020)** "Learning to Simulate Complex Physics with Graph Networks." ICML 2020. — GNS architecture, relative displacement encoding, absolute position masking.
  - [arXiv](https://arxiv.org/abs/2002.09405)
  - [ICML proceedings](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf)

- **Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia (2021)** "Learning Mesh-Based Simulation with Graph Networks." ICLR 2021. — MeshGraphNets, relative mesh features.
  - [arXiv](https://arxiv.org/abs/2010.03409)
  - [OpenReview](https://openreview.net/forum?id=roNqYL0_XP)

- **Gebhardt et al. (2024)** "Physics-augmented neural networks for constitutive modeling of hyperelastic geometrically exact beams." CMAME 2024. — Strain/curvature inputs in material frame for beam neural constitutive models.
  - [arXiv](https://arxiv.org/html/2407.00640v1)
  - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782524008466)

- **Kaba et al. (ICML 2023)** "Equivariance with Learned Canonicalization Functions." — Proves MLP + canonicalization is a universal approximator for equivariant functions.
  - [arXiv](https://arxiv.org/abs/2211.06489)
  - [ICML proceedings](https://proceedings.mlr.press/v202/kaba23a/kaba23a.pdf)

- **MuJoCo / Gymnasium documentation** — Standard RL observation spaces exclude global position for locomotion.
  - [Gymnasium MuJoCo envs](https://gymnasium.farama.org/environments/mujoco/)

### Secondary (MEDIUM confidence)

- **Kim et al. (CoRL 2023)** "SE(2)-Equivariant Pushing Dynamics Models for Tabletop Object Manipulation." — Pose decomposition for SE(2)-equivariant dynamics.
  - [Proceedings](https://proceedings.mlr.press/v205/kim23b/kim23b.pdf)

- **Jiang et al. (2024)** "Hierarchical RL-Guided Large-scale Navigation of a Snake Robot." — COBRA snake robot ego-centric observations.
  - [arXiv](https://arxiv.org/html/2312.03223)
  - [Paper PDF](https://www.ccs.neu.edu/home/lsw/papers/aim2024-snake.pdf)

- **Liu et al. (2023)** "Reinforcement Learning of CPG-regulated Locomotion Controller for a Soft Snake Robot." — Body-relative observations for snake RL.
  - [arXiv](https://arxiv.org/html/2207.04899)

- **Naughton et al. (2021)** "Elastica: A Compliant Mechanics Environment for Soft Robotic Control." — PyElastica RL interface, observation design for Cosserat rods.
  - [GitHub](https://github.com/GazzolaLab/Elastica-RL-control)
  - [IEEE](https://ieeexplore.ieee.org/document/9369003/)

### Project-Internal (HIGH confidence)

- `knowledge/surrogate-mathematical-formulation.md` — Current 124-dim state layout, action space
- `knowledge/surrogate-architecture-comparison.md` — MLP architecture justification, 500-substep global coupling
- `knowledge/surrogate-time-encoding-and-elastic-propagation.md` — Phase encoding, elastic propagation
- `issues/surrogate-omega-z-poor-prediction.md` — omega_z R²=0.23, per-element phase encoding fix
- `issues/surrogate-spatial-structure-analysis.md` — 500-substep global coupling, MLP adequacy
- `src/observations/virtual_chassis.py` — Existing body-frame feature extractors (CoG, orientation)
- `papers/aprx_model_elastica/state.py` — Current RodState2D packing, named slices
- `report/system-formulation.tex` — Formal Cosserat rod governing equations
