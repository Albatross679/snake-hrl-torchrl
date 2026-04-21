---
name: Symmetry Analysis for ODE/PDE State Simplification
description: Formal procedure for identifying which state variables in an ODE/PDE system can be replaced by relative quantities without loss of representational power, applied to Cosserat rod dynamics
type: knowledge
created: 2026-03-16
updated: 2026-03-16
tags:
  - ode
  - pde
  - symmetry
  - invariance
  - state-representation
  - cosserat-rod
  - surrogate-model
  - noether
aliases:
  - state simplification
  - relative vs absolute state
  - symmetry-based feature reduction
  - ODE invariance analysis
---

# Symmetry Analysis for ODE/PDE State Simplification

When building a neural surrogate for an ODE/PDE system, the choice of state representation determines generalization, normalization stability, and sample efficiency. This document formalizes the procedure for identifying which state variables can be replaced by relative quantities **without loss of representational power**.

---

## 1. Core Question

Given an ODE system $\dot{\mathbf{s}} = \mathbf{g}(\mathbf{s}, \mathbf{a}, t)$, can we replace some absolute state variables with relative quantities and still represent the dynamics exactly?

**Answer**: Yes, if and only if the RHS $\mathbf{g}$ is invariant under the corresponding transformation. This is a direct consequence of Noether's theorem (informally applied): continuous symmetries of the dynamics imply conserved quantities and redundant coordinates.

---

## 2. General Procedure

### Step 1: Write out RHS dependencies

For each equation $\dot{s}_i = g_i(\mathbf{s}, \mathbf{a}, t)$, list which state variables $g_i$ **actually** depends on — not which are in the state vector, but which appear in the computation of forces/accelerations.

### Step 2: Check for symmetries

A state variable $q$ can be replaced by relative quantities if the RHS is invariant under a global shift:

$$\mathbf{g}(\mathbf{s}) = \mathbf{g}(\mathbf{s} + \delta) \quad \forall \delta \in \text{symmetry group}$$

| Symmetry | Test | Simplification |
|---|---|---|
| Translation invariance | Does $\mathbf{g}$ depend on $x_i$ only through $(x_{i+1} - x_i)$? | Replace absolute positions with relative displacements |
| Rotation invariance | Does $\mathbf{g}$ depend on $\psi_e$ only through $(\psi_{e+1} - \psi_e)$? | Replace absolute angles with curvatures |
| Time-shift invariance | Does $\mathbf{g}$ depend on $t$ only through the action $\mathbf{a}(t)$? | System is autonomous given $\mathbf{a}(t)$ |
| Scale invariance | Does $\mathbf{g}(\lambda\mathbf{s}) = \lambda^k \mathbf{g}(\mathbf{s})$? | Nondimensionalize |

### Step 3: Check for broken symmetries

Sometimes a force **breaks** an invariance. This is the critical step — if any single force term depends on the absolute quantity, the symmetry is broken and the simplification is invalid.

Examples of symmetry-breaking terms:
- **Spatially varying fields**: gravity $g(z)$ instead of constant $g$ breaks translational invariance in $z$
- **Spatially varying friction**: $c_t(x, y)$ (different terrain) breaks translational invariance in $(x, y)$
- **Fixed obstacles/walls**: boundary interactions at world positions break translation invariance
- **External potential fields**: a target at $(x^*, y^*)$ breaks invariance for the reward (but **not** for the dynamics — this distinction matters)

**Key insight**: The dynamics and the reward function have different symmetries. The surrogate approximates the dynamics $\mathbf{g}$, so only the dynamics symmetries matter for the surrogate's input representation.

### Step 4: Construct the transformation

If the symmetry holds, define the body-frame transformation explicitly:
1. Choose a reference frame (e.g., center of mass + mean heading)
2. Write the forward transform: absolute → body-frame
3. Write the inverse transform: body-frame + global pose → absolute
4. Verify bijectivity (the transform must be invertible)

---

## 3. Application to Cosserat Rod Snake Robot

### 3.1 The ODE system

From system-formulation.tex, the semi-discrete system is:

$$\dot{\mathbf{s}} = \mathbf{g}(\mathbf{s}, \mathbf{a}, t), \quad \mathbf{s} \in \mathbb{R}^{124}, \quad \mathbf{a} \in \mathbb{R}^5$$

State vector:
$$\mathbf{s} = (x_1, \ldots, x_{21},\ y_1, \ldots, y_{21},\ \dot{x}_1, \ldots, \dot{x}_{21},\ \dot{y}_1, \ldots, \dot{y}_{21},\ \psi_1, \ldots, \psi_{20},\ \omega_{z,1}, \ldots, \omega_{z,20})$$

### 3.2 Force-by-force symmetry analysis

| Force term | What it uses | Depends on absolute $(x, y)$? | Depends on absolute $\psi$? |
|---|---|---|---|
| Internal shear-stretch | $\sigma_e = Q_e^T (x_{e+1} - x_e) / \ell_{\text{rest}}$ — differences | **No** | **No** (uses $Q_e$ from $\psi_e$, but only relative to tangent) |
| Internal bending-twist | $\kappa_v - \kappa_{\text{rest}}$ — curvature from angle differences | **No** | **No** |
| Gravity | $m_i \mathbf{g}$ — constant vector | **No** | **No** |
| RFT friction | $-c_t \mathbf{v}_t - c_n \mathbf{v}_n$ — velocity decomposed along tangent/normal | **No** | **No** (tangent direction from position differences; friction uses angle between velocity and tangent, which is relative) |
| Numerical damping | $-\gamma \mathbf{v}_i$ — velocity only | **No** | **No** |
| CPG control | $\kappa_e^{\text{rest}} = A \sin(2\pi k s_e + 2\pi f t + \phi_0) + b$ — arc-length parametrized | **No** | **No** |

**Result**: Not a single force depends on absolute position or absolute orientation. The system has **full SE(2) symmetry** (planar translation + rotation).

### 3.3 The body-frame transformation

Define the reference frame:
- Center of mass: $\bar{x} = \frac{1}{21}\sum_i x_i$, $\bar{y} = \frac{1}{21}\sum_i y_i$
- Mean heading: $\bar{\psi} = \frac{1}{20}\sum_e \psi_e$
- CoM velocity: $\bar{v}_x = \frac{1}{21}\sum_i \dot{x}_i$, $\bar{v}_y = \frac{1}{21}\sum_i \dot{y}_i$

**Forward transform** (absolute → body-frame):

$$\tilde{x}_i, \tilde{y}_i = R(-\bar{\psi}) \begin{pmatrix} x_i - \bar{x} \\ y_i - \bar{y} \end{pmatrix}$$

$$\tilde{v}_{x,i}, \tilde{v}_{y,i} = R(-\bar{\psi}) \begin{pmatrix} \dot{x}_i - \bar{v}_x \\ \dot{y}_i - \bar{v}_y \end{pmatrix}$$

$$\tilde{\psi}_e = \psi_e - \bar{\psi}$$

$$\tilde{\omega}_e = \omega_e \quad \text{(unchanged)}$$

where $R(\alpha)$ is the 2D rotation matrix.

**Inverse transform** (body-frame + global pose → absolute):

$$\begin{pmatrix} x_i \\ y_i \end{pmatrix} = R(\bar{\psi}) \begin{pmatrix} \tilde{x}_i \\ \tilde{y}_i \end{pmatrix} + \begin{pmatrix} \bar{x} \\ \bar{y} \end{pmatrix}$$

The transform is **bijective** and **differentiable**. Zero information loss.

### 3.4 Dimensions

| Representation | Dims | Bounded? |
|---|---|---|
| Absolute state | 124 | No — positions grow without bound |
| Body-frame state | 124 | Yes — positions bounded by rod length (~1m) |
| Global pose (separate) | 4 | $(\bar{x}, \bar{y}, \bar{\psi}, \bar{\omega})$ |

Same total information. The body-frame state has bounded, well-normalized features.

---

## 4. Why This Matters for Surrogates

1. **Generalization**: A surrogate trained on data from one region of the plane generalizes to all regions, because body-frame features are position-independent.

2. **Normalization**: Body-frame positions are bounded by the rod length L = 1.0m. Absolute positions are unbounded, making normalization unstable over long rollouts.

3. **Sample efficiency**: The network does not need to learn translation/rotation invariance from data — it is built into the representation.

4. **Physics alignment**: The surrogate's input features match what the physics actually uses (strains, curvatures, relative velocities), giving the network an appropriate inductive bias.

---

## 5. When This Simplification Fails

The body-frame simplification is **invalid** if any of these conditions hold:

- **Spatially varying environment**: friction coefficients depend on $(x, y)$, e.g., different terrain types at different locations
- **Fixed obstacles**: walls, barriers at specific world coordinates that create position-dependent contact forces
- **External fields**: electromagnetic fields, flow fields that vary with world position
- **Multiple interacting agents**: if another robot's absolute position matters for contact/interaction forces

In all these cases, some absolute position information must be retained (at minimum, the global pose). The internal body-shape representation can still be relative.

---

## 6. General Checklist for Any ODE/PDE System

When designing a surrogate for a new system, apply this checklist:

- [ ] List every force/source term in the RHS
- [ ] For each term, identify whether it uses absolute or relative quantities
- [ ] If ALL terms use only relative quantities → full invariance, use body-frame representation
- [ ] If SOME terms use absolute quantities → partial invariance, use body-frame for internal state but retain global pose as input
- [ ] If the system has no spatial symmetry → absolute coordinates are necessary
- [ ] Verify the transform is invertible (can reconstruct absolute state from body-frame + global pose)
- [ ] Check that the output representation is consistent with the input (predict body-frame residuals, not absolute residuals)
