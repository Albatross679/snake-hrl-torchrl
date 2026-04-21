---
name: Choi 2025 Four Tasks — Full PDE System
description: Complete PDE specification for all four soft manipulator tasks (Follow Target, 4D IK, 2D Tight Obstacles, 3D Random Obstacles) in the PINN framework format from pinn-clean-pde-system-framework.md
type: knowledge
created: 2026-03-19
updated: 2026-03-19
tags: [choi2025, dismech, pde, pinn, soft-manipulator, cosserat, der, tasks]
aliases: [choi2025-tasks-pde, four-tasks-pde]
---

# Choi & Tong (2025) — Four Tasks: Full PDE System

All four tasks share the same underlying physics (Discrete Elastic Rod). They differ in:
- **Dimensionality** of control (2D bending vs 3D bending vs 3D bending + twist)
- **Target type** (moving vs static, position-only vs position + orientation)
- **Contact forces** (none vs obstacle contact)
- **Reward function**

---

## 1. Shared Physics: Discrete Elastic Rod (DER)

### 1.1 Well-Posedness Tuple

$\mathcal{S} = (\mathcal{L}, \Omega, \text{BC}, \text{IC}, \boldsymbol{\theta})$

| Component | Name | Specification |
|-----------|------|---------------|
| $\mathcal{L}$ | Governing equations | DER: stretch + bend + twist elastic energies |
| $\Omega$ | Domain | Discrete centerline: $N = 21$ nodes, $N-1 = 20$ edges, arc length $L = 1.0$ m $\times$ time $t \in [0, T]$ |
| BC | Boundary conditions | Clamped at node 0: $\mathbf{q}_0 = \mathbf{q}_0^{\text{fixed}}$, $\dot{\mathbf{q}}_0 = \mathbf{0}$ |
| IC | Initial conditions | Straight rod along $x$-axis at rest |
| $\boldsymbol{\theta}$ | Parameters | $L = 1.0$ m, $r = 0.05$ m, $\rho = 1000$ kg/m$^3$, $E = 10$ MPa, $\nu = 0.5$ |
| Control | Delta natural curvature | $C = 5$ control points, Voronoi-smoothed to $N-2 = 19$ bend springs |

### 1.2 Degrees of Freedom

The generalized coordinate vector is:

$$
\mathbf{q} = [\mathbf{q}_0, \theta^0, \mathbf{q}_1, \theta^1, \ldots, \theta^{N-2}, \mathbf{q}_{N-1}]^T \in \mathbb{R}^{4N-1}
$$

where $\mathbf{q}_i \in \mathbb{R}^3$ is the position of node $i$ and $\theta^i$ is the signed angle (twist) of edge $i$.

For $N = 21$: $\dim(\mathbf{q}) = 4(21) - 1 = 83$ DOF.

### 1.3 Elastic Energies

**Stretching energy** (edge extension/compression):

$$
\epsilon^i = \|\mathbf{e}^i\| / \bar{\lambda} - 1, \qquad E_s = \frac{1}{2} K_s \sum_{i=0}^{N-2} (\epsilon^i)^2 \bar{\lambda}
$$

where $\mathbf{e}^i = \mathbf{q}_{i+1} - \mathbf{q}_i$, $\bar{\lambda}$ is the rest edge length, $K_s = EA$ is the stretching stiffness.

**Bending energy** (curvature at interior nodes):

Curvature binormal at interior node $i$:

$$
(\kappa\mathbf{b})_i = \frac{2\mathbf{t}^{i-1} \times \mathbf{t}^i}{1 + \mathbf{t}^{i-1} \cdot \mathbf{t}^i}
$$

where $\mathbf{t}^i = \mathbf{e}^i / \|\mathbf{e}^i\|$ is the unit tangent of edge $i$.

Integrated curvature components along material frame:

$$
\kappa_{1,i} = \frac{1}{2}(\kappa\mathbf{b})_i \cdot (\mathbf{m}_2^{i-1} + \mathbf{m}_2^i), \qquad \kappa_{2,i} = \frac{1}{2}(\kappa\mathbf{b})_i \cdot (\mathbf{m}_1^{i-1} + \mathbf{m}_1^i)
$$

$$
E_b = \frac{1}{2} K_b \sum_{i=1}^{N-2} (\boldsymbol{\kappa}_i - \bar{\boldsymbol{\kappa}}_i)^2 \frac{1}{V_i}
$$

where $\boldsymbol{\kappa}_i = [\kappa_{1,i}, \kappa_{2,i}]$, $\bar{\boldsymbol{\kappa}}_i$ is the natural (rest) curvature set by the controller, $V_i$ is the Voronoi length, and $K_b = EI$ is the bending stiffness with $I = \pi r^4 / 4$.

**Twisting energy** (relative rotation of material frames):

$$
\psi_i = \theta^i - \theta^{i-1} + \beta^i, \qquad E_t = \frac{1}{2} K_t \sum_{i=1}^{N-2} (\psi^i - \bar{\psi}^i)^2 \frac{1}{V_i}
$$

where $\beta^i$ is the reference frame rotation angle between consecutive edges and $K_t = GJ$ is the twisting stiffness with $G = E / 2(1+\nu)$.

**Total elastic energy:**

$$
E_{\text{elastic}} = E_s + E_b + E_t
$$

### 1.4 Equations of Motion

$$
\mathbf{M} \ddot{\mathbf{q}} = -\nabla_\mathbf{q} E_{\text{elastic}}(\mathbf{q}) + \mathbf{F}_{\text{ext}}(\mathbf{q}, t)
$$

where $\mathbf{M}$ is the lumped mass matrix and $\mathbf{F}_{\text{ext}}$ includes gravity and (for contact tasks) contact forces.

**Residual form:**

$$
\mathcal{R}(\mathbf{q}) = \mathbf{M} \ddot{\mathbf{q}} + \nabla_\mathbf{q} E_{\text{elastic}}(\mathbf{q}) - \mathbf{F}_{\text{ext}}(\mathbf{q}) = \mathbf{0}
$$

Expanding the elastic gradient into its three components:

$$
\nabla_\mathbf{q} E_{\text{elastic}} = \nabla_\mathbf{q} E_s + \nabla_\mathbf{q} E_b + \nabla_\mathbf{q} E_t
$$

### 1.5 External Forces

**Gravity** (all tasks):

$$
\mathbf{F}_{\text{grav},i} = m_i \mathbf{g}, \qquad \mathbf{g} = (0, 0, -9.81) \text{ m/s}^2
$$

where $m_i = \rho A \bar{\lambda}$ (interior nodes) or $m_i = \rho A \bar{\lambda} / 2$ (boundary nodes).

**Contact forces** (Tasks 3 & 4 only):

Contact between the rod and spherical obstacles is modeled via a smooth barrier potential. The penetration distance between node $i$ and obstacle $j$ is:

$$
\epsilon_{ij} = R_{\text{obs},j} + r_{\text{rod}} - \|\mathbf{q}_i - \mathbf{p}_{\text{obs},j}\|
$$

where $\epsilon_{ij} > 0$ indicates penetration.

**Elastica contact force** (explicit penalty):

$$
\mathbf{F}_{\text{contact}}^{\text{elastica}} = H(\epsilon)(-\mathbf{F}_\perp + k\epsilon + \mathbf{d})\hat{\mathbf{u}}
$$

where $H(\cdot)$ is the Heaviside function, $k$ is contact stiffness, $\mathbf{d}$ is damping, and $\hat{\mathbf{u}}$ is the contact normal.

**DisMech contact energy** (smooth barrier / Implicit Contact Method):

$$
E_{\text{contact}} = \sum_j \sum_i k \left( \frac{1}{K} \log(1 + e^{K \epsilon_{ij}}) \right)^2
$$

where $K = 15/\delta$, $\delta = 0.005$ m. The contact force is derived as:

$$
\mathbf{F}_{\text{contact}} = -\nabla_\mathbf{q} E_{\text{contact}}
$$

### 1.6 Control: Delta Natural Curvature

At each RL step, the agent outputs $\Delta\bar{\boldsymbol{\kappa}} \in [-1, 1]^{d_a}$ at $C = 5$ control points. These are:

1. Scaled by $\Delta\kappa_{\max}$
2. Voronoi-interpolated to all $N-2 = 19$ bend springs via weight matrix $\mathbf{W} \in \mathbb{R}^{19 \times 5}$
3. Accumulated: $\bar{\boldsymbol{\kappa}}_i \leftarrow \bar{\boldsymbol{\kappa}}_i + \mathbf{W} \Delta\bar{\boldsymbol{\kappa}}$

The natural curvature $\bar{\boldsymbol{\kappa}}_i$ enters the bending energy $E_b$ and drives the rod to bend.

### 1.7 Boundary Conditions (All Tasks)

**Clamped base** (node 0 fixed in space):

$$
\mathbf{q}_0(t) = \mathbf{q}_0^{\text{fixed}} = (0, 0, 0), \qquad \dot{\mathbf{q}}_0(t) = \mathbf{0}
$$

### 1.8 Initial Conditions (All Tasks)

$$
\mathbf{q}_i(0) = i \cdot \bar{\lambda} \cdot \hat{\mathbf{e}}_x, \quad i = 0, \ldots, N-1 \qquad \text{(straight rod along } x\text{-axis)}
$$

$$
\dot{\mathbf{q}}(0) = \mathbf{0}, \qquad \theta^i(0) = 0, \quad i = 0, \ldots, N-2
$$

### 1.9 Observation Space (All Tasks)

$$
\mathbf{o} = [\underbrace{\mathbf{q}_0, \ldots, \mathbf{q}_{N-1}}_{3N}, \underbrace{\dot{\mathbf{q}}_0, \ldots, \dot{\mathbf{q}}_{N-1}}_{3N}, \underbrace{\kappa_1, \ldots, \kappa_{N-2}}_{N-2}, \underbrace{\mathbf{p}_{\text{target}}}_{3}, \underbrace{(\text{task-specific})}_{\cdot\cdot\cdot}]
$$

---

## 2. Task 1: FOLLOW TARGET

### 2.1 Task Specification

| Property | Value |
|----------|-------|
| Dimensionality | 3D |
| Control type | 3D bending ($\kappa_1, \kappa_2$) |
| Action dim | $2C = 10$ |
| Contact | None |
| Target | Moving (bouncing within workspace) |
| Control frequency | 10 Hz (2 substeps at $\Delta t = 0.05$ s) |

### 2.2 Target Dynamics

The target is an independent dynamical system (not coupled to the rod):

$$
\dot{\mathbf{p}}_{\text{target}} = \mathbf{v}_{\text{target}}
$$

$$
\mathbf{v}_{\text{target}} \leftarrow \mathbf{v}_{\text{target}} - 2(\mathbf{v}_{\text{target}} \cdot \hat{\mathbf{n}}) \hat{\mathbf{n}} \quad \text{when } \|\mathbf{p}_{\text{target}}\| > r_{\max}
$$

where $\hat{\mathbf{n}} = \mathbf{p}_{\text{target}} / \|\mathbf{p}_{\text{target}}\|$ is the outward normal of the spherical workspace boundary, $|\mathbf{v}_{\text{target}}| = 0.05$ m/s, and $r_{\max} = 0.9$ m.

Initial target position sampled uniformly in spherical coordinates:

$$
\mathbf{p}_{\text{target}}(0) = r(\sin\phi\cos\theta, \sin\phi\sin\theta, \cos\phi), \quad r \sim U[0.3, 0.9], \; \theta \sim U[0, 2\pi], \; \phi \sim U[0, \pi]
$$

### 2.3 Governing Equations

$$
\mathbf{M} \ddot{\mathbf{q}} + \nabla_\mathbf{q} (E_s + E_b + E_t) - \mathbf{F}_{\text{grav}} = \mathbf{0}
$$

No contact forces.

### 2.4 Reward

$$
r_t = \underbrace{e^{-5 d_t}}_{\text{distance}} + \underbrace{10 (d_{t-1} - d_t)}_{\text{improvement}}
$$

where $d_t = \|\mathbf{q}_{N-1}(t) - \mathbf{p}_{\text{target}}(t)\|$ is the tip-to-target distance.

---

## 3. Task 2: 4D INVERSE KINEMATICS

### 3.1 Task Specification

| Property | Value |
|----------|-------|
| Dimensionality | 3D |
| Control type | 3D bending + twist ($\kappa_1, \kappa_2, \bar{\psi}$) |
| Action dim | $3C = 15$ |
| Contact | None |
| Target | Static position $(x, y, z)$ + yaw orientation $\theta_{\text{yaw}}$ |
| Control frequency | 10 Hz (2 substeps at $\Delta t = 0.05$ s) |

### 3.2 Target Specification

Static 4D target sampled once per episode:

$$
\mathbf{p}_{\text{target}} = r(\sin\phi\cos\theta, \sin\phi\sin\theta, \cos\phi), \quad r \sim U[0.3, 0.9]
$$

$$
\hat{\mathbf{d}}_{\text{target}} \sim \text{Uniform}(S^2) \quad \text{(random unit orientation vector)}
$$

The observation includes both $\mathbf{p}_{\text{target}} \in \mathbb{R}^3$ and $\hat{\mathbf{d}}_{\text{target}} \in \mathbb{R}^3$.

### 3.3 Governing Equations

Identical to Task 1 (no contact):

$$
\mathbf{M} \ddot{\mathbf{q}} + \nabla_\mathbf{q} (E_s + E_b + E_t) - \mathbf{F}_{\text{grav}} = \mathbf{0}
$$

The difference is in control: the agent also modulates the natural twist $\bar{\psi}_i$, which enters the twisting energy $E_t$. This provides an additional actuation mode (revolute joint collinear with the edge).

### 3.4 Reward

$$
r_t = 0.7 \underbrace{e^{-5 d_{\text{pos}}}}_{\text{position}} + 0.3 \underbrace{\frac{1 + \cos\alpha}{2}}_{\text{orientation}}
$$

where:

$$
d_{\text{pos}} = \|\mathbf{q}_{N-1} - \mathbf{p}_{\text{target}}\|
$$

$$
\cos\alpha = \text{clip}(\hat{\mathbf{t}}_{\text{tip}} \cdot \hat{\mathbf{d}}_{\text{target}}, -1, 1)
$$

$$
\hat{\mathbf{t}}_{\text{tip}} = \frac{\mathbf{q}_{N-1} - \mathbf{q}_{N-2}}{\|\mathbf{q}_{N-1} - \mathbf{q}_{N-2}\|}
$$

---

## 4. Task 3: 2D TIGHT OBSTACLES

### 4.1 Task Specification

| Property | Value |
|----------|-------|
| Dimensionality | 2D |
| Control type | 2D bending ($\kappa_1$ only) |
| Action dim | $C = 5$ |
| Contact | Two fixed obstacles forming narrow gap |
| Target | Static position |
| Control frequency | 2 Hz (10 substeps at $\Delta t = 0.05$ s) |

### 4.2 Obstacle Configuration

Two spherical obstacles placed symmetrically, forming a narrow gap:

$$
\mathbf{p}_{\text{obs},1} = (d, \; g/2 + R_{\text{obs}}, \; 0), \qquad \mathbf{p}_{\text{obs},2} = (d, \; -(g/2 + R_{\text{obs}}), \; 0)
$$

where $d = (d_{\min} + d_{\max})/2$ is the mid-range distance, $g = 0.15$ m is the gap width (0.12 m clearance for $2r = 0.1$ m diameter rod), and $R_{\text{obs}} = 0.05$ m is the obstacle radius.

### 4.3 Governing Equations (with Contact)

$$
\mathbf{M} \ddot{\mathbf{q}} + \nabla_\mathbf{q} (E_s + E_b + E_t) + \nabla_\mathbf{q} E_{\text{contact}} - \mathbf{F}_{\text{grav}} = \mathbf{0}
$$

where $E_{\text{contact}}$ and $\epsilon_{ij}$ are defined in Section 1.5.

### 4.4 Reward

$$
r_t = \underbrace{e^{-5 d_t}}_{\text{distance}} + \underbrace{10 (d_{t-1} - d_t)}_{\text{improvement}} - \underbrace{c_p \sum_i \max(0, \epsilon_i)}_{\text{contact penalty}}
$$

where $c_p = 10$ is the contact penalty scale and $\epsilon_i = \sum_j \max(0, R_{\text{obs},j} - \|\mathbf{q}_i - \mathbf{p}_{\text{obs},j}\|)$ is the total penetration at node $i$.

---

## 5. Task 4: 3D RANDOM OBSTACLES

### 5.1 Task Specification

| Property | Value |
|----------|-------|
| Dimensionality | 3D |
| Control type | 3D bending ($\kappa_1, \kappa_2$) |
| Action dim | $2C = 10$ |
| Contact | Random spherical obstacles (re-sampled each episode) |
| Target | Static position (fixed across episodes) |
| Control frequency | 2 Hz (10 substeps at $\Delta t = 0.05$ s) |

### 5.2 Obstacle Configuration

$n_{\text{obs}}$ spherical obstacles (paper uses 8 in Fig. 1), each with radius $R_{\text{obs}} = 0.05$ m, sampled uniformly in a contact-free manner:

$$
\mathbf{p}_{\text{obs},j} = r_j(\sin\phi_j\cos\theta_j, \sin\phi_j\sin\theta_j, \cos\phi_j)
$$

$$
r_j \sim U[0.2, 0.8], \quad \theta_j \sim U[0, 2\pi], \quad \phi_j \sim U[0.3, \pi - 0.3]
$$

> $\phi$ avoids poles to prevent clustering near the clamped base.

### 5.3 Governing Equations (with Contact)

Same as Task 3, but with randomized obstacle positions and 3D control:

$$
\mathbf{M} \ddot{\mathbf{q}} + \nabla_\mathbf{q} (E_s + E_b + E_t) + \nabla_\mathbf{q} E_{\text{contact}} - \mathbf{F}_{\text{grav}} = \mathbf{0}
$$

### 5.4 Reward

Identical to Task 3:

$$
r_t = e^{-5 d_t} + 10 (d_{t-1} - d_t) - c_p \sum_i \max(0, \epsilon_i)
$$

---

## 6. Summary: Four Tasks Comparison

| | Task 1: Follow Target | Task 2: 4D IK | Task 3: 2D Tight Obs | Task 4: 3D Random Obs |
|---|---|---|---|---|
| **Control DOF** | $2C = 10$ (3D bend) | $3C = 15$ (3D bend + twist) | $C = 5$ (2D bend) | $2C = 10$ (3D bend) |
| **Contact** | No | No | Yes (2 fixed) | Yes ($n$ random) |
| **Target** | Moving | Static (pos + orient) | Static (pos only) | Static (pos only) |
| **Control freq** | 10 Hz | 10 Hz | 2 Hz | 2 Hz |
| **Substeps/action** | 2 | 2 | 10 | 10 |
| **Governing eqn** | $E_s + E_b + E_t + \mathbf{F}_g$ | $E_s + E_b + E_t + \mathbf{F}_g$ | $E_s + E_b + E_t + E_c + \mathbf{F}_g$ | $E_s + E_b + E_t + E_c + \mathbf{F}_g$ |

### Canonical Form

All four tasks are governed by the same equation of motion:

$$
\boxed{
\mathbf{M} \ddot{\mathbf{q}} + \nabla_\mathbf{q} \left( E_s + E_b + E_t + \underbrace{E_c}_{\text{Tasks 3,4 only}} \right) = \mathbf{F}_{\text{grav}}
}
$$

The tasks differ in:
1. Which components of $\bar{\boldsymbol{\kappa}}_i$ and $\bar{\psi}_i$ the agent controls (determines action space)
2. Whether $E_c$ (contact energy) is present (determines physics complexity)
3. The reward signal $r_t$ (determines the RL objective)
4. The target dynamics (moving vs static, position-only vs position + orientation)

## 7. Physical Parameters (Table A.3)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Rod length | $L$ | 1.0 m |
| Rod radius | $r$ | 0.05 m |
| Cross-section area | $A = \pi r^2$ | $7.854 \times 10^{-3}$ m$^2$ |
| Second moment of area | $I = \pi r^4 / 4$ | $4.909 \times 10^{-6}$ m$^4$ |
| Polar moment | $J = \pi r^4 / 2$ | $9.817 \times 10^{-6}$ m$^4$ |
| Density | $\rho$ | 1000 kg/m$^3$ |
| Young's modulus | $E$ | $10 \times 10^6$ Pa |
| Poisson's ratio | $\nu$ | 0.5 |
| Shear modulus | $G = E/2(1+\nu)$ | $3.333 \times 10^6$ Pa |
| Stretching stiffness | $K_s = EA$ | $7.854 \times 10^4$ N |
| Bending stiffness | $K_b = EI$ | $4.909 \times 10^1$ N m$^2$ |
| Twisting stiffness | $K_t = GJ$ | $3.272 \times 10^1$ N m$^2$ |
| Number of nodes | $N$ | 21 |
| Number of edges | $N-1$ | 20 |
| Rest edge length | $\bar{\lambda} = L/(N-1)$ | 0.05 m |
| Gravity | $\mathbf{g}$ | $(0, 0, -9.81)$ m/s$^2$ |
| Contact stiffness | $k$ | $10^6$ |
| Contact tolerance | $\delta$ | 0.005 m |
| Control points | $C$ | 5 |
