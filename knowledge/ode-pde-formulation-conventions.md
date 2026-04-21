---
name: ODE/PDE System Formulation Conventions
description: Established conventions and best practices for cleanly formulating ODE and PDE systems mathematically, covering notation, standard forms, boundary/initial conditions, LaTeX typesetting, and computational perspectives
type: knowledge
created: 2026-03-16
updated: 2026-03-16
tags:
  - math
  - ode
  - pde
  - formulation
  - notation
  - latex
  - differential-equations
aliases:
  - ODE notation
  - PDE formulation
  - mathematical writing conventions
---

# ODE/PDE System Formulation Conventions

A comprehensive guide to the established conventions for writing down systems of ordinary and partial differential equations cleanly, unambiguously, and in a form suitable for both publication and implementation.

---

## 1. Guiding Principles

A clean formulation satisfies five criteria:

1. **Unambiguous** -- every symbol is defined exactly once, every equation has a clear domain
2. **Self-contained** -- a reader can understand the system without flipping to other documents
3. **Implementable** -- a programmer can translate each equation directly into code
4. **Dimensionally consistent** -- every term in every equation has matching physical units
5. **Separated** -- physics (governing equations) is cleanly separated from numerics (discretization, solver details)

---

## 2. ODE Systems

### 2.1 Standard Forms

#### Explicit First-Order System (State-Space Form)

The most universal form. Higher-order ODEs are always reducible to this.

$$
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, t), \qquad \mathbf{x}(t_0) = \mathbf{x}_0
$$

where $\mathbf{x} \in \mathbb{R}^n$ is the state vector, $\mathbf{f}: \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}^n$ is the right-hand side, and $\mathbf{x}_0$ is the initial condition.

**Conventions:**
- Bold lowercase for vectors: $\mathbf{x}$, $\mathbf{f}$, $\mathbf{u}$
- Dot notation for time derivatives: $\dot{\mathbf{x}} = d\mathbf{x}/dt$
- Subscript notation for components: $x_i$ is the $i$-th component of $\mathbf{x}$
- The system is *autonomous* if $\mathbf{f}$ does not depend explicitly on $t$: $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$

#### Linear Time-Invariant (LTI) State-Space Form

$$
\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}, \qquad \mathbf{y} = C\mathbf{x} + D\mathbf{u}
$$

where $A \in \mathbb{R}^{n \times n}$ is the state matrix, $B \in \mathbb{R}^{n \times m}$ the input matrix, $C \in \mathbb{R}^{p \times n}$ the output matrix, $D \in \mathbb{R}^{p \times m}$ the feedthrough matrix. The shorthand $(A, B, C, D)$ denotes a complete LTI system.

**Conventions:**
- Bold uppercase for matrices: $A$, $B$, $M$, $K$ (or sometimes $\mathbf{A}$, $\mathbf{B}$)
- Bold lowercase for column vectors: $\mathbf{x}$, $\mathbf{u}$, $\mathbf{y}$
- In control theory: $\mathbf{u}$ = input, $\mathbf{y}$ = output, $\mathbf{x}$ = state

#### Second-Order Mechanical System (Newton/Euler-Lagrange Form)

$$
M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + K\mathbf{q} = \mathbf{f}_{\text{ext}}
$$

or equivalently from the Euler-Lagrange equations:

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = Q_i, \qquad i = 1, \ldots, n
$$

where $L = T - V$ is the Lagrangian, $\mathbf{q}$ are generalized coordinates, and $Q_i$ are generalized forces.

**Reduction to first-order form:**

$$
\mathbf{x} = \begin{pmatrix} \mathbf{q} \\ \dot{\mathbf{q}} \end{pmatrix}, \qquad
\dot{\mathbf{x}} = \begin{pmatrix} \dot{\mathbf{q}} \\ M^{-1}(\mathbf{f}_{\text{ext}} - C\dot{\mathbf{q}} - K\mathbf{q}) \end{pmatrix}
$$

#### Hamiltonian Form

$$
\dot{q}_i = \frac{\partial H}{\partial p_i}, \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}, \qquad i = 1, \ldots, n
$$

where $H(\mathbf{q}, \mathbf{p}) = T + V$ is the Hamiltonian, $\mathbf{q}$ are generalized coordinates, and $\mathbf{p}$ are conjugate momenta. This form is preferred when symplectic structure matters (energy conservation, geometric integration).

#### Implicit ODE Form

$$
\mathbf{F}(t, \mathbf{x}, \dot{\mathbf{x}}) = \mathbf{0}
$$

Used when the derivative cannot be easily isolated on the left-hand side. This is the entry point for DAE theory.

### 2.2 Differential-Algebraic Equations (DAEs)

DAEs arise naturally in constrained mechanical systems, circuit simulation, and chemical kinetics.

#### General Implicit Form

$$
\mathbf{F}(t, \mathbf{y}, \dot{\mathbf{y}}) = \mathbf{0}
$$

where $\mathbf{F}: \mathbb{R} \times \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}^n$ and the Jacobian $\partial \mathbf{F}/\partial \dot{\mathbf{y}}$ is singular (otherwise it is just an implicit ODE).

#### Semi-Explicit Index-1 Form

$$
\dot{\mathbf{y}} = \mathbf{f}(\mathbf{y}, \mathbf{z}, t), \qquad \mathbf{0} = \mathbf{g}(\mathbf{y}, \mathbf{z}, t)
$$

where $\mathbf{y}$ are *differential variables* and $\mathbf{z}$ are *algebraic variables*. The system is index-1 if $\partial \mathbf{g}/\partial \mathbf{z}$ is nonsingular.

**Index definition:** The *differentiation index* is the minimum number of times the algebraic constraints must be differentiated with respect to $t$ to obtain a system of pure ODEs.

**Conventions from Hairer/Wanner (Solving ODEs II) and Brenan/Campbell/Petzold:**
- Index-0: A pure ODE system
- Index-1: One differentiation needed (most common in practice)
- Index-2: Two differentiations (e.g., multibody dynamics with position constraints)
- Index-3: Three differentiations (e.g., position-constrained multibody with Lagrange multipliers)

#### Constrained Mechanical System (Index-3 DAE)

$$
M\ddot{\mathbf{q}} = \mathbf{f}(\mathbf{q}, \dot{\mathbf{q}}, t) - G^T(\mathbf{q})\boldsymbol{\lambda}, \qquad \mathbf{g}(\mathbf{q}) = \mathbf{0}
$$

where $G = \partial \mathbf{g}/\partial \mathbf{q}$ is the constraint Jacobian and $\boldsymbol{\lambda}$ are Lagrange multipliers. Common in multibody dynamics (robotics, vehicle simulation).

### 2.3 Presenting an ODE System: The Complete Template

A publication-ready ODE system presentation includes:

```
1. Governing equation(s)
2. State vector definition
3. Domain and time interval
4. Initial conditions
5. Parameters with units and values
6. Auxiliary definitions (constitutive laws, force models)
```

**Example (damped oscillator):**

> Consider the damped harmonic oscillator:
> $$m\ddot{x} + c\dot{x} + kx = f(t)$$
> where $x(t) \in \mathbb{R}$ is the displacement, subject to initial conditions
> $$x(0) = x_0, \qquad \dot{x}(0) = v_0.$$
> The parameters are mass $m > 0$, damping coefficient $c \geq 0$, and stiffness $k > 0$.

---

## 3. PDE Systems

### 3.1 Strong Form (Classical/Pointwise Form)

The strong form states the equation at every point in the domain.

#### Scalar PDE

$$
\mathcal{L}[u] = f \qquad \text{in } \Omega
$$

where $\mathcal{L}$ is a differential operator, $u: \Omega \to \mathbb{R}$ is the unknown, $f$ is a source term, and $\Omega \subset \mathbb{R}^d$ is the spatial domain.

#### System of PDEs

$$
\mathcal{L}_i[\mathbf{u}] = f_i \qquad \text{in } \Omega, \quad i = 1, \ldots, m
$$

or in compact vector notation:

$$
\boldsymbol{\mathcal{L}}[\mathbf{u}] = \mathbf{f} \qquad \text{in } \Omega
$$

#### Canonical Examples

**Heat equation (parabolic):**
$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u + f \qquad \text{in } \Omega \times (0, T]
$$

**Wave equation (hyperbolic):**
$$
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u + f \qquad \text{in } \Omega \times (0, T]
$$

**Laplace/Poisson equation (elliptic):**
$$
-\nabla^2 u = f \qquad \text{in } \Omega
$$

**Notation convention:** The negative sign in the Laplacian is standard in the elliptic PDE community because it makes the operator positive-definite. Applied math and physics sometimes omit it.

### 3.2 Conservation Law Form

Preferred for hyperbolic systems (gas dynamics, shallow water, elastodynamics). Following LeVeque's conventions:

#### Scalar Conservation Law

$$
\frac{\partial u}{\partial t} + \nabla \cdot \mathbf{F}(u) = 0
$$

where $\mathbf{F}$ is the *flux function*.

#### System of Conservation Laws

$$
\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = \mathbf{0}
$$

In 1D:

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = \mathbf{0}
$$

**Conventions:**
- $\mathbf{U} \in \mathbb{R}^m$ is the vector of conserved quantities
- $\mathbf{F}: \mathbb{R}^m \to \mathbb{R}^m$ is the flux function
- The *flux Jacobian* $A(\mathbf{U}) = \partial \mathbf{F}/\partial \mathbf{U}$ determines hyperbolicity: the system is hyperbolic if $A$ has $m$ real eigenvalues with a complete set of eigenvectors
- Strictly hyperbolic if all eigenvalues are distinct

#### Euler Equations (Prototypical Example)

$$
\frac{\partial}{\partial t}\begin{pmatrix} \rho \\ \rho u \\ E \end{pmatrix} + \frac{\partial}{\partial x}\begin{pmatrix} \rho u \\ \rho u^2 + p \\ u(E + p) \end{pmatrix} = \mathbf{0}
$$

This illustrates the convention of writing conserved variables on the left and flux on the right as column vectors.

### 3.3 Weak (Variational) Form

Required for finite element methods. Obtained by multiplying the strong form by a test function and integrating.

#### Derivation Pattern

Given a strong form: $-\nabla \cdot (a \nabla u) = f$ in $\Omega$,

1. Multiply by test function $v \in V$:

$$
-\int_\Omega (\nabla \cdot (a \nabla u)) \, v \, dx = \int_\Omega f \, v \, dx
$$

2. Integrate by parts (move one derivative onto $v$):

$$
\int_\Omega a \nabla u \cdot \nabla v \, dx - \int_{\partial\Omega} a \frac{\partial u}{\partial n} v \, ds = \int_\Omega f \, v \, dx
$$

3. State: Find $u \in V$ such that

$$
a(u, v) = \ell(v) \qquad \forall v \in V
$$

where $a(u, v) = \int_\Omega a \nabla u \cdot \nabla v \, dx$ is the *bilinear form* and $\ell(v) = \int_\Omega f v \, dx$ is the *linear functional*.

**Conventions:**
- $V$ (or $H^1_0(\Omega)$) is the trial/test space
- $a(\cdot, \cdot)$ for the bilinear form, $\ell(\cdot)$ for the linear form
- "Find ... such that ... for all ..." is the canonical phrasing
- This is the formulation that FEniCS, deal.II, and other FE codes expect

### 3.4 Coupled Multi-Physics Systems

When multiple physical processes interact, the system can be written in block form:

$$
\begin{pmatrix}
\mathcal{L}_{11} & \mathcal{L}_{12} \\
\mathcal{L}_{21} & \mathcal{L}_{22}
\end{pmatrix}
\begin{pmatrix}
u_1 \\
u_2
\end{pmatrix}
=
\begin{pmatrix}
f_1 \\
f_2
\end{pmatrix}
$$

where the off-diagonal operators $\mathcal{L}_{12}$, $\mathcal{L}_{21}$ represent coupling.

**Operator splitting** is the standard numerical approach: solve each sub-problem sequentially, iterating to convergence. Common splitting methods:
- **Lie splitting** (first-order): solve $\mathcal{L}_1$, then $\mathcal{L}_2$
- **Strang splitting** (second-order): half-step $\mathcal{L}_1$, full step $\mathcal{L}_2$, half-step $\mathcal{L}_1$

---

## 4. Boundary and Initial Conditions

### 4.1 Types of Boundary Conditions

| Name | Form | Physical Meaning |
|------|------|------------------|
| Dirichlet | $u = g$ on $\partial\Omega_D$ | Prescribed value |
| Neumann | $\frac{\partial u}{\partial n} = h$ on $\partial\Omega_N$ | Prescribed flux |
| Robin (mixed) | $\alpha u + \beta \frac{\partial u}{\partial n} = g$ on $\partial\Omega_R$ | Convective transfer |
| Periodic | $u(0, t) = u(L, t)$ | Wrapping |

### 4.2 Presentation Convention

The canonical way to present a complete PDE problem (an initial-boundary value problem, or IBVP):

$$
\begin{aligned}
\frac{\partial u}{\partial t} - \alpha \nabla^2 u &= f && \text{in } \Omega \times (0, T], \\
u &= g_D && \text{on } \Gamma_D \times (0, T], \\
\frac{\partial u}{\partial n} &= g_N && \text{on } \Gamma_N \times (0, T], \\
u(\mathbf{x}, 0) &= u_0(\mathbf{x}) && \text{in } \Omega,
\end{aligned}
$$

where $\Gamma_D \cup \Gamma_N = \partial\Omega$ and $\Gamma_D \cap \Gamma_N = \emptyset$.

**Key conventions:**
- "in $\Omega$" for the governing equation (interior of domain)
- "on $\partial\Omega$" or "on $\Gamma$" for boundary conditions
- BCs and ICs follow immediately after the governing equation
- The domain decomposition $\Gamma_D \cup \Gamma_N = \partial\Omega$ must be stated explicitly

### 4.3 Number of Conditions

The required number of BCs and ICs depends on the order of the PDE:
- **Elliptic** (e.g., Poisson): BCs on entire boundary, no ICs
- **Parabolic** (e.g., heat): BCs on boundary + 1 IC in time
- **Hyperbolic** (e.g., wave): BCs on boundary + 2 ICs in time (value + velocity)

For ODE IVPs: $n$-th order system needs $n$ initial conditions.

---

## 5. Notation Conventions by Community

### 5.1 Time Derivatives

| Notation | Community | Example |
|----------|-----------|---------|
| $\dot{x}$, $\ddot{x}$ | Mechanics, control theory, ODEs | $\dot{\mathbf{q}} = \partial \mathbf{q}/\partial t$ |
| $x'$, $x''$ | Pure math, general ODEs | $y'' + y = 0$ |
| $\partial_t u$, $u_t$ | PDE community | $u_t = \alpha u_{xx}$ |
| $\frac{du}{dt}$, $\frac{d^2u}{dt^2}$ | Explicit/pedagogical | Clear but verbose |
| $D_t u$ | Operator notation | Used in abstract settings |

### 5.2 Spatial Derivatives

| Notation | Meaning | Community |
|----------|---------|-----------|
| $u_x$, $u_{xx}$ | Subscript notation | Concise PDE writing |
| $\partial_x u$, $\partial_{xx} u$ | Partial derivative | Explicit and clear |
| $\frac{\partial u}{\partial x}$, $\frac{\partial^2 u}{\partial x^2}$ | Leibniz notation | Pedagogical, dimensional analysis |
| $\nabla u$, $\nabla^2 u$, $\Delta u$ | Gradient, Laplacian | Coordinate-free (preferred for multi-d) |
| $\nabla \cdot \mathbf{F}$, $\text{div}\,\mathbf{F}$ | Divergence | Conservation laws |
| $\nabla \times \mathbf{F}$, $\text{curl}\,\mathbf{F}$ | Curl | Electromagnetics, fluid mechanics |

### 5.3 Vectors and Matrices

| Convention | Usage |
|-----------|-------|
| **Bold lowercase** $\mathbf{x}$ | Column vectors |
| **Bold uppercase** $\mathbf{A}$ or plain uppercase $A$ | Matrices |
| $\|\cdot\|$ | Norm (specify which: $\|\cdot\|_2$, $\|\cdot\|_\infty$) |
| $\langle \cdot, \cdot \rangle$ | Inner product |
| $\otimes$ | Tensor/outer product |
| $\mathbb{R}^n$ | State space (specify dimension) |

### 5.4 Function Spaces (PDE Community)

| Space | Meaning |
|-------|---------|
| $C^k(\Omega)$ | $k$-times continuously differentiable |
| $L^2(\Omega)$ | Square-integrable |
| $H^1(\Omega)$ | Sobolev space (one weak derivative in $L^2$) |
| $H^1_0(\Omega)$ | $H^1$ with zero trace on $\partial\Omega$ |

---

## 6. Nondimensionalization

### 6.1 Why It Matters

Nondimensionalization reveals the relative importance of terms, reduces the number of parameters, and prevents unit-related implementation errors. Every physical system should be nondimensionalized before analysis or simulation.

### 6.2 Standard Procedure

1. **Identify characteristic scales**: length $L$, time $T$, velocity $U$, etc.
2. **Define dimensionless variables**: $\tilde{x} = x/L$, $\tilde{t} = t/T$, $\tilde{u} = u/U$
3. **Substitute** into the governing equations
4. **Group** dimensional parameters into dimensionless numbers (Reynolds, Peclet, Froude, etc.)
5. **Verify** every term in every equation is dimensionless

### 6.3 Example

Starting from the Navier-Stokes momentum equation:

$$
\rho \left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}\right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g}
$$

After nondimensionalization with scales $U$, $L$, $T = L/U$:

$$
\frac{\partial \tilde{\mathbf{u}}}{\partial \tilde{t}} + \tilde{\mathbf{u}} \cdot \tilde{\nabla} \tilde{\mathbf{u}} = -\tilde{\nabla} \tilde{p} + \frac{1}{\text{Re}} \tilde{\nabla}^2 \tilde{\mathbf{u}} + \frac{1}{\text{Fr}^2} \hat{\mathbf{g}}
$$

where $\text{Re} = \rho U L / \mu$ (Reynolds number) and $\text{Fr} = U/\sqrt{gL}$ (Froude number). This reveals that only two dimensionless numbers govern the entire problem.

---

## 7. Computational Framework Perspectives

### 7.1 How Solvers Expect Problems

| Framework | Expected Form | Interface |
|-----------|--------------|-----------|
| **SciPy / MATLAB** | $\dot{\mathbf{y}} = \mathbf{f}(t, \mathbf{y})$ (explicit first-order) | `f(t, y) -> dydt` |
| **Julia DifferentialEquations.jl** | Same, plus mass matrix $M\dot{\mathbf{y}} = \mathbf{f}(t, \mathbf{y})$ and DAE forms | `ODEProblem(f, u0, tspan)` |
| **SUNDIALS IDA** | $\mathbf{F}(t, \mathbf{y}, \dot{\mathbf{y}}) = \mathbf{0}$ (implicit/DAE) | Residual form |
| **FEniCS** | Weak form: $a(u,v) = \ell(v)$ | `solve(a == L, u, bc)` |
| **deal.II** | Weak form with manual assembly | Bilinear/linear form classes |
| **PETSc** | Residual $\mathbf{F}(\mathbf{x}) = \mathbf{0}$ or $\mathbf{F}(t, \mathbf{x}, \dot{\mathbf{x}}) = \mathbf{0}$ | `FormFunction`, `FormIFunction` |

### 7.2 The Mass Matrix Form

Many real systems are naturally in the form:

$$
M(t, \mathbf{y})\dot{\mathbf{y}} = \mathbf{f}(t, \mathbf{y})
$$

where $M$ is the *mass matrix*. If $M$ is invertible, this is an ODE. If $M$ is singular, this is a DAE. Modern solvers (Julia DifferentialEquations.jl, SUNDIALS) handle this natively.

### 7.3 Method of Lines (MOL)

The bridge from PDEs to ODEs: discretize space, keep time continuous.

$$
\text{PDE: } \frac{\partial u}{\partial t} = \mathcal{L}[u]
\quad \xrightarrow{\text{spatial discretization}} \quad
\text{ODE: } \dot{\mathbf{U}} = L_h \mathbf{U}
$$

where $\mathbf{U}(t) \in \mathbb{R}^N$ is the vector of nodal values and $L_h$ is the discrete spatial operator. This is the standard approach in computational science: let the spatial discretization produce an ODE system, then use a standard ODE integrator for time.

---

## 8. Domain-Specific Patterns

### 8.1 Elastic Rod Dynamics (Cosserat/Kirchhoff Rods)

Directly relevant to the snake robot simulation. Cosserat rod theory formulates the dynamics as a system of PDEs in arc-length $s$ and time $t$.

**Primary variables:**
- Centerline position $\mathbf{r}(s, t) \in \mathbb{R}^3$
- Director frame $\{\mathbf{d}_1(s,t), \mathbf{d}_2(s,t), \mathbf{d}_3(s,t)\}$ (local material frame)

**Strain measures:**
- Shear/extension strain: $\boldsymbol{\sigma} = \mathbf{Q} \mathbf{r}_s - \mathbf{d}_3$ (where $\mathbf{Q}$ rotates from lab to local frame)
- Curvature/twist vector: $\boldsymbol{\kappa}$ defined by $\partial_s \mathbf{d}_j = \boldsymbol{\kappa} \times \mathbf{d}_j$

**Governing equations (conservation of linear and angular momentum):**

$$
\begin{aligned}
\rho A \, \ddot{\mathbf{r}} &= \partial_s \mathbf{n} + \mathbf{f}_{\text{ext}}, \\
\rho I \, \dot{\boldsymbol{\omega}} &= \partial_s \mathbf{m} + \mathbf{r}_s \times \mathbf{n} + \mathbf{c}_{\text{ext}},
\end{aligned}
$$

where $\mathbf{n}(s,t)$ is the internal force, $\mathbf{m}(s,t)$ is the internal moment, and subscript $s$ denotes $\partial/\partial s$.

**Constitutive relations (linear elastic):**

$$
\mathbf{n} = S(\boldsymbol{\sigma} - \boldsymbol{\sigma}^0), \qquad \mathbf{m} = B(\boldsymbol{\kappa} - \boldsymbol{\kappa}^0)
$$

where $S$ and $B$ are diagonal stiffness matrices, and superscript $0$ denotes the rest (reference) configuration.

**Kirchhoff rod** is the special case with $\boldsymbol{\sigma} = \mathbf{0}$ (inextensible, unshearable). This eliminates the shear strain and constrains $|\mathbf{r}_s| = 1$, turning the system into a DAE.

**Discrete Elastic Rods (DER)** -- the approach used in DisMech -- discretizes the rod into nodes and edges, turning the PDE system into a large ODE system:

$$
M\ddot{\mathbf{q}} = -\frac{\partial E}{\partial \mathbf{q}} + \mathbf{f}_{\text{ext}}(\mathbf{q}, \dot{\mathbf{q}}, t)
$$

where $\mathbf{q}$ collects all node positions and twist angles, $E$ is the elastic energy (stretching + bending + twisting), and $\mathbf{f}_{\text{ext}}$ includes gravity, contact, friction, and actuation forces.

### 8.2 CPG (Central Pattern Generator) as a Coupled ODE System

The CPG in this project generates coordinated joint curvatures via coupled oscillators. The Matsuoka oscillator for a single unit:

$$
\begin{aligned}
\tau \dot{x}_i &= -x_i - \beta v_i - \sum_{j \neq i} w_{ij} y_j + u_i, \\
\tau' \dot{v}_i &= -v_i + y_i, \\
y_i &= \max(0, x_i),
\end{aligned}
$$

where $x_i$ is the membrane potential, $v_i$ is the adaptation variable, $y_i$ is the output, and $w_{ij}$ are coupling weights. The $\max$ introduces a nonlinearity that makes this a piecewise-linear system.

### 8.3 Contact Mechanics

Contact and friction are typically formulated as complementarity conditions (not smooth ODEs):

$$
0 \leq g_n \perp f_n \geq 0
$$

meaning: gap $g_n \geq 0$, normal force $f_n \geq 0$, and $g_n \cdot f_n = 0$ (complementarity). In practice, these are regularized (penalty method) or solved as optimization problems (LCP).

---

## 9. LaTeX Typesetting Conventions

### 9.1 Environment Selection

| Use | Environment | Notes |
|-----|-------------|-------|
| Single numbered equation | `equation` | One number per equation |
| Aligned system | `align` | `&` at alignment points, `\\` between lines |
| Sub-equations | `subequations` + `align` | One parent number: (1a), (1b), ... |
| Unnumbered display | `equation*` or `align*` | For intermediate steps |
| Cases/piecewise | `cases` inside `equation` | For BVPs with different regions |

**Never use `eqnarray`.** It is deprecated and produces inconsistent spacing. Use `align` from `amsmath` instead.

### 9.2 Alignment Conventions

Align at the relation symbol (`=`, `\leq`, etc.):

```latex
\begin{align}
  \dot{\mathbf{x}} &= \mathbf{f}(\mathbf{x}, t), \label{eq:state} \\
  \mathbf{y}       &= \mathbf{h}(\mathbf{x}),     \label{eq:output}
\end{align}
```

For governing equation + BCs + ICs, use `aligned` inside `equation` for a single number, or `subequations` + `align` for individual numbers:

```latex
\begin{subequations}\label{eq:heat-ibvp}
\begin{align}
  \frac{\partial u}{\partial t} - \alpha \nabla^2 u &= f
    && \text{in } \Omega \times (0, T], \label{eq:heat-pde} \\
  u &= g_D && \text{on } \Gamma_D, \label{eq:heat-dirichlet} \\
  \frac{\partial u}{\partial n} &= g_N
    && \text{on } \Gamma_N, \label{eq:heat-neumann} \\
  u(\mathbf{x}, 0) &= u_0(\mathbf{x})
    && \text{in } \Omega. \label{eq:heat-ic}
\end{align}
\end{subequations}
```

### 9.3 Operator Formatting

```latex
% Correct
\operatorname{div}  \quad \nabla \cdot
\operatorname{grad} \quad \nabla
\operatorname{curl} \quad \nabla \times

% Vectors
\mathbf{x}  % upright bold (most common)
\boldsymbol{x}  % italic bold (for Greek)
\vec{x}     % arrow notation (less common in modern papers)

% Matrices
\mathbf{A}  % or just A (context makes it clear)

% Partial derivatives
\partial_t u  % compact
\frac{\partial u}{\partial t}  % explicit

% Function names
\sin, \cos, \exp, \log  % use \operatorname for custom: \operatorname{Re}
```

### 9.4 Consistent Style Checklist

- [ ] All vectors are bold (or all have arrows -- pick one and be consistent)
- [ ] All matrices are distinguishable from scalars (bold uppercase or calligraphic)
- [ ] Derivative notation is consistent throughout (do not mix $\dot{x}$ and $dx/dt$ for the same variable)
- [ ] Domain and codomain specified for each function
- [ ] Every symbol is defined before or immediately after first use
- [ ] Physical units stated for dimensional quantities
- [ ] Equation numbering is sequential and referenced in text

---

## 10. Common Mistakes and Anti-Patterns

### 10.1 In Formulation

1. **Mixing derivative notations**: Using $\dot{x}$ in one equation and $\frac{dx}{dt}$ in the next for the same variable. Pick one and be consistent.

2. **Undefined symbols**: Introducing a symbol in an equation without defining it. Every symbol must be defined.

3. **Missing domain specification**: Writing $\nabla^2 u = f$ without stating "in $\Omega$" and the boundary conditions. A PDE without BCs is not a well-posed problem.

4. **Dimensional inconsistency**: Terms in an equation having different physical units. Always verify with dimensional analysis.

5. **Ambiguous state vectors**: Writing $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ without specifying what $\mathbf{x}$ contains. Always define the state vector explicitly: $\mathbf{x} = (q_1, q_2, \dot{q}_1, \dot{q}_2)^T$.

6. **Forgetting initial/boundary conditions**: The governing equation alone does not define a unique solution. ICs/BCs are part of the mathematical problem.

7. **Wrong index in DAE**: Formulating a DAE without analyzing its index. Index-3 DAEs require special solvers and index reduction; failing to recognize this leads to numerical instabilities or wrong results.

8. **Nonlinear terms in "linear" systems**: Claiming a system is linear when constitutive laws or force models introduce nonlinearity. Be explicit about linearity assumptions.

### 10.2 In LaTeX

1. **Using `eqnarray` instead of `align`**: The `eqnarray` environment has known spacing bugs and is deprecated by the AMS.

2. **Inconsistent vector notation**: Switching between $\vec{x}$, $\mathbf{x}$, and $\underline{x}$ within the same document.

3. **Not using `\operatorname`**: Writing `sin` instead of `\sin` or `div` instead of `\operatorname{div}`, which produces incorrect italic formatting.

4. **Missing `\left`/`\right` for tall delimiters**: Or overusing them where fixed-size delimiters suffice. Use `\bigl`/`\bigr` for manual sizing when `\left`/`\right` produce poor results.

5. **Breaking equations at wrong points**: Break before binary operators (`+`, `-`), not after, per standard mathematical typesetting convention.

### 10.3 In Implementation

1. **Not reducing to first order**: Implementing a second-order ODE directly instead of converting to the standard first-order form that solvers expect.

2. **Ignoring solver requirements**: Using an explicit integrator for a stiff system, or a non-DAE solver for a DAE system.

3. **Dimensional quantities in code**: Passing physical quantities with units into a nondimensional solver without proper scaling, or mixing SI and CGS units.

4. **Sign convention errors**: The most common source of bugs. Be explicit about whether forces are positive in the direction of motion or against it. Document your sign convention.

---

## 11. Presentation Order for a Complete System

The standard order, used consistently by Strogatz (Nonlinear Dynamics and Chaos), Hairer/Norsett/Wanner (Solving ODEs I/II), Evans (Partial Differential Equations), and LeVeque (Numerical Methods for Conservation Laws):

### For an ODE System

1. State the physical problem in words (one sentence)
2. Define the state variables and their meaning
3. Write the governing equation(s)
4. State initial conditions
5. Define all parameters (table with symbol, description, units, default values)
6. State auxiliary relationships (constitutive laws, force models)
7. Note any constraints, conservation laws, or symmetries

### For a PDE System

1. State the physical problem in words
2. Define the domain $\Omega$ and its boundary $\partial\Omega$
3. Define the unknown(s) and function spaces
4. Write the governing equation(s) with "in $\Omega$" or "in $\Omega \times (0,T]$"
5. State boundary conditions with "on $\Gamma_D$", "on $\Gamma_N$"
6. State initial conditions with "in $\Omega$" (for time-dependent problems)
7. Define all coefficients, source terms, and parameters
8. State well-posedness assumptions if relevant

### For a Computational Method

1. Reference the continuous problem (by equation number)
2. Describe the discretization (spatial mesh, time stepping)
3. Write the discrete system
4. State the solver and convergence criteria
5. Note stability/CFL conditions if applicable

---

## 12. Summary Table: Which Form to Use When

| Situation | Recommended Form | Why |
|-----------|-----------------|-----|
| Standard ODE simulation | Explicit first-order $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, t)$ | Universal solver interface |
| Mechanical systems | Second-order $M\ddot{\mathbf{q}} = \mathbf{f}$ then reduce | Natural physics, then convert for solver |
| Energy-conserving simulation | Hamiltonian $\dot{\mathbf{q}} = H_{\mathbf{p}}$, $\dot{\mathbf{p}} = -H_{\mathbf{q}}$ | Enables symplectic integrators |
| Constrained mechanics | DAE: $M\ddot{\mathbf{q}} = \mathbf{f} - G^T\boldsymbol{\lambda}$, $\mathbf{g}(\mathbf{q}) = \mathbf{0}$ | Preserves constraints exactly |
| Finite element analysis | Weak form: $a(u,v) = \ell(v)$ | Required by FE software |
| Shock/wave propagation | Conservation law: $\mathbf{U}_t + \mathbf{F}(\mathbf{U})_x = \mathbf{0}$ | Correct weak solutions, Rankine-Hugoniot |
| Semi-discrete (MOL) | $\dot{\mathbf{U}} = L_h \mathbf{U}$ | Spatial discretization to ODE |
| Multi-physics coupling | Block operator form + splitting | Modular solvers |

---

## Sources

### Primary
- Evans, L.C. *Partial Differential Equations* (AMS, GSM 19). Standard PDE reference.
  - [AMS Bookstore](https://bookstore.ams.org/gsm-19-r)
  - [Berkeley lecture notes](https://math.berkeley.edu/~evans/evans_pcam.pdf)
- Hairer, E., Norsett, S.P., Wanner, G. *Solving Ordinary Differential Equations I/II*. Standard ODE reference, especially for stiff systems and DAEs.
- LeVeque, R.J. *Numerical Methods for Conservation Laws*. Standard conservation law reference.
  - [PDF](http://tevza.org/home/course/modelling-II_2016/books/Leveque%20-%20Numerical%20Methods%20for%20Conservation%20Laws.pdf)
- Strogatz, S. *Nonlinear Dynamics and Chaos*. Standard reference for qualitative ODE analysis.
- Moser, S.M. *How to Typeset Equations in LaTeX*. ETH Zurich guide for equation typesetting.
  - [PDF](https://moser-isi.ethz.ch/docs/typeset_equations.pdf)

### Domain-Specific
- [Elastica/PyElastica Cosserat Rod Theory](https://www.cosseratrods.org/cosserat_rods/theory/): Cosserat rod governing equations, notation, and constitutive models
- [DAE formulation on Scholarpedia](http://www.scholarpedia.org/article/Differential-algebraic_equations): DAE index theory and standard forms
- [COMSOL Weak Form Introduction](https://www.comsol.com/blogs/brief-introduction-weak-form): Accessible overview of strong vs. weak form
- [Nondimensionalization (UW Madison)](https://people.math.wisc.edu/~angenent/519.2016s/notes/non-dimensionalization.html): Systematic nondimensionalization procedure
- [State-Space Representation (Swarthmore)](https://lpsa.swarthmore.edu/Representations/SysRepSS.html): LTI state-space conventions
- [MIT State-Space Handout](http://web.mit.edu/2.14/www/Handouts/StateSpace.pdf): Standard $(A,B,C,D)$ notation

### LaTeX
- [IEEE Math Typesetting Guide](https://conferences.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/8/IEEE-Math-Typesetting-Guide-for-LaTeX-Users.pdf): IEEE conventions for mathematical typesetting
- [MFEM Weak Formulations](https://mfem.org/fem_weak_form/): Practical weak form reference for FE implementation
