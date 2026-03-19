---
name: Clean PDE System Framework for PINNs
description: Math-focused reference covering canonical PDE forms, PINN loss construction, concrete examples (Navier-Stokes, Cosserat rod), Python pseudocode, and framework comparison
type: knowledge
created: 2026-03-19
updated: 2026-03-19
tags: [pinn, pde, physics-informed, framework, deep-learning, surrogate]
aliases: [pde-system, pinn-framework]
---

# Clean PDE System Framework for PINNs

## 1. Core Abstraction

A PINN requires three components:

| Component | Symbol | Role |
|-----------|--------|------|
| Residual function | $\mathcal{R}(\mathbf{u}; \boldsymbol{\lambda})$ | PDE operator evaluated on network output |
| Boundary conditions | $\mathcal{B}(\mathbf{u}) = \mathbf{g}$ | Constraints on $\partial\Omega$ |
| Domain | $\Omega \subset \mathbb{R}^d$ | Spatial-temporal region of interest |

The neural network $\hat{\mathbf{u}}_\theta(\mathbf{x})$ approximates the solution. Training minimizes the PDE residual at collocation points sampled from $\Omega$.

$$
\hat{\mathbf{u}}_\theta : \Omega \to \mathbb{R}^m, \quad \mathbf{x} \mapsto \hat{\mathbf{u}}
$$

Derivatives $\nabla \hat{\mathbf{u}}, \nabla^2 \hat{\mathbf{u}}, \ldots$ computed via automatic differentiation.

## 2. Canonical Form

### General PDE System

$$
\mathcal{N}_i[\mathbf{u}; \boldsymbol{\lambda}](\mathbf{x}) = f_i(\mathbf{x}), \quad \mathbf{x} \in \Omega, \quad i = 1, \ldots, N_{\text{eq}}
$$

$$
\mathcal{B}_j[\mathbf{u}](\mathbf{x}) = g_j(\mathbf{x}), \quad \mathbf{x} \in \partial\Omega, \quad j = 1, \ldots, N_{\text{bc}}
$$

$$
\mathcal{I}_k[\mathbf{u}](\mathbf{x}, t=0) = h_k(\mathbf{x}), \quad k = 1, \ldots, N_{\text{ic}}
$$

Rewrite in residual form:

$$
\mathcal{R}_i(\mathbf{x}, \mathbf{u}, \nabla\mathbf{u}, \nabla^2\mathbf{u}, \ldots) = 0
$$

### PINN Loss

$$
\mathcal{L} = w_r \mathcal{L}_r + w_b \mathcal{L}_b + w_i \mathcal{L}_i + w_d \mathcal{L}_d
$$

| Term | Definition | Collocation set |
|------|-----------|----------------|
| $\mathcal{L}_r$ | $\frac{1}{N_r}\sum_{k=1}^{N_r}\sum_{i=1}^{N_{\text{eq}}} \|\mathcal{R}_i(\mathbf{x}_k)\|^2$ | Interior points $\{\mathbf{x}_k\} \subset \Omega$ |
| $\mathcal{L}_b$ | $\frac{1}{N_b}\sum_{k=1}^{N_b}\sum_{j} \|\mathcal{B}_j(\mathbf{x}_k) - g_j\|^2$ | Boundary points $\{\mathbf{x}_k\} \subset \partial\Omega$ |
| $\mathcal{L}_i$ | $\frac{1}{N_i}\sum_{k=1}^{N_i}\sum_{l} \|\mathcal{I}_l(\mathbf{x}_k) - h_l\|^2$ | Initial condition points |
| $\mathcal{L}_d$ | $\frac{1}{N_d}\sum_{k=1}^{N_d} \|\hat{\mathbf{u}}(\mathbf{x}_k) - \mathbf{u}_k^*\|^2$ | Observation/data points |

Weights $w_r, w_b, w_i, w_d$ balance loss terms. Common strategies: fixed, NTK-based, inverse Dirichlet, self-adaptive.

## 2.1 Worked Example: PDE → Residual Form

**Given:** 2D unsteady advection-diffusion with Dirichlet BC and initial condition.

### Step 1: Write the PDE system in standard form

$$
\underbrace{\frac{\partial c}{\partial t} + \mathbf{v} \cdot \nabla c = D\nabla^2 c}_{\text{PDE}}, \quad
\underbrace{c = 0 \;\text{ on }\; \partial\Omega}_{\text{BC}}, \quad
\underbrace{c(x,y,0) = c_0(x,y)}_{\text{IC}}
$$

### Step 2: Identify the three components

| Component | Standard form | Meaning |
|-----------|--------------|---------|
| PDE operator $\mathcal{N}$ | $c_t + v_x c_x + v_y c_y - D(c_{xx} + c_{yy}) = 0$ | Governing equation |
| BC operator $\mathcal{B}$ | $c = 0$ on $\partial\Omega$ | Boundary constraint |
| IC operator $\mathcal{I}$ | $c(x,y,0) = c_0(x,y)$ | Initial state |

### Step 3: Convert each to residual form ($= 0$)

$$
\mathcal{R}_{\text{PDE}}(x, y, t) = c_t + v_x c_x + v_y c_y - D(c_{xx} + c_{yy})
$$

$$
\mathcal{R}_{\text{BC}}(x, y, t) = c(x, y, t) - 0 = c(x, y, t) \quad \text{on } \partial\Omega
$$

$$
\mathcal{R}_{\text{IC}}(x, y) = c(x, y, 0) - c_0(x, y)
$$

### Step 4: Assemble the PINN loss

$$
\mathcal{L} = \underbrace{w_r \frac{1}{N_r}\sum_k \mathcal{R}_{\text{PDE}}^2(\mathbf{x}_k)}_{\text{interior}} + \underbrace{w_b \frac{1}{N_b}\sum_k \mathcal{R}_{\text{BC}}^2(\mathbf{x}_k)}_{\text{boundary}} + \underbrace{w_i \frac{1}{N_i}\sum_k \mathcal{R}_{\text{IC}}^2(\mathbf{x}_k)}_{\text{initial}}
$$

### Step 5: Derivative requirements

| Residual | Derivatives needed | AD calls |
|----------|--------------------|----------|
| $\mathcal{R}_{\text{PDE}}$ | $c_t, c_x, c_y, c_{xx}, c_{yy}$ | 1st-order: 3, 2nd-order: 2 |
| $\mathcal{R}_{\text{BC}}$ | none (just $c$) | 0 |
| $\mathcal{R}_{\text{IC}}$ | none (just $c$) | 0 |

### Pattern

For any PDE system, the conversion follows:

$$
\boxed{
\begin{aligned}
&\text{PDE: } \mathcal{N}[u] = f \;\;\longrightarrow\;\; \mathcal{R}_{\text{PDE}} = \mathcal{N}[u] - f \\
&\text{BC: } \mathcal{B}[u] = g \;\;\longrightarrow\;\; \mathcal{R}_{\text{BC}} = \mathcal{B}[u] - g \\
&\text{IC: } u(t{=}0) = h \;\;\longrightarrow\;\; \mathcal{R}_{\text{IC}} = u(t{=}0) - h
\end{aligned}
}
$$

Each residual is zero when the constraint is satisfied. The PINN minimizes $\sum \|\mathcal{R}\|^2$ over collocation points.

## 3. Concrete Example: 2D Navier-Stokes

### Outputs

$$
\hat{\mathbf{u}}_\theta(x, y, t) = \begin{pmatrix} u(x,y,t) \\ v(x,y,t) \\ p(x,y,t) \end{pmatrix}
$$

- $u, v$: velocity components
- $p$: pressure
- Inputs: $(x, y, t) \in \Omega \times [0, T]$

### Residuals

**R1 — x-momentum:**

$$
\mathcal{R}_1 = \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial x} - \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)
$$

**R2 — y-momentum:**

$$
\mathcal{R}_2 = \frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial y} - \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)
$$

**R3 — continuity:**

$$
\mathcal{R}_3 = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}
$$

### Derivative Requirements

| Residual | Required Derivatives | AD Order |
|----------|---------------------|----------|
| $\mathcal{R}_1$ | $u_t, u_x, u_y, u_{xx}, u_{yy}, p_x$ | 2nd |
| $\mathcal{R}_2$ | $v_t, v_x, v_y, v_{xx}, v_{yy}, p_y$ | 2nd |
| $\mathcal{R}_3$ | $u_x, v_y$ | 1st |

Total: 13 unique partial derivatives. AD graph built once, reused across collocation points.

### Mapping to Canonical Form

$$
\mathcal{R}_i(\mathbf{x}, \mathbf{u}, \nabla\mathbf{u}, \nabla^2\mathbf{u}) = 0, \quad i \in \{1, 2, 3\}
$$

- $\mathbf{x} = (x, y, t)$
- $\mathbf{u} = (u, v, p)$
- Parameters $\boldsymbol{\lambda} = (\rho, \nu)$ — known (forward) or learnable (inverse)

## 4. Clean Python Structure

### PDESystem Abstract Class

```python
from abc import ABC, abstractmethod
import torch

class PDESystem(ABC):
    """Abstract PDE system for PINN training."""

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Independent variables, e.g. ['x', 'y', 't']."""

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Dependent variables, e.g. ['u', 'v', 'p']."""

    @property
    @abstractmethod
    def params(self) -> dict[str, float]:
        """Physical parameters, e.g. {'nu': 1e-3, 'rho': 1.0}."""

    @abstractmethod
    def residuals(self, u: dict[str, torch.Tensor],
                  grads: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Compute PDE residuals R_i.

        Args:
            u: Network outputs keyed by output_names.
            grads: Partial derivatives keyed by 'u_x', 'u_xx', etc.

        Returns:
            List of residual tensors, each (N,).
        """

    @abstractmethod
    def boundary_conditions(self, x: dict[str, torch.Tensor],
                            u: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Boundary condition residuals B_j(u) - g_j."""

    @abstractmethod
    def domain(self) -> dict[str, tuple[float, float]]:
        """Domain bounds per input dimension."""
```

### NavierStokes2D Implementation

```python
class NavierStokes2D(PDESystem):
    input_names = ['x', 'y', 't']
    output_names = ['u', 'v', 'p']

    def __init__(self, nu: float = 1e-3, rho: float = 1.0):
        self._params = {'nu': nu, 'rho': rho}

    @property
    def params(self):
        return self._params

    def residuals(self, u, grads):
        nu = self._params['nu']
        rho = self._params['rho']

        R1 = (grads['u_t'] + u['u']*grads['u_x'] + u['v']*grads['u_y']
               + grads['p_x']/rho - nu*(grads['u_xx'] + grads['u_yy']))

        R2 = (grads['v_t'] + u['u']*grads['v_x'] + u['v']*grads['v_y']
               + grads['p_y']/rho - nu*(grads['v_xx'] + grads['v_yy']))

        R3 = grads['u_x'] + grads['v_y']

        return [R1, R2, R3]

    def boundary_conditions(self, x, u):
        # No-slip walls: u=0, v=0 on boundary
        return [u['u'], u['v']]

    def domain(self):
        return {'x': (0.0, 1.0), 'y': (0.0, 1.0), 't': (0.0, 1.0)}
```

### Training Loop Skeleton

```python
def train_pinn(pde: PDESystem, net: torch.nn.Module, epochs: int = 10000):
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    domain = pde.domain()

    for epoch in range(epochs):
        # 1. Sample collocation points
        x_interior = sample_interior(domain, N=2048)
        x_boundary = sample_boundary(domain, N=512)
        x_initial = sample_initial(domain, N=512)

        # 2. Forward pass + AD for derivatives
        x_interior.requires_grad_(True)
        u_hat = net(x_interior)
        grads = compute_derivatives(u_hat, x_interior, pde)

        # 3. Compute losses
        residuals = pde.residuals(u_hat, grads)
        L_r = sum(r.pow(2).mean() for r in residuals)
        L_b = boundary_loss(pde, net, x_boundary)
        L_i = initial_loss(pde, net, x_initial)

        loss = w_r * L_r + w_b * L_b + w_i * L_i

        # 4. Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 5. Existing Frameworks Comparison

| Framework | Language | PDE Spec | AD Backend | Geometry | Inverse Support | Notes |
|-----------|----------|----------|------------|----------|----------------|-------|
| DeepXDE | Python | Symbolic / callable | TensorFlow / PyTorch / JAX | Built-in primitives + CSG | Yes (trainable params) | Most mature; good docs |
| NeuralPDE.jl | Julia | Symbolic (ModelingToolkit) | Zygote / ForwardDiff | IntervalSets | Yes | Julia ecosystem; composable |
| NVIDIA Modulus | Python | Config YAML + Python | PyTorch | STL import, primitives | Yes | GPU-optimized; industrial |
| PyDEns | Python | Symbolic (SymPy) | TensorFlow 1.x | Rectangular | Limited | Lightweight; unmaintained |

### Key Differentiators

| Feature | DeepXDE | NeuralPDE.jl | Modulus | PyDEns |
|---------|---------|-------------|---------|--------|
| Hard BC enforcement | Yes | Yes | Yes | No |
| Adaptive collocation (RAR) | Yes | No | Yes | No |
| Multi-GPU training | Via backend | Yes | Yes | No |
| Custom loss weighting | Yes | Yes | Yes | Manual |
| Time-dependent PDEs | Yes | Yes | Yes | Yes |

## 6. Application: Cosserat Rod in Elastica

### 6.1 Well-Posedness Tuple

The Elastica system is fully specified by $\mathcal{S} = (\mathcal{L}, \Omega, \text{BC}, \text{IC}, \boldsymbol{\theta})$:

| Component | Name | Specification |
|-----------|------|---------------|
| $\mathcal{L}$ | Governing equations | Cosserat rod PDEs: conservation of linear and angular momentum with linear elastic constitutive law |
| $\Omega$ | Domain | $s \in [0, L]$, $L = 0.5$ m (arc length) $\times$ $t \in [0, T]$ (time) |
| BC | Boundary conditions | Free–free: $\mathbf{F} = \mathbf{M} = \mathbf{0}$ at $s = 0$ and $s = L$ |
| IC | Initial conditions | Straight rod at rest: $\mathbf{x}(s,0) = \mathbf{x}_0 + s\hat{\mathbf{e}}_x$, $\mathbf{v} = \boldsymbol{\kappa} = \boldsymbol{\omega} = \mathbf{0}$ |
| $\boldsymbol{\theta}$ | Parameters | $E = 10^5$ Pa, $\nu = 0.5$, $\rho = 1200$ kg/m³, $R = 0.02$ m, $c_t = 0.01$, $c_n = 0.05$, $\gamma = 0.002$ |
| Control | CPG forcing | $\mathbf{a} = (A, f, k, \phi_0, b) \in \mathbb{R}^5$ enters as time-varying rest curvature $\boldsymbol{\kappa}_0(s,t)$ |

### 6.2 Governing PDEs (Continuous)

**PDE 1 — Linear momentum** (force balance on each cross-section):

$$
\rho A \frac{\partial \mathbf{v}}{\partial t} = \frac{\partial \mathbf{F}}{\partial s} + \mathbf{f}_{\text{ext}}
$$

> $\rho A$ = mass per unit length, $\mathbf{F}$ = internal force resultant, $\mathbf{f}_{\text{ext}}$ = external distributed force (friction)

**PDE 2 — Angular momentum** (torque balance on each cross-section):

$$
\rho \mathbf{I} \frac{\partial \boldsymbol{\omega}}{\partial t} = \frac{\partial \mathbf{M}}{\partial s} + \mathbf{m}_{\text{ext}} + \frac{\partial \mathbf{x}}{\partial s} \times \mathbf{F}
$$

> $\rho \mathbf{I}$ = rotational inertia per unit length, $\mathbf{M}$ = internal moment, $\frac{\partial \mathbf{x}}{\partial s} \times \mathbf{F}$ = coupling between shear force and moment

**Constitutive law** (linear elastic closure):

$$
\mathbf{F} = \mathbf{S}(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_0), \qquad \mathbf{M} = \mathbf{B}(\boldsymbol{\kappa} - \boldsymbol{\kappa}_0)
$$

$$
\mathbf{S} = \text{diag}(GA, GA, EA), \qquad \mathbf{B} = \text{diag}(EI, EI, GJ)
$$

> $\mathbf{S}$ = shear/stretch stiffness matrix, $\mathbf{B}$ = bend/twist stiffness matrix, $\boldsymbol{\kappa}_0$ = rest curvature (set by CPG)

### 6.3 Semi-Discrete ODE (Staggered Grid)

Spatial discretization on a staggered grid ($N_n = 21$ nodes, $N_e = 20$ elements) converts the PDE to 124 coupled first-order ODEs:

$$
\dot{\mathbf{s}} = \mathbf{g}(\mathbf{s}, \mathbf{a}, t), \qquad \mathbf{s} \in \mathbb{R}^{124}, \quad \mathbf{a} \in \mathbb{R}^5
$$

**State vector layout:**

| Slice | Variable | Count | Meaning |
|-------|----------|-------|---------|
| `[0:21]` | $x_i$ | 21 | Node $x$-positions (m) |
| `[21:42]` | $y_i$ | 21 | Node $y$-positions (m) |
| `[42:63]` | $\dot{x}_i$ | 21 | Node $x$-velocities (m/s) |
| `[63:84]` | $\dot{y}_i$ | 21 | Node $y$-velocities (m/s) |
| `[84:104]` | $\psi_e$ | 20 | Element yaw angle (rad) |
| `[104:124]` | $\omega_{z,e}$ | 20 | Element angular velocity (rad/s) |

### 6.4 Boundary Conditions

Free–free (no external force or moment at either end):

$$
\mathbf{F}(s=0, t) = \mathbf{0}, \qquad \mathbf{F}(s=L, t) = \mathbf{0}
$$

$$
\mathbf{M}(s=0, t) = \mathbf{0}, \qquad \mathbf{M}(s=L, t) = \mathbf{0}
$$

> In discrete form: elastic forces $F^{\text{elastic}} = 0$ at the two boundary nodes ($i = 0$ and $i = N_n - 1$). Boundary nodes use half-mass lumping: $m_{\text{boundary}} = \rho A \Delta\ell / 2$.

### 6.5 Initial Conditions

$$
\mathbf{x}(s, 0) = \mathbf{x}_0 + s\hat{\mathbf{e}}_x \quad \text{(straight rod along } x\text{-axis)}
$$

$$
\mathbf{v}(s, 0) = \mathbf{0}, \qquad \boldsymbol{\kappa}(s, 0) = \mathbf{0}, \qquad \boldsymbol{\omega}(s, 0) = \mathbf{0}
$$

> All velocities and curvatures start at zero. The rod is straight and at rest.

### 6.6 Differentiable Right-Hand Side (2D Planar)

For the PINN, an inextensible rod approximation is used (bending only, no stretching). The RHS computes $\dot{\mathbf{x}} = f(\mathbf{x})$ for all 124 state dimensions.

**Kinematic equations** (trivial — velocity equals rate of change of position):

$$
\dot{x}_i = v_{x,i}, \qquad \dot{y}_i = v_{y,i}, \qquad \dot{\psi}_e = \omega_{z,e}
$$

> For $i = 0, \ldots, N_n - 1$ (nodes) and $e = 0, \ldots, N_e - 1$ (elements).

**Discrete curvature** at internal joint $j$ (connecting elements $e = j$ and $e = j+1$):

$$
\kappa_j = \frac{\psi_{j+1} - \psi_j}{\Delta\ell}, \qquad j = 0, \ldots, N_e - 2
$$

> $\Delta\ell = L / N_e$ is the element length.

**Bending moment:**

$$
M_j = EI \frac{\psi_{j+1} - \psi_j}{\Delta\ell}
$$

> $EI$ = bending stiffness, $I = \pi r^4 / 4$ = second moment of area.

**Elastic torque** on each element (moment gradient):

$$
\tau_e^{\text{elastic}} =
\begin{cases}
M_0 / \Delta\ell, & e = 0 \text{ (first element)} \\
(M_e - M_{e-1}) / \Delta\ell, & 1 \leq e \leq N_e - 2 \\
-M_{N_e-2} / \Delta\ell, & e = N_e - 1 \text{ (last element)}
\end{cases}
$$

**Anisotropic RFT friction forces:**

Element tangent (regularized normalization for smooth gradients):

$$
\hat{t}_{x,e} = \frac{x_{e+1} - x_e}{\sqrt{(x_{e+1} - x_e)^2 + (y_{e+1} - y_e)^2 + \varepsilon^2}}, \qquad \varepsilon = 10^{-6}
$$

Node tangents by averaging neighboring elements:

$$
\hat{t}_{x,i}^{\text{node}} =
\begin{cases}
\hat{t}_{x,0}, & i = 0 \\
\frac{1}{2}(\hat{t}_{x,i-1} + \hat{t}_{x,i}), & 1 \leq i \leq N_n - 2 \\
\hat{t}_{x,N_e-1}, & i = N_n - 1
\end{cases}
$$

Velocity decomposition into tangential and normal components:

$$
v_{\parallel,i} = v_{x,i} \hat{t}_{x,i}^{\text{node}} + v_{y,i} \hat{t}_{y,i}^{\text{node}}
$$

$$
\mathbf{v}_{\text{tan},i} = v_{\parallel,i} \hat{\mathbf{t}}_i^{\text{node}}, \qquad \mathbf{v}_{\text{norm},i} = \mathbf{v}_i - \mathbf{v}_{\text{tan},i}
$$

RFT friction force at each node:

$$
F_{x,i}^{\text{fric}} = -c_t v_{\text{tan},x,i} - c_n v_{\text{norm},x,i}, \qquad F_{y,i}^{\text{fric}} = -c_t v_{\text{tan},y,i} - c_n v_{\text{norm},y,i}
$$

> $c_t$ = tangential drag, $c_n$ = normal drag. Anisotropy ratio $c_n / c_t = 5$ is what enables snake-like locomotion.

**Elastic bending forces** at internal joints:

$$
F_{x,i}^{\text{elastic}} = \frac{M_j \hat{n}_{x,j}}{\Delta\ell^2}, \qquad F_{y,i}^{\text{elastic}} = \frac{M_j \hat{n}_{y,j}}{\Delta\ell^2}
$$

> Normal direction: $\hat{n}_{x,j} = -\hat{t}_{y,j}^{\text{joint}}$, $\hat{n}_{y,j} = \hat{t}_{x,j}^{\text{joint}}$ (90° rotation of joint tangent). Zero at boundary nodes.

**Newton's second law** (translational acceleration):

$$
\dot{v}_{x,i} = \frac{F_{x,i}^{\text{fric}} + F_{x,i}^{\text{elastic}}}{m_i}, \qquad \dot{v}_{y,i} = \frac{F_{y,i}^{\text{fric}} + F_{y,i}^{\text{elastic}}}{m_i}
$$

> $m_i = \rho A \Delta\ell$ (interior), $m_i = \rho A \Delta\ell / 2$ (boundary, half-mass lumping).

**Angular momentum balance:**

$$
\dot{\omega}_{z,e} = \frac{\tau_e^{\text{elastic}}}{\rho I \Delta\ell}
$$

**Complete RHS:**

$$
f(\mathbf{x}) = (\dot{x}_i, \dot{y}_i, \dot{v}_{x,i}, \dot{v}_{y,i}, \dot{\psi}_e, \dot{\omega}_{z,e}) \in \mathbb{R}^{124}
$$

### 6.7 Residual Form for PINN

Converting the six ODE blocks to residual form ($\mathcal{R} = 0$):

| Residual | Equation | AD derivatives needed |
|----------|----------|-----------------------|
| $\mathcal{R}_1$ | $\dot{x}_i - v_{x,i} = 0$ | $\partial x_i / \partial t$ |
| $\mathcal{R}_2$ | $\dot{y}_i - v_{y,i} = 0$ | $\partial y_i / \partial t$ |
| $\mathcal{R}_3$ | $\dot{\psi}_e - \omega_{z,e} = 0$ | $\partial \psi_e / \partial t$ |
| $\mathcal{R}_4$ | $m_i \dot{v}_{x,i} - F_{x,i}^{\text{fric}} - F_{x,i}^{\text{elastic}} = 0$ | $\partial v_{x,i} / \partial t$ |
| $\mathcal{R}_5$ | $m_i \dot{v}_{y,i} - F_{y,i}^{\text{fric}} - F_{y,i}^{\text{elastic}} = 0$ | $\partial v_{y,i} / \partial t$ |
| $\mathcal{R}_6$ | $\rho I \Delta\ell \dot{\omega}_{z,e} - \tau_e^{\text{elastic}} = 0$ | $\partial \omega_{z,e} / \partial t$ |

> $\mathcal{R}_1$–$\mathcal{R}_3$ are kinematic (1st-order AD only). $\mathcal{R}_4$–$\mathcal{R}_6$ are dynamic (1st-order AD on velocities, but forces depend on spatial derivatives of position/angle which are handled by the discrete stencil, not AD).

### 6.8 PINN Loss Function

**Standard PINN loss** (three terms):

$$
\mathcal{L}(\theta) = \lambda_{\text{IC}} \mathcal{L}_{\text{IC}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{data}} \mathcal{L}_{\text{data}}
$$

**Initial condition loss:**

$$
\mathcal{L}_{\text{IC}} = \| \mathbf{h}_\theta(\mathbf{x}_0, \mathbf{u}, 0) - \mathbf{x}_0 \|^2
$$

> Penalizes deviation from known starting state at $t = 0$.

**Physics residual loss** at $N_c$ collocation points $\{t_k\} \subset [0, T_s]$:

$$
\mathcal{L}_{\text{phys}} = \frac{1}{N_c} \sum_{k=1}^{N_c} \left\| \frac{\partial \hat{\mathbf{x}}_{t_k}}{\partial t} - f(\hat{\mathbf{x}}_{t_k}, \mathbf{u}_{t_k}) \right\|^2
$$

> Each collocation point evaluates: (autodiff derivative of network output) minus (physics RHS at predicted state). The residual is zero when the network output satisfies the ODE.

**Data loss** against simulator trajectories:

$$
\mathcal{L}_{\text{data}} = \frac{1}{N_d} \sum_{k=1}^{N_d} \| \hat{\mathbf{x}}_{t_k} - \mathbf{x}_{\text{data},k} \|^2
$$

> Optional — PINNs can train with $N_d = 0$ if physics is fully specified.

### 6.9 DD-PINN Loss (Ansatz-Based)

The DD-PINN removes time as a network input. Instead, the network outputs ansatz parameters:

$$
\hat{\mathbf{x}}_t = \mathbf{g}(\mathbf{a}, t) + \mathbf{x}_0, \qquad \mathbf{a} = f_\theta(\mathbf{x}_0, \mathbf{u}_0)
$$

> $\mathbf{g}(\mathbf{a}, 0) \equiv \mathbf{0}$ by construction, so $\hat{\mathbf{x}}_0 = \mathbf{x}_0$ exactly. No IC loss needed.

**Ansatz function** (damped sinusoidal basis, per state variable $j$):

$$
g_j(t) = \sum_{i=1}^{n_g} \alpha_{ij} e^{-\delta_{ij} t} [\sin(\beta_{ij} t + \gamma_{ij}) - \sin(\gamma_{ij})]
$$

> $\alpha$ = amplitude, $\beta$ = frequency, $\gamma$ = phase offset, $\delta \geq 0$ = damping rate. Network outputs $\mathbf{a} = (\boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\gamma}, \boldsymbol{\delta}) \in \mathbb{R}^{4 m n_g}$.

**Closed-form time derivative** (no autodiff needed):

$$
\dot{g}_j(t) = \sum_{i=1}^{n_g} \alpha_{ij} e^{-\delta_{ij} t} [\beta_{ij} \cos(\beta_{ij} t + \gamma_{ij}) - \delta_{ij} \sin(\beta_{ij} t + \gamma_{ij})]
$$

**DD-PINN loss** (two terms only, since $\mathcal{L}_{\text{IC}} \equiv 0$):

$$
\mathcal{L}(\theta) = \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{data}} \mathcal{L}_{\text{data}}
$$

**Physics residual** (ansatz derivative vs. Cosserat RHS):

$$
\mathcal{L}_{\text{phys}} = \frac{1}{N_c} \sum_{k=1}^{N_c} \| \dot{\mathbf{g}}(\mathbf{a}, t_k) - f(\mathbf{g}(\mathbf{a}, t_k) + \mathbf{x}_0, \mathbf{u}_{t_k}) \|^2
$$

> $\dot{\mathbf{g}}$ is computed analytically (no computational graph). The network is called once to produce $\mathbf{a}$; the ansatz is evaluated at $N_c$ collocation points using elementary operations.

**Data loss** (predicted delta vs. ground-truth transition):

$$
\mathcal{L}_{\text{data}} = \frac{1}{N_d} \sum_{k=1}^{N_d} \| \mathbf{g}(\mathbf{a}^{(k)}, T_s) - (\mathbf{x}_{T_s}^{(k)} - \mathbf{x}_0^{(k)}) \|^2
$$

> Compares predicted state delta at end of control interval against simulator ground truth.

### 6.10 Physics-Regularized Surrogate Loss (Algebraic Constraints)

No differentiable simulator needed. Constraints derived from kinematic identities on $\Delta\mathbf{x} = \mathbf{x}_{t+1} - \mathbf{x}_t$. All use trapezoidal integration.

**C1 — Translational kinematic consistency:**

$$
\mathcal{L}_{\text{kin},x} = \frac{1}{N_n} \sum_{i=0}^{N_n-1} \left( \Delta x_i - (v_{x,i} + \tfrac{1}{2} \Delta v_{x,i}) \Delta t \right)^2
$$

$$
\mathcal{L}_{\text{kin},y} = \frac{1}{N_n} \sum_{i=0}^{N_n-1} \left( \Delta y_i - (v_{y,i} + \tfrac{1}{2} \Delta v_{y,i}) \Delta t \right)^2
$$

> Position change must equal integrated velocity. Trapezoidal: uses average velocity $\bar{v} = v + \frac{1}{2}\Delta v$.

**C2 — Angular kinematic consistency:**

$$
\mathcal{L}_{\text{ang}} = \frac{1}{N_e} \sum_{e=0}^{N_e-1} \left( \Delta\psi_e - (\omega_{z,e} + \tfrac{1}{2} \Delta\omega_{z,e}) \Delta t \right)^2
$$

> Yaw change must equal integrated angular velocity.

**C3 — Curvature–moment consistency** (weighted $0.1\times$):

$$
\mathcal{L}_{\text{curv}} = 0.1 \cdot \frac{1}{N_e - 1} \sum_{j=0}^{N_e-2} \left( \frac{\psi_{j+1}' - \psi_j'}{\Delta\ell} - \frac{\psi_{j+1} - \psi_j}{\Delta\ell} - \frac{\Delta\omega_{z,j+1} - \Delta\omega_{z,j}}{\Delta\ell} \Delta t \right)^2
$$

> Spatial curvature change should be consistent with the spatial gradient of predicted angular velocity change.

**C4 — Energy bound:**

$$
\mathcal{L}_{\text{energy}} = \text{ReLU}(|\Delta\text{KE}| - \tau)^2
$$

> Penalizes kinetic energy changes exceeding threshold $\tau$. Only activates for unphysically large energy jumps.

**Combined regularizer loss:**

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{kin},x} + \mathcal{L}_{\text{kin},y} + \mathcal{L}_{\text{ang}} + \mathcal{L}_{\text{curv}} + \mathcal{L}_{\text{energy}}
$$

### 6.11 Comparison: Three Physics-Informed Approaches

| | Standard PINN | DD-PINN | Physics Regularizer |
|---|---|---|---|
| Differentiable simulator | Required | Required | Not needed |
| IC enforcement | Soft (penalty) | Hard (ansatz) | N/A (single-step) |
| Time derivative | Autodiff (expensive) | Closed-form (cheap) | N/A |
| Collocation points | ~100–500 | ~1000–5000 (cheap) | N/A |
| Physics fidelity | Full Cosserat RHS | Full Cosserat RHS | Kinematic + energy only |
| Implementation | `src/pinn/` | `src/pinn/` | `src/pinn/` |

### 6.12 3D vs 2D Simplifications

| Full 3D Cosserat | 2D Planar (this project) |
|------------------|--------------------------|
| $\mathbf{r} \in \mathbb{R}^3$ | $(x, y)$ only |
| Quaternion $\mathbf{q} \in S^3$ | Yaw angle $\psi$ |
| 13 output dims per node | 6 output dims per node |
| Stretching + shearing + bending + torsion | Bending only (inextensible) |
| Full Cosserat constitutive law | $M = EI \cdot \kappa$ |
| Director-based curvature on Voronoi domain | Finite-difference curvature $(\psi_{j+1} - \psi_j) / \Delta\ell$ |

### 6.13 Connection to Existing Implementation

| File | Class/Function | Role |
|------|---------------|------|
| `physics_residual.py` | `CosseratRHS` | Differentiable $f(\mathbf{x})$ for 2D rod (bending + RFT friction) |
| `ansatz.py` | `DampedSinusoidalAnsatz` | Hard IC satisfaction via $g(\mathbf{a}, 0) = 0$ |
| `models.py` | `DDPINNModel` | Fourier features + MLP + ansatz |
| `collocation.py` | Sobol/uniform + RAR | Residual-adaptive refinement of collocation points |

## 7. Learning Extensions

### Constraint Enforcement

- **Soft constraints** — penalty terms in loss (standard PINN approach)
- **Hard constraints** — ansatz design guaranteeing $g(\mathbf{a}, 0) = 0$ (see `DampedSinusoidalAnsatz`)
- **Lagrange multipliers** — augmented Lagrangian for strict constraint satisfaction

### Problem Variants

- **Inverse problems** — estimate $\boldsymbol{\lambda}$ (material params) from data; make params `nn.Parameter`
- **Transfer learning** — pre-train on simple geometry, fine-tune on complex
- **Domain decomposition (DD-PINN)** — split $\Omega$ into subdomains, train local networks, enforce interface continuity
- **Adaptive collocation** — RAR concentrates points where $|\mathcal{R}|$ is large (implemented in `collocation.py`)
- **Curriculum training** — start with coarse resolution / simple BCs, progressively increase complexity
- **Multi-fidelity** — combine sparse high-fidelity data with cheap low-fidelity simulation

### Neural Operators (Beyond PINNs)

- **DeepONet** — learn operator $\mathcal{G}: f \mapsto u$ mapping forcing to solution
- **FNO (Fourier Neural Operator)** — spectral convolution in Fourier space; resolution-independent
- **Both** are mesh-free and can incorporate physics loss as regularizer

### DeepXDE Walkthrough

```python
import deepxde as dde

# 1. Define geometry + time
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Define PDE residual
def navier_stokes(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    du = dde.grad.jacobian(u, x)  # (N, 3, 3)
    d2u = dde.grad.hessian(u, x, component=0)
    # ... assemble R1, R2, R3
    return [R1, R2, R3]

# 3. Define BCs
bc_u = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on: on, component=0)
bc_v = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on: on, component=1)

# 4. Assemble + train
data = dde.data.TimePDE(geomtime, navier_stokes, [bc_u, bc_v],
                         num_domain=2048, num_boundary=512)
net = dde.maps.FNN([3] + [64]*4 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(epochs=20000)
```

### Cosserat Rod as PINN

Mapping the 2D planar rod to the PINN framework:

```python
class CosseratRod2D(PDESystem):
    input_names = ['s', 't']
    output_names = ['x', 'y', 'vx', 'vy', 'theta', 'omega']

    def __init__(self, E=2e6, rho=1200, r=0.001, L=1.0, n_elem=20):
        self._params = {'E': E, 'rho': rho, 'r': r, 'L': L, 'n_elem': n_elem}

    def residuals(self, u, grads):
        # R1: dx/dt = vx  =>  grads['x_t'] - u['vx'] = 0
        R1 = grads['x_t'] - u['vx']
        # R2: dy/dt = vy
        R2 = grads['y_t'] - u['vy']
        # R3: dtheta/dt = omega
        R3 = grads['theta_t'] - u['omega']
        # R4: rho*A*dvx/dt = dFx/ds + f_friction_x
        R4 = self.rho_A * grads['vx_t'] - grads['Fx_s'] - self.f_friction_x(u)
        # R5: rho*A*dvy/dt = dFy/ds + f_friction_y
        R5 = self.rho_A * grads['vy_t'] - grads['Fy_s'] - self.f_friction_y(u)
        # R6: rho*I*domega/dt = dM/ds
        R6 = self.rho_I * grads['omega_t'] - grads['M_s']
        return [R1, R2, R3, R4, R5, R6]

    def domain(self):
        return {'s': (0.0, self._params['L']), 't': (0.0, 0.5)}
```
