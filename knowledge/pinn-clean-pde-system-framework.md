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

## 6. Application: Cosserat Rod

### State Variables

Inputs: $(s, t)$ where $s \in [0, L]$ is arc length, $t \in [0, T]$.

| Output | Symbol | Dimension | Description |
|--------|--------|-----------|-------------|
| Position | $\mathbf{r}(s,t)$ | 3 | Centerline position $(r_1, r_2, r_3)$ |
| Orientation | $\mathbf{q}(s,t)$ | 4 | Unit quaternion $(q_0, q_1, q_2, q_3)$ |
| Velocity | $\mathbf{v}(s,t)$ | 3 | Translational velocity |
| Angular velocity | $\boldsymbol{\omega}(s,t)$ | 3 | Body-frame angular velocity |

Full output: $\hat{\mathbf{u}}_\theta(s,t) \in \mathbb{R}^{13}$.

### Residuals

**R1 — Linear momentum:**

$$
\rho A \frac{\partial \mathbf{v}}{\partial t} = \frac{\partial \mathbf{n}}{\partial s} + \mathbf{f}_{\text{ext}}
$$

**R2 — Angular momentum:**

$$
\rho \mathbf{I} \frac{\partial \boldsymbol{\omega}}{\partial t} = \frac{\partial \mathbf{m}}{\partial s} + \frac{\partial \mathbf{r}}{\partial s} \times \mathbf{n} + \boldsymbol{\tau}_{\text{ext}}
$$

**R3 — Kinematic compatibility:**

$$
\frac{\partial \mathbf{r}}{\partial t} = \mathbf{v}, \qquad \frac{\partial \mathbf{q}}{\partial t} = \frac{1}{2}\boldsymbol{\omega} \otimes \mathbf{q}
$$

### Quaternion Constraint

$$
\|\mathbf{q}\|^2 = q_0^2 + q_1^2 + q_2^2 + q_3^2 = 1
$$

Added as penalty: $\mathcal{L}_{\text{quat}} = \frac{1}{N}\sum_k (\|\mathbf{q}_k\|^2 - 1)^2$.

### Challenges

| Challenge | Description |
|-----------|-------------|
| SO(3) manifold | Quaternion normalization constraint; singularity-free but adds algebraic constraint |
| Material frame | Directors $\mathbf{d}_i$ must remain orthonormal; difficult to enforce softly |
| Stiffness | High bending stiffness $EI$ creates stiff ODEs; collocation needs fine temporal resolution |
| Coupled fields | Position, orientation, velocity, angular velocity all coupled through constitutive laws |

### Connection to Existing Implementation

The `src/pinn/` package implements a **DD-PINN** (data-driven PINN) variant for the 2D planar snake:

- `physics_residual.py` — `CosseratRHS`: differentiable $f(\mathbf{x})$ for 2D rod (bending + RFT friction)
- `ansatz.py` — `DampedSinusoidalAnsatz`: hard IC satisfaction via $g(\mathbf{a}, 0) = 0$
- `models.py` — `DDPINNModel`: Fourier features + MLP + ansatz
- `collocation.py` — Sobol/uniform sampling + RAR (residual-adaptive refinement)

Simplifications in the 2D implementation vs. full 3D Cosserat:

| Full 3D | 2D Planar (this project) |
|---------|-------------------------|
| $\mathbf{r} \in \mathbb{R}^3$ | $(x, y)$ only |
| Quaternion $\mathbf{q} \in S^3$ | Yaw angle $\theta$ |
| 13 output dims per node | 6 output dims per node |
| Stretching + shearing + bending + torsion | Bending only (inextensible) |
| Full Cosserat constitutive law | $M = EI \cdot \kappa$ |

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
