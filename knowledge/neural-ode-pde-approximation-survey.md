---
name: Neural Network Approximation of ODE/PDE Systems
description: Survey of methods for using neural networks to approximate, solve, or learn ODE and PDE systems — PINNs, Neural ODEs, Neural Operators, surrogate modeling, differentiable physics
type: knowledge
created: 2026-03-16
updated: 2026-03-16
tags:
  - neural-ode
  - pinn
  - neural-operator
  - surrogate-model
  - differentiable-physics
  - deep-learning
  - pde
  - ode
aliases:
  - neural ODE survey
  - PINN survey
  - neural PDE solver
---

# Neural Network Approximation of ODE/PDE Systems

A survey of methods for using neural networks to approximate, solve, or learn ODE and PDE systems, with emphasis on relevance to snake robot HRL with Cosserat rod dynamics, CPG oscillator ODEs, and contact mechanics.

---

## Table of Contents

1. [Physics-Informed Neural Networks (PINNs)](#1-physics-informed-neural-networks-pinns)
2. [Neural ODEs](#2-neural-odes)
3. [Neural Operators](#3-neural-operators)
4. [Surrogate Modeling](#4-surrogate-modeling)
5. [Differentiable Physics / Differentiable Simulation](#5-differentiable-physics--differentiable-simulation)
6. [Graph Neural Networks for Physics](#6-graph-neural-networks-for-physics)
7. [Structure-Preserving Neural Networks](#7-structure-preserving-neural-networks)
8. [Applications to Robotics and Soft Body Dynamics](#8-applications-to-robotics-and-soft-body-dynamics)
9. [Software Libraries](#9-software-libraries)
10. [Common Pitfalls](#10-common-pitfalls)
11. [Open Problems and Active Research](#11-open-problems-and-active-research)
12. [Relevance to Snake Robot HRL Project](#12-relevance-to-snake-robot-hrl-project)
13. [Maturity Assessment](#13-maturity-assessment)
14. [Sources](#14-sources)

---

## 1. Physics-Informed Neural Networks (PINNs)

### 1.1 Seminal Work

**Raissi, Perdikaris, and Karniadakis (2019)** -- "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations," *Journal of Computational Physics*.

**Core idea:** Train a neural network $u_\theta(\mathbf{x}, t)$ to satisfy a PDE by incorporating the PDE residual into the loss function:

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{IC}} \mathcal{L}_{\text{IC}}
$$

where $\mathcal{L}_{\text{PDE}} = \frac{1}{N_r}\sum_{i=1}^{N_r} \|\mathcal{L}[u_\theta](\mathbf{x}_i, t_i) - f(\mathbf{x}_i, t_i)\|^2$ is computed at collocation points using automatic differentiation.

**Contributions:**
- Meshfree solution of forward and inverse PDE problems
- Simultaneous solution and parameter estimation
- Works with noisy/sparse data
- No discretization of the domain required

### 1.2 Current State of the Art (2024--2025)

**Key advances:**

- **Causal PINNs (Wang, Perdikaris, 2024):** Enforce temporal causality during training to prevent propagation failures. Vanilla PINNs can minimize PDE residuals at later times even when earlier predictions are wrong.

- **Failure-Informed Adaptive Sampling (Wu et al., SIAM J. Sci. Comput., 2023):** Dynamically place collocation points in regions of high PDE residual, addressing the critical problem that uniform sampling wastes capacity on easy regions.

- **Retain-Resample-Release (R3) Sampling (Daw et al., ICML 2023):** Incrementally accumulates collocation points in high-residual regions with low computational overhead. An extension (causal R3) respects temporal causality for time-dependent PDEs.

- **Fourier Feature Enhancement (FRES, 2024):** Addresses spectral bias by dynamically generating high-frequency features through Fourier embedding and multi-layer residual correction.

- **Separable PINNs (Cho et al., 2024):** Factorize the network to achieve linear scaling with dimensionality -- for a 4D heat equation, training time reduced from 289 hours to 2.5 hours (100x speedup).

- **Domain Decomposition PINNs (DD-PINN):** Decompose the spatial domain into sub-regions, each handled by a separate smaller network. Particularly effective for Cosserat rod problems (see Section 8).

### 1.3 Known Failure Modes

| Failure Mode | Description | Mitigation |
|---|---|---|
| **Spectral bias** | Networks preferentially learn low-frequency components, struggling with multi-scale or high-frequency solutions | Fourier features, multi-scale architectures, adaptive activation functions |
| **Propagation failures** | Solution fails to propagate from IC/BC into the interior, converging to trivial solutions | Causal training, time-marching PINNs, adaptive sampling |
| **Loss imbalance** | PDE loss and BC/IC loss converge at different rates, causing one to dominate | Adaptive loss weighting (e.g., learning rate annealing, NTK-based balancing) |
| **Optimization difficulty** | Ill-conditioned loss landscape from differential operators | Gradient descent with momentum (GDM), second-order optimizers, curriculum learning |
| **Scalability** | Computational cost grows rapidly with problem dimensionality and complexity | Separable PINNs, domain decomposition, transfer learning |

### 1.4 Strengths and Limitations

**Strengths:**
- Meshfree, can handle complex geometries
- Natural framework for inverse problems
- Works with sparse/noisy data
- Encodes known physics directly

**Limitations:**
- Training is expensive and often unstable
- Poor for time-dependent problems without causal training
- Struggles with sharp gradients, shocks, and multi-scale phenomena
- Accuracy generally inferior to classical solvers for well-posed forward problems
- Each new problem instance requires retraining (unless combined with neural operators)

### 1.5 Relevance to Snake Robot

**MEDIUM-HIGH.** PINNs are directly applicable to learning Cosserat rod static equilibria and steady-state solutions. For dynamic simulation during RL training, PINNs are too slow (each evaluation requires optimization). However, PINNs are excellent for:
- Inverse parameter identification (estimating rod stiffness from trajectory data)
- Generating training data for faster surrogate models
- Solving static/quasi-static sub-problems offline

---

## 2. Neural ODEs

### 2.1 Seminal Work

**Chen, Rubanova, Bettencourt, and Duvenaud (NeurIPS 2018, Best Paper)** -- "Neural Ordinary Differential Equations."

**Core idea:** Parameterize the derivative of the hidden state with a neural network and use an ODE solver for the forward pass:

$$
\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}(t), t), \qquad \mathbf{h}(t_0) = \mathbf{h}_0
$$

The output is $\mathbf{h}(t_1) = \mathbf{h}_0 + \int_{t_0}^{t_1} f_\theta(\mathbf{h}(t), t) \, dt$, computed by a black-box ODE solver.

**Key contributions:**
- Continuous-depth networks as the infinite-depth limit of ResNets
- O(1) memory backpropagation via the adjoint sensitivity method
- Adaptive computation -- the ODE solver automatically adjusts step sizes
- Continuous normalizing flows for generative modeling

### 2.2 The Adjoint Method

Instead of backpropagating through solver steps (memory-intensive, numerically unstable), the adjoint method solves a backward ODE:

$$
\frac{d\mathbf{a}}{dt} = -\mathbf{a}(t)^T \frac{\partial f_\theta}{\partial \mathbf{h}}, \qquad \mathbf{a}(t_1) = \frac{\partial L}{\partial \mathbf{h}(t_1)}
$$

where $\mathbf{a}(t) = \partial L / \partial \mathbf{h}(t)$ is the adjoint state. This gives O(1) memory cost regardless of the number of solver steps.

### 2.3 Current Advances (2024--2025)

- **Temporal Adaptive Batch Normalization (TA-BN, NeurIPS 2024):** Addresses the mismatch between standard BN (designed for discrete layers) and the continuous-time nature of Neural ODEs. Enables stacking more layers within Neural ODEs.

- **ODE-RNN, GRU-ODE, ODE-LSTM:** Replace discrete recurrence with continuous ODE evolution for irregular time series. ODE-RNN (Rubanova et al., NeurIPS 2019) updates a hidden state via an ODE between observations.

- **Hybrid Physics-AI approaches (2025):** Integrate Neural ODEs solved with implicit schemes into differentiable, regionalizable, spatially distributed models for data assimilation.

- **Augmented Neural ODEs (Dupont et al., NeurIPS 2019):** Augment the state space to allow trajectories to cross, addressing the topological limitation that standard Neural ODEs preserve the topology of the input space.

### 2.4 Knowledge-Based Neural ODEs (KNODE)

A critical variant for our project. KNODE augments a known (possibly imperfect) physics model with a learned residual:

$$
\dot{\mathbf{x}} = f_{\text{physics}}(\mathbf{x}, t) + g_\theta(\mathbf{x}, t)
$$

where $f_{\text{physics}}$ encodes known dynamics (e.g., Cosserat rod equations) and $g_\theta$ is a neural network that learns the model error. This is the architecture used in the KNODE-Cosserat paper (Section 8).

### 2.5 Strengths and Limitations

**Strengths:**
- Memory-efficient training via adjoint method
- Adaptive computation (solver adjusts to problem difficulty)
- Natural framework for continuous-time dynamics
- Composable with known physics (KNODE)
- Well-supported in PyTorch (torchdiffeq)

**Limitations:**
- Training can be slow (requires ODE solve in forward and backward pass)
- Adjoint method can be numerically unstable for stiff systems
- Standard formulation limited to ODEs (not directly applicable to PDEs without spatial discretization)
- Stiff systems require implicit solvers, which are more expensive

### 2.6 Relevance to Snake Robot

**HIGH.** Neural ODEs are directly relevant because:
1. The DER formulation is already an ODE system ($M\ddot{\mathbf{q}} = -\partial E/\partial \mathbf{q} + \mathbf{f}_{\text{ext}}$)
2. KNODE allows augmenting the imperfect Cosserat rod model with learned corrections
3. CPG oscillators are already ODEs -- Neural ODE augmentation is natural
4. Compatible with TorchRL training loop via torchdiffeq

---

## 3. Neural Operators

### 3.1 Core Concept

Neural operators learn mappings between function spaces (infinite-dimensional), not between finite-dimensional vectors. They learn the solution operator $\mathcal{G}: \mathcal{U} \to \mathcal{V}$ that maps input functions (e.g., initial conditions, forcing terms, boundary conditions) to output functions (e.g., PDE solutions).

**Key advantage over PINNs:** Once trained, a neural operator generalizes to new problem instances (new ICs, BCs, parameters) without retraining.

### 3.2 DeepONet

**Lu, Jin, Pang, Zhang, and Karniadakis (Nature Machine Intelligence, 2021)** -- "Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators."

**Architecture:** Two sub-networks:
- **Branch network:** Encodes the input function $u$ (evaluated at sensor points)
- **Trunk network:** Encodes the query location $(\mathbf{x}, t)$

Output: $\mathcal{G}(u)(\mathbf{x}, t) \approx \sum_{k=1}^{p} b_k(u) \cdot t_k(\mathbf{x}, t)$

**Variants:**
- **POD-DeepONet:** Uses Proper Orthogonal Decomposition basis for the trunk
- **MIONet:** Multi-input operator network for multiple input functions
- **Fourier-DeepONet:** Incorporates Fourier features
- **DeepOKAN (2024):** Replaces standard networks with Kolmogorov-Arnold Networks

### 3.3 Fourier Neural Operator (FNO)

**Li, Kovachki, Azizzadenesheli, Liu, Bhatt, Stuart, and Anandkumar (ICLR 2021)** -- "Fourier Neural Operator for Parametric Partial Differential Equations."

**Architecture:** Iterative layers of global convolutions in Fourier space:

$$
v_{l+1}(x) = \sigma\left(W_l v_l(x) + \mathcal{F}^{-1}(R_l \cdot \mathcal{F}(v_l))(x)\right)
$$

where $\mathcal{F}$ is the FFT, $R_l$ is a learnable filter in Fourier space (truncated to $k_{\max}$ modes), and $W_l$ is a local linear transform.

**Key property:** Resolution invariance -- trained on coarse grids, evaluated on fine grids.

**Variants (2023--2024):**
- **TFNO (Tensorized FNO):** Tucker tensor factorization reduces parameters to ~10% of dense FNO
- **SFNO (Spherical FNO, Bonev et al., 2023):** Extension to spherical domains using spherical harmonics
- **GINO (Geometry-Informed NO, Li et al., 2023):** Handles varying, irregular geometries
- **U-FNO:** U-Net-inspired architecture for multiphase flow
- **PINO (Physics-Informed NO):** Combines operator learning with physics-informed function optimization

### 3.4 Comparison: DeepONet vs FNO

| Feature | DeepONet | FNO |
|---|---|---|
| **Input flexibility** | Handles irregular sensor locations | Requires structured grid |
| **Architecture** | Branch-trunk factorization | Spectral convolutions |
| **Resolution invariance** | Yes (via trunk network) | Yes (via spectral truncation) |
| **Geometry handling** | More flexible | Requires regular domains (without GINO) |
| **Training data** | Needs paired input-output functions | Same |
| **Best for** | Irregular geometries, sparse sensors | Structured domains, periodic/smooth solutions |

### 3.5 Strengths and Limitations

**Strengths:**
- Amortized inference: once trained, evaluation is a single forward pass (~1000x faster than PDE solvers)
- Generalize to new problem instances
- Resolution invariance (especially FNO)
- Can learn operators from data alone or with physics constraints

**Limitations:**
- Require large amounts of training data (typically from classical solvers)
- FNO limited to structured grids without geometric extensions
- Accuracy degrades for out-of-distribution inputs
- Training is expensive (but inference is cheap)
- Struggle with discontinuities and shocks

### 3.6 Relevance to Snake Robot

**MEDIUM.** Neural operators could learn the mapping from (CPG parameters, initial rod configuration) to (rod trajectory), providing a fast surrogate for the full DER simulation. However:
- Training data must come from the expensive DER solver
- The rod geometry (1D arc-length domain) is relatively simple, so the full power of neural operators may be overkill
- More relevant if many forward evaluations with varying parameters are needed (e.g., RL policy optimization across parameter sweeps)

---

## 4. Surrogate Modeling

### 4.1 Core Concept

Use a neural network as a fast proxy for an expensive numerical solver. The surrogate is trained on input-output pairs from the solver and then replaces it during optimization, design exploration, or real-time control.

### 4.2 Key Approaches

**Data-Driven Surrogates:**
- Train an MLP, CNN, or RNN on (input, output) pairs from the simulator
- Simplest approach, no physics knowledge required
- Risk of poor generalization outside training distribution

**Physics-Enhanced Deep Surrogates (PEDS):**
- **Pestourie et al. (Nature Machine Intelligence, 2023):** Combines a low-fidelity physics simulator with a neural network generator, trained end-to-end. The physics simulator provides a cheap approximate solution; the neural network learns the correction.

**Latent Neural PDE Solvers (2024):**
- Encode high-fidelity PDE solutions into a reduced low-dimensional latent space
- Learn dynamics in latent space via a latent ODE
- Decode back to full solution
- Achieves significant speedups with controlled accuracy loss

**Autoencoder + Neural ODE:**
- Encoder maps full state $\mathbf{q} \in \mathbb{R}^N$ to latent $\mathbf{z} \in \mathbb{R}^d$ where $d \ll N$
- Neural ODE evolves $\mathbf{z}(t)$ in latent space
- Decoder reconstructs full state
- Particularly suitable for the DER system where $\mathbf{q}$ has many DOFs

### 4.3 Relevance to Snake Robot

**HIGH.** Surrogate modeling is the most directly applicable paradigm:
- The DER solver is expensive (the bottleneck in RL training)
- The RL agent needs many forward evaluations
- A surrogate trained on DER trajectories could replace the solver during policy optimization
- The KNODE approach (Section 2.4) is a physics-informed surrogate

---

## 5. Differentiable Physics / Differentiable Simulation

### 5.1 Core Concept

Make the physics simulator differentiable so that gradients of the simulation output with respect to inputs (control parameters, initial conditions, design variables) can be computed via backpropagation. This enables direct gradient-based optimization through the simulation.

### 5.2 Key Works

**DiffTaichi (Hu et al., ICLR 2020):**
- Differentiable programming language for physical simulation
- Source code transformations preserve arithmetic intensity and parallelism
- 4.2x faster than hand-engineered CUDA, 188x faster than TensorFlow
- Supports elastic bodies, fluids, rigid bodies
- Neural network controllers optimized within tens of iterations

**DiffPD (Du et al., ACM TOG, 2022):**
- Differentiable Projective Dynamics for soft-body simulation
- Exploits prefactorized Cholesky decomposition for fast backpropagation
- 4--19x faster than standard Newton's method for backprop
- Applications: system identification, inverse design, trajectory optimization, closed-loop control

**JAX MD (Schoenholz and Cubuk, NeurIPS 2020):**
- Differentiable molecular dynamics in JAX
- Automatic differentiation through entire simulation trajectories
- GPU-accelerated

**Brax (Freeman et al., 2021):**
- Differentiable rigid-body simulator in JAX
- Designed for RL policy training
- Massively parallel simulation on accelerators

### 5.3 Stabilizing RL in Differentiable Simulation

**Xing (CMU, 2025):** "Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation." Addresses the challenge that naive backpropagation through long simulation horizons causes exploding/vanishing gradients. Key insight: short-horizon differentiable rollouts combined with RL value function estimation provides stable training.

**Soft Analytic Policy Optimization (SAPO, RoboSoft 2025):** Combines RL with first-order analytic gradients from differentiable simulation, scaling to tasks involving rigid bodies and deformables.

### 5.4 Strengths and Limitations

**Strengths:**
- Exact gradients through the simulation
- Sample-efficient optimization (vs. model-free RL)
- Enables end-to-end training of controller + simulator
- Can combine with RL for long-horizon tasks

**Limitations:**
- Gradients through long rollouts can explode/vanish
- Contact and friction are inherently non-smooth (gradients undefined at discontinuities)
- Requires re-implementing the simulator in a differentiable framework
- Memory cost grows with simulation length (checkpointing helps)

### 5.5 Relevance to Snake Robot

**MEDIUM-HIGH.** Differentiable simulation would enable gradient-based policy optimization for the snake robot, but:
- The existing DER solver (DisMech) is in C++ and not differentiable
- Re-implementing DER in a differentiable framework (JAX, PyTorch) is a significant engineering effort
- Contact mechanics (penalty-based in DisMech) introduces non-smoothness
- A more practical route: use a differentiable surrogate (Section 4) trained on DisMech outputs

---

## 6. Graph Neural Networks for Physics

### 6.1 Seminal Works

**Graph Network-based Simulators (GNS) -- Sanchez-Gonzalez et al. (ICML 2020):**
"Learning to Simulate Complex Physics with Graph Networks."
- Represents physical systems as particles (graph nodes)
- Learns dynamics via message-passing
- Encode-process-decode architecture
- Generalizes to different particle counts, initial conditions, and thousands of timesteps
- Handles fluids, rigid solids, deformable materials

**MeshGraphNets -- Pfaff et al. (ICLR 2021):**
"Learning Mesh-Based Simulation with Graph Networks."
- Operates on simulation meshes (nodes = mesh vertices, edges = mesh connectivity)
- Learns to predict updates to mesh vertex positions
- Applications: aerodynamics, structural mechanics, cloth simulation

### 6.2 Recent Advances (2024--2025)

- **X-MeshGraphNet (2024):** Scalable, multi-scale extension of MeshGraphNet. Addresses scalability to large meshes and long-range interactions. Accurate predictions of surface pressure and wall shear stresses.

- **Edge-Augmented GNN (EA-GNN):** Introduces "virtual" edges for faster information propagation, improving computational efficiency.

- **GNN + Contact Dynamics (Allen et al., CoRL 2023):** "Graph Network Simulators Can Learn Discontinuous, Rigid Contact Dynamics." Demonstrates that GNNs can handle the non-smooth nature of contact.

- **PhymPGN (arXiv 2024):** Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems.

### 6.3 Strengths and Limitations

**Strengths:**
- Handle irregular, unstructured meshes and particle systems
- Naturally encode local interactions (message passing = local physics)
- Scale to large systems
- Can learn contact and collision dynamics
- Architecture mirrors the structure of physical systems

**Limitations:**
- Long-range interactions require many message-passing steps (or multi-scale approaches)
- Training requires simulation data
- Error accumulation over long rollouts
- Less accurate than classical solvers for well-resolved problems

### 6.4 Relevance to Snake Robot

**MEDIUM.** The DER discretization naturally maps to a graph (nodes = rod vertices, edges = rod segments). A GNN could:
- Learn the DER dynamics as a message-passing update
- Handle contact by adding edges between contacting nodes
- Generalize to different rod lengths/discretizations

However, the 1D rod structure is relatively simple, and the overhead of a full GNN may not be justified compared to simpler surrogate approaches.

---

## 7. Structure-Preserving Neural Networks

### 7.1 Hamiltonian Neural Networks (HNN)

**Greydanus, Dzamba, and Sohl-Dickstein (NeurIPS 2019):**

Learn a Hamiltonian $H_\theta(\mathbf{q}, \mathbf{p})$ such that:

$$
\dot{\mathbf{q}} = \frac{\partial H_\theta}{\partial \mathbf{p}}, \qquad \dot{\mathbf{p}} = -\frac{\partial H_\theta}{\partial \mathbf{q}}
$$

**Key property:** The learned dynamics exactly conserve the learned Hamiltonian, preventing energy drift over long simulations.

### 7.2 Lagrangian Neural Networks (LNN)

**Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho (ICLR 2020 Workshop):**

Learn a Lagrangian $L_\theta(\mathbf{q}, \dot{\mathbf{q}})$ and derive equations of motion via Euler-Lagrange:

$$
\frac{d}{dt}\frac{\partial L_\theta}{\partial \dot{\mathbf{q}}} - \frac{\partial L_\theta}{\partial \mathbf{q}} = \mathbf{0}
$$

**Advantage over HNN:** Works directly in generalized coordinates (no need to compute conjugate momenta).

### 7.3 Recent Extensions (2023--2025)

- **Dissipative HNN/LNN:** Add learned dissipation (Rayleigh dissipation function) for non-conservative systems -- essential for the snake robot which has friction and damping.
- **Port-Hamiltonian Neural Networks (Eidnes et al., 2023):** Learn port-Hamiltonian systems that naturally handle energy exchange with the environment (actuation, damping).
- **Constrained HNN:** Enforce holonomic constraints while preserving Hamiltonian structure.
- **Symplectic Neural Networks:** Use symplectic integrators in the network architecture to exactly preserve phase-space volume.

### 7.4 Relevance to Snake Robot

**MEDIUM.** The DER formulation is derived from elastic energy minimization (Lagrangian mechanics), making LNN a natural fit. However:
- The snake robot has significant dissipation (friction, damping) -- pure Hamiltonian/Lagrangian formulations need extension
- Port-Hamiltonian formulations could handle actuation and damping
- Most useful for long-horizon simulation accuracy (preventing energy drift)

---

## 8. Applications to Robotics and Soft Body Dynamics

### 8.1 KNODE-Cosserat (Hsieh et al., 2024)

**"Knowledge-based Neural Ordinary Differential Equations for Cosserat Rod-based Soft Robots"** (arXiv:2408.07776)

**The most directly relevant paper for our project.**

**Key contributions:**
- Adopts KNODE framework for Cosserat rod-based soft robots
- Uses NODE to improve accuracy of *spatial* derivatives (not just temporal)
- Combines first-principle Cosserat rod physics with neural ODE corrections
- Validated on tendon-driven continuum robots
- Works with imperfect physics models + training data from real robot trajectories
- Code available: [github.com/hsiehScalAR/KNODE-Cosserat](https://github.com/hsiehScalAR/KNODE-Cosserat)

**Architecture:**

$$
\dot{\mathbf{x}} = f_{\text{Cosserat}}(\mathbf{x}) + g_\theta(\mathbf{x})
$$

where $f_{\text{Cosserat}}$ encodes the Cosserat rod equations and $g_\theta$ learns unmodeled dynamics (friction, contact, manufacturing defects).

**Results:** Significant improvement over both pure physics models and pure neural network models in both simulation and real-world experiments.

### 8.2 DD-PINN for Soft Continuum Robots (2025)

**"Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory"** (arXiv:2508.12681)

- Domain-Decomposed PINN (DD-PINN) as surrogate for dynamic Cosserat rod model
- **44,000x speedup** over direct numerical simulation
- End-effector position errors below 3 mm in simulation
- Real-time MPC for soft continuum robots
- Real-world experiments with accelerations up to 3.55 m/s^2

### 8.3 C-PINN for Deformable Linear Objects (2024)

- Embeds Cosserat rod theory directly into PINN loss function
- Curriculum learning strategy for training stability
- Online sim-to-real residual adaptation module
- Efficient, real-time DLO modeling and automatic shape control

### 8.4 Generalizable PINN Surrogates (2025)

**"Generalizable and Fast Surrogates: Model Predictive Control of Articulated Soft Robots using Physics-Informed Neural Networks"** (arXiv:2502.01916)

- First application of original PINNs to real multi-DOF soft robot
- Extended PINC and DD-PINN architectures for generalizability despite system changes after training

### 8.5 Differentiable Simulation for Soft Robots (2024--2025)

- **RoboSoft 2025:** Differentiable simulation of soft robots with frictional contacts
- Various backends now available: PyElastica (Cosserat rods), SoMo (soft body), DiffPD (projective dynamics)
- End-to-end optimization of controllers and robot designs

### 8.6 Key Insight for Our Project

The KNODE-Cosserat approach is the most directly transferable method:
1. We already have a Cosserat rod solver (DisMech/DER)
2. KNODE can learn corrections to our imperfect model
3. The hybrid approach works with limited training data
4. It preserves the physical structure of the equations

---

## 9. Software Libraries

### 9.1 Core Libraries

| Library | Purpose | Backend | Maturity | Latest Version |
|---|---|---|---|---|
| **torchdiffeq** | Neural ODE solving | PyTorch | Mature | ~0.2.4 (6.4k stars) |
| **DeepXDE** | PINNs, DeepONet, operator learning | TF/PyTorch/JAX/Paddle | Mature | 1.15.0 |
| **neuraloperator** | FNO, TFNO, neural operators | PyTorch | Mature | 2.0.0 (Oct 2025) |
| **NVIDIA PhysicsNeMo** | PINNs, FNO, GNN, diffusion | PyTorch | Production | Renamed from Modulus |
| **DiffTaichi** | Differentiable physics | Taichi | Mature | Part of Taichi |
| **JAX** | Autodiff + JIT compilation | XLA | Mature | -- |

### 9.2 torchdiffeq (Most Relevant)

**Installation:** `pip install torchdiffeq`

**Key features:**
- Differentiable ODE solvers with full GPU support
- O(1) memory backpropagation via adjoint method
- Adaptive-step solvers: `dopri5` (default), `dopri8`, `bosh3`, `fehlberg2`, `adaptive_heun`
- Fixed-step solvers: `euler`, `midpoint`, `rk4`, `explicit_adams`, `implicit_adams`
- SciPy solver wrappers

**Basic usage:**
```python
from torchdiffeq import odeint, odeint_adjoint

# Define dynamics
class ODEFunc(nn.Module):
    def forward(self, t, y):
        return self.net(y)  # dy/dt = f_theta(y)

func = ODEFunc()
y0 = torch.tensor([1.0, 0.0])
t = torch.linspace(0, 1, 100)
y = odeint(func, y0, t)  # or odeint_adjoint for O(1) memory
```

**Critical note:** When using `odeint_adjoint`, `func` must be an `nn.Module` so parameters can be collected for the adjoint backward pass.

### 9.3 DeepXDE

**Installation:** `pip install deepxde`

**Key features:**
- PINNs for forward/inverse problems
- DeepONet, POD-DeepONet, MIONet, Fourier-DeepONet
- Five backend support (TensorFlow 1.x/2.x, PyTorch, JAX, PaddlePaddle)
- Stochastic PDEs
- Developed by Lu Lu (Yale), maintained since 2018

### 9.4 neuraloperator (PyTorch Ecosystem)

**Installation:** `pip install neuraloperator`

**Key features:**
- Official FNO implementation
- TFNO with Tucker factorization (10% parameters of dense FNO)
- Part of PyTorch ecosystem
- Version 2.0.0 released October 2025

### 9.5 NVIDIA PhysicsNeMo (formerly Modulus)

**Key features:**
- Production-grade framework for physics-AI models at scale
- PINNs, FNO, PINO, GNNs, diffusion models
- Multi-GPU/multi-node training
- Extensive documentation and examples

### 9.6 Specialized Libraries

| Library | Focus | URL |
|---|---|---|
| **torchsde** | Stochastic differential equations in PyTorch | github.com/google-research/torchsde |
| **diffrax** | Differential equations in JAX | github.com/patrick-kidger/diffrax |
| **Brax** | Differentiable rigid-body simulation in JAX | github.com/google/brax |
| **PyElastica** | Cosserat rod simulation (Python) | github.com/GazzolaLab/PyElastica |

---

## 10. Common Pitfalls

### 10.1 Training Neural ODEs

1. **Stiff systems require implicit solvers.** The DER system can be stiff (high stiffness coefficients). Using `dopri5` (explicit) on a stiff system leads to tiny step sizes and divergence. Use implicit solvers or reformulate.

2. **Adjoint method instability.** For stiff or chaotic systems, the adjoint ODE can diverge backward in time. Remedy: use `odeint` with checkpointing instead of `odeint_adjoint`, or use the seminorm adjoint (available in torchdiffeq).

3. **NFE explosion.** The number of function evaluations (NFE) can grow during training as the learned dynamics become complex. Monitor NFE; if it grows unbounded, regularize (e.g., kinetic energy regularization on $f_\theta$).

### 10.2 Training PINNs

4. **Loss balancing is critical.** Naive equal weighting of PDE/BC/IC losses almost always fails. Use adaptive weighting (e.g., Wang et al.'s learning rate annealing, or GradNorm).

5. **Collocation point placement matters.** Uniform random sampling is suboptimal. Use adaptive sampling (R3, failure-informed) or Latin hypercube sampling.

6. **Spectral bias prevents learning high-frequency solutions.** Use Fourier feature inputs $\gamma(\mathbf{x}) = [\sin(2\pi B \mathbf{x}), \cos(2\pi B \mathbf{x})]$ with random $B$.

### 10.3 Surrogate Modeling

7. **Distribution shift in RL.** A surrogate trained on trajectories from one policy will be inaccurate for trajectories from a different policy. Solution: online fine-tuning, ensemble uncertainty estimates, or Dagger-style data aggregation.

8. **Error accumulation in autoregressive rollouts.** Small per-step errors compound over long trajectories. Use teacher forcing during training and scheduled sampling.

9. **Overfitting to training distribution.** Surrogates can memorize training data rather than learning physics. Regularize with physics constraints or use physics-informed architectures.

### 10.4 General

10. **Ignoring units and scaling.** Neural networks work best with normalized inputs/outputs (~unit variance). Nondimensionalize the ODE/PDE system before training (see `knowledge/ode-pde-formulation-conventions.md`, Section 6).

11. **Wrong time discretization.** Don't mix the Neural ODE's continuous-time formulation with discrete-time training data without proper handling of measurement timestamps.

12. **Contact discontinuities.** Contact mechanics involves non-smooth dynamics (complementarity conditions). Neural networks (smooth functions) struggle to represent sharp transitions. Use event-detection or piecewise models.

---

## 11. Open Problems and Active Research

### 11.1 Active Research Directions

1. **Operator learning for parametric PDEs with complex geometries** -- extending FNO/DeepONet beyond structured grids to real-world geometries (GINO, graph-based operators).

2. **Foundation models for PDEs** -- training a single large model on diverse PDE families, then fine-tuning for specific problems (analogous to LLMs for language). Active area at NVIDIA, Caltech, Brown.

3. **Reliable uncertainty quantification** -- knowing *when* the surrogate's prediction is unreliable. Ensemble methods, Bayesian Neural ODEs, conformal prediction.

4. **Multi-scale and multi-physics** -- learning coupled systems where different physics operate at different scales (e.g., Cosserat rod + contact + CPG in our case).

5. **Long-horizon stability** -- preventing error accumulation in autoregressive rollouts of learned simulators. Spectral methods, conservation-aware architectures.

6. **Efficient training of PINNs** -- reducing the training cost gap between PINNs and classical solvers. Curriculum learning, transfer learning, meta-learning.

7. **Non-smooth dynamics** -- learning contact, friction, and impact dynamics that involve discontinuities. Hybrid smooth/discrete models, complementarity-aware architectures.

### 11.2 Open Questions for Snake Robot

1. **Can KNODE-Cosserat handle dynamic simulation (not just quasi-static)?** The existing paper focuses on spatial derivatives; temporal dynamics for RL training is an open question.

2. **How to handle contact in the surrogate?** Contact is the most challenging non-smooth component. Options: regularize (penalty method) then learn, or use a separate contact model.

3. **What is the minimum training data required?** Physics-informed approaches need less data, but the exact amount for our DER system is unknown.

4. **Can the surrogate be differentiable end-to-end with the RL policy?** This would enable gradient-based policy optimization through the learned dynamics.

---

## 12. Relevance to Snake Robot HRL Project

### 12.1 The Bottleneck

The DER solver (DisMech) is the computational bottleneck in RL training. Each environment step requires solving the elastic rod equations with contact, which is expensive. The HRL framework requires many environment interactions (approach policy + coil policy + meta-controller).

### 12.2 Recommended Approach Hierarchy

Ordered by practicality and expected impact:

| Priority | Approach | Why | Effort |
|---|---|---|---|
| **1** | **KNODE-Cosserat surrogate** | Directly applicable, proven on Cosserat rods, hybrid physics+learning | Medium |
| **2** | **Autoencoder + latent Neural ODE** | Compress DER state, evolve cheaply in latent space | Medium |
| **3** | **DD-PINN for static sub-problems** | 44,000x speedup demonstrated for Cosserat rods | Medium |
| **4** | **GNN surrogate (MeshGraphNets-style)** | Natural fit for DER graph structure, handles contact | High |
| **5** | **FNO for parameter sweeps** | Useful if varying rod properties across training | High |
| **6** | **Full differentiable DER reimplementation** | Maximum benefit but enormous engineering effort | Very High |

### 12.3 Specific Component Mapping

| Snake Robot Component | Best NN Method | Library |
|---|---|---|
| DER elastic rod dynamics | KNODE (Neural ODE + Cosserat physics) | torchdiffeq |
| CPG oscillator | Neural ODE or KNODE | torchdiffeq |
| Contact mechanics | GNN or learned penalty model | PyG / custom |
| Static equilibrium solving | DD-PINN | DeepXDE |
| Full surrogate for RL | Autoencoder + latent Neural ODE | torchdiffeq + PyTorch |
| Parameter identification | PINNs (inverse problem) | DeepXDE |

---

## 13. Maturity Assessment

| Method | Maturity | Ready for Production? | Best For |
|---|---|---|---|
| **PINNs** | Mature (research) | Partially -- inverse problems and simple forward problems work well | Parameter identification, steady-state solutions |
| **Neural ODEs** | Mature | Yes -- torchdiffeq is stable and well-tested | Continuous-time dynamics, KNODE hybrid models |
| **FNO** | Mature | Yes -- neuraloperator 2.0, PhysicsNeMo | Structured-grid PDE surrogates |
| **DeepONet** | Mature | Yes -- DeepXDE provides stable implementation | Operator learning with irregular inputs |
| **Differentiable simulation** | Research | Partially -- DiffTaichi/Brax work; custom solvers require effort | When gradients through sim are essential |
| **GNN for physics** | Research | Partially -- demonstrated in papers, limited reusable tooling | Complex geometries, contact dynamics |
| **HNN/LNN** | Research | No -- mainly academic demonstrations | Long-horizon energy conservation |
| **KNODE-Cosserat** | Early research | No -- single paper, small-scale demos | Hybrid physics-learning for rod robots |

---

## 14. Sources

### Seminal Papers

- Raissi, M., Perdikaris, P., and Karniadakis, G.E. (2019). "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations." *Journal of Computational Physics*, 378, 686--707.
- Chen, R.T.Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. (2018). "Neural Ordinary Differential Equations." *NeurIPS 2018* (Best Paper). [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- Lu, L., Jin, P., Pang, G., Zhang, Z., and Karniadakis, G.E. (2021). "Learning Nonlinear Operators via DeepONet." *Nature Machine Intelligence*, 3, 218--229.
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhatt, K., Stuart, A., and Anandkumar, A. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.
- Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J., and Battaglia, P. (2020). "Learning to Simulate Complex Physics with Graph Networks." *ICML 2020*. [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)
- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., and Battaglia, P. (2021). "Learning Mesh-Based Simulation with Graph Networks." *ICLR 2021*. [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- Greydanus, S., Dzamba, M., and Sohl-Dickstein, J. (2019). "Hamiltonian Neural Networks." *NeurIPS 2019*.
- Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., and Ho, S. (2020). "Lagrangian Neural Networks." *ICLR 2020 Workshop*.

### Directly Relevant to Snake Robot / Cosserat Rods

- Hsieh et al. (2024). "Knowledge-based Neural Ordinary Differential Equations for Cosserat Rod-based Soft Robots." [arXiv:2408.07776](https://arxiv.org/abs/2408.07776). Code: [github.com/hsiehScalAR/KNODE-Cosserat](https://github.com/hsiehScalAR/KNODE-Cosserat)
- "Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory." (2025). [arXiv:2508.12681](https://arxiv.org/abs/2508.12681)
- "Generalizable and Fast Surrogates: Model Predictive Control of Articulated Soft Robots using Physics-Informed Neural Networks." (2025). [arXiv:2502.01916](https://arxiv.org/abs/2502.01916)
- "Physics-Informed Neural Networks for Continuum Robots: Towards Fast Approximation of Static Cosserat Rod Theory." (2024). [IEEE](https://ieeexplore.ieee.org/iel8/10609961/10609862/10610742.pdf)

### PINN Training and Failure Modes

- Wang, S. and Perdikaris, P. (2024). "Challenges in Training PINNs: A Loss Landscape Perspective." [arXiv:2402.01868](https://arxiv.org/pdf/2402.01868)
- Daw, A. et al. (2023). "Mitigating Propagation Failures in Physics-informed Neural Networks." *ICML 2023*.
- Wu, C. et al. (2023). "Failure-Informed Adaptive Sampling for PINNs." *SIAM Journal on Scientific Computing*.
- "A simple remedy for failure modes in physics informed neural networks." *Neural Networks* (2024).

### Differentiable Simulation

- Hu, Y. et al. (2020). "DiffTaichi: Differentiable Programming for Physical Simulation." *ICLR 2020*. [arXiv:1910.00935](https://arxiv.org/abs/1910.00935)
- Du, T. et al. (2022). "DiffPD: Differentiable Projective Dynamics." *ACM TOG*.
- Xing (2025). "Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation." CMU Thesis. [arXiv:2412.12089](https://arxiv.org/abs/2412.12089)

### Neural Operators

- Li, Z. et al. (2023). "Geometry-Informed Neural Operator (GINO)."
- Bonev, B. et al. (2023). "Spherical Fourier Neural Operator (SFNO)."
- Pestourie, R. et al. (2023). "Physics-enhanced deep surrogates for PDEs." *Nature Machine Intelligence*.
- "Latent neural PDE solver: A reduced-order modeling framework for partial differential equations." *JCP* (2024). [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999124009537)

### Software

- torchdiffeq: [github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) (6.4k stars, MIT license)
- DeepXDE: [github.com/lululxvi/deepxde](https://github.com/lululxvi/deepxde) (v1.15.0)
- neuraloperator: [github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) (v2.0.0)
- NVIDIA PhysicsNeMo: [developer.nvidia.com/modulus](https://developer.nvidia.com/modulus)
- Physics-Based Deep Learning list: [github.com/thunil/Physics-Based-Deep-Learning](https://github.com/thunil/Physics-Based-Deep-Learning)
- Neural PDE Solver collection: [github.com/bitzhangcy/Neural-PDE-Solver](https://github.com/bitzhangcy/Neural-PDE-Solver)
