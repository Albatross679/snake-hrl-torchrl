---
id: 9e189ec1-d2fd-44cb-b857-46360b0de102
name: neural-surrogates-cosserat-rod-dynamics
description: Survey of neural network architectures (PINN, GNN, neural-ODE) for approximating Cosserat rod physics
type: knowledge
created: 2026-03-09T00:00:00
updated: 2026-03-09T00:00:00
tags: [neural-network, cosserat-rod, surrogate-model, PINN, GNN, neural-ODE, world-model, reinforcement-learning]
aliases: []
---

# Neural Network Surrogates for Cosserat Rod Dynamics

Comprehensive research survey on using neural networks to approximate Cosserat rod / elastic rod physics, covering architectures, performance benchmarks, and applicability to RL environments.

---

## 1. Neural Surrogates for Cosserat Rod Simulation

### DD-PINN: Domain-Decoupled Physics-Informed Neural Network

**Paper:** Stolzle et al., "Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory" (arXiv:2508.12681)

- **Architecture:** Domain-decoupled PINN (DD-PINN) that decouples the time domain from the feed-forward NN to construct an Ansatz function, computing gradients in closed form rather than via automatic differentiation.
- **State space:** 72 states -- the largest state-space model learned so far using PINNs for controls. Predicts the entire state of a continuum robot (strains and velocities).
- **Speedup: 44,000x** over numerical Cosserat rod integration.
- **Accuracy:** End-effector position errors below 3 mm (2.3% of actuator length) in simulation; similar accuracy in real-world experiments with accelerations up to 3.55 m/s^2.
- **Control frequency:** Nonlinear evolutionary MPC runs at **70 Hz** on GPU.
- **Online adaptation:** DD-PINN used within an unscented Kalman filter for estimating model states and bending compliance from end-effector position measurements, updating parameters online.

**Paper:** Stolzle et al., "Domain-decoupled Physics-informed Neural Networks with Closed-form Gradients for Fast Model Learning of Dynamical Systems" (arXiv:2408.14951)

- DD-PINN inherently fulfills initial conditions and supports higher-order excitation inputs.
- Significantly reduced training times vs. standard PINC (physics-informed neural controller) which relies on graph-based automatic differentiation.

### KNODE-Cosserat: Knowledge-Based Neural ODEs

**Paper:** Hsieh et al., "Knowledge-based Neural Ordinary Differential Equations for Cosserat Rod-based Soft Robots" (arXiv:2408.07776)

- **Architecture:** Combines first-principle Cosserat rod physics models with neural ODE residual terms.
- **Key innovation:** First application of any NODE variant to continuous-space systems for robotics.
- **Accuracy improvement: 58.7%** model accuracy improvement in real-world experiments over physics-only model, even for long-horizon trajectories.
- **Training data:** Simulated trajectories from PyElastica's implicit-shooting solver.
- **Speed:** Once trained, provides rapid predictions suitable for real-time control (specific FPS not reported vs. solver).
- **Code:** https://github.com/hsiehScalAR/KNODE-Cosserat

### Generalizable and Fast Surrogates (PINN for Articulated Soft Robots)

**Paper:** "Generalizable and Fast Surrogates: Model Predictive Control of Articulated Soft Robots using Physics-Informed Neural Networks" (arXiv:2502.01916)

- **Speedup: 467x** over accurate first-principles (FP) model at slightly reduced accuracy.
- **Data efficiency:** Reduces expensive real-world training data to a minimum of **one dataset in one system domain**.
- **Generalization:** High generalizability compared to RNNs, using only 2 hours of data across domains.
- **MPC frequency: 47 Hz** in six dynamic experiments on a pneumatic articulated soft robot.

### PINN for Static Cosserat Rod Theory (Continuum Robots)

**Paper:** "Physics-Informed Neural Networks for Continuum Robots: Towards Fast Approximation of Static Cosserat Rod Theory" (IEEE, 2024)

- Computes entire shape of tendon-driven continuum robot using Cosserat rod PDE as loss.
- **Accuracy:** Median position deviation below **1 mm (0.5% of robot length)** from reference model.

### Neural Network Kinematics Solver

**Paper:** "Fast Real-Time Neural Network-Based Kinematics Solving of the Cosserat Rod Model for a Parallel Continuum Surgical Manipulator" (IEEE, 2025)

- NN replaces iterative BVP solver for inverse kinematics of Cosserat rod models.
- Enables real-time surgical manipulator control.

---

## 2. Graph Neural Network Simulators for Deformable Bodies

### MeshGraphNets (DeepMind)

**Paper:** Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks" (ICLR 2021, arXiv:2010.03409)

- **Architecture:** GNN operating on simulation mesh directly. Message passing in mesh-space approximates differential operators (internal dynamics), while world-space edges handle collision/contact.
- **Domains:** Aerodynamics, structural mechanics, cloth -- handles 1D structures (rods/beams) as a special case of mesh-based simulation.
- **Generalization:** Resolution-independent dynamics; can scale to more complex state spaces at test time.
- **Code:** https://github.com/echowve/meshGraphNets_pytorch (PyTorch port)

### X-MeshGraphNet (Scalable Multi-Scale)

**Paper:** "X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation" (arXiv:2411.17164)

- Partitions large graphs with halo regions for seamless message passing across partitions.
- Addresses scalability limitations of original MeshGraphNets.

### BSMS-GNN (Bi-Stride Multi-Scale)

**Paper:** Cao et al., "Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN" (ICML 2023, arXiv:2210.02573)

- **Pooling strategy:** Novel bi-stride pooling on BFS frontiers -- no manual coarse meshes needed.
- **Efficiency:** Only **31-51% of computation** vs. standard GraphMeshNets. Cuts RAM by ~50%.
- **Domains:** Cylinder flow, airfoil, **elastic plate**, inflating elastic surface.
- Fastest inference time among compared methods; stable global rollouts.
- **Code:** https://github.com/Eydcao/BSMS-GNN

### GNS: Graph Network Simulator (DeepMind)

**Paper:** Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph Networks" (ICML 2020, arXiv:2002.09405)

- **Architecture:** Particle-based GNN with encoder-processor-decoder. Edges connect particles within radius R.
- **Domains:** Fluids, rigid solids, **deformable materials** interacting with one another.
- **Generalization:** Trains on thousands of particles, generalizes to 10x more particles and thousands of timesteps.
- **Error mitigation:** Training with noise corruption makes model robust to rollout error accumulation.
- **Speedup:** Up to **1000x** for benchmark problems at inference (claimed in geoelements/gns implementation).
- **Code:** https://github.com/geoelements/gns

### GNN for Beam Dynamics

**Paper:** "Predicting dynamic responses of continuous deformable bodies: A graph-based learning approach" (CMAME, 2023)

- Applied GNN to learn dynamics of 2D simply supported elastic beams.
- Validated generalization to structures with material nonlinearity (elastoplastic beams).

### Physics-Encoded GNN for Contact Deformation

**Paper:** "Physics-Encoded Graph Neural Networks for Deformation Prediction under Contact" (arXiv:2402.03466)

- Models rigid-to-deformable contact using cross-attention between object graphs.
- Jointly learns geometry and physics for consistent deformation reconstruction.

---

## 3. Neural ODEs and Structure-Preserving Architectures

### Hamiltonian Neural Networks (HNN)

**Paper:** Greydanus et al., "Hamiltonian Neural Networks" (NeurIPS 2019)

- Learns the Hamiltonian H(q,p) as a neural network; uses autodiff to get Hamilton's equations.
- **Energy conservation:** Conserves energy along collision-free trajectories.
- **Limitation:** Requires canonical coordinates (generalized positions and momenta).

### Lagrangian Neural Networks (LNN)

**Paper:** Cranmer et al., "Lagrangian Neural Networks" (ICLR 2020 Workshop)

- Parameterizes arbitrary Lagrangians via neural networks.
- **Advantage over HNN:** Does not require canonical coordinates; works when canonical momenta are unknown.
- Better prediction and generalization due to energy conservation.

### Extending to Contact

**Paper:** Zhong et al., "Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models" (NeurIPS 2021, arXiv:2102.06794)

- Extends HNN/LNN to systems with collisions and contact.
- Validated on pendulum, rigid body, and quadrotor systems.

### Symplectic Neural Networks

**Paper:** "Nonseparable Symplectic Neural Networks" (NSSNNs)

- Embeds symplectomorphism into network design, strictly preserving symplectic structure.
- **Long-term stability:** Stable predictions over **tens of thousands of iterations**.
- Conserves unknown, nonseparable Hamiltonian energy.

**Paper:** "Symplectic Learning for Hamiltonian Neural Networks" (JCP, 2023)

- Symplectic Hamiltonian Neural Networks (SHNNs) learn modified Hamiltonians to arbitrary precision.
- Using symplectic integrator during training significantly improves performance.

**Paper:** "Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control" (ICLR 2020)

- Combines neural ODE framework with symplectic integration.
- Rigorous handle on discretization error.

### Hamiltonian Neural ODE on SE(3)

**Paper:** Duong & Atanasov, "Hamiltonian-based Neural ODE Networks on the SE(3) Manifold" (RSS 2021, arXiv:2106.12782)

- Applies Hamiltonian neural ODE to rigid body dynamics on SE(3) manifold.
- Relevant for rod segments modeled as rigid body chains.

### Key Takeaway for Rod Dynamics

Structure-preserving architectures (symplectic, Hamiltonian, Lagrangian) provide:
- Bounded energy error over long rollouts (critical for multi-second rod simulations)
- Better generalization from small training sets
- Stable long-horizon predictions (10,000+ steps)

However, Cosserat rods are **dissipative** systems (with friction/damping), so pure Hamiltonian/symplectic methods need extension. Port-Hamiltonian or dissipative Lagrangian formulations are more appropriate.

---

## 4. PINNs for Rod and Beam Dynamics

### PINN for Euler-Bernoulli Beams

**Paper:** "A Gentle Introduction to Physics-Informed Neural Networks, with Applications in Static Rod and Beam Problems" (JAACM, 2024)

- Good convergence between PINN and closed-form analytical solutions for beam deflection.

**Paper:** "A-PINN: Auxiliary Physics-informed Neural Networks for Structural Vibration Analysis in Continuous Euler-Bernoulli Beam" (arXiv:2601.00866)

- Addresses challenge of 4th-order PDEs in beam vibration.
- PINNs struggle with high-order differential equations -- A-PINN introduces auxiliary variables to reduce order.

### PINN for Nonlinear Beam Bending

**Paper:** Bazmara et al., "Physics-informed neural networks for nonlinear bending of 3D functionally graded beam" (2023)

- **Speedup: 37x** on average over finite difference method for nonlinear bending response.
- **Speedup: 71x** for nonlinear buckling behavior prediction.

### PINN for Beam Vibration

**Paper:** "A Physics-Informed Deep Neural Network based beam vibration framework" (Eng. Appl. AI, 2024)

- Framework for simulation and parameter identification of beam vibrations.

### PINN for Cable Vibration with Bending Stiffness

**Paper:** "Physics-Informed Neural Networks for Solving Free Vibration Response of Cables" (JVET, 2025)

- Enhanced PINN methodology for cables incorporating bending stiffness.
- Addresses PINNs' limited accuracy with high-order PDEs in structural dynamics.

### Multi-level PINN for Structural Mechanics

**Paper:** "Multi-level physics informed deep learning for solving partial differential equations in computational structural mechanics" (Comms. Eng., 2024)

- **Accuracy:** Relative errors below **2%** for deformation of bending shells.

### Accuracy vs. Traditional Solvers -- Summary

| Method | Domain | Speedup | Accuracy |
|--------|--------|---------|----------|
| DD-PINN | Cosserat rod (dynamic) | 44,000x | <3 mm tip error (2.3%) |
| PINN static | Cosserat rod (static) | ~100x | <1 mm (0.5%) |
| PINN beam bending | FG beam (nonlinear) | 37-71x | Comparable to FDM |
| PINN articulated SR | Soft robot ODE | 467x | Slightly reduced |
| Multi-level PINN | Shell deformation | Not reported | <2% relative error |

### Limitations of PINNs as Surrogates

- Struggle with 4th-order PDEs (beam vibration requires auxiliary variable tricks).
- Training can be slow and sensitive to hyperparameters.
- Once trained for a specific geometry/BC, retraining needed for new configurations (unless using meta-learning or neural operators).
- Inference is fast but not as fast as a compiled forward pass of a simple MLP.

---

## 5. Learned Simulators for RL Environments

### SoRoLEX: Soft Robotics Learned Environment in JAX

**Paper:** "Towards Reinforcement Learning Controllers for Soft Robots using Learned Environments" (arXiv:2410.18519)

- **Architecture:** LSTM learns forward dynamics mapping (actuation -> task space).
- **Framework:** Fully JAX-based; compiled with XLA, executed in parallel on GPU.
- **RL approach:** Actor-critic networks learn from multiple parallel trajectories sampled from task space.
- **Key innovation:** Replaces physics simulator with learned LSTM model for RL training, enabling massive parallelization.
- **Relevance:** Directly applicable to our snake robot -- LSTM learns Cosserat rod dynamics, RL trains in learned model.
- **Code:** https://github.com/uljad/SoRoLEX

### PIN-WM: Physics-Informed World Models

**Paper:** "PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation" (RSS 2025, arXiv:2504.16693)

- World model of 3D rigid body dynamics learned from visual observations via Gaussian Splatting.
- Few-shot, task-agnostic physical interaction trajectories sufficient.
- Currently rigid-body only; authors note extending to MPM (Material Point Method) for deformable objects is future work.

### DreamerV3

**Paper:** Hafner et al., "Mastering Diverse Control Tasks through World Models" (2023)

- Learns latent dynamics model (RSSM) for imagination-based RL.
- Works across 150+ diverse tasks without hyperparameter tuning.
- Latent model accuracy sufficient for policy learning despite compounding errors.
- Not specifically applied to deformable rod dynamics, but architecture is general.

### TD-MPC2

**Paper:** Hansen et al., "TD-MPC2: Scalable, Robust World Models" (2024)

- Learned latent dynamics model for short-horizon trajectory optimization + terminal value function.
- Joint learning of model and value function mitigates model error exploitation.
- Primarily demonstrated on rigid-body locomotion and manipulation.

### MBPO: Model-Based Policy Optimization

**Paper:** Janner et al., "When to Trust Your Model" (NeurIPS 2019)

- Uses learned model for **short rollouts** from real states (not full-length rollouts from initial state).
- Matches model-free asymptotic performance with **10x less data**.
- Key insight: short-horizon model predictions limit error accumulation.

### Neural Motion Simulator (MoSim)

**Paper:** "Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning" (arXiv:2504.07095)

- World model that predicts future physical state from observations + actions.
- Enables zero-shot RL when model is accurate enough.

### Accuracy Requirements for RL

Based on the literature, the following principles emerge:

1. **Short-horizon model accuracy matters most.** MBPO showed that using the model only for short rollouts (1-5 steps) from real states limits error accumulation. Even models with ~5-10% per-step error can work.
2. **Latent-space models can tolerate more error** than state-space models, because the policy learns to cope with systematic model bias (DreamerV3, TD-MPC2).
3. **Training data noise injection** (GNS strategy) makes policies robust to model error.
4. **Physics-informed models generalize better** with less data and produce more physically consistent predictions, reducing out-of-distribution failures during RL exploration.

---

## 6. Practical Performance Numbers

### Inference Speed Comparison

| System | Speedup vs. Solver | Inference Speed | GPU Batch? |
|--------|-------------------|-----------------|------------|
| DD-PINN (Cosserat) | 44,000x | 70 Hz MPC | Yes |
| PINN (articulated SR) | 467x | 47 Hz MPC | Yes |
| GNS (particle-based) | up to 1000x | ~ms per step | Yes |
| MeshGraphNets | ~100x (varies) | ~ms per step | Yes |
| BSMS-GNN | 2-3x faster than MeshGraphNets | Sub-ms per step | Yes |
| FNO (Geo-FNO) | 100,000x | Sub-ms | Yes |
| PINN (beam bending) | 37-71x | Not reported | Possible |
| KNODE-Cosserat | Not quantified vs. solver | Real-time capable | Not reported |

### GPU-Parallel Environment Performance (Reference Points)

| Framework | Envs in Parallel | Steps/Second | Hardware |
|-----------|-----------------|--------------|----------|
| Isaac Gym | 10,000+ | 100,000s/sec | Single GPU |
| Brax (JAX) | 10,000+ | Millions/sec | TPU/GPU (V100) |
| MuJoCo XLA (MJX) | Thousands | 950K (A100), 2.7M (TPU v5) | Per single humanoid |
| SoRoLEX (learned LSTM) | Parallel (JAX) | Not reported | GPU |
| Our PyElastica | 16 | ~57 FPS | CPU (V100 unused) |

### Training Data Requirements

| Method | Data Needed | Notes |
|--------|-------------|-------|
| DD-PINN | Physics equations only (unsupervised) | No simulation data; PDE loss only |
| KNODE-Cosserat | Simulated trajectories | Amount not quantified; uses PyElastica |
| PINN (articulated SR) | **1 dataset minimum** | 2 hours across domains for comparison |
| GNS | 1000s of trajectories | ~1000 rollouts typical |
| MeshGraphNets | 1000s of simulations | Standard supervised learning |
| PI-DeepONet | **5-10x less than data-only** | Physics regularization reduces data needs |
| SoRoLEX (LSTM) | Simulated trajectories | Dataset from simulator exploration |

---

## 7. Relevance to Our Snake Robot Project

### Current Bottleneck
- PyElastica Cosserat rod simulation at 57 FPS with 16 CPU envs.
- GPU (V100) entirely unused for physics.
- Training limited by physics simulation speed, not RL compute.

### Most Promising Approaches (Ranked)

**1. SoRoLEX-style Learned LSTM Environment (Highest impact, most direct)**
- Train LSTM on PyElastica trajectories to learn forward dynamics.
- Port to JAX, run 1000s of parallel envs on GPU.
- Expected speedup: 100-1000x (from CPU -> GPU parallelism alone).
- Risk: LSTM may not capture all rod dynamics accurately for RL.

**2. DD-PINN Surrogate (Proven for Cosserat rods)**
- 44,000x speedup demonstrated specifically for Cosserat rod dynamics.
- Physics-informed (no need for massive training data).
- Can be integrated into MPC or used as RL environment dynamics.
- Risk: Architecture complexity; may need adaptation for our specific rod configuration.

**3. KNODE-Cosserat Hybrid (Best accuracy/data tradeoff)**
- Keeps physics model as backbone, neural ODE corrects residual.
- 58.7% accuracy improvement over physics-only.
- Could use our existing PyElastica as the "knowledge" base.
- Risk: May not be as fast as pure NN (still solves ODE with NN residual).

**4. GNN Simulator (Most general, scalable)**
- Treat rod nodes as particles, learn message-passing dynamics.
- Proven for deformable materials.
- Scalable to different rod configurations.
- Risk: Requires more training data; overkill for 1D rod structure.

**5. Neural Operator (FNO) (Fastest inference)**
- 10^5x speedup potential.
- Best for parametric studies (varying material properties, boundary conditions).
- Risk: Designed for PDE solution fields, may not map directly to RL step function.

### Recommended Implementation Path

1. **Phase 1 (Quick win):** Collect trajectory dataset from current PyElastica env (10K-100K episodes). Train simple MLP or LSTM to predict next-state from (state, action). Use as RL environment in JAX with batched inference on GPU.

2. **Phase 2 (Physics-informed):** Add Cosserat rod PDE terms to the loss function (PINN-style). This should improve accuracy and reduce training data needs.

3. **Phase 3 (Structure-preserving):** If energy/stability issues arise, switch to port-Hamiltonian or dissipative Lagrangian neural architecture for better long-horizon stability.

---

## Key Papers Reference List

1. Stolzle et al. (2025) - DD-PINN for Cosserat Rod MPC [arXiv:2508.12681]
2. Stolzle et al. (2024) - Domain-decoupled PINNs [arXiv:2408.14951]
3. Hsieh et al. (2024) - KNODE-Cosserat [arXiv:2408.07776]
4. (2025) - Generalizable Fast Surrogates PINN [arXiv:2502.01916]
5. (2024) - PINN for Static Cosserat Rod [IEEE ICRA 2024]
6. Pfaff et al. (2021) - MeshGraphNets [arXiv:2010.03409]
7. (2024) - X-MeshGraphNet [arXiv:2411.17164]
8. Cao et al. (2023) - BSMS-GNN [arXiv:2210.02573]
9. Sanchez-Gonzalez et al. (2020) - GNS [arXiv:2002.09405]
10. Greydanus et al. (2019) - Hamiltonian Neural Networks [NeurIPS 2019]
11. Cranmer et al. (2020) - Lagrangian Neural Networks [ICLR 2020 WS]
12. Zhong et al. (2021) - HNN/LNN with Contact [arXiv:2102.06794]
13. Uljad et al. (2024) - SoRoLEX [arXiv:2410.18519]
14. (2025) - PIN-WM [arXiv:2504.16693]
15. Hafner et al. (2023) - DreamerV3
16. Hansen et al. (2024) - TD-MPC2
17. Janner et al. (2019) - MBPO [NeurIPS 2019]
18. Li et al. (2021) - Fourier Neural Operator [arXiv:2207.05209]
19. Makoviychuk et al. (2021) - Isaac Gym [arXiv:2108.10470]
20. Freeman et al. (2021) - Brax [arXiv:2106.13281]
