---
name: neural-surrogate-simulators-for-rl
description: Neural network surrogates of physics simulators for RL training with focus on soft robotics
type: knowledge
created: 2026-03-09T00:00:00
updated: 2026-03-09T00:00:00
tags: [neural-surrogate, learned-simulator, world-model, reinforcement-learning, soft-robotics, differentiable-simulation, physics-informed, Cosserat-rod]
aliases: []
---

# Neural Network Surrogates of Physics Simulators for RL Training

Deep research into learned/neural simulators that can replace or augment physics engines for reinforcement learning, with emphasis on soft robot applications and Cosserat rod dynamics relevant to our snake robot project.

---

## 1. Learned Simulators for Soft Robot RL

### 1.1 LSTM-Based Learned Environments for Soft Robots

**Paper:** "Towards Reinforcement Learning Controllers for Soft Robots using Learned Environments" (arXiv:2410.18519, October 2024)

The most directly relevant work to our project. Pipeline:
1. Explore state-action space using a physically safe **mean-reverting random walk** in actuation space
2. Collect dataset from ground-truth simulator
3. Train a **forward dynamics model using LSTM** on the collected transitions
4. Use the trained LSTM as a surrogate environment
5. Train RL policy (actor-critic / PPO) inside the learned environment using **JAX** for GPU-parallelized rollouts (PureJaxRL)

**Key results:**
- Order-of-magnitude speedup from PureJaxRL running agents and environments jointly on GPU
- Multiple parallel environments from the trained LSTM model enable simultaneous learning from many trajectories
- Safety-oriented exploration protocol prevents damage during data collection

**Relevance to our project:** This is essentially what we would do -- train an LSTM on PyElastica rollouts, then use it as a fast surrogate for PPO training. The JAX parallelization is the key speedup enabler.

### 1.2 Bridging High-Fidelity FEM and RL via Surrogates

**Paper:** Hong et al., "Bridging High-Fidelity Simulations and Physics-Based Learning using a Surrogate Model for Soft Robot Control" (Advanced Intelligent Systems, Wiley, 2026)

- Problem: FEM models are accurate but too slow for RL
- Solution: Train a surrogate on real-world + FEM-generated datasets
- The surrogate captures complex dynamics while maintaining efficiency for RL training
- Validated with sim2real: trajectory tracking and force control with high accuracy

### 1.3 DNN Surrogate for Underwater Soft Fin Robot

**Paper:** "Underwater Soft Fin Flapping Motion with Deep Neural Network Based Surrogate Model" (arXiv:2502.03135, February 2025)

- DNN surrogate acts as simulator for RL agent training
- Enables force control of fin-actuated underwater robots
- Integrates DNN surrogate model directly into the RL training loop

### 1.4 SofaGym + Domain Randomization

**Framework:** SOFA (Simulation Open Framework Architecture) with SofaGym

- OpenAI Gym interfaces for soft robot digital twins
- Domain Randomization (DR) enhances RL policy robustness
- DR drastically reduces training time while improving sim-to-real transfer
- Parallelization via SubprocVecEnv for speedup
- Code: github.com/andreaprotopapa/sofa-dr-rl

---

## 2. World Models and Model-Based RL

### 2.1 Robotic World Model (RWM)

**Paper:** Li, Krause, Hutter (ETH Zurich), "Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics" (arXiv:2501.10100, January 2025, CoRL)

- **First framework** to reliably train policies on a learned neural network simulator without domain-specific knowledge, then deploy on physical hardware
- **Dual-autoregressive mechanism** + self-supervised training for reliable long-horizon predictions
- No domain-specific inductive biases needed
- Trained with PPO, deployed on real robots with minimal performance loss
- Extension (RWM-U) adds epistemic uncertainty estimation for temporally consistent multi-step rollouts
- Code: github.com/leggedrobotics/robotic_world_model

### 2.2 Neural Motion Simulator (MoSim)

**Paper:** "Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning" (CVPR 2025)

- World model that predicts future physical state from current observations + actions
- Trained on randomized data as a **complete surrogate for dm_control**
- Agents train entirely inside the world model, achieving scores closely approximating real environment
- Enables **zero-shot RL** when predictions are accurate enough
- Transforms any model-free RL algorithm into model-based by substituting the environment
- State-of-the-art in physical state prediction across dm_control benchmarks

### 2.3 DreamerV3 / DayDreamer

**Paper:** "Mastering Diverse Domains through World Models" (Nature, 2025)

- **Data efficiency:** Larger models consistently improve both final performance AND data efficiency
- **Real-world results:** Quadruped learns to stand + walk from scratch in 1 hour; pick-and-place in 10 hours
- Scaling properties: more gradient steps = more data efficiency
- General-purpose: works across Atari, DM Control, Minecraft without hyperparameter tuning

### 2.4 Neural Robot Dynamics (NeRD)

**Paper:** "Neural Robot Dynamics" (arXiv:2508.15755, 2025)

- Replaces analytical forward dynamics + contact solvers with neural networks
- Stable and accurate simulation over **thousands of timesteps**
- Fine-tunable from real-world data; generalizes across tasks, environments, controllers
- ANYmal quadruped: PPO policy trained inside NeRD, zero-shot transfer to analytical simulator with **<0.1% error** in accumulated reward over 1000-step trajectories
- Integrated into NVIDIA Newton physics engine
- Key insight: only replace application-agnostic simulation modules (forward dynamics, contact), keep everything else

---

## 3. DiffTaichi and Differentiable Simulation

### 3.1 DiffTaichi

**Paper:** "DiffTaichi: Differentiable Programming for Physical Simulation" (ICLR 2020)

- Source-code-level automatic differentiation preserving arithmetic intensity and parallelism
- **ChainQueen** (elastic object simulator): **188x faster** than TensorFlow implementation
- 10 differentiable simulators included (mass-spring, elastic, fluid, rigid body)
- JIT compilation + megakernel fusion for GPU/CPU efficiency
- Gradient-based optimization beats RL by **one order of magnitude** in optimization speed for soft robot control

### 3.2 ChainQueen

**Paper:** "ChainQueen: A Real-Time Differentiable Physical Simulator for Soft Robotics" (ICRA 2019)

- Real-time differentiable hybrid Lagrangian-Eulerian simulator based on MLS-MPM
- Handles problems with nearly 3,000 decision variables
- Gradient-based descent outperforms state-of-the-art RL by 10x in optimization speed

### 3.3 PlasticineLab

**Paper:** "PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics" (ICLR 2021)

- Built on ChainQueen with plasticity + contact gradients
- 10 soft-body manipulation tasks
- Gradient-based approaches find solutions in tens of iterations
- **Limitation:** gradient methods fall short on multi-stage tasks requiring long-term planning

### 3.4 DiffPD (Differentiable Projective Dynamics)

**Paper:** Du et al., "DiffPD: Differentiable Projective Dynamics" (ACM TOG, 2022)

- Differentiable soft-body simulator with implicit time integration
- **4-19x faster** than standard Newton's method via prefactorized Cholesky decomposition
- Applications: system identification, inverse design, trajectory optimization, closed-loop control
- Supports penalty-based and complementarity-based contact models

### 3.5 Differentiable Simulation vs. RL: Key Trade-offs

| Aspect | Differentiable Simulation | Reinforcement Learning |
|--------|--------------------------|----------------------|
| Sample efficiency | >5x better (needs <20% of samples) | Data-hungry |
| Training speed | >7x faster | Slower convergence |
| Gradient quality | Low-variance, analytical | High-variance (REINFORCE-style) |
| Reward functions | Must be differentiable | Any reward (binary, sparse, etc.) |
| Contact-rich tasks | Challenging (discontinuities) | Handles naturally |
| Long-horizon planning | Falls short on multi-stage | Better for complex planning |

**Bottom line for our project:** Differentiable simulation cannot directly replace RL for our snake locomotion problem because (a) contact dynamics introduce non-differentiable discontinuities, and (b) our reward involves distance-to-goal which is inherently non-smooth. A neural surrogate approach is more appropriate.

### 3.6 GPU-Accelerated Cosserat Rod Simulation

**Paper:** "Massively-Parallel Implementation of Inextensible Elastic Rods Using Inter-block GPU Synchronization" (arXiv:2509.04277, 2025)

- GPU implementation of inextensible Cosserat rod model
- **Average 15x speedup** over CPU version
- Real-time simulation at haptic interactive rates (0.5-1kHz)
- Relevant: could accelerate PyElastica directly rather than replacing it

---

## 4. Neural Network Architectures for Sequential Physics

### 4.1 Recurrent Architectures (LSTM/GRU)

**Strengths for physics surrogates:**
- Natural fit for time-stepping dynamics (state -> next_state)
- GRU: fewer parameters than LSTM, minimal quality loss
- Can handle variable time steps without accuracy degradation
- Successfully used for vessel dynamics, chemical reactor modeling, soft robot forward dynamics

**Practical considerations:**
- LSTM slightly outperforms GRU in most physics applications
- GRU is viable for resource-constrained settings
- Both can be deployed for online prediction at a fraction of a second

### 4.2 Graph Neural Networks (GNS)

**Paper:** Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph Networks" (ICML 2020, DeepMind)

- Encode-process-decode architecture with learned message-passing
- **5,000x faster** than MPM simulation (2.5 hours vs 20 seconds for granular flow)
- Trains on ~1,000 trajectories; generalizes to **30x more particles** and **5,000 timesteps**
- Accuracy: within **5% error** of ground-truth MPM
- Generalizes across particle counts, initial conditions, and spatial extents
- 15 message-passing steps is the sweet spot for accuracy vs. compute

**Training data:**
- 256 real-world trajectories sufficient for cube tossing
- As few as 64 trajectories for rotation prediction
- 8,096 simulated MuJoCo trajectories for more complex tasks

### 4.3 Transformer / Attention-Based Models

- MoSim (CVPR 2025) uses attention mechanisms for physical state prediction
- Robotic World Model uses dual-autoregressive mechanism
- Generally more data-hungry than recurrent models but better at long-range dependencies

---

## 5. Error Accumulation and Rollout Stability

### 5.1 The Core Problem

Autoregressive neural simulators feed predictions back as inputs, causing errors to compound exponentially. This is the primary failure mode for learned physics surrogates.

### 5.2 Mitigation Strategies

#### 5.2.1 Training Noise Injection
- Add Gaussian noise to training inputs to simulate inference-time errors
- Reduces distribution shift between training (clean states) and inference (noisy predicted states)
- Used in DeepMind's GNS as a key determinant of long-term performance
- Simple to implement, effective baseline

#### 5.2.2 Temporal Unrolling During Training

**Paper:** "How Temporal Unrolling Supports Neural Physics Simulators" (Thuerey Group, TUM)

Quantitative results:
- Non-differentiable but unrolled training: **4.5x improvement** over fully differentiable without unrolling
- Fully differentiable + unrolled: **38% improvement** on average over baselines
- **Critical threshold:** models with <=4 unrolling steps are unstable; >=8 steps is sufficient
- Trade-off: longer unrolling = more GPU memory but more stable rollouts
- Sweet spot: maximize rollout length while fitting in GPU memory

#### 5.2.3 PDE-Refiner (Iterative Refinement)

**Paper:** "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers" (NeurIPS 2023)

- Inspired by diffusion models: iteratively refine predictions at different frequency scales
- **30% improvement** in accurate prediction horizon (100s vs 75s for standard neural operator)
- Increasing model parameters 4x only improves by 5s; PDE-Refiner improves by 25s
- Implicitly provides spectral data augmentation (improves data efficiency)
- Enables uncertainty quantification via diffusion model connection

#### 5.2.4 Scheduled Sampling
- Randomly replace ground-truth tokens with model predictions during training
- Gradually increases replacement probability over training
- Mitigates train-inference mismatch

#### 5.2.5 Hybrid Neural-Numerical Correction
- Couple neural operators with classical numerical solvers
- Invoke solver adaptively when surrogate error exceeds threshold
- Enables bounded error growth while preserving computational efficiency

**Recommendation for our project:** Use noise injection (simple, effective) + temporal unrolling with >=8 steps during LSTM training. If accuracy degrades, add periodic correction using simplified physics.

---

## 6. Training Data Generation

### 6.1 Data Requirements (Quantitative)

| System | Data Size | Architecture | Accuracy |
|--------|-----------|-------------|----------|
| Granular flow (GNS) | ~1,000 trajectories | Graph Network | <5% error vs MPM |
| Real cube tossing | 256 trajectories | GNS | Good position prediction |
| Rotation prediction | 64 trajectories | GNS | Good rotation prediction |
| MuJoCo tasks | 8,096 trajectories | GNS | Matches simulation |
| Soft robot (LSTM) | Random walk exploration | LSTM | Sufficient for policy |
| PINNs for Cosserat | 1 dataset in 1 domain | PINN | High accuracy |

### 6.2 Active Learning / Online Data Collection

- **Online training** of neural surrogates yields **~7% better performance** than offline training (Meyer et al., ICML 2023)
- Active learning reduces need for pre-defined training sets by intelligently selecting which simulations to run
- Adaptive strategies balance exploration (new regions) and exploitation (refining known regions)
- **Transfer learning** can reduce required training data by **1-2 orders of magnitude**

### 6.3 Domain of Validity and Generalization

- **GNS generalizes** to 30x larger spatial extents and 30x more particles than training
- **PINNs generalize** to out-of-distribution data better than pure data-driven RNNs
- **Surrogate-only approaches** have limited generalization outside training distribution
- **Physics-informed approaches** dramatically reduce data requirements and improve extrapolation

### 6.4 Practical Strategy for Our Project

1. **Data collection:** Run PyElastica with randomized CPG parameters (amplitude, frequency, wave number) + randomized goal positions. Use mean-reverting random walks in action space for safe exploration.
2. **Target dataset:** ~5,000-10,000 trajectories of 200 steps each (based on GNS and LSTM literature)
3. **State representation:** Rod node positions, velocities, curvatures, goal-relative features
4. **Validation:** Hold out 20% of trajectories; measure per-step MSE and multi-step rollout divergence
5. **Active learning:** After initial training, identify high-error regions and collect targeted data

---

## 7. Hybrid / Residual Physics Approaches

### 7.1 Residual Model-Based RL

**Paper:** "Residual Model-Based Reinforcement Learning for Physical Dynamics" (NeurIPS 2022)

- Model = simplified physics ODE + learned neural residual
- The residual captures unmodeled dynamics, process noise, environmental perturbations
- Benefits: sample efficiency of physics model + flexibility of neural network
- Strongly regularized by the given ODE, preventing overfitting

### 7.2 KNODE-Cosserat (Neural ODE Residuals on Cosserat Rods)

**Paper:** "Knowledge-based Neural Ordinary Differential Equations for Cosserat Rod-based Soft Robots" (arXiv:2408.07776, August 2024)

- **Directly relevant:** Neural ODE residuals augment Cosserat rod physics
- NODE improves accuracy of spatial derivatives (not temporal -- a key insight)
- Given an imperfect Cosserat model as prior knowledge + real trajectory data, learns the true dynamics
- Metrics: DTW-aligned tip Euclidean distance + MSE on all segment positions/orientations
- Retains differentiability for downstream control
- Addresses the core challenge: soft robots have high spatial dimensionality making pure model-based control difficult

**Relevance to our project:** We could use simplified PyElastica (faster settings, fewer elements) as the physics prior and train a neural ODE residual to match high-fidelity PyElastica. This would give us a fast + accurate surrogate.

### 7.3 DD-PINN for Cosserat Rod Dynamics

**Paper:** "Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory" (arXiv:2508.12681, 2025)

- Domain-decoupled PINN (DD-PINN) as surrogate for dynamic Cosserat rod model
- **Speed-up factor: 44,000x** over first-principles simulation
- Used within unscented Kalman filter for state estimation
- Nonlinear evolutionary MPC runs at **70 Hz on GPU**
- Simulation accuracy: end-effector position errors **<3 mm** (2.3% of actuator length)
- Real-world: accelerations up to 3.55 m/s^2
- Uses time-dependent ansatz functions for improved training speed

### 7.4 Physics-Informed Neural Networks for Continuum Robots

**Paper:** "Physics-Informed Neural Networks for Continuum Robots: Towards Fast Approximation of Static Cosserat Rod Theory" (IEEE, 2024)

- PINNs incorporate Cosserat rod theory directly into loss function
- Outperform data-driven RNNs in both accuracy and generalization
- Require **minimal real-world training data** (1 dataset in 1 domain)
- Orders of magnitude faster than accurate physics-driven models

### 7.5 Hybrid Neural-Numerical Correction

**Paper:** "The Best of Both Worlds: Hybridizing Neural Operators and Solvers for Stable Long-Horizon Inference" (arXiv:2512.19643, 2025)

- Neural operator runs fast for most steps
- Classical solver invoked adaptively when error exceeds threshold
- Bounded error growth over time while preserving computational efficiency

---

## 8. Concrete NN Surrogates Replacing Physics in RL

### 8.1 Speedup Factors (Summary Table)

| System | Surrogate Type | Speedup | Source |
|--------|---------------|---------|--------|
| DD-PINN Cosserat rod | PINN | 44,000x | arXiv:2508.12681 |
| FourCastNet weather | Adaptive Fourier Neural Operator | 45,000-80,000x | NVIDIA |
| GNS granular flow | Graph Network | 5,000x | DeepMind ICML 2020 |
| DiffTaichi ChainQueen | Differentiable MPM | 188x (vs TensorFlow) | ICLR 2020 |
| Newton/MuJoCo-Warp humanoid | GPU-accelerated simulation | 70-100x | NVIDIA 2025 |
| Cosserat rod GPU | GPU parallelization | 15x | arXiv:2509.04277 |
| DiffPD soft body | Differentiable PD | 4-19x | ACM TOG 2022 |
| RF circuit design | DNN surrogate | ~2,400x (4min -> 100ms) | arXiv:2603.00104 |
| PINN smart grid | Physics-informed NN | 1.5x (50% faster training) | arXiv:2510.17380 |
| Brax physics engine | GPU-accelerated sim | 100-1,000x | Google Research |

### 8.2 NVIDIA PhysicsNeMo (formerly SimNet/Modulus)

- AI toolkit for physics-informed neural networks
- Applications: weather forecasting, turbine simulation, heatsink design, semiconductor TCAD
- Builds AI surrogate models combining physics causality with data
- Enables real-time parametric predictions (multiple configurations simultaneously)
- Used in industrial digital twins

### 8.3 FourCastNet (Weather Forecasting)

- Week-long forecast in <2 seconds (vs hours for NWP)
- FourCastNet3 (2025): 60-day rollout in <4 minutes on single H100
- Matches ECMWF IFS accuracy at short lead times
- 12,000x less energy per forecast
- Demonstrates that neural surrogates can match state-of-the-art numerical methods at scale

### 8.4 NeRD in NVIDIA Newton (Robotics)

- Neural dynamics module integrated into Newton physics engine
- ANYmal quadruped: policy trained in NeRD, zero-shot transfer with <0.1% reward error
- Pre-train NeRD once, reuse for multiple downstream tasks
- NVIDIA Newton: open-source, GPU-accelerated, built on Warp

---

## 9. Practical Lessons and Recommendations for Our Project

### 9.1 Feasibility Assessment

Our current bottleneck is PyElastica running at ~57 FPS with 16 parallel environments. A neural surrogate could potentially provide:

- **Conservative estimate:** 100-1,000x speedup (based on GNS, LSTM surrogate literature)
- **Optimistic estimate:** 5,000-44,000x speedup (based on GNS granular flow, DD-PINN)
- **Realistic for LSTM surrogate:** ~500-2,000x speedup when running on GPU with JAX/PyTorch batched inference

This would translate from ~57 FPS to potentially 5,700-114,000 FPS, making long training runs feasible.

### 9.2 Recommended Approach (Ranked by Practicality)

**Option A: LSTM Forward Dynamics Model (Lowest Risk)**
1. Collect 5,000-10,000 trajectories from PyElastica with randomized actions
2. Train LSTM to predict next_state from (state, action) sequences
3. Wrap as TorchRL environment
4. Train PPO inside learned environment
5. Periodically validate against ground-truth PyElastica
- **Pros:** Proven approach (arXiv:2410.18519), straightforward implementation
- **Cons:** Limited generalization, error accumulation over long rollouts

**Option B: KNODE-Cosserat Residual (Best Accuracy)**
1. Build simplified fast Cosserat model (fewer elements, larger timestep)
2. Train neural ODE residual to correct toward high-fidelity PyElastica
3. Use corrected model as RL environment
- **Pros:** Physics-informed, better generalization, less training data needed
- **Cons:** More complex implementation, requires simplified physics baseline

**Option C: DD-PINN Cosserat Surrogate (Highest Speedup)**
1. Train domain-decoupled PINN on Cosserat rod dynamics
2. Incorporate rod theory into PINN loss function
3. Use as 44,000x faster surrogate
- **Pros:** Extreme speedup, physics-constrained, proven for Cosserat rods
- **Cons:** Requires PINN expertise, may not handle CPG actuation patterns well

**Option D: Dreamer-Style World Model (Most General)**
1. Train latent world model (e.g., DreamerV3 architecture) on PyElastica interactions
2. Learn policy entirely in latent imagination
- **Pros:** Proven at scale, handles partial observability, uncertainty-aware
- **Cons:** Complex implementation, may be overkill for our state-based setup

### 9.3 Critical Implementation Details

1. **Noise injection during training** is essential -- add Gaussian noise to states during surrogate training to prevent rollout divergence
2. **Temporal unrolling >= 8 steps** during training significantly improves long-horizon stability
3. **Periodic ground-truth validation** -- every N policy updates, run a few episodes in real PyElastica to detect surrogate drift
4. **State normalization** -- normalize all state features to similar scales before surrogate training
5. **Action space coverage** -- ensure training data covers the full action space the RL agent might explore; use domain randomization
6. **Rollout horizon** -- match surrogate training rollout length to RL episode length

### 9.4 Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Surrogate diverges on long rollouts | High | Noise injection, unrolling, periodic correction |
| Policy exploits surrogate inaccuracies | Medium | Domain randomization, ensemble of surrogates |
| Insufficient action space coverage | Medium | Active data collection, mean-reverting random walks |
| Sim-to-real gap (surrogate-to-simulator) | Low-Medium | Periodic validation, fine-tuning on new data |
| Implementation complexity | Medium | Start with Option A (LSTM), graduate to B/C |

---

## 10. Key References

1. "Towards RL Controllers for Soft Robots using Learned Environments" (arXiv:2410.18519, 2024) -- LSTM surrogate for soft robot RL
2. "Robotic World Model" (arXiv:2501.10100, 2025) -- Neural simulator for robust policy optimization
3. "Neural Motion Simulator (MoSim)" (CVPR 2025) -- World model replacing dm_control
4. "Knowledge-based Neural ODEs for Cosserat Rod-based Soft Robots" (arXiv:2408.07776, 2024) -- KNODE residuals on Cosserat models
5. "DD-PINN for Cosserat Rod MPC" (arXiv:2508.12681, 2025) -- 44,000x speedup PINN surrogate
6. "Learning to Simulate Complex Physics with Graph Networks" (ICML 2020) -- GNS framework, 5,000x speedup
7. "DiffTaichi" (ICLR 2020) -- Differentiable simulation, ChainQueen 188x faster
8. "PlasticineLab" (ICLR 2021) -- Differentiable soft-body benchmark
9. "DreamerV3" (Nature 2025) -- World models for diverse domains
10. "Neural Robot Dynamics (NeRD)" (arXiv:2508.15755, 2025) -- Neural dynamics in NVIDIA Newton
11. "How Temporal Unrolling Supports Neural Physics Simulators" (TUM, 2024) -- Rollout stability analysis
12. "PDE-Refiner" (NeurIPS 2023) -- Iterative refinement for long rollouts
13. "Residual MBRL for Physical Dynamics" (NeurIPS 2022) -- Physics + neural residual
14. "PINNs for Continuum Robots" (IEEE 2024) -- Fast Cosserat rod approximation
15. "DiffPD" (ACM TOG 2022) -- Differentiable soft-body simulation
16. Hong et al. (Advanced Intelligent Systems, 2026) -- Surrogate bridging FEM and RL for soft robots
17. "ChainQueen" (ICRA 2019) -- Differentiable MPM for soft robotics
18. FourCastNet (NVIDIA, 2022-2025) -- 45,000x speedup weather surrogate
19. NVIDIA Newton/NeRD (2025) -- 70-100x GPU-accelerated robotics simulation
20. "GNS: A Generalizable Graph Neural Network-based Simulator" (JOSS, 2022) -- Open-source GNS implementation
