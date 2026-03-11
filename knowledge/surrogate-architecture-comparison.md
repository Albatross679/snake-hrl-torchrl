---
name: surrogate-architecture-comparison
description: Comprehensive comparison of neural network architectures beyond MLP for physics surrogate modeling of Cosserat rod dynamics in RL
type: knowledge
created: 2026-03-09T00:00:00
updated: 2026-03-09T18:00:00
tags: [neural-network, surrogate-model, GNN, transformer, FNO, DeepONet, neural-ODE, LSTM, Mamba, equivariant, 1D-CNN, architecture-comparison, cosserat-rod]
aliases: []
---

# Neural Network Architectures Beyond MLP for Physics Surrogate Modeling

## Context

Our snake locomotion RL system uses PyElastica (Cosserat rod, 20 segments, 21 nodes) running at ~57 FPS with 16 CPU envs. The existing feasibility analysis (`knowledge/neural-surrogate-cosserat-rod.md`) recommends an MLP surrogate predicting 2D rod state deltas (124D input/output). This document evaluates **nine alternative architectures** that could offer advantages over a plain MLP for this specific problem.

**Our surrogate task**: Given current rod state (124 floats: positions, velocities, angles, angular velocities) + action (5 floats) + phase encoding (2 floats) = 131 inputs, predict the next rod state (124 floats) after one macro-step (500 internal PyElastica integration steps).

### Verified: Per-Substep Locality vs Per-RL-Step Global Coupling

**Per-substep physics is strictly local (3-node stencil)**, verified against PyElastica source:

- **Curvature** at Voronoi point k: computed from relative rotation of director frames at nodes k and k+1 only (2-node stencil, `_rotations.py`).
- **Bending torque** on node k: `torque[k] = couple[k] - couple[k-1]`, where `couple[k]` depends on `kappa[k]` (nodes k, k+1) and `couple[k-1]` depends on `kappa[k-1]` (nodes k-1, k). Gives a **3-node stencil** (k-1, k, k+1).
- **No global coupling**: PyElastica uses explicit symplectic integrators (PositionVerlet/PEFRL). No tridiagonal solves, no implicit matrix inversions.

The project's own `get_curvatures()` in `src/physics/elastica_snake_robot.py` confirms:
```python
for i in range(1, self.num_nodes - 1):
    v1 = self.positions[i] - self.positions[i - 1]
    v2 = self.positions[i + 1] - self.positions[i]
```

**However**, the surrogate predicts 500 substeps in one shot. Information propagates ±1 node per substep → a perturbation at node 0 reaches node 20 after just 20 substeps, then bounces back and forth **~25 times** within a single RL step. **Every node influences every other node.** See `issues/surrogate-spatial-structure-analysis.md` for full analysis.

---

## 1. Graph Neural Networks (GNNs) / Message Passing Neural Networks

### How It Works for Rod Surrogate

The Cosserat rod is naturally a graph: each of the 20-21 nodes is a graph node, with edges connecting neighbors along the rod backbone. Message passing propagates information between neighboring segments, mimicking how forces transmit through the rod. Each message-passing step is analogous to a spatial differential operator.

**Architecture**: Encode per-node features (position, velocity, angle, angular velocity) -> multiple rounds of message passing between neighboring nodes -> decode per-node state deltas. The action and phase encoding are injected as global context features.

### Key Papers

- **GNS** (Sanchez-Gonzalez et al., ICML 2020, [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)): Particle-based GNN for fluids, rigid solids, and deformable materials. Up to 1000x speedup. Noise injection during training is critical for stable long rollouts. Uses M message-passing steps as a key hyperparameter; the paper found that "a greater number of message-passing steps yielded improved performance in both one-step and rollout accuracy, likely because increasing M allows computing longer-range, and more complex, interactions among particles." Computation scales linearly with M.
- **MeshGraphNets** (Pfaff et al., ICLR 2021, [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)): Operates on simulation meshes directly. Resolution-independent dynamics. Handles cloth, aerodynamics, structural mechanics. Processor typically uses **10-15 message passing layers** (NVIDIA's implementation documentation confirms this range). Each layer uses MLPs for edge and node updates with residual connections.
- **BSMS-GNN** (Cao et al., ICML 2023, [arXiv:2210.02573](https://arxiv.org/abs/2210.02573)): Bi-stride multi-scale pooling. Only 31-51% of computation vs. standard GNNs. Tested on elastic plates and surfaces.
- **Physics-Informed GNN for Soft Tissue** (Dalton et al., CMAME 2023, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782523004759)): GNN trained with minimum potential energy principle. Handles patient-specific geometries without low-order approximations.
- **PhysGNN** ([arXiv:2109.04352](https://arxiv.org/abs/2109.04352)): Predicts soft tissue deformation in ~4ms on GPU vs. ~11.5 seconds for FEM. ~2700x speedup.
- **Dynami-CAL GraphNet** (Nature Comms, 2025, [link](https://www.nature.com/articles/s41467-025-67802-5)): Physics-informed GNN conserving linear and angular momentum. Edge-local reference frames that are equivariant to rotations.
- **GNS vs Neural Operators** ([arXiv:2509.06154](https://arxiv.org/html/2509.06154v1)): GNS achieves <1% relative error with only 30 training trajectories. 82% less error accumulation than FNO autoregressive, 99.86% less than DeepONet autoregressive. But inference is ~60x slower than FNO.

### Pros for Rod Dynamics

- **Structural inductive bias**: Naturally encodes the 1D topology of the rod (node-edge chain). Forces between segment i and segment j must pass through all intermediate segments, matching physical causality.
- **Resolution generalization**: Could train on 20-segment rod and potentially generalize to different discretizations.
- **Data efficiency**: GNS achieves <1% error with only 30 training trajectories -- far fewer than MLP.
- **Contact handling**: Physics-encoded GNNs can handle deformable-rigid contact via cross-attention between object graphs.

### Cons for Rod Dynamics

- **Overkill for 20 nodes**: A 1D chain of 20 nodes is extremely simple as a graph. The message-passing machinery adds overhead for very little structural benefit over an MLP that already sees all 20 nodes simultaneously.
- **Inference slower than MLP**: GNNs with message passing require sequential computation through rounds. For 20 nodes, 2-3 message passing steps suffice, but each step involves scatter/gather operations that are less GPU-efficient than dense matrix multiplies in an MLP.
- **Parameter overhead**: Encoder + processor + decoder architecture has more components to tune.
- **Benchmark context**: GNN inference ~180 seconds for 1000 test cases vs. ~3 seconds for FNO and ~1.3 seconds for DeepONet on the same benchmark.

### Verdict for Our Problem

**Not recommended as the primary architecture.** The 1D chain of 20 nodes is too simple to benefit from graph structure. An MLP with 124 inputs already has full connectivity to all nodes. GNNs shine when the number of nodes is large (100+), the topology is complex, or resolution generalization is needed. However, GNN is a natural upgrade path if we ever move to more complex multi-rod assemblies or 3D configurations.

---

## 2. Neural ODEs / Neural Differential Equations

### How It Works for Rod Surrogate

Instead of directly predicting next_state from current_state, a neural ODE parameterizes the time derivative: dx/dt = f_theta(x, u). Integration is performed with a numerical ODE solver (e.g., Dormand-Prince). The network learns the continuous dynamics, and the ODE solver handles time-stepping.

**KNODE variant**: Use a known (but imperfect) physics model as the base dynamics, and add a neural network residual: dx/dt = f_physics(x, u) + f_theta(x, u). The NN only needs to learn the correction, not the entire dynamics.

### Key Papers

- **KNODE-Cosserat** (Hsieh et al., 2024, [arXiv:2408.07776](https://arxiv.org/abs/2408.07776)): Neural ODE residual corrects simplified Cosserat rod model. 58.7% accuracy improvement over physics-only. Trained on PyElastica data. Code: [github.com/hsiehScalAR/KNODE-Cosserat](https://github.com/hsiehScalAR/KNODE-Cosserat).
- **Neural ODEs for Model Order Reduction** (Wiley, 2025, [link](https://onlinelibrary.wiley.com/doi/10.1002/nme.70060)): Neural ODEs for surrogate modeling of stiff systems, directly relevant to rod dynamics with high-frequency modes.
- **Neural ODE Surrogate for Power Systems** ([arXiv:2405.06827](https://arxiv.org/abs/2405.06827)): Deep equilibrium layer + neural ODE for dynamic simulations. Accuracy comparable to physics-based surrogates.

### Pros for Rod Dynamics

- **Physics hybrid (KNODE)**: The correction-only approach is extremely data-efficient and naturally respects physics structure.
- **Continuous-time representation**: Can predict at arbitrary time resolutions, not just fixed dt.
- **Interpretable**: The residual term directly shows where the physics model is wrong.
- **Proven on Cosserat rods**: KNODE-Cosserat is the most directly relevant architecture to our problem.
- **Adjoint method**: Memory-efficient backpropagation through ODE solves.

### Cons for Rod Dynamics

- **Slower than MLP at inference**: The ODE solver requires multiple neural network evaluations per step (Dormand-Prince typically uses 6 evaluations per step). If the NN is 10x slower than a single MLP forward pass, a single step is ~60x slower.
- **KNODE requires a simplified physics model**: Need to implement a fast but inaccurate Cosserat model as the base -- significant engineering effort.
- **Stiffness issues**: Cosserat rod dynamics can be stiff, requiring implicit solvers or very small step sizes. Neural ODEs with stiff dynamics are an active research challenge.
- **Training complexity**: Backpropagation through ODE solvers (adjoint method) can have numerical stability issues.

### Verdict for Our Problem

**Strong option for Phase 2 (accuracy upgrade).** The KNODE-Cosserat approach is directly proven on our problem domain and uses PyElastica data. However, it is slower than a pure MLP surrogate at inference time and requires implementing a simplified physics backbone. Best used as an upgrade if MLP accuracy proves insufficient, not as the first approach.

---

## 3. Transformers for Dynamics Modeling

### How It Works for Rod Surrogate

Treat the rod's 20 segments as a sequence of tokens. Each token contains the per-segment state (position, velocity, angle, angular velocity). Self-attention lets each segment attend to every other segment, capturing long-range interactions along the rod. Action and phase information serve as conditioning tokens.

### Key Papers

- **PDE-Transformer** (May 2025, [arXiv:2505.24717](https://arxiv.org/html/2505.24717v1)): Multi-purpose transformer for PDEs. Works on regular grids. Deep conditioning mechanisms for PDE-specific information.
- **Graph Transformer Surrogate** (CMAME 2024, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782524006650)): Multi-head Graph Attention encoder + GRU decoder for spatiotemporal dynamics.
- **AB-UPT** (Feb 2025, [arXiv:2502.09692](https://arxiv.org/html/2502.09692v2)): Anchored-Branched Universal Physics Transformers. Multi-branch architecture separating geometry encoding, surface simulation, and volume simulation.
- **Physics-Embedded Transformer-CNN** (2025, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0997754625001530)): Hybrid CNN for local patterns + Transformer for long-range dependencies, informed by Navier-Stokes equations.
- **Multiple Physics Pretraining** (MPP, [OpenReview](https://openreview.net/forum?id=fH9eqpCcR3)): Large surrogate models pretrained on multiple heterogeneous physical systems simultaneously.

### Pros for Rod Dynamics

- **Global context**: Self-attention captures interactions between distant segments instantly (no multi-hop message passing).
- **Parallelizable training**: Unlike RNNs, all positions process simultaneously.
- **Flexible conditioning**: Can condition on actions, phase, material properties, etc. via cross-attention or token concatenation.
- **Pretrained foundation models**: Could potentially benefit from physics foundation models in the future.

### Cons for Rod Dynamics

- **Quadratic complexity**: O(N^2) attention for N segments. With N=20, this is only 400 operations -- negligible compared to MLP. But the overhead of attention heads, layer norms, and positional encodings adds constant-factor cost.
- **No spatial inductive bias**: Standard transformers don't know that segment 3 is between segments 2 and 4. This must be learned from data, requiring more training samples.
- **Heavy architecture**: Even a small transformer (4 layers, 4 heads, 128 dim) has more parameters than a comparably effective MLP for this problem size.
- **Diminishing returns at small scale**: Transformers excel at large-scale sequence modeling. For a 20-token sequence, the attention mechanism provides little advantage over an MLP that already processes all 124 features jointly.
- **Not proven for rod dynamics**: No papers applying transformers specifically to Cosserat rod surrogate modeling.

### Verdict for Our Problem

**Not recommended.** The sequence length (20 segments) is too short for transformers to provide meaningful advantages over MLPs. Transformers would add architectural complexity without proportional benefit. They become relevant if: (a) we move to much longer rods (100+ segments), (b) we need to process multi-modal inputs (vision + state), or (c) we want to leverage pretrained physics foundation models.

---

## 4. Fourier Neural Operators (FNO)

### How It Works for Rod Surrogate

FNO operates in spectral space: the rod state (as a function of arc-length) is transformed to Fourier space, multiplied by learnable spectral weights, then transformed back. This learns operators on function spaces rather than finite-dimensional vector spaces.

For our rod: treat the 20-segment state as a discrete sampling of a continuous function along the rod's arc-length. The FNO learns the operator that maps (current rod state function, action) to (next rod state function).

### Key Papers

- **Physics-Informed FNO for Beam Bending** (2025, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0141029625014750)): PIFNO for non-prismatic Euler-Bernoulli beams with variable cross-sections. Data-free (uses PDE loss only). Reformulates 4th-order beam equation as two 2nd-order equations.
- **LITEFNO** (NeurIPS ML4PS 2025, [PDF](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_287.pdf)): Lightweight FNO addressing spatio-temporal dynamics challenges.
- **Physics-Encoded FNO** (PeFNO, 2025, [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5280314)): Divergence-free stress field surrogate for solids.
- **FNO vs GNN Comparison** ([arXiv:2509.06154](https://arxiv.org/html/2509.06154v1)): FNO inference ~3 seconds for 1000 test cases vs ~180 seconds for GNS. But GNS achieves lower error with less data.
- **Mamba Neural Operator** (NeurIPS 2024, [arXiv:2410.02113](https://arxiv.org/abs/2410.02113)): SSM-based alternative to FNO that captures long-range dependencies better. See Section 8 below.

### Pros for Rod Dynamics

- **Spectral efficiency**: Rod dynamics have strong spectral structure (serpenoid wave is nearly sinusoidal). FNO is naturally suited to capture this.
- **Resolution independence**: Learns operators, not discretization-specific mappings. Can train on 20-segment data and evaluate on finer discretization.
- **Blazing fast inference**: Potentially 100,000x speedup over PDE solvers, via GPU-accelerated FFT.
- **Data-free training possible**: PIFNO can train using only PDE residual loss, no simulation data needed.
- **Directly applicable to beam/rod problems**: PIFNO for non-prismatic beams is the closest published work to our problem.

### Cons for Rod Dynamics

- **Spectral bias**: FNO preferentially captures low-frequency modes and may miss high-frequency features (e.g., sharp curvature gradients at rod endpoints or contact points).
- **Regular grid requirement**: Standard FNO assumes uniform spatial sampling. Our 20-segment rod satisfies this, but curved rods with non-uniform arc-length discretization would need Geo-FNO extensions.
- **1D specificity**: Most FNO work is on 2D/3D PDEs. A 1D rod with only 20 points is a very short "signal" for spectral methods. With N=20, only ~10 meaningful Fourier modes exist.
- **Action conditioning is awkward**: FNO operates on spatial functions, but our action (5 floats) is not a spatial function. Requires architectural modifications to inject non-spatial inputs.
- **Overkill for our problem size**: The operator-learning framework shines for high-resolution PDE fields. For 20 discrete points, a standard MLP already operates at effectively the same level of expressiveness.

### Verdict for Our Problem

**Not recommended for current problem scale.** The 20-segment rod provides too few spatial points for FNO's spectral approach to offer meaningful advantages. FNO would become valuable if: (a) we need resolution-independent dynamics (transfer across different rod discretizations), (b) we move to much finer discretization (100+ segments), or (c) we want data-free training via PDE loss only. The PIFNO for beam bending is conceptually relevant and worth monitoring.

---

## 5. DeepONet (Deep Operator Networks)

### How It Works for Rod Surrogate

DeepONet has two sub-networks: a **branch net** that encodes the input function (e.g., current rod state evaluated at fixed sensor locations) and a **trunk net** that encodes query locations (e.g., where to predict the output). The output is the dot product of their representations. For dynamics prediction, the branch net takes the current state + action, and the trunk net takes the time at which we want the prediction.

### Key Papers

- **DeepONet** (Lu et al., Nature Machine Intelligence 2021, [link](https://www.nature.com/articles/s42256-021-00302-5)): Universal approximation theorem for operators. Significantly reduces generalization error vs. fully-connected networks.
- **PI-DeepONet** ([topic](https://www.emergentmind.com/topics/physics-informed-deep-operator-network-pi-deeponet)): Physics-informed variant. 1-2 orders of magnitude lower prediction error when labeled data is scarce. Generalizes to out-of-distribution inputs.
- **DeepOMamba** ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999125005558)): Combines DeepONet with Mamba SSM. Best computational cost-to-performance ratio for spatio-temporal PDEs.
- **RaNN-DeepONet** ([arXiv:2503.00317](https://arxiv.org/abs/2503.00317)): Randomized hidden layers replace nonconvex optimization with linear least-squares solve. Orders-of-magnitude faster training.
- **One-shot operator learning** (Feng et al., 2025): Self-supervised + meta-learning achieves 5-10% relative error from a single solution trajectory.

### Pros for Rod Dynamics

- **Continuous output**: Can predict state at arbitrary spatial locations and times, not just at fixed grid points.
- **Generalization**: Can generalize across different input functions (e.g., different initial conditions, action profiles) without retraining.
- **Data efficiency with physics**: PI-DeepONet needs 5-10x less data than data-only approaches.
- **Flexible querying**: Can ask "what is the rod state at arc-length s at time t" for any (s, t) -- useful for adaptive resolution.
- **Fast training variant**: RaNN-DeepONet reduces training to a single linear solve.

### Cons for Rod Dynamics

- **Lower accuracy than GNN/FNO in autoregressive mode**: DeepONet autoregressive has 99.86% more error accumulation than GNS over long rollouts (from the GNS vs Neural Operators benchmark).
- **Not designed for autoregressive stepping**: DeepONet naturally maps input functions to output functions. Using it as a step-by-step dynamics model requires feeding its output back as input, which it was not designed for.
- **Architecture complexity**: Two separate networks (branch + trunk) with their own hyperparameters. More complex than a single MLP.
- **Dot-product output can be limiting**: The bilinear output structure may not capture complex nonlinear interactions as well as a direct MLP mapping.
- **No specific Cosserat rod results**: While DeepONet has been applied to various PDEs, no published work applies it specifically to Cosserat rod dynamics.

### Verdict for Our Problem

**Not recommended for autoregressive dynamics stepping.** DeepONet's strength is in mapping between function spaces (e.g., given initial condition function, predict solution function), not in step-by-step dynamics prediction for RL. Its error accumulation in autoregressive mode is the worst among tested architectures. However, PI-DeepONet could be valuable for a different formulation: predicting the entire trajectory given an action sequence, bypassing autoregressive stepping entirely.

---

## 6. Recurrent Architectures (LSTM/GRU)

### How It Works for Rod Surrogate

An LSTM or GRU processes a sequence of (state, action) pairs and predicts the next state. The hidden state captures temporal context -- information about the dynamics history that affects future evolution. This is particularly relevant when the single-step state is not fully Markov (e.g., if the observation doesn't capture all internal rod dynamics).

**SoRoLEX approach**: Train an LSTM on real/simulated robot trajectories to learn forward dynamics. Deploy in JAX with GPU parallelism for RL training.

### Key Papers

- **SoRoLEX** (Uljad et al., RoboSoft 2024, [arXiv:2410.18519](https://arxiv.org/abs/2410.18519)): LSTM learns soft robot dynamics from data. JAX-based parallel RL. Achieves convergence within 3mm of target position. The most directly relevant system architecture to our problem.
- **LSTM/GRU for MPC** (Neurocomputing 2025, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231225003844)): Comprehensive review of recurrent architectures in model predictive control. GRU reduces computation by ~29% and memory by 50% vs. LSTM.
- **RNN for Articulated Soft Robots** ([arXiv:2411.05616](https://arxiv.org/html/2411.05616v1)): LSTM for single actuators, GRU for articulated soft robots in nonlinear MPC.
- **Encoder-Decoder GRU for Structural Dynamics** (Comp. Mech. 2023, [Springer](https://link.springer.com/article/10.1007/s00466-023-02317-8)): Modified attention + GRU encoder-decoder for predicting dynamic response of shock-loaded plates. Best result among compared methods.

### Pros for Rod Dynamics

- **Temporal context**: Hidden state captures dynamics history, handling any non-Markov effects in the observation.
- **Proven for soft robot surrogate**: SoRoLEX demonstrated successful RL training with LSTM-learned environment.
- **Captures hysteresis**: RNNs can model rate-dependent and history-dependent behavior, relevant for viscoelastic materials.
- **Moderate computational cost**: GRU inference is O(hidden_dim^2) per step -- comparable to MLP for similar hidden sizes.
- **GPU-parallelizable for RL**: JAX/PyTorch can batch thousands of independent LSTM rollouts on GPU.

### Cons for Rod Dynamics

- **Our state IS Markov**: We verified that the full 2D rod state (124D) is Markov -- the dynamics depend only on current state + action, not history. A recurrent architecture adds unnecessary complexity when the state is Markov.
- **Sequential training bottleneck**: LSTM/GRU must process training sequences sequentially (though batching across sequences helps).
- **Vanishing/exploding gradients for long sequences**: Training on 500-step episodes can be challenging.
- **Hidden state initialization**: At episode start, the hidden state is zero (untrained), leading to poor initial predictions. Requires burn-in period.
- **Non-parallelizable per-episode**: Within a single episode, each step depends on the previous hidden state, preventing temporal parallelism.
- **More parameters than equivalent MLP**: LSTM with 512 hidden units has ~4x more parameters than an MLP layer of the same width.

### Verdict for Our Problem

**Not recommended as primary architecture because our state is Markov.** The full 124D rod state contains all the information needed for prediction -- no hidden state is required. Adding recurrence would increase parameters and training complexity without improving accuracy. However, LSTM/GRU becomes relevant if we reduce the surrogate to operate on the 14D observation space (which IS non-Markov), where the hidden state would implicitly reconstruct unmeasured state variables.

---

## 7. 1D Convolutional Approaches

### How It Works for Rod Surrogate

Treat the rod's per-segment state as a 1D signal along the body axis. Apply 1D convolution kernels that slide along the rod, capturing local spatial patterns (e.g., curvature waves, bending modes). Multiple layers with increasing receptive field capture progressively longer-range spatial interactions.

**Architecture**: Each segment has a feature vector (position, velocity, angle, angular velocity ~6 features). Stack as (batch, segments=20, features=6) tensor. Apply 1D conv layers along the segment axis. Action/phase injected via FiLM conditioning or concatenation after global pooling.

### Key Papers

- **LaDEEP for Slender Solids** ([arXiv:2506.06001](https://arxiv.org/html/2506.06001v1)): Encodes partitioned regions of slender solids into token sequences, maintaining spatial order. Uses 1D convolution along the spatial axis.
- **TCN vs LSTM for Structural Dynamics** (Comp. Mech. 2023, [Springer](https://link.springer.com/article/10.1007/s00466-023-02317-8)): Temporal Convolutional Networks compared with LSTM/GRU for shock-loaded plate dynamics.
- **1D CNN for Rod/Wave Dynamics** ([ResearchGate](https://www.researchgate.net/publication/360020248_A_deep_learning_based_surrogate_modelling_for_wave_propagation_in_structures)): Wave propagation in rod structures using deep learning surrogate.

### Pros for Rod Dynamics

- **Spatial locality inductive bias**: Convolution kernels naturally encode that neighboring segments interact more strongly than distant ones, matching the local nature of Cosserat rod equations (stresses depend on local strain gradients).
- **Weight sharing**: The same kernel applies at every position along the rod, enforcing translational equivariance along the body axis. This is physically meaningful: the dynamics equations are the same at every segment (assuming uniform material).
- **Parameter efficient**: A 1D conv layer with kernel size 3 and 64 channels has only 64*64*3 = 12K parameters, much less than an equivalent dense layer.
- **GPU-efficient**: 1D convolution is highly optimized on GPUs via cuDNN.
- **Multi-scale features via pooling/dilated convs**: Hierarchical 1D conv can capture both local bending and global body shape.

### Cons for Rod Dynamics

- **Limited receptive field**: A convolution kernel of size 3 only sees 3 adjacent segments per layer. To propagate information across the full 20-segment rod requires ~10 layers (or dilated convolutions). An MLP sees all segments simultaneously in one layer.
- **Boundary effects**: The rod endpoints have different physics (free ends, or constrained). 1D conv applies the same kernel everywhere, including boundaries, which is physically wrong at the ends.
- **Action injection is awkward**: The action (5 global floats) affects the entire rod but is not spatially distributed. Requires conditioning mechanisms (FiLM, concatenation after global pooling, etc.).
- **Only 20 segments**: The signal is extremely short. With kernel size 3, there are only 18 valid positions. Standard deep 1D conv architectures (WaveNet, TCN) are designed for sequences of hundreds to thousands of points.
- **Breaks for non-uniform properties**: If the rod has varying material properties (e.g., different stiffness at head vs. tail), weight sharing becomes a disadvantage.

### Verdict for Our Problem

**Not recommended — wrong inductive bias.** While per-substep physics is local (3-node stencil), the surrogate predicts 500 substeps integrated together, during which information traverses the full rod ~25 times. A narrow-kernel CNN would impose locality constraints that contradict the globally-coupled target mapping. Additionally, the 20-segment rod is too short for 1D convolution to provide meaningful advantages — a kernel of size 21 degenerates to a fully-connected layer. A 1D conv first layer could still serve as a lightweight spatial feature extractor before an MLP trunk, but the expected benefit is marginal.

---

## 8. State Space Models (S4, Mamba)

### How It Works for Rod Surrogate

State Space Models (SSMs) model sequences through a continuous-time linear dynamical system: dx/dt = Ax + Bu, y = Cx + Du. S4 constrains the state matrix A to have special structure (HiPPO initialization) enabling efficient computation. Mamba adds input-dependent (selective) parameterization, making the model adaptive to each input.

For rod dynamics, SSMs can operate in two ways:
1. **Temporal SSM**: Process a time series of rod states, predicting forward dynamics autoregressively.
2. **Spatial SSM**: Process the spatial sequence of segments (20 nodes along the rod body), treating arc-length as the "sequence dimension."

### Key Papers

- **Mamba Neural Operator** (NeurIPS 2024, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5ee553ec47c31e46a1209bb858b30aa5-Abstract-Conference.html)): O(N) complexity neural operator using Mamba. Alias-free architecture. New SOTA on PDE benchmarks with fewer parameters and better efficiency. Proves continuous-discrete equivalence for operator approximation.
- **MNO: Transformers vs SSMs for PDEs** ([arXiv:2410.02113](https://arxiv.org/abs/2410.02113)): Mamba Neural Operator achieves 40-65% RMSE reduction vs. transformers. Better stability in frequency response. Demonstrates Mamba as a drop-in replacement for attention in neural operators.
- **DeepOMamba** ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999125005558)): Best computational cost/performance ratio by combining DeepONet with Mamba for spatio-temporal PDEs.
- **S4 to Mamba Survey** (March 2025, [arXiv:2503.18970](https://www.arxiv.org/pdf/2503.18970)): Comprehensive survey covering evolution from S4 to Mamba-2, including applications in time-series, RL, and physics.

### Pros for Rod Dynamics

- **Linear complexity**: O(N) vs. O(N^2) for transformers. But with N=20 segments, this advantage is negligible.
- **Superior frequency response**: Mamba maintains stable frequency profile while transformers show sharp fluctuations, important for capturing serpenoid wave dynamics accurately.
- **Strong with limited data**: Mamba Neural Operator achieved lower nRMSE with only 1K training samples than transformers with more data.
- **Continuous-time formulation**: SSMs are inherently continuous-time systems, matching the continuous dynamics of the Cosserat rod ODE/PDE.
- **Proven for PDE surrogate**: MambaNO achieves SOTA on multiple PDE benchmarks.

### Cons for Rod Dynamics

- **Very new architecture**: Mamba was introduced in late 2023. Physics applications are emerging but limited. Less mature tooling than MLP/LSTM/GNN.
- **Designed for long sequences**: S4/Mamba's advantages emerge for sequences of 1000+ tokens. Our 20-segment rod does not benefit from their long-range efficiency.
- **Selective mechanism overhead**: Mamba's input-dependent parameterization adds computation per element. For very short sequences, this overhead dominates any efficiency gains.
- **Hardware requirements**: Mamba's custom CUDA kernels require recent GPU architectures. Our V100 may not be optimal.
- **No Cosserat rod applications**: No published work applies Mamba/S4 to rod or beam dynamics.
- **Complexity of implementation**: Mamba requires specialized CUDA kernels for efficient hardware utilization.

### Verdict for Our Problem

**Not recommended for current problem.** SSMs are designed for long sequences (1000+ tokens) and our rod has only 20 segments. The architecture adds complexity without proportional benefit at this scale. However, **Mamba Neural Operator is a promising direction to watch** for future PDE surrogate work -- its superior frequency response and data efficiency could become relevant if we move to higher-resolution discretizations or multi-rod systems.

---

## 9. Equivariant Neural Networks

### How It Works for Rod Surrogate

Equivariant neural networks embed symmetry groups (e.g., SE(2) for 2D translations and rotations, SE(3) for 3D) directly into the network architecture. If the input is rotated/translated, the output transforms consistently. For rod dynamics, SE(2)-equivariance means: if the entire rod configuration is rotated by angle theta, the predicted dynamics rotate by the same angle.

### Key Papers

- **NequIP** (Batzner et al., [arXiv:2101.03164](https://arxiv.org/abs/2101.03164)): SE(3)-equivariant GNN for interatomic potentials. **3 orders of magnitude fewer training data** than conventional approaches. Outperforms models with 1000x more data.
- **GeoNorm for SE(3) GNNs** (IJCAI 2024, [link](https://www.ijcai.org/proceedings/2024/661)): Normalization techniques for SE(3)-equivariant GNNs. Addresses scalability to large particle systems.
- **Equivariant Neural Simulators** ([arXiv:2305.14286](https://arxiv.org/abs/2305.14286)): Equivariant architecture for stochastic spatiotemporal dynamics. Improves simulation quality, data efficiency, rollout stability, and uncertainty quantification.
- **SE(2)-Equivariant Pushing Dynamics** (CoRL 2022, [PMLR](https://proceedings.mlr.press/v205/kim23b/kim23b.pdf)): Equivariant model for tabletop manipulation dynamics. Directly relevant to our 2D setting.
- **Topology-Integrating GNN** (Nature Comms 2025, [link](https://www.nature.com/articles/s41467-025-62250-7)): Higher-order topology complexes with physics-informed message passing for rigid body dynamics.

### Pros for Rod Dynamics

- **Dramatic data efficiency**: NequIP achieves equivalent accuracy with 3 orders of magnitude fewer training samples. This is the single biggest advantage -- we could potentially train an accurate surrogate with 1K transitions instead of 1M.
- **Physical consistency**: SE(2)-equivariance means the model inherently respects the rotational and translational symmetry of the physics. A snake robot performing the same gait in different orientations will produce physically consistent results without needing data augmentation.
- **Better generalization**: The network cannot learn spurious orientation-dependent features. It generalizes perfectly across all orientations from any single orientation's training data.
- **Rollout stability**: Equivariant simulators show improved long-horizon prediction stability.
- **Directly applicable to our 2D problem**: SE(2) is the relevant symmetry group for our 2D rod. Implementation is simpler than full SE(3).

### Cons for Rod Dynamics

- **Implementation complexity**: Equivariant layers (Clebsch-Gordan tensor products, Wigner D-matrices) are significantly more complex than standard linear layers. Libraries exist (e3nn, escnn) but integration with custom architectures requires expertise.
- **Computational overhead**: Equivariant layers are 2-5x more expensive per parameter than standard layers, due to tensor product operations.
- **SE(2) symmetry may already be handled**: If we train with data augmentation (random rotations/translations of training data), a standard MLP can learn approximate equivariance. The question is whether built-in equivariance is worth the architectural complexity.
- **Action symmetry is broken**: The action space (CPG parameters) may have orientation-dependent meaning. If action[0] = "turn left" (absolute direction), the dynamics are not SE(2)-equivariant. However, if actions are body-relative (e.g., "increase left curvature"), equivariance holds.
- **Small system**: For 20 nodes in 2D, the symmetry group SE(2) is relatively simple. A standard MLP with data augmentation may approximate equivariance adequately.

### Verdict for Our Problem

**Promising for data efficiency, but high implementation cost.** The 1000x reduction in training data is extremely attractive -- it would reduce our data collection from 24 minutes to ~1.5 seconds. However, the implementation complexity of SE(2)-equivariant layers is significant. **Recommended approach**: First try MLP with SE(2) data augmentation (random rotations/translations of training trajectories). If this achieves sufficient accuracy with available data, equivariant networks are unnecessary. If data efficiency is a bottleneck (unlikely given cheap PyElastica data), equivariant networks become compelling.

**Key consideration**: Our actions are body-relative (CPG amplitude, frequency, wave number) which means the dynamics ARE SE(2)-equivariant. This makes equivariant networks theoretically well-suited.

---

## 10. Multi-Scale / Hierarchical Approaches

### How It Works for Rod Surrogate

Multi-scale architectures explicitly handle the local-to-global information transition that occurs in our problem: per-substep physics is local (3-node stencil), but over 500 substeps information propagates globally across all 21 nodes. These architectures process the spatial domain at multiple resolutions simultaneously, with coarse-scale representations capturing long-range interactions and fine-scale representations preserving local detail.

### Key Papers and Architectures

- **X-MeshGraphNet** (NVIDIA, Nov 2024, [arXiv:2411.17164](https://arxiv.org/abs/2411.17164)): Multi-scale extension of MeshGraphNet. Builds multi-scale graphs by iteratively combining coarse and fine-resolution point clouds, where each level refines the previous. Enables efficient long-range interactions via coarse-level message passing while preserving fine-scale detail. Incorporates halo regions for seamless cross-partition message passing with gradient aggregation.
- **DCNO — Dilated Convolution Neural Operator** (Aug 2024, [arXiv:2408.00775](https://arxiv.org/abs/2408.00775)): Combines Fourier layers (low-frequency global components) with dilated convolution layers at multiple dilation rates (high-frequency local details). Dilated convolutions expand receptive field without extra parameters by introducing gaps between kernel elements. Tested on multiscale elliptic, Navier-Stokes, and Helmholtz equations.
- **MSPT — Multi-Scale Patch Transformer** (Dec 2025, [arXiv:2512.01738](https://arxiv.org/html/2512.01738v1)): Partitions domain into spatial patches using ball trees, applies local self-attention within patches and global attention to pooled representations (PMSA). Captures fine-grained local interactions and long-range global dependencies in parallel.
- **BSMS-GNN** (Cao et al., ICML 2023, [arXiv:2210.02573](https://arxiv.org/abs/2210.02573)): Bi-stride multi-scale pooling for mesh-based GNNs. Only 31-51% of computation vs. standard GNNs. Tested on elastic plates and surfaces.
- **Neural Modular Physics (NMP)** (Dec 2025, [arXiv:2512.15083](https://arxiv.org/abs/2512.15083)): Decomposes elastic simulation into modular components: a neural constitutive module (material law) and a neural integration module (time-stepper), connected through intermediate physical quantities. Two-stage training: independent module training with physics supervision, then joint fine-tuning. Modules can be swapped with classical solvers ("interchange inference").
- **Hierarchical Deep Learning Time-Steppers** (Phil. Trans. R. Soc. A, 2022, [link](https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0200)): Learns flow-maps at different time scales explicitly, with each sub-network focusing on its intrinsic range of interest. Maintains computational efficiency through vectorized computation.

### Pros for Rod Dynamics

- **Directly addresses the local-to-global problem**: The 500-substep integration propagates information from a 3-node stencil globally. Multi-scale architectures explicitly model this transition at multiple spatial scales.
- **DCNO is conceptually appealing**: The rod has both smooth global motion (captured by low Fourier modes) and local curvature variations (captured by dilated convolutions). DCNO's hybrid approach naturally separates these.
- **NMP's modularity**: Could replace only the constitutive model (material response) with a neural network while keeping analytical time integration, or vice versa. This separation is natural for Cosserat rods where the constitutive law (stress-strain) and kinematics (geometry) are distinct.

### Cons for Rod Dynamics

- **Overkill at 20 nodes**: Multi-scale approaches shine when there are hundreds or thousands of spatial DOFs. With only 20-21 nodes, the "coarse" and "fine" scales are barely distinguishable.
- **X-MeshGraphNet complexity**: Designed for large 3D meshes with millions of nodes. The graph construction, partitioning, and halo management overhead far exceeds what a 21-node 1D chain needs.
- **DCNO requires spatial grid**: Dilated convolutions assume regular spatial sampling. Our 20-segment rod satisfies this, but the signal is very short (only ~10 meaningful dilation levels before wrapping around).
- **NMP two-stage training**: More complex training pipeline. The separation into constitutive and integration modules may not provide clear benefits when the entire state is only 124D.

### Verdict for Our Problem

**Not recommended at current scale, but DCNO-inspired hybrid is worth noting.** The idea of combining a Fourier/spectral component (for global wave shape) with a local convolution component (for per-node details) could be implemented as a lightweight modification to the MLP -- e.g., appending a few Fourier basis features of the node positions to the input. NMP's modular decomposition is intellectually elegant but over-engineered for 21 nodes. These approaches become relevant for higher-resolution discretizations (100+ nodes) or multi-rod assemblies.

---

## 11. DD-PINN and Physics-Augmented Constitutive Networks

### Specific to Cosserat Rod Surrogates

Two recently published architectures are specifically designed for Cosserat rod dynamics and deserve separate mention:

#### DD-PINN (Domain-Decomposed PINN for Cosserat MPC)
- **Paper**: Stolzle et al. (2025), [arXiv:2508.12681](https://arxiv.org/abs/2508.12681)
- **Speedup**: 44,000x over dynamic Cosserat rod simulation
- **Architecture**: Domain-decomposed PINN serving as a surrogate for the dynamic Cosserat rod model, used within nonlinear MPC with an unscented Kalman filter for state estimation
- **Accuracy**: End-effector position errors below 3mm (2.3% of actuator length)
- **Key feature**: Adaptable bending stiffness, meaning the surrogate can handle varying material properties without retraining

#### Physics-Augmented NNs for Geometrically Exact Beams
- **Paper**: Benady et al. (2024), [arXiv:2407.00640](https://arxiv.org/abs/2407.00640), published in CMAME
- **Architecture**: Feed-forward NNs represent an effective hyperelastic beam potential; forces and moments are obtained as gradients of this potential, ensuring thermodynamic consistency
- **Key feature**: Symmetry conditions implemented via invariant-based approach (transverse isotropy, point symmetry), providing physics-correct constitutive behavior by construction
- **Relevance**: This learns the constitutive law (stress-strain relationship) rather than the full dynamics. Could be combined with an analytical integrator for a physics-augmented surrogate.

#### Latent Neural ODE with Autoencoder
- **Paper**: (March 2026), [arXiv:2603.03238](https://arxiv.org/html/2603.03238)
- **Architecture**: Autoencoder for nonlinear dimensionality reduction + neural ODE for latent dynamics. Investigates geometric regularization (near-isometry, stochastic decoder gain penalty, curvature penalty, Stiefel projection).
- **For soft robots**: Attention Broadcast Decoder (ABCD) achieves 5.7x error reduction for Koopman operators and 3.5x for oscillator networks on two-segment soft robot ([arXiv:2511.18322](https://arxiv.org/html/2511.18322)).
- **Relevance**: Could compress the 124D rod state to a lower-dimensional latent space where dynamics are easier to learn. The autoencoder captures the rod's constraint manifold (e.g., inextensibility, unit quaternion constraints).

### Verdict for Our Problem

**DD-PINN is impressive but targets a different use case** (real-time MPC for continuum robots, not RL training). The 44,000x speedup is over a full Cosserat solver, not over an MLP surrogate. **Physics-augmented constitutive NNs** could inform a more principled surrogate design where the NN learns only the material response while kinematics are handled analytically. **Latent Neural ODE** is interesting if the 124D state has significant redundancy that an autoencoder could exploit.

---

## Architecture Comparison Matrix

| Architecture | Inference Speed | Data Efficiency | Accuracy | Handles Rod Topology | Implementation Complexity | Recommended? |
|---|---|---|---|---|---|---|
| **MLP** (baseline) | Fastest | Moderate (500K-1M) | Good | No (flat vector) | Simplest | **Yes (Phase 1)** |
| **GNN** | Moderate | High (30-1K trajectories) | Best | Yes (1D chain) | Moderate | No (overkill for 20 nodes) |
| **Neural ODE (KNODE)** | Slow (multi-eval) | Very high (few trajectories) | Best (physics-informed) | Via physics model | High | **Yes (Phase 2)** |
| **Transformer** | Moderate | Low | Good | No (needs learned) | Moderate-High | No (sequence too short) |
| **FNO** | Very fast | Moderate | Good (low-freq) | Yes (spectral) | Moderate | No (too few spatial points) |
| **DeepONet** | Fast | High (with PI) | Poor (autoregressive) | Via branch/trunk | Moderate | No (poor for step-by-step) |
| **LSTM/GRU** | Moderate | Moderate | Good (temporal) | No | Low-Moderate | No (state is Markov) |
| **1D CNN** | Fast | Moderate | Good (local) | Yes (spatial conv) | Low | No (wrong bias for 500-substep global coupling) |
| **Mamba/S4** | Very fast (long seq) | High | Best (freq response) | Yes (spatial SSM) | High | No (sequence too short) |
| **Equivariant NN** | Moderate | **1000x better** | Very good | Yes (SE(2) symmetry) | Very high | Data-limited only |
| **Multi-scale (DCNO/X-MGN)** | Fast | Moderate | Very good (multi-freq) | Yes (spectral+local) | High | No (too few nodes) |
| **DD-PINN** | Moderate | High (physics-informed) | Very good | Yes (domain decomp) | Very high | No (targets MPC, not RL) |
| **Latent Neural ODE** | Moderate | High | Good (compressed) | Via autoencoder | High | Interesting if state redundant |

---

## Recommended Architecture Strategy for Our Problem

### Phase 1: MLP with Enhancements (Immediate, ~8 days)
- **Architecture**: 3-layer MLP (512 units, SiLU), state-delta prediction
- **Enhancement 1**: SE(2) data augmentation -- randomly rotate/translate training trajectories
- **Enhancement 2**: Multi-step rollout loss (5-10 steps)
- **Enhancement 3**: Training noise injection (Gaussian, sigma=0.01 * state_std)
- **Expected throughput**: ~6M steps/sec on V100 (17,000x speedup)

### Phase 2: Hybrid Physics-NN (If MLP accuracy insufficient, +2-3 weeks)
- **Architecture**: KNODE-Cosserat style -- simplified physics backbone + MLP residual
- **Alternative**: 1D CNN first layer (captures spatial structure) + MLP trunk
- **Expected benefit**: Better accuracy with less data, more stable long rollouts

### Phase 3: Advanced Architecture (Future research, +1-2 months)
- **Option A**: SE(2)-equivariant MLP (if data efficiency becomes critical)
- **Option B**: Mamba spatial operator (if moving to higher-resolution rods)
- **Option C**: GNN (if moving to multi-rod or 3D assemblies)
- **Option D**: Structure-preserving (port-Hamiltonian) for guaranteed energy stability

### Cross-Cutting Insight: The 500-Substep Global Coupling Problem

The core challenge is that per-substep physics is local (3-node stencil), but over 500 substeps information propagates globally across all 21 nodes. This is the analog of "how many message-passing steps do I need?" in GNN terms.

For a 21-node 1D chain, information from one end reaches the other in ~10 message-passing steps (each step covers 1 edge hop). GNS/MeshGraphNets typically use 10-15 processor layers precisely to ensure global information propagation across the mesh. Our MLP achieves this "for free" -- a single dense layer connects all 124 inputs to all hidden units, providing instant global coupling.

The MLP's flat architecture is actually well-matched to this problem because:
1. The spatial domain is small (21 nodes) -- global connectivity is cheap
2. The temporal aggregation (500 substeps -> 1 macro-step) is the hard part, and all architectures must learn this mapping from data regardless of spatial inductive bias
3. The MLP's lack of spatial structure is a feature, not a bug, at this scale -- it allows the network to discover whatever spatial patterns matter without being constrained by assumptions

**The real leverage points for improving on MLP are**: (a) delta/residual prediction (already implemented), (b) multi-step rollout loss during training (already implemented), (c) training noise injection for rollout stability (already implemented), (d) SE(2) data augmentation (not yet implemented), and (e) per-feature-group normalization exploiting the known state structure (not yet implemented).

### What NOT to Pursue
- **Transformer**: Sequence too short, no advantage over MLP
- **FNO**: Too few spatial points for spectral methods
- **DeepONet for step-by-step**: Poor autoregressive error accumulation
- **LSTM/GRU**: State is Markov, recurrence unnecessary

---

## Key Quantitative References

| Metric | Value | Source |
|---|---|---|
| GNS vs FNO inference | GNS: 180s, FNO: 3s, DeepONet: 1.3s (1000 test cases) | [arXiv:2509.06154](https://arxiv.org/html/2509.06154v1) |
| GNS data efficiency | <1% error with 30 training trajectories | [arXiv:2509.06154](https://arxiv.org/html/2509.06154v1) |
| DD-PINN Cosserat speedup | 44,000x | [arXiv:2508.12681](https://arxiv.org/abs/2508.12681) |
| KNODE accuracy improvement | 58.7% over physics-only | [arXiv:2408.07776](https://arxiv.org/abs/2408.07776) |
| Equivariant data efficiency | 1000x fewer samples (NequIP) | [arXiv:2101.03164](https://arxiv.org/abs/2101.03164) |
| Mamba vs Transformer (Darcy Flow) | 40% RMSE reduction | [arXiv:2410.02113](https://arxiv.org/abs/2410.02113) |
| PhysGNN soft tissue | 4ms GPU vs 11.5s FEM CPU (~2700x) | [arXiv:2109.04352](https://arxiv.org/abs/2109.04352) |
| PINN articulated soft robot | 467x speedup | [arXiv:2502.01916](https://arxiv.org/abs/2502.01916) |
| GRU vs LSTM | 29% less computation, 50% less memory | [Neurocomputing 2025](https://www.sciencedirect.com/science/article/pii/S0925231225003844) |
| PINN surrogate RL training speedup | 50% faster policy training | [arXiv:2510.17380](https://arxiv.org/abs/2510.17380) |
| NeurIPS ML4CFD GNN speedup | 300-600x over CFD solver | [arXiv:2506.08516](https://arxiv.org/html/2506.08516) |

---

## References

1. Sanchez-Gonzalez et al. (2020). "Learning to Simulate Complex Physics with Graph Networks." [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)
2. Pfaff et al. (2021). "Learning Mesh-Based Simulation with Graph Networks." [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
3. Cao et al. (2023). "BSMS-GNN: Efficient Mesh-Based Physical Simulation." [arXiv:2210.02573](https://arxiv.org/abs/2210.02573)
4. Dalton et al. (2023). "Physics-informed GNN emulation of soft-tissue mechanics." [CMAME](https://www.sciencedirect.com/science/article/pii/S0045782523004759)
5. Hsieh et al. (2024). "KNODE-Cosserat." [arXiv:2408.07776](https://arxiv.org/abs/2408.07776)
6. Stolzle et al. (2025). "DD-PINN for Cosserat Rod MPC." [arXiv:2508.12681](https://arxiv.org/abs/2508.12681)
7. Stolzle et al. (2025). "Generalizable and Fast Surrogates for Articulated Soft Robots." [arXiv:2502.01916](https://arxiv.org/abs/2502.01916)
8. (2025). "PIFNO for Non-Prismatic Beam Bending." [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0141029625014750)
9. Lu et al. (2021). "DeepONet." [Nature MI](https://www.nature.com/articles/s42256-021-00302-5)
10. Cheng et al. (2024). "Mamba Neural Operator." [arXiv:2410.02113](https://arxiv.org/abs/2410.02113)
11. Zheng et al. (2024). "Alias-Free Mamba Neural Operator." [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5ee553ec47c31e46a1209bb858b30aa5-Abstract-Conference.html)
12. Batzner et al. (2021). "NequIP: E(3)-Equivariant GNN." [arXiv:2101.03164](https://arxiv.org/abs/2101.03164)
13. Uljad et al. (2024). "SoRoLEX." [arXiv:2410.18519](https://arxiv.org/abs/2410.18519)
14. Greydanus et al. (2019). "Hamiltonian Neural Networks." NeurIPS 2019.
15. Cranmer et al. (2020). "Lagrangian Neural Networks." ICLR 2020 Workshop.
16. (2025). "Dynami-CAL GraphNet." [Nature Comms](https://www.nature.com/articles/s41467-025-67802-5)
17. (2024). "Physics-Encoded GNN for Contact Deformation." [arXiv:2402.03466](https://arxiv.org/html/2402.03466v1)
18. (2025). "PhysGNN for Soft Tissue Deformation." [arXiv:2109.04352](https://arxiv.org/abs/2109.04352)
19. (2025). "GNS vs Neural Operators for PDEs." [arXiv:2509.06154](https://arxiv.org/html/2509.06154v1)
20. (2025). "PIFNO for Beam Bending." [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0141029625014750)
21. (2025). "DeepOMamba." [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999125005558)
22. (2024). "LSTM/GRU for MPC Review." [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231225003844)
23. (2023). "Encoder-Decoder GRU for Structural Dynamics." [Comp. Mech.](https://link.springer.com/article/10.1007/s00466-023-02317-8)
24. (2025). "PDE-Transformer." [arXiv:2505.24717](https://arxiv.org/html/2505.24717v1)
25. (2024). "Graph Transformer Surrogate." [CMAME](https://www.sciencedirect.com/science/article/abs/pii/S0045782524006650)
26. (2024). "LaDEEP for Slender Solids." [arXiv:2506.06001](https://arxiv.org/html/2506.06001v1)
27. Gu et al. (2022). "S4: Structured State Spaces for Sequence Modeling."
28. Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
29. Kim et al. (2022). "SE(2)-Equivariant Pushing Dynamics." [CoRL](https://proceedings.mlr.press/v205/kim23b/kim23b.pdf)
30. (2025). "S4 to Mamba Survey." [arXiv:2503.18970](https://www.arxiv.org/pdf/2503.18970)
31. (2024). "X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation." [arXiv:2411.17164](https://arxiv.org/abs/2411.17164)
32. (2024). "DCNO: Dilated Convolution Neural Operator for Multiscale PDEs." [arXiv:2408.00775](https://arxiv.org/abs/2408.00775)
33. (2025). "Neural Modular Physics for Elastic Simulation." [arXiv:2512.15083](https://arxiv.org/abs/2512.15083)
34. Benady et al. (2024). "Physics-augmented NNs for constitutive modeling of hyperelastic geometrically exact beams." [arXiv:2407.00640](https://arxiv.org/abs/2407.00640)
35. (2026). "Geometry Regularization in Autoencoder ROMs with Latent Neural ODE." [arXiv:2603.03238](https://arxiv.org/html/2603.03238)
36. (2025). "Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots." [arXiv:2511.18322](https://arxiv.org/html/2511.18322)
37. (2025). "MSPT: Multi-Scale Patch Transformer." [arXiv:2512.01738](https://arxiv.org/html/2512.01738v1)
38. (2025). "GeoMaNO: Geometric Mamba Neural Operator." [arXiv:2505.12020](https://arxiv.org/html/2505.12020v1)
39. (2025). "GNSS: Graph Network-based Structural Simulator." [arXiv:2510.25683](https://arxiv.org/html/2510.25683)
40. (2025). "Non-Linear Spectral GNN Simulator for Stable Rollouts." [arXiv:2601.05860](https://arxiv.org/abs/2601.05860)
41. (2024). "Mamba Neural Operator: Transformers vs. SSMs for PDEs." [Journal of Comp. Physics](https://www.sciencedirect.com/science/article/pii/S0021999125008496)
