---
name: physics-informed-reinforcement-learning
description: Survey of physics-informed RL approaches for snake robot locomotion
type: knowledge
created: 2026-02-17T00:00:00
updated: 2026-02-17T00:00:00
tags: [reinforcement-learning, physics-informed, survey, PIRL, locomotion, soft-robotics, snake-robot]
aliases: []
---

# Physics-Informed Reinforcement Learning (PIRL)

A survey of approaches for injecting physics priors into reinforcement learning, with emphasis on applicability to snake robot locomotion using Discrete Elastic Rods (DER) simulation and PPO/SAC training.

## 1. Taxonomy Overview

The definitive survey is Banerjee et al. (2023), "A Survey on Physics Informed Reinforcement Learning: Review and Open Problems" (arXiv:2309.01909, published in Expert Systems with Applications, 2025). It organizes PIRL along two axes:

**Axis 1 -- Where physics enters the RL pipeline:**
- Observation/state representation
- Action/policy structure
- Reward function
- Transition/dynamics model
- Loss/optimization objective

**Axis 2 -- Type of physics incorporation bias:**
- **Observational bias**: physics shapes what the agent sees (e.g., physics-derived features, coordinate transforms)
- **Inductive bias**: physics constrains the model architecture (e.g., Lagrangian/Hamiltonian networks, equivariant networks, CPG layers)
- **Learning bias**: physics modifies the learning objective (e.g., physics-regularized losses, energy-conservation penalties, physics-informed reward shaping)

---

## 2. Main Categories of PIRL Approaches

### 2.1 Physics-Informed Dynamics Models (Transition Function)

**Core idea**: Learn a dynamics model that respects known physics, then use it for model-based RL (planning, policy gradient through the model, or Dyna-style augmentation).

**Key approaches:**

- **Lagrangian Neural Networks (LNNs)** -- Cranmer et al. (2020, arXiv:2003.04630). Learn a scalar Lagrangian L(q, q_dot) with a neural network; derive equations of motion via Euler-Lagrange equations. Does not require canonical coordinates (unlike HNNs). Deep Lagrangian Networks (DeLaNs) were the robotics-focused variant.

- **Hamiltonian Neural Networks (HNNs)** -- Greydanus et al. (2019). Learn the Hamiltonian H(q, p) and derive dynamics via Hamilton's equations. Conserves energy by construction. Requires canonical momenta, which limits practical applicability.

- **Physics-Informed Model-Based RL** -- Ramesh & Ravindran (2023, L4DC, arXiv:2212.02179). Uses LNN-based dynamics models within a model-based RL algorithm. The LNN version uses one network for potential energy and another for the mass matrix (lower-triangular decomposition). Key result: physics-informed models significantly outperform standard DNNs in environments sensitive to initial conditions; achieve better sample efficiency universally; and outperform SAC in challenging environments by computing policy gradients analytically through the differentiable model.

- **Extending LNN/HNN with Contact** -- Zhong et al. (2021, arXiv:2102.06794). Addresses the main limitation of energy-conserving models for robotics: contacts and dissipation. Extends Lagrangian/Hamiltonian networks with differentiable contact models.

**Applicability to snake-hrl**: Moderate-to-high. The DER simulation already provides a physics model; an LNN-based surrogate could be trained for faster rollouts or for analytical policy gradients. However, contacts and friction (critical for snake locomotion) remain challenging for LNN/HNN approaches.

### 2.2 Differentiable Physics / Simulation in the Loop

**Core idea**: Make the physics simulator itself differentiable, then backpropagate policy gradients through the simulation (First-order Policy Gradient, FoPG).

**Key approaches:**

- **PODS (Policy Optimization via Differentiable Simulation)** -- Mora et al. (2021, ICML). Backpropagation through time using differentiable simulators. Dramatically improves sample efficiency.

- **SHAC (Short-Horizon Actor-Critic)** -- Xu et al. Uses a smooth critic to handle non-smooth contact dynamics and truncated windows to avoid exploding/vanishing gradients. Effective for contact-rich locomotion.

- **Differentiable Discrete Elastic Rods (DDER)** -- ROAHM Lab (2024, arXiv:2406.05931). Reformulates DER to be fully differentiable with respect to all model variables. Enables gradient-based system identification and integration with deep learning pipelines. Runs in real-time.

- **DisMech Gradient Descent** -- StructuresComp (2024). Proposes gradient descent through DisMech's natural curvature parameters for real2sim mapping of soft manipulators.

- **Residual Policy Learning via Differentiable Simulation** -- Luo et al. (2024, arXiv:2410.03076). Combines differentiable simulation with residual policies for quadruped locomotion. Demonstrates sim-to-real transfer.

**Key challenge**: FoPG methods struggle with contact-rich dynamics (local minima, exploding gradients). The SHAC approach partially mitigates this. Stabilizing RL in differentiable multiphysics simulation is an active research area (Xing, 2025, CMU thesis; ICLR 2025).

**Applicability to snake-hrl**: HIGH. The DER simulation (DisMech) already has differentiable variants (DDER). This is the most natural fit: backpropagating through the DER simulation could yield analytical policy gradients, dramatically improving sample efficiency over PPO/SAC. The main barrier is implementing the differentiable backward pass through the implicit time-stepping solver.

### 2.3 Physics-Structured Policy Networks (Action Space)

**Core idea**: Embed physics knowledge directly into the policy architecture, constraining the action space to physically plausible outputs.

**Key approaches:**

- **CPG-RL** -- Bellegarda & Ijspeert (2022, IEEE RA-L, arXiv:2211.00458). Embeds Central Pattern Generators as differentiable layers in the policy network. The RL agent learns to modulate CPG parameters (amplitude, frequency, phase offsets) rather than raw joint commands. Results in smooth, energy-efficient locomotion. Extended to Visual CPG-RL (2024) with exteroceptive sensing.

- **Bio-inspired CPG Networks** -- Nature Scientific Reports (2025). Reformulates CPGs into fully-differentiable, stateless network layers, enabling joint end-to-end training of CPGs and MLP using gradient-based learning. Supports multi-skill locomotion.

- **CPG-regulated Soft Snake Robot** -- Arachchige et al. (2023, IEEE RA-L, arXiv:2207.04899). RL module for goal-tracking + CPG with Matsuoka oscillators for generating stable locomotion patterns. Specifically for soft snake robots.

- **Action Smoothness Regularization (CAPS)** -- Mysore et al. (2021, ICRA). Regularizes policy networks to produce smooth action trajectories based on state-to-action mapping continuity.

- **Symmetry-Equivariant Policies** -- Multiple recent works (2024-2025). Encodes morphological symmetries (bilateral, rotational) directly into policy network architecture using equivariant neural networks. MS-PPO (2024) uses morphology-informed graph neural architectures. Up to 40% improvement in tracking accuracy and dramatically better sample efficiency.

**Applicability to snake-hrl**: VERY HIGH. The project already uses CPG actuators (physics/cpg/). CPG-RL is the most directly applicable approach: wrapping the existing serpenoid curve CPG as a differentiable policy layer, with RL modulating amplitude/frequency/phase. Snake robots have bilateral symmetry that can be exploited with equivariant networks.

### 2.4 Physics-Informed Reward Functions

**Core idea**: Augment or replace hand-crafted rewards with physics-based quantities that encode domain knowledge about desired behavior.

**Key approaches:**

- **Energy-Based Reward Shaping / Cost of Transport (CoT)** -- Multiple works. Penalize mechanical energy expenditure or CoT (energy per unit distance per unit weight). Up to 32% energy efficiency improvement. Promotes natural gaits without motion capture data.

- **Impact Mitigation Factor (IMF)** -- arXiv:2510.09543 (2025). A physics-informed metric quantifying a robot's ability to passively mitigate ground impacts. Used as reward bonus alongside Adversarial Motion Priors.

- **Gait Potential Reward Shaping** -- Widely used in locomotion RL. Reward = change in potential function (e.g., distance to goal, velocity alignment). Already implemented in this project (src/rewards/gait_potential.py, src/rewards/shaping.py).

- **Lyapunov-Based Reward Shaping** -- Physics-Informed DDPG. Uses Lyapunov stability theory to shape rewards, enhancing stability and reliability without compromising optimality.

- **Physics-Informed Reward Machines** -- arXiv:2508.14093 (2025). Integrates physical dynamics directly into formal reward machine structures for structured, high-fidelity reward specifications.

**Applicability to snake-hrl**: VERY HIGH (already partially implemented). The project uses gait potential and shaping rewards. Additional physics-informed rewards could include: (1) elastic energy stored in the DER body, (2) curvature smoothness penalties derived from the rod's bending energy, (3) friction work as a proxy for propulsive efficiency, (4) CoT for energy-efficient locomotion.

### 2.5 Residual Physics / Sim-to-Real

**Core idea**: Combine a physics model (possibly coarse) with learned residual corrections to handle unmodeled dynamics.

**Key approaches:**

- **Residual Physics for Soft Robots** -- ResearchGate (2024). Learns residual forces to compensate for state-to-state prediction errors in soft robot simulation, matching sparse motion markers between sim and reality.

- **Residual Policy Learning** -- Train a base policy in simulation, then learn a residual correction on the real robot. Combines the structure of the physics model with the adaptability of learning.

- **Domain Randomization + Physics Structure** -- Standard approach: randomize physics parameters (friction, mass, stiffness) during training. Physics-informed variant: randomize only parameters the physics model identifies as uncertain.

**Applicability to snake-hrl**: Moderate. Relevant if/when transferring to a physical snake robot. Residual physics could correct DER simulation errors (e.g., unmodeled viscoelasticity, manufacturing defects, hysteresis).

### 2.6 Physics-Informed State Representations (Observations)

**Core idea**: Transform raw observations into physics-meaningful features before feeding them to the policy.

**Key approaches:**

- **Curvature-based observations**: Instead of raw node positions, use curvatures (natural representation for elastic rods). Already relevant to snake-hrl via the DER curvature state.

- **Energy-based features**: Include kinetic/potential/elastic energy as observation features.

- **Coordinate-free representations**: Use body-frame quantities (relative angles, local curvatures) instead of global coordinates. Improves generalization.

**Applicability to snake-hrl**: HIGH (already partially implemented). The observation extractors in src/observations/ likely already use some physics-derived features. Additional options: include elastic energy, curvature rates, friction force estimates.

---

## 3. Most Relevant Work for Snake Robot + DER + PPO/SAC

### 3.1 Directly Applicable (High Priority)

| Approach | Why | Effort |
|---|---|---|
| **CPG-RL** (Bellegarda 2022) | Project already has CPG; embed as differentiable policy layer | Medium |
| **Physics-informed rewards** (energy, CoT, curvature smoothness) | DER provides all needed quantities; extend existing reward shaping | Low |
| **Symmetry-equivariant policy** | Snake has bilateral symmetry; proven sample efficiency gains | Medium |
| **Back-stepping Experience Replay** (2024) | Designed specifically for soft snake robots with anisotropic friction; 48% speed improvement over baselines | Medium |

### 3.2 High Potential, More Engineering (Medium Priority)

| Approach | Why | Effort |
|---|---|---|
| **Differentiable DER (DDER)** | Analytical policy gradients through the physics; huge sample efficiency gains | High |
| **LNN-based surrogate model** | Fast approximate rollouts for model-based RL; compute analytical gradients | High |
| **Physics-informed state features** | Add elastic energy, friction work to observations | Low |

### 3.3 Future / Sim-to-Real (Lower Priority)

| Approach | Why | Effort |
|---|---|---|
| **Residual physics** | Correct DER simulation errors for real robot transfer | High |
| **Domain randomization with physics structure** | Randomize DER parameters (stiffness, friction) informed by uncertainty | Medium |

---

## 4. Specific Recommendations for snake-hrl

### Immediate wins (modify existing code):

1. **Enrich reward function** with DER-derived physics quantities:
   - Elastic bending energy penalty (already computable from rod curvatures)
   - Curvature rate smoothness penalty (temporal derivative of curvatures)
   - Cost of Transport = total energy / (mass * g * distance)
   - Friction work efficiency = useful propulsive work / total friction dissipation

2. **Add physics features to observations**:
   - Total elastic energy (bending + twisting + stretching)
   - Per-segment curvature rates
   - Ground reaction force estimates

3. **Exploit bilateral symmetry** via data augmentation:
   - Mirror state-action pairs during training (doubles effective data)
   - Or use equivariant network architecture

### Medium-term (new components):

4. **CPG-RL integration**: Make the CPG in physics/cpg/ a differentiable layer in the policy network. The RL agent outputs CPG modulation parameters (amplitude_scale, frequency_scale, phase_offsets) rather than raw curvature commands.

5. **Differentiable DER**: Investigate DDER (arXiv:2406.05931) for gradient-based policy optimization. This would replace PPO's zeroth-order gradient estimation with analytical first-order gradients through the simulation.

### Long-term (research):

6. **LNN surrogate**: Train a Lagrangian Neural Network on DER rollout data as a fast, differentiable surrogate for model-based RL or planning.

7. **Residual physics for sim-to-real**: Learn residual corrections to the DER model from real robot data.

---

## 5. Key References

### Surveys
- Banerjee, Nguyen, Fookes, Raissi. "A Survey on Physics Informed Reinforcement Learning: Review and Open Problems." arXiv:2309.01909 (2023), Expert Systems with Applications (2025). [arXiv](https://arxiv.org/abs/2309.01909) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417425017865)

### Lagrangian/Hamiltonian Networks
- Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho. "Lagrangian Neural Networks." arXiv:2003.04630 (2020). [arXiv](https://arxiv.org/abs/2003.04630)
- Zhong, Leonardis, Posa. "Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models." arXiv:2102.06794 (2021). [arXiv](https://arxiv.org/abs/2102.06794)
- Ramesh, Ravindran. "Physics-Informed Model-Based Reinforcement Learning." L4DC 2023. [arXiv](https://arxiv.org/abs/2212.02179) | [PDF](https://proceedings.mlr.press/v211/ramesh23a/ramesh23a.pdf)

### Differentiable Physics
- Mora, Peychev, Ha, Vechev, Coros. "PODS: Policy Optimization via Differentiable Simulation." ICML 2021. [PDF](http://proceedings.mlr.press/v139/mora21a/mora21a.pdf)
- Xing. "Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation." CMU Thesis (2025). [PDF](https://www.ri.cmu.edu/app/uploads/2025/05/xing2025msrthesis.pdf)
- ROAHM Lab. "Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects." arXiv:2406.05931 (2024). [arXiv](https://arxiv.org/abs/2406.05931) | [Project](https://roahmlab.github.io/DEFORM/)

### CPG-RL
- Bellegarda, Ijspeert. "CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion." IEEE RA-L, 2022. [arXiv](https://arxiv.org/abs/2211.00458)
- Bellegarda, Ijspeert. "Visual CPG-RL: Learning Central Pattern Generators for Visually-Guided Quadruped Locomotion." 2024. [arXiv](https://arxiv.org/abs/2212.14400)

### Snake/Soft Robot RL
- Arachchige et al. "Reinforcement Learning of CPG-regulated Locomotion Controller for a Soft Snake Robot." IEEE RA-L, 2023. [arXiv](https://arxiv.org/abs/2207.04899)
- "Back-stepping Experience Replay with Application to Model-free Reinforcement Learning for a Soft Snake Robot." IEEE RA-L, 2024. [arXiv](https://arxiv.org/abs/2401.11372)
- "Physics-informed reinforcement learning for motion control of a fish-like swimming robot." Scientific Reports, 2023. [Nature](https://www.nature.com/articles/s41598-023-36399-4)

### DER / Soft Robot Simulation
- Naughton, Sun, Tekinalp, Parthasarathy, Chowdhary, Gazzola. "Elastica: A compliant mechanics environment for soft robotic control." IEEE RA-L, 2021. [arXiv](https://arxiv.org/abs/2009.08422)
- StructuresComp. "DisMech: A Discrete Differential Geometry-Based Physical Simulator." [GitHub](https://github.com/StructuresComp/dismech-rods) | [PDF](https://par.nsf.gov/servlets/purl/10524090)
- "Rod models in continuum and soft robot control: a review." arXiv:2407.05886 (2024). [arXiv](https://arxiv.org/html/2407.05886v1)

### Symmetry in RL
- "MS-PPO: Morphological-Symmetry-Equivariant Policy for Legged Robot Locomotion." arXiv:2512.00727 (2024). [arXiv](https://arxiv.org/pdf/2512.00727)
- "Leveraging Symmetry in RL-based Legged Locomotion Control." 2024. [arXiv](https://arxiv.org/abs/2403.17320)

### Energy-Efficient Locomotion
- "Guiding Energy-Efficient Locomotion through Impact Mitigation Rewards." arXiv:2510.09543 (2025). [arXiv](https://arxiv.org/abs/2510.09543)

### Residual Physics
- "Sim-to-Real of Soft Robots with Learned Residual Physics." 2024. [ResearchGate](https://www.researchgate.net/publication/383361106_Sim-to-Real_of_Soft_Robots_with_Learned_Residual_Physics)

### Physics-Informed Reward Shaping
- "Physics-informed reward shaped reinforcement learning control of a robot manipulator." 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2090447925003363)
- "Physics-Informed Reward Machines." arXiv:2508.14093 (2025). [arXiv](https://arxiv.org/pdf/2508.14093)
