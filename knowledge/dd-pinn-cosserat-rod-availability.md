---
title: DD-PINN for Cosserat Rod Dynamics — Practical Availability Assessment
date: 2026-03-09
class: knowledge
tags: [dd-pinn, cosserat-rod, pinn, neural-surrogate, soft-robotics]
---

# DD-PINN for Cosserat Rod Dynamics — Practical Availability Assessment

## Paper Identity

**arXiv:2508.12681** — "Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory"

- **Authors:** Johann Licher, Max Bartholdt, Henrik Krauss, Tim-Lukas Habich, Thomas Seel, Moritz Schappler
- **Affiliation:** Institute of Mechatronic Systems, Leibniz University Hannover, Germany
- **Submitted:** August 18, 2025 (v1); revised January 15, 2026 (v2)
- **Funding:** DFG grant no. 405032969
- **NOT authored by Maximilian Stolzle** — Stolzle (TU Delft / Disney Research) works on related soft robot modeling but is not on this paper.

## What DD-PINN Actually Is

The DD-PINN (Domain-Decoupled Physics-Informed Neural Network) is a neural surrogate for the dynamic Cosserat rod model. It achieves a 44,000x speed-up over the full physics model.

### Key Technical Properties

| Property | Status |
|---|---|
| Static or dynamic? | **Dynamic** — predicts full temporal evolution |
| State space size | 72 states (largest PINN-learned state-space for robot control) |
| States predicted | Full robot state: strains AND velocities |
| Actuation type | **Pneumatic pressure** (3 air chambers), NOT rest curvature |
| External forces | **Not handled** — "external forces are neglected" |
| Friction/contact | **Not handled** |
| Robot geometry | Soft pneumatic continuum arm (~130 mm length), NOT a snake |
| Spatial discretization | Collocation method with implicit midpoint rule |
| Training data source | Custom Cosserat rod solver (symbolic CAS), NOT PyElastica |
| Real-world validation | Yes — hardware experiments with 3 mm tip error, 3.55 m/s^2 accel |

### How It Works

1. DD-PINN decouples the time domain from the neural network, using ansatz functions for closed-form gradient computation (no autodiff needed for time derivatives)
2. Embedded in an Unscented Kalman Filter for online state + parameter estimation
3. Used inside a Nonlinear Evolutionary MPC running at 70 Hz on GPU
4. Bending stiffness is an adaptable parameter updated online

## Code Availability

### This paper (arXiv:2508.12681): **NO PUBLIC CODE**

- No GitHub link in the paper
- No code availability statement
- No supplementary materials referenced
- Authors' GitHub profiles do not contain related repositories

### Authors' Related Repositories

| Author | GitHub | DD-PINN Code? |
|---|---|---|
| Tim-Lukas Habich | [tlhabich](https://github.com/tlhabich) | No — 4 repos, all MATLAB/C++ robotics, nothing PINN-related |
| Moritz Schappler | [SchapplM](https://github.com/SchapplM) | No — 39 repos, mainly MATLAB kinematics/dynamics toolboxes |
| Johann Licher | Not found on GitHub | N/A |
| Henrik Krauss | Not found on GitHub | N/A |

### Related Code That DOES Exist

1. **[Martin-Bensch/tdcr-pinn](https://github.com/Martin-Bensch/tdcr-pinn)** (10 stars)
   - ICRA 2024 paper: "Physics-Informed Neural Networks for Continuum Robots: Towards Fast Approximation of Static Cosserat Rod Theory"
   - Same research group (Bensch, Job, Habich, Seel, Schappler)
   - **Static** Cosserat rod PINN (predecessor to the dynamic DD-PINN)
   - Python (62%) + C++ (38%), uses pybind11 binding to Cosserat rod statics solver
   - Status: "Work in progress" as of March 2024
   - Framework: Not documented (likely PyTorch based on Python ecosystem)

2. **DD-PINN foundational paper** (arXiv:2408.14951)
   - "Domain-decoupled Physics-informed Neural Networks with Closed-form Gradients for Fast Model Learning of Dynamical Systems"
   - Authors: Krauss, Habich, Bartholdt, Seel, Schappler (August 2024)
   - General method paper — tested on mass-spring-damper, five-mass-chain, two-link robot
   - **No code released**

## Framework Assessment

The paper does not disclose which deep learning framework is used. Circumstantial evidence:
- The predecessor (tdcr-pinn) is Python-based
- The MPC runs on GPU at 70 Hz, suggesting PyTorch or JAX
- The Hannover group uses MATLAB extensively for kinematics but Python for ML
- The DD-PINN method uses closed-form gradients (no autodiff for time), which could work in any framework

## Relevance to Snake Robot Locomotion RL

### Blockers for Our Use Case

1. **No rest curvature actuation** — designed for pneumatic pressure input, not the rest curvature κ(s,t) that CPG-based snake locomotion requires
2. **No external forces** — explicitly neglects them; we need RFT friction for ground locomotion
3. **No contact/friction** — essential for snake-ground interaction
4. **Wrong geometry** — tested on a short (~130 mm) pneumatic arm, not a long slender snake
5. **No code available** — cannot adapt or extend
6. **72-state system** — may be larger than needed for snake locomotion (approach might still apply conceptually)

### What Could Be Salvaged (Conceptually)

- The DD-PINN architecture idea (time-domain decoupling with ansatz functions) is general
- The 44,000x speed-up over Cosserat rod simulation is impressive
- The UKF integration for online parameter adaptation is clever
- Could inspire building a custom PINN for our rod dynamics — but would need to:
  - Add rest curvature as input
  - Add RFT friction forces
  - Retrain from scratch on snake-geometry data
  - Implement from scratch (no code to start from)

## Maximilian Stolzle's Actual Work

Stolzle (now at Disney Research, PhD from TU Delft) works on related but different topics:

- **[tud-phi/jsrm](https://github.com/tud-phi/jax-soft-robot-modelling)** — JAX-based kinematic/dynamic models of continuum soft robots (15 stars)
- **[tud-phi/HSA-PyElastica](https://github.com/tud-phi/HSA-PyElastica)** — PyElastica plugin for Handed Shearing Auxetics simulation
- **NeurIPS 2024** — Input-to-State Stable Coupled Oscillator Networks for latent-space dynamics
- Uses JAX extensively for differentiable soft robot simulation
- Does NOT have PINN-specific code

## Bottom Line

The DD-PINN paper (arXiv:2508.12681) is a real, published, peer-reviewed paper with impressive results, but:

1. **No public code exists** (as of March 2026)
2. **Not applicable to snake locomotion** without major modifications (no friction, no rest curvature actuation, no contact)
3. **Not by Stolzle** — it's by the Hannover group (Licher, Schappler et al.)
4. The closest available code is the static PINN predecessor ([tdcr-pinn](https://github.com/Martin-Bensch/tdcr-pinn)), which is incomplete
