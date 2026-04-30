---
created: 2026-03-11T13:28:43.255Z
title: Explore pure ODE data generation as alternative to Elastica
area: general
files:
  - locomotion_elastica/env.py
  - src/physics/
---

## Problem

Currently, surrogate model training data is generated through PyElastica simulation, which involves a complex Cosserat rod PDE solver with 500 substeps per RL step at ~3 FPS per worker. This is slow and introduces significant Python interpreter overhead. The snake's dynamics might be adequately captured by a simpler ODE system (e.g., a reduced-order model of the snake's center-of-mass trajectory and body shape parameters), which could generate training data orders of magnitude faster.

## Solution

Investigate whether a pure ODE formulation can replace Elastica for data generation:

1. **Derive reduced-order ODE** — model the snake as a set of coupled ODEs for COM position, heading, and body curvature modes, driven by CPG inputs with anisotropic friction forces
2. **Validate against Elastica** — compare ODE trajectories to Elastica ground truth across a range of CPG parameters (amplitude, frequency, wave number) to quantify approximation error
3. **Benchmark speed** — measure data generation throughput with ODE solver (e.g., `scipy.integrate.solve_ivp` or a simple RK4) vs Elastica
4. **Assess surrogate model quality** — if ODE data is sufficiently accurate, train a surrogate on ODE-generated data and compare validation metrics to the Elastica-trained surrogate
5. **Determine feasibility threshold** — define acceptable error bounds for the RL use case (the surrogate only needs to be good enough for policy learning, not exact physics)
