---
name: KNODE-Cosserat Hybrid Surrogate Report
description: LaTeX report on KNODE-Cosserat hybrid physics+NN surrogate for snake robot Cosserat rod dynamics, with pseudo-code algorithms and literature survey
type: knowledge
created: 2026-03-16
updated: 2026-03-16
tags: [knode, cosserat-rod, hybrid-surrogate, neural-ode, pseudo-code, latex-report]
aliases: [knode-report, hybrid-surrogate-report]
---

# KNODE-Cosserat Hybrid Physics+NN Surrogate Report

## Document Location

`report/knode-cosserat-hybrid-surrogate.tex`

## Contents

1. **Cost decomposition** of the Verlet substep — identifies Stages 2b–2c (internal force assembly, 42% of computation) as the NN replacement target
2. **KNODE framework adaptation** — hybrid equation where NN predicts internal forces/torques while physics handles kinematics, external forces, and time integration
3. **Architecture** — 131D input → 62D output MLP (~135K params), replacing the 654K full-state surrogate
4. **Four pseudo-code algorithms**:
   - Algorithm 1: Hybrid Verlet Substep (the core innovation)
   - Algorithm 2: Multi-substep RL environment step
   - Algorithm 3: Two-phase training (single-step MSE → multi-step rollout BPTT)
   - Algorithm 4: Residual correction variant (simplified physics + NN correction)
5. **Comparison table** — MLP vs KNODE (replace) vs KNODE (residual) vs DD-PINN vs PyElastica
6. **Reduced substep analysis** — physics backbone enables larger dt (0.05s vs 0.001s), cutting substeps from 500 to ~10
7. **Literature survey** — KNODE-Cosserat, DD-PINN, Kasaei ANODE, Gao residual, CLPNets, SoftAE, etc.
8. **Decision framework** — when to use each approach based on accuracy/throughput/data requirements

## Key Findings from Research

- **No follow-up work** on KNODE-Cosserat itself (2024–2026) — it remains a one-off paper
- **DD-PINN (Licher et al., 2025)** achieves 44,000x speedup with open-source code from companion paper
- **Kasaei et al. (2025)** — two papers on Augmented Neural ODEs with Cosserat prior for continuum robots, closest analog to KNODE
- **CLPNets (Eldred et al., 2024)** — Lie-Poisson preserving networks theoretically ideal for SE(3) rod chains but untested on rods
- **No paper benchmarks KNODE against pure MLP** — this is an open comparison opportunity
