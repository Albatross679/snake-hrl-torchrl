# Neural Surrogate & RL for Snake Robot Locomotion

## What This Is

Build a neural surrogate model of PyElastica Cosserat rod snake robot dynamics, then use it to train a reinforcement learning agent for locomotion — with validation against ground-truth Elastica trajectories, architecture experiments, and a LaTeX research report comparing surrogate-based RL against direct Elastica RL baselines.

## Core Value

Train a neural surrogate model accurate enough to replace Elastica simulation during RL training, enabling faster iteration and producing publishable research comparing surrogate-based and direct-simulation RL.

## Requirements

### Validated

- ✓ Data collection pipeline with Sobol quasi-random action sampling — existing (`aprx_model_elastica/collect_data.py`)
- ✓ W&B integration for logging collection metrics — existing (in `collect_data.py`)
- ✓ Density-weighted sampling for training — existing (`aprx_model_elastica/dataset.py`)
- ✓ Pre-flight smoke test — existing (commit d3e5277)
- ✓ Disk space pre-check — existing (commit bdbf093)
- ✓ Post-collection data validation — existing (commit 9a54b11)

### Active

- [ ] Autonomous periodic monitoring agent that checks collection health
- [ ] Multi-criteria stop condition: min 8 hours AND min sample count AND state-action grid coverage target
- [ ] Auto-diagnosis and fix of runtime issues (crashes, bad data, performance degradation)
- [ ] Markdown documentation of all issues encountered and fixes applied (issues/, logs/)
- [ ] W&B dashboard monitoring with key health metrics
- [ ] State-action grid coverage tracking and reporting

### Out of Scope

- Surrogate model training — separate project phase after data is collected
- RL training with surrogate — depends on trained surrogate model
- New physics backends — using PyElastica only for data collection
- Multi-GPU collection — single V100 is sufficient for overnight runs

## Context

- **Existing code:** `aprx_model_elastica/` package has collection, dataset, model, training, and validation modules
- **Physics:** PyElastica Cosserat rod simulation, ~57 FPS with 16 parallel envs
- **Hardware:** Tesla V100-PCIE-16GB, 48 CPUs available
- **W&B:** Already integrated in collect_data.py with basic metric logging
- **Coverage strategy:** Sobol quasi-random sampling for 5D action hypercube, density bins for reweighting rare states
- **Branch:** `ralph/surrogate-data-collection` — active development branch

## Constraints

- **Runtime:** Collection must run 8+ hours unattended (overnight)
- **Hardware:** Single Tesla V100, 48 CPUs, standard disk space
- **Physics:** PyElastica only — ~57 FPS throughput ceiling with 16 parallel envs
- **Monitoring:** Agent checks periodically (not continuous), fixes issues autonomously

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Sobol quasi-random sampling | Better 5D coverage than uniform random | — Pending |
| Multi-criteria stop condition | Ensures both quantity and quality of data | — Pending |
| Autonomous agent fixes issues | Overnight run can't wait for human intervention | — Pending |
| Density-weighted training data | Compensate for uneven coverage in state space | — Pending |

## Current State

Phase 15 complete — Operator-Theoretic Policy Gradient (OTPG) from arXiv:2603.17875 implemented alongside PPO and SAC. OTPGTrainer with MMD-RKHS trust region loss, Choi2025 benchmark integration, and 100K-frame validation showing learning signal (reward 9→21 on follow_target). Three RL algorithms now available for comparison.

---
*Last updated: 2026-03-19 after Phase 15 completion*
