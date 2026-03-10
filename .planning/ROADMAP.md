# Roadmap: Surrogate Data Collection & Validation

## Overview

Build a high-quality dataset of snake robot dynamics transitions for training a neural surrogate model. Phase 1 (data collection) is complete — 2.3 GB of transition data collected from 16 parallel PyElastica workers with Sobol quasi-random actions, state perturbation, and health monitoring. Phase 2 validates the collected data: checks distribution evenness across state-action space, identifies quality issues, and produces actionable recommendations.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Collection** - Collect snake robot dynamics transitions from PyElastica with parallel workers, health monitoring, NaN filtering, and Sobol quasi-random exploration
- [ ] **Phase 2: Data Validation** - Analyze collected dataset for distribution evenness, data quality, anomalies, and coverage gaps to determine readiness for surrogate training
- [ ] **Phase 5: Train RL agent using surrogate model** - Train a reinforcement learning agent using the validated surrogate model as the environment
- [ ] **Phase 6: Write research report in LaTeX** - Write up findings, methods, and results in a LaTeX research report
- [ ] **Phase 7: Foundation model exploration** - Explore foundation model approaches for snake robot dynamics to discover interesting generalizable findings
- [ ] **Phase 8: Elastica RL baseline** - Train RL directly on Elastica with identical reward/observation setup as surrogate RL for controlled comparison

## Phase Details

### Phase 1: Data Collection
**Goal**: Collect a large, diverse dataset of snake robot dynamics transitions from PyElastica
**Depends on**: Nothing (first phase)
**Status**: COMPLETE
**Result**: 28 batch files across 16 workers, ~2.3 GB total. Each batch contains states (N, 124), actions (N, 5), serpenoid_times (N), next_states (N, 124), episode_ids, step_indices, and force snapshots. Collected with Sobol quasi-random actions, 30% perturbation fraction, 50% random action fraction.

### Phase 2: Data Validation
**Goal**: Understand the quality and coverage of the collected dataset, identify any issues that would affect surrogate model training, and produce clear recommendations
**Depends on**: Phase 1
**Requirements**: DVAL-01, DVAL-02, DVAL-03, DVAL-04, DVAL-05
**Success Criteria** (what must be TRUE):
  1. Distribution analysis shows how evenly the state-action space is covered, with per-dimension histograms and joint coverage heatmaps
  2. Data quality report identifies NaN/Inf values, duplicate transitions, constant features, and outliers with counts and locations
  3. Temporal analysis shows episode length distribution and whether transitions are biased toward early/late timesteps
  4. Action space coverage analysis shows whether Sobol quasi-random + perturbation achieved good 5D coverage vs gaps
  5. A summary report with clear pass/fail assessment and actionable recommendations (recollect, augment, proceed) is produced
**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md — Build validation analysis module (distributions, quality, temporal, coverage, figures, report)
- [ ] 02-02-PLAN.md — Run validation on dataset and human review of results

### Phase 02.1: re-collect surrogate data with per-node phase encoding (INSERTED)

**Goal:** Re-collect snake robot dynamics transitions incorporating per-node CPG phase as an explicit state feature, addressing the coverage and encoding gaps identified in Phase 2 validation. Produces a new dataset that replaces Phase 1 data for surrogate training.
**Requirements**: RCOL-01, RCOL-02, RCOL-03, RCOL-04
**Depends on:** Phase 2
**Plans:** 3/3 plans complete

Plans:
- [ ] 02.1-01-PLAN.md — Add encode_per_element_phase() to state.py and update INPUT_DIM to 189
- [ ] 02.1-02-PLAN.md — Rewrite collection pipeline (checkpoint format, perturb_omega_std=1.5), archive Phase 1 data, launch collection
- [ ] 02.1-03-PLAN.md — Add OverlappingPairDataset to dataset.py (on-the-fly pair formation, 189-dim input)

### Phase 3: Train surrogate model using supervised learning
**Goal**: Train an MLP surrogate model on the Phase 02.1 dataset via hyperparameter sweep (LR x model size), select the best model by validation MSE, and produce per-component error analysis with diagnostic plots
**Depends on**: Phase 02.1
**Requirements**: SURR-01, SURR-02, SURR-03, SURR-04, SURR-05
**Success Criteria** (what must be TRUE):
  1. 5 sweep configurations trained to convergence with W&B logging
  2. Best model selected by lowest single-step validation MSE
  3. Per-component errors reported in physical units (position mm, velocity mm/s, angle rad, angular velocity rad/s)
  4. Diagnostic plots saved: error histograms, predicted-vs-actual, sweep comparison
  5. Best model checkpoint ready at output/surrogate/best/ for Phase 4
**Plans:** 2 plans

Plans:
- [ ] 03-01-PLAN.md — Sweep infrastructure and execute hyperparameter sweep (5 configs)
- [ ] 03-02-PLAN.md — Analyze sweep results, select best model, generate diagnostic plots

### Phase 03.1: Surrogate Model Architecture Experiments — Rollout Loss, Residual, History Window (INSERTED)

**Goal:** Run architecture experiments comparing 3 improvements to the 512x3 MLP baseline: (A) rollout loss weight/horizon tuning across 4 variants, (B) residual MLP, (C) history window K=2 if A+B fall short. Select the best architecture for Phase 4 validation. If none improve significantly over baseline (val_loss=0.2161, R²=0.784), proceed with existing checkpoint.
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04, ARCH-05
**Depends on:** Phase 3
**Plans:** 2/3 plans executed

Plans:
- [ ] 03.1-01-PLAN.md — Add architectural variants to code (ResidualSurrogateModel, HistorySurrogateModel, HistoryDataset, CLI args) with unit tests
- [ ] 03.1-02-PLAN.md — Create arch_sweep.py and run Experiments A+B (rollout loss variants + residual)
- [ ] 03.1-03-PLAN.md — Analyze results, human selection gate, copy winner to output/surrogate/best/

### Phase 4: Validate surrogate model against Elastica solver trajectories

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 03.1
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 4 to break down)

### Phase 5: Train RL agent using surrogate model

**Goal:** Train a reinforcement learning agent using the validated surrogate model as the environment, achieving locomotion performance comparable to or better than direct Elastica training
**Requirements**: TBD
**Depends on:** Phase 4
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 5 to break down)

### Phase 6: Write research report in LaTeX

**Goal:** Write a comprehensive research report documenting the surrogate modeling pipeline, RL training results, and comparisons with direct simulation training
**Requirements**: TBD
**Depends on:** Phase 5
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 6 to break down)

### Phase 7: Foundation model exploration for snake robot dynamics

**Goal:** Explore whether a foundation model approach (pretrained on diverse snake robot dynamics) yields interesting generalizable findings beyond task-specific surrogates
**Requirements**: TBD
**Depends on:** Phase 3 (uses collected data; independent of RL training)
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 7 to break down)

### Phase 8: Train RL baseline directly on Elastica for controlled comparison

**Goal:** Train RL (PPO) directly on Elastica using the same reward function, observation space, and hyperparameters as Phase 5's surrogate RL, producing a controlled baseline for wall-clock time, sample efficiency, and final performance comparison
**Requirements**: TBD
**Depends on:** Phase 2 (needs finalized env/reward setup; can run in parallel with Phases 3-5)
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 8 to break down)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6
Phase 7 is exploratory and can run in parallel after Phase 3.
Phase 8 (Elastica baseline) can run in parallel with Phases 3-5 after Phase 2.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Collection | - | COMPLETE | 2026-03-10 |
| 2. Data Validation | 0/2 | Not started | - |
| 02.1. Re-collect with Per-Node Phase | 3/3 | Complete    | 2026-03-10 |
| 3. Surrogate Training | 0/2 | Not started | - |
| 3.1. Arch Experiments | 1/3 | In Progress|  |
| 4. Surrogate Validation | 0/0 | Not planned | - |
| 5. RL Training | 0/0 | Not planned | - |
| 6. LaTeX Report | 0/0 | Not planned | - |
| 7. Foundation Model | 0/0 | Not planned | - |
| 8. Elastica RL Baseline | 0/0 | Not planned | - |
