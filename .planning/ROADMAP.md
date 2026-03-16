# Roadmap: Neural Surrogate & RL for Snake Robot Locomotion

## Overview

Build a high-quality dataset of snake robot dynamics transitions for training a neural surrogate model. Phase 1 (data collection) and Phase 2 (data validation) are complete, including re-collection with per-node phase encoding (Phase 2.1) and RL-step-only collection (Phase 2.2). Phase 3 (surrogate model training) is in progress — training MLP/Residual/Transformer surrogates via 15-config architecture sweep on Phase 2.2 data. Phases 4-5 will validate the surrogate and train an RL agent. Phase 6 (LaTeX report) is complete with Background/Methods/Discussion written; Results sections await Phases 4/5/8.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Collection** - Collect snake robot dynamics transitions from PyElastica with parallel workers, health monitoring, NaN filtering, and Sobol quasi-random exploration
- [x] **Phase 2: Data Validation** - Analyze collected dataset for distribution evenness, data quality, anomalies, and coverage gaps to determine readiness for surrogate training
- [x] **Phase 2.1: Re-collect with per-node phase encoding** - Re-collect transitions with per-node CPG phase as explicit state feature (INSERTED)
- [x] **Phase 2.2: Collect RL-step-only transitions** - Collect only RL-level steps, no sub-steps (INSERTED)
- [ ] **Phase 3: Train surrogate model** - Train MLP/Residual/Transformer surrogates via 15-config sweep, select best model (1/3 plans complete)
- [ ] **Phase 4: Validate surrogate model** - Validate against Elastica solver trajectories
- [ ] **Phase 5: Train RL agent using surrogate model** - Train a reinforcement learning agent using the validated surrogate model as the environment
- [x] **Phase 6: Write research report in LaTeX** - Write up findings, methods, and results in a LaTeX research report (Background/Methods/Discussion complete, Results pending)
- [ ] **Phase 7: Foundation model exploration** - Explore foundation model approaches for snake robot dynamics to discover interesting generalizable findings
- [ ] **Phase 8: Elastica RL baseline** - Train RL directly on Elastica with identical reward/observation setup as surrogate RL for controlled comparison
- [ ] **Phase 9: Physics framework comparison** - Compare DisMech, Genesis, FEM vs Elastica for snake simulation
- [ ] **Phase 10: Tunnel/pipe navigation** - Train RL agent for snake traversal of building infrastructure
- [ ] **Phase 11: Model-based RL** - Explore surrogate as world model for planning and policy optimization
- [ ] **Phase 12: Hamiltonian/Lagrangian NNs** - Explore physics-informed neural networks for snake dynamics

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

### Phase 2.1: Re-collect surrogate data with per-node phase encoding (INSERTED)

**Goal:** Re-collect snake robot dynamics transitions incorporating per-node CPG phase as an explicit state feature, addressing the coverage and encoding gaps identified in Phase 2 validation. Produces a new dataset that replaces Phase 1 data for surrogate training.
**Requirements**: RCOL-01, RCOL-02, RCOL-03, RCOL-04
**Depends on:** Phase 2
**Plans:** 3/3 plans complete

Plans:
- [x] 02.1-01-PLAN.md — Add encode_per_element_phase() to state.py and update INPUT_DIM to 189
- [x] 02.1-02-PLAN.md — Rewrite collection pipeline (checkpoint format, perturb_omega_std=1.5), archive Phase 1 data, launch collection
- [x] 02.1-03-PLAN.md — Add OverlappingPairDataset to dataset.py (on-the-fly pair formation, 189-dim input)

### Phase 2.2: Collect RL-step-only transitions (minimal change from 2.1) (INSERTED)

**Goal:** Minimal change from Phase 02.1: collect only the RL-level step (no sub-steps smaller than the RL step), keeping all other pipeline changes from 02.1 intact.
**Requirements**: RLDC-01, RLDC-02, RLDC-03
**Depends on:** Phase 2.1
**Plans:** 1/1 plans complete

Plans:
- [x] 02.2-01-PLAN.md — Test dataset compatibility with steps_per_run=1, write launch script, smoke test, launch 16-worker collection to data/surrogate_rl_step/

### Phase 3: Train surrogate model using supervised learning
**Goal**: Train surrogate models (MLP, Residual MLP, Wide/Deep MLP, FT-Transformer) on Phase 02.2 dataset via 15-config architecture sweep, select best model by val_loss with human review, and produce confirmed checkpoint at output/surrogate/best/ for Phase 4.
**Depends on**: Phase 02.2
**Requirements**: SURR-01, SURR-02, SURR-03, SURR-04, SURR-05
**Success Criteria** (what must be TRUE):
  1. 15 sweep configurations (5 MLP + 3 Residual + 3 Wide/Deep + 4 FT-Transformer) trained to convergence with W&B logging
  2. Diagnostic plots saved: error histograms, predicted-vs-actual, sweep comparison, per-component RMSE
  3. Best model selected by human review of val_loss and per-component RMSE in physical units
  4. Best model checkpoint confirmed at output/surrogate/best/ with selection.json for Phase 4
**Plans:** 1/3 complete

Plans:
- [x] 03-01-PLAN.md — Implement TransformerSurrogateModel, wire FlatStepDataset, add --arch CLI, expand W&B logging, update sweep.py with 15 configs
- [ ] 03-02-PLAN.md — Smoke test all architectures, launch full 15-config sweep in tmux
- [ ] 03-03-PLAN.md — Build analysis script, generate diagnostic plots, human review, promote best model to output/surrogate/best/

### Phase 4: Validate surrogate model against Elastica solver trajectories

**Goal:** Validate surrogate model accuracy against ground-truth PyElastica trajectories across 4 action scenarios (random, forward crawl, slow/fast gaits, trained PPO policy) with per-scenario PASS/WARN/FAIL verdicts, diagnostic figures, and structured report
**Requirements**: SVAL-01, SVAL-02, SVAL-03, SVAL-04, SVAL-05, SVAL-06
**Depends on:** Phase 3
**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md — Architecture dispatch, action generators, dynamic model discovery, scenario engine, verdict logic, tests
- [ ] 04-02-PLAN.md — Figure generation (trajectory overlays, heatmap, bars), structured report, CLI wiring, human verify

### Phase 5: Train RL agent using surrogate model

**Goal:** Train a reinforcement learning agent using the validated surrogate model as the environment, achieving locomotion performance comparable to or better than direct Elastica training
**Requirements**: TBD
**Depends on:** Phase 4
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 5 to break down)

### Phase 6: Write research report in LaTeX

**Goal:** Write a comprehensive research report documenting the surrogate modeling pipeline, RL training results, and comparisons with direct simulation training. Writing begins immediately with Background/Related Work/Methods sections; Results sections use placeholders until Phases 4/5/8 complete.
**Requirements**: (none — report writing has no formal requirement IDs)
**Depends on:** Phase 5 (for Results); can begin immediately (Background/Methods)
**Status**: COMPLETE (Background/Methods/Discussion written; Results sections pending Phases 4/5/8)
**Plans:** 4/4 complete

Plans:
- [x] 06-01-PLAN.md — Create report/ scaffold: report.tex skeleton (8 sections + preamble), references.bib (11 papers), Makefile (Docker latexmk)
- [x] 06-02-PLAN.md — Write Background section (Cosserat rod PDEs, RFT, CPG, PyElastica) and Related Work section (DD-PINN, KNODE, SoRoLEX, MBPO)
- [x] 06-03-PLAN.md — Write Methods section (data collection, per-element encoding, MLP architecture, training) and Discussion subsections (physics calibration, data pipeline)
- [x] 06-04-PLAN.md — Compile PDF via Docker, human review checkpoint

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

### Phase 9: Physics framework comparison experiment — DisMech, Genesis, FEM vs Elastica

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 8
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 9 to break down)

### Phase 10: Train RL agent for tunnel and pipe navigation — snake traverses building infrastructure

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 9
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 10 to break down)

### Phase 11: Explore model-based RL using surrogate as world model for planning and policy optimization

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 10
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 11 to break down)

### Phase 12: Explore Hamiltonian or Lagrangian neural networks for snake dynamics

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 11
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 12 to break down)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 2.1 -> 2.2 -> 3 -> 4 -> 5 -> 6
Phase 7 is exploratory and can run in parallel after Phase 3.
Phase 8 (Elastica baseline) can run in parallel with Phases 3-5 after Phase 2.
Phases 9-12 are future research directions.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Collection | - | COMPLETE | 2026-03-10 |
| 2. Data Validation | 0/2 | Superseded by 02.1/02.2 | - |
| 02.1. Re-collect with Per-Node Phase | 3/3 | COMPLETE | 2026-03-10 |
| 02.2. Collect RL-step-only (min change) | 1/1 | COMPLETE (8.5 GB) | 2026-03-10 |
| 3. Surrogate Training (Phase 02.2 data) | 1/3 | **IN PROGRESS** | - |
| 4. Surrogate Validation | 0/2 | **PLANNED** | - |
| 5. RL Training | 0/0 | Not planned | - |
| 6. LaTeX Report | 4/4 | COMPLETE | 2026-03-11 |
| 7. Foundation Model | 0/0 | Not planned | - |
| 8. Elastica RL Baseline | 0/0 | Not planned | - |
| 9. Physics Framework Comparison | 0/0 | Not planned | - |
| 10. Tunnel/Pipe Navigation | 0/0 | Not planned | - |
| 11. Model-Based RL | 0/0 | Not planned | - |
| 12. Hamiltonian/Lagrangian NNs | 0/0 | Researched | - |
