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

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Collection | - | COMPLETE | 2026-03-10 |
| 2. Data Validation | 0/2 | Not started | - |
