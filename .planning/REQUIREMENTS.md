# Requirements: Surrogate Data Collection & Validation

**Defined:** 2026-03-09
**Updated:** 2026-03-10
**Core Value:** Produce a high-quality, well-covered dataset of snake robot dynamics transitions ready for surrogate model training

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Collection (Phase 1 — COMPLETE)

- [x] **DCOL-01**: Parallel data collection from PyElastica with 16 workers
- [x] **DCOL-02**: Sobol quasi-random actions for better 5D coverage
- [x] **DCOL-03**: State perturbation (30% of episodes) for diverse initial conditions
- [x] **DCOL-04**: NaN/Inf filtering, atomic saves, graceful shutdown
- [x] **DCOL-05**: Health monitoring with crash/stall detection and W&B alerts

### Data Validation (Phase 2 — COMPLETE)

- [x] **DVAL-01**: Per-dimension distribution analysis — histograms for all 124 state dims and 5 action dims showing uniformity or skew
- [x] **DVAL-02**: Data quality checks — NaN/Inf count, duplicate transitions, constant/near-constant features, outlier detection (>5 sigma)
- [x] **DVAL-03**: Temporal analysis — episode length distribution, step index bias (early vs late timestep over-representation)
- [x] **DVAL-04**: Action space coverage — 5D coverage metric (binned fill fraction), identification of under-sampled action regions
- [x] **DVAL-05**: Summary report with pass/fail assessment per metric and actionable recommendations for surrogate training readiness

### Re-collection with Per-Node Phase Encoding (Phase 02.1)

- [x] **RCOL-01**: Per-element CPG phase encoding — 60-dim feature replacing 2-dim global phase: sin/cos/kappa for each of 20 rod elements, computed from action and serpenoid time
- [x] **RCOL-02**: Checkpoint-format collection — each run calls env.step(action) 4 times with the same action, saving rod state at each macro-step boundary (5 states, 4 valid pairs)
- [x] **RCOL-03**: Improved omega_z coverage — perturb_omega_std increased to 1.5 rad/s (from 0.05), forces collection disabled; 25 GB target dataset
- [x] **RCOL-04**: OverlappingPairDataset — loads checkpoint batch files, forms (state, per_element_phase, next_state) pairs on-the-fly with density-weighted sampling

### RL-Step Data Collection (Phase 02.2)

- [x] **RLDC-01**: Flat-format RL-step collection — steps_per_run=1, producing flat states(N,124)/next_states(N,124) instead of substep_states(N,K+1,124); auto-enabled when steps_per_run=1 and collect_forces=True
- [x] **RLDC-02**: Force/torque capture in each batch file — external_forces(N,3,21), internal_forces(N,3,21), external_torques(N,3,20), internal_torques(N,3,20); captured via RodState2D.pack_forces() after each env._step()
- [x] **RLDC-03**: FlatStepDataset class loading Phase 02.2 flat .pt batches with forces, train/val split by episode_id, per-item forces dict in __getitem__; 8 automated tests

### Surrogate Model Training (Phase 3)

- [ ] **SURR-01**: Hyperparameter sweep infrastructure — sweep runner script that launches training with different LR x model size configs, tracks results per run
- [ ] **SURR-02**: Execute sweep — train 5 configurations (LR={1e-4, 3e-4, 1e-3} x hidden_dims={256x3, 512x3}) with W&B logging, early stopping, density weighting
- [ ] **SURR-03**: Best model selection — select model with lowest single-step validation MSE across sweep runs
- [ ] **SURR-04**: Per-component error analysis in physical units — compute RMSE/MAE per state component (pos mm, vel mm/s, yaw rad, omega rad/s)
- [ ] **SURR-05**: Diagnostic plots — error histograms per component, predicted-vs-actual overlays, sweep comparison chart saved to figures/surrogate_training/

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Analysis

- **ADVN-01**: State-action joint coverage heatmaps (PCA-reduced 2D projections)
- **ADVN-02**: Coverage gap analysis identifying specific regions needing follow-up collection
- **ADVN-03**: Automatic perturbation parameter adjustment based on coverage gaps

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time web dashboard | W&B already provides dashboards |
| Automatic recollection | Validate first, then decide manually |
| Distributed multi-machine | Single machine with 48 CPUs sufficient |
| Architecture alternatives (1D-CNN, GNN) | Only if MLP fails after expanded sweep |
| Multi-step trajectory validation | Phase 4 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DCOL-01 | Phase 1 | Complete |
| DCOL-02 | Phase 1 | Complete |
| DCOL-03 | Phase 1 | Complete |
| DCOL-04 | Phase 1 | Complete |
| DCOL-05 | Phase 1 | Complete |
| DVAL-01 | Phase 2 | Complete |
| DVAL-02 | Phase 2 | Complete |
| DVAL-03 | Phase 2 | Complete |
| DVAL-04 | Phase 2 | Complete |
| DVAL-05 | Phase 2 | Complete |
| RCOL-01 | Phase 02.1 | Planned |
| RCOL-02 | Phase 02.1 | Planned |
| RCOL-03 | Phase 02.1 | Planned |
| RCOL-04 | Phase 02.1 | Planned |
| RLDC-01 | Phase 02.2 | Planned |
| RLDC-02 | Phase 02.2 | Planned |
| RLDC-03 | Phase 02.2 | Planned |
| SURR-01 | Phase 3 | Planned |
| SURR-02 | Phase 3 | Planned |
| SURR-03 | Phase 3 | Planned |
| SURR-04 | Phase 3 | Planned |
| SURR-05 | Phase 3 | Planned |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-10 — added Phase 02.1 re-collection requirements (RCOL-01 through RCOL-04); added Phase 02.2 RL-step data collection requirements (RLDC-01 through RLDC-03)*
