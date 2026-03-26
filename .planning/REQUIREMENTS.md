# Requirements: Surrogate Data Collection & Validation

**Defined:** 2026-03-09
**Updated:** 2026-03-26
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

- [x] **SURR-01**: Hyperparameter sweep infrastructure — sweep runner script that launches training with different LR x model size configs, tracks results per run
- [x] **SURR-02**: Execute sweep — train 5 configurations (LR={1e-4, 3e-4, 1e-3} x hidden_dims={256x3, 512x3}) with W&B logging, early stopping, density weighting
- [ ] **SURR-03**: Best model selection — select model with lowest single-step validation MSE across sweep runs
- [x] **SURR-04**: Per-component error analysis in physical units — compute RMSE/MAE per state component (pos mm, vel mm/s, yaw rad, omega rad/s)
- [ ] **SURR-05**: Diagnostic plots — error histograms per component, predicted-vs-actual overlays, sweep comparison chart saved to figures/surrogate_training/

### Surrogate Model Validation (Phase 4)

- [ ] **SVAL-01**: Architecture-aware model loading — dispatch model class (MLP/Residual/Transformer) from config.json arch field; error on missing arch
- [ ] **SVAL-02**: Dynamic checkpoint discovery — scan directory for subdirectories with valid config.json; model-count agnostic
- [ ] **SVAL-03**: Four validation scenarios — random actions (10 ep), forward crawl (10 ep), slow/fast gaits (5+5 ep), trained PPO policy (10 ep), each up to 500 steps
- [ ] **SVAL-04**: Per-scenario PASS/WARN/FAIL verdicts — RMSE@50 and CoM drift@50 thresholds with FAIL-SOFT/FAIL-HARD tiers, NaN detection, flagged components (>2x average)
- [ ] **SVAL-05**: Validation figures — trajectory overlays (best/median/worst per scenario), per-component error heatmap, scenario comparison bars, RMSE/CoM drift over time
- [ ] **SVAL-06**: Structured validation report — markdown report at output/surrogate/validation_report.md with summary table, per-scenario details, diagnosis section, threshold documentation

### Choi2025 Soft Robot Replication (Phase 14)

- [x] **CHOI-01**: PPO config dataclass — `Choi2025PPOConfig(PPOConfig)` with clip=0.2, epochs=10, minibatch=64, 3x256 ReLU MLP, W&B project `choi2025-replication`
- [x] **CHOI-02**: PPO training entry point — `train_ppo.py` with same CLI interface as SAC `train.py`, using `PPOTrainer` from `src/trainers/ppo.py`
- [x] **CHOI-03**: Experiment matrix runner — `run_experiment.py` orchestrating 8 sequential runs (4 tasks x 2 algos) with GPU cleanup between runs, `--quick` flag for 100K validation
- [x] **CHOI-04**: Quick validation — all 8 configs run for 100K frames without crashes, W&B runs visible in `choi2025-replication` project
- [x] **CHOI-05**: Full training — all 8 configs run for 1M frames, launched in tmux with wall-time limits
- [x] **CHOI-06**: Video rollouts — 1-2 episode videos per task from best SAC and PPO checkpoints, saved to `media/choi2025/`
- [x] **CHOI-07**: Results documentation — comprehensive experiment report with learning signal assessment (reward improves over training) for all 8 runs

### MM-RKHS Operator-Theoretic Policy Gradient (Phase 15)

- [x] **MMRKHS-01**: MMRKHSConfig dataclass — `MMRKHSConfig(RLConfig)` with beta=1.0, eta=1.0, mmd_bandwidth=1.0, mmd_num_samples=16, gae_lambda=0.95, value_coef=0.5; no entropy_coef or clip_epsilon
- [x] **MMRKHS-02**: MMRKHSTrainer class — follows PPOTrainer pattern (`__init__`/`train()`/`_update()`), creates actor via `create_actor()`, critic via `create_critic()`, single Adam optimizer, GAE, SyncDataCollector
- [x] **MMRKHS-03**: MMD penalty computation — RBF kernel with linear-time unbiased estimator (O(n)), configurable bandwidth and sample count, returns finite non-negative scalar
- [x] **MMRKHS-04**: MM-RKHS loss function — `loss = -E[ratio*A] + beta*MMD^2 + (1/eta)*KL + value_coef*critic_loss`; log-ratio clamped [-20,20], advantage normalized, NaN guards
- [x] **MMRKHS-05**: Choi2025 benchmark integration — `Choi2025MMRKHSConfig(MMRKHSConfig)` + `train_mmrkhs.py` entry point; 100K-frame quick validation on follow_target completes without crash
- [x] **MMRKHS-06**: Checkpoint save/load — atomic saves with backup, round-trip restores actor/critic/optimizer state; follows PPOTrainer checkpoint pattern

### PINN Debugging Pipeline (Phase 17)

- [ ] **PDIAG-01**: Probe PDE validation suite — 4 generic probe PDEs (heat, advection, Burgers, reaction-diffusion) with analytical solutions and automated pass/fail criteria; each probe tests one additional PINN capability progressively; `ALL_PROBES` list and `run_probe_validation()` runner in `src/pinn/probe_pdes.py`
- [ ] **PDIAG-02**: PDE system analysis — `analyze_pde_system()` function evaluating CosseratRHS per-term residual magnitudes, nondimensionalization quality (good/acceptable/poor), stiffness indicator (Jacobian condition number), and magnitude spread; validates actual physics setup before training
- [ ] **PDIAG-03**: Diagnostic failure detection metrics — `PINNDiagnostics` middleware class computing loss component ratios, per-loss-term gradient norms, residual spatial distribution, per-component physics violation magnitudes, and ReLoBRaLo weight health; returns metrics dict each epoch for W&B logging (log-only, no `wandb.alert()` per D-07)
- [ ] **PDIAG-04**: Probe pre-flight integration — `train_pinn.py` auto-runs `run_probe_validation()` before training (opt-out via `--skip-probes`); prints PASS/FAIL per probe with warning if any fail
- [ ] **PDIAG-05**: NTK eigenvalue diagnostics — `compute_ntk_eigenvalues()` standalone function computing approximate NTK spectrum (eigenvalue_max, eigenvalue_min, condition_number, spectral_decay_rate) via parameter subsampling (n_params_sample=500); runs every 50 epochs via PINNDiagnostics middleware
- [x] **PDIAG-06**: pinn-debug Claude Code skill — `.claude/skills/pinn-debug/SKILL.md` with 4-phase diagnostic workflow (probe validation, dashboard metrics, loss decision tree, physics sub-tree), inline decision tree for fault isolation, quick symptom lookup table; `.claude/skills/pinn-debug/references/failure-modes.md` with 7 documented failure modes and literature citations

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
| RCOL-01 | Phase 02.1 | Complete |
| RCOL-02 | Phase 02.1 | Complete |
| RCOL-03 | Phase 02.1 | Complete |
| RCOL-04 | Phase 02.1 | Complete |
| RLDC-01 | Phase 02.2 | Complete |
| RLDC-02 | Phase 02.2 | Complete |
| RLDC-03 | Phase 02.2 | Complete |
| SURR-01 | Phase 3 | Planned |
| SURR-02 | Phase 3 | Planned |
| SURR-03 | Phase 3 | Planned |
| SURR-04 | Phase 3 | Planned |
| SURR-05 | Phase 3 | Planned |
| SVAL-01 | Phase 4 | Planned |
| SVAL-02 | Phase 4 | Planned |
| SVAL-03 | Phase 4 | Planned |
| SVAL-04 | Phase 4 | Planned |
| SVAL-05 | Phase 4 | Planned |
| SVAL-06 | Phase 4 | Planned |
| CHOI-01 | Phase 14 | Complete |
| CHOI-02 | Phase 14 | Complete |
| CHOI-03 | Phase 14 | Complete |
| CHOI-04 | Phase 14 | Complete |
| CHOI-05 | Phase 14 | Complete |
| CHOI-06 | Phase 14 | Complete |
| CHOI-07 | Phase 14 | Complete |
| MMRKHS-01 | Phase 15 | Planned |
| MMRKHS-02 | Phase 15 | Planned |
| MMRKHS-03 | Phase 15 | Planned |
| MMRKHS-04 | Phase 15 | Planned |
| MMRKHS-05 | Phase 15 | Planned |
| MMRKHS-06 | Phase 15 | Planned |
| PDIAG-01 | Phase 17 | Planned |
| PDIAG-02 | Phase 17 | Planned |
| PDIAG-03 | Phase 17 | Planned |
| PDIAG-04 | Phase 17 | Planned |
| PDIAG-05 | Phase 17 | Planned |
| PDIAG-06 | Phase 17 | Planned |

**Coverage:**
- v1 requirements: 47 total
- Mapped to phases: 47
- Unmapped: 0

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-26 — added Phase 17 PINN diagnostics requirements (PDIAG-01 through PDIAG-06)*
