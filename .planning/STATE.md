---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Ready to execute
stopped_at: Completed 17-03-PLAN.md
last_updated: "2026-03-26T13:25:08.581Z"
progress:
  total_phases: 21
  completed_phases: 7
  total_plans: 33
  completed_plans: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** Train and validate a surrogate model of snake robot dynamics for use in RL training
**Current focus:** Phase 17 — design-pinn-debugging-workflow-and-diagnostic-pipeline

## Current Position

Phase: 17 (design-pinn-debugging-workflow-and-diagnostic-pipeline) — EXECUTING
Plan: 2 of 3

## Dataset Summary

### V1 (Phase 1 Output — Archived)

- **Location:** `data/surrogate_v1/` (archived — use for reference only)
- **Size:** 2.3 GB, 28 batch files from 16 workers (w00–w15)
- **Per batch:** states (N, 124), actions (N, 5), serpenoid_times (N), next_states (N, 124), episode_ids, step_indices, forces dict

### V2 (Phase 02.1 Output — Active Collection)

- **Location:** `data/surrogate/` (active — collection running in tmux gsd-collect)
- **Target size:** ~25 GB (50M transition pairs)
- **Per batch:** substep_states (N, 5, 124), actions (N, 5), t_start (N,), episode_ids, step_ids
- **Format:** Checkpoint-style — 5 rod boundary states per run (start + 4 post-step)
- **State vector (124-dim):** pos_x(21), pos_y(21), vel_x(21), vel_y(21), yaw(20), omega_z(20)
- **Actions (5-dim):** amplitude, frequency, wave_number, phase_offset, direction_bias
- **Collection config:** Sobol quasi-random, 30% perturbation, 50% random action fraction, perturb_omega_std=1.5 rad/s

### V2.2 (Phase 02.2 Output — Active Collection)

- **Location:** `data/surrogate_rl_step/` (active — collection running in tmux gsd-collect-rl, stop at 10 GB)
- **Target size:** ≥10 GB minimum (~5M transitions at ~2055 bytes/transition)
- **Per batch:** states (N, 124), next_states (N, 124), actions (N, 5), t_start (N,), episode_ids, step_ids, forces dict
- **Format:** Flat — one row per RL step; ~2055 bytes/transition (with forces)
- **Forces:** external_forces (N,3,21), internal_forces (N,3,21), external_torques (N,3,20), internal_torques (N,3,20)
- **Collection config:** Sobol, 30% perturbation, steps_per_run=1, collect_forces=True, perturb_omega_std=1.5 rad/s
- **Dataset class:** `FlatStepDataset` (aprx_model_elastica/dataset.py)

## Performance Metrics

- Total plans completed: 9
- Total execution time: ~3.6 hours

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 02.2-collect-rl-step-only-minimal-change-from-2-1 | 01 | 16 min | 6 | 4 |
| Phase 06-write-research-report-in-latex P04 | 5 | 2 tasks | 3 files |
| Phase 06.1 P01 | 11 min | 2 tasks | 1 files |
| Phase 03 P01 | 7 min | 2 tasks | 5 files |
| Phase 03 P04 | 10 min | 2 tasks | 5 files |
| Phase 03 P02 | 95 min | 2 tasks | 3 files |
| Phase 13 P01 | 3 min | 1 tasks | 5 files |
| Phase 13 P02 | 3 min | 1 tasks | 4 files |
| Phase 14 P01 | 8 min | 3 tasks | 8 files |
| Phase 14 P02 | 81 min | 3 tasks | 6 files |
| Phase 14 P03 | 20 min | 2 tasks | 10 files |
| Phase 15 P01 | 6 min | 2 tasks | 5 files |
| Phase 15 P02 | 18min | 2 tasks | 3 files |
| Phase 17 P03 | 4min | 2 tasks | 2 files |

## Accumulated Context

### Decisions

- [Restructure]: Phase 1 = data collection (complete), Phase 2 = data validation (current focus)
- [Restructure]: Removed old Phase 2 (coverage tracking) and Phase 3 (quality reporting) — merged relevant concerns into Phase 2 validation
- [Phase 02]: Load all batch files directly without SurrogateDataset to validate full unfiltered dataset
- [Phase 02]: 8-metric pass/fail rubric with PASS/WARN/FAIL thresholds for data quality assessment
- [Phase 03]: Sweep design — 5 runs, all 3 LRs (1e-4, 3e-4, 1e-3), all 3 model sizes (256x3, 512x3, 512x4)
- [Phase 03]: Best model = sweep_lr1e3_h512x3 (lr=1e-3, 512x3, val_loss=0.2161, R²=0.784, epoch 124)
- Phase 02.1 inserted after Phase 2: re-collect surrogate data with per-node phase encoding (URGENT)
- [Restructure]: Phase 03.1 merged into Phase 03 — architecture experiments are now Plans 03-05 of Phase 3
- [Phase 03, Plan 03]: ResidualSurrogateModel asserts uniform hidden_dims to prevent shape mismatches in skip connections
- [Phase 03, Plan 03]: HistoryDataset extends TrajectoryDataset(rollout_length=history_k+1) — reuses window builder
- [Phase 03, Plan 03]: History training loop deferred to arch_sweep.py (Plan 04) — only CLI arg wiring in train_surrogate.py
- [Phase 03, Plan 04]: ARCH_SWEEP_CONFIGS has 5 entries (A1/A3/A4/A5/B1); A2 injected from Plan 01 metrics.json as baseline
- [Phase 03, Plan 04]: Default output-base is output/surrogate/arch_sweep — isolates arch sweep from Plan 01 runs
- [Phase 02.1]: wave_number denorm range [0.5,3.5] hardcoded from perturb_rod_state(); TIME_ENC_DIM deprecated but kept; INPUT_DIM 131→189 breaks old models intentionally
- [Phase 02.1]: [Phase 02.1-03]: OverlappingPairDataset computes per-element phase on-the-fly from (action, t_start) — not pre-stored in batch files
- [Phase 02.1]: V2 checkpoint format: 5 boundary states per run enables flexible overlapping training pairs without episode boundary data loss
- [Phase 02.1]: perturb_omega_std increased 0.05→1.5 rad/s to cover operational CPG omega_z range of 1-10 rad/s
- [Phase 02.1]: Phase 1 surrogate data archived to data/surrogate_v1/ (not deleted) — rollback capability preserved
- [Phase 02.2]: --skip-disk-check required at runtime: estimator uses 1 KB/transition but forces add ~2x overhead; actual ~2055 bytes/transition
- [Phase 02.2]: W&B per-batch logging in _collection_loop() (single-process) and monitor loop (multi-process); W&B runs not serializable across subprocess boundaries
- [Phase 02.2]: Collection stops manually at 10 GB (not 50M), meeting RLDC-03 minimum with ~5M transitions at ~2055 bytes each
- [Phase 02.2]: flat_output auto-set True when steps_per_run==1 and collect_forces=True
- [Phase 06-write-research-report-in-latex]: report/ subdirectory chosen for report.tex with graphicspath set to ../figures/
- [Phase 06-write-research-report-in-latex]: [Phase 06, Plan 01]: natbib round,sort,authoryear with hyperref loaded last; 11 BibTeX entries for all cited papers
- [Phase 06]: [Phase 06, Plan 02]: CPG equation uses explicit 2pi factors for unambiguous notation
- [Phase 06-write-research-report-in-latex]: [Phase 06, Plan 03]: Methods section uses two-phase training narrative (single-step MSE then combined rollout loss) with xrightarrow architecture pipeline in equation environment
- [Phase 06-write-research-report-in-latex]: [Phase 06, Plan 03]: Discussion challenges use textbf{} labels within subsection paragraphs rather than nested subsubsections
- [Phase 06]: Human approved PDF despite DreamerV3/Janner attribution error — deferred to future content revision
- [Phase quick-2]: Introduction uses \\Cref for sentence-initial section references; 4-paragraph structure (problem, physics detail, contribution, roadmap)
- [Phase quick-4]: Added Governing PDEs subsubsection and paragraph-level I/O summary to section 2.1
- [Phase 06.1]: Staggered Grid and External Forces demoted to subsubsection under PyElastica Backend for clean hierarchy
- [Phase 06.1]: DisMech CPG condensed to paragraph in shared CPG Control subsection rather than separate subsection
- [Phase quick-260317-lb3]: Structured Method of Lines subsubsection with paragraph-level organization and compact explicit/implicit table
- [Phase 03]: RMSNorm for all transformer normalization (not LayerNorm)
- [Phase 03]: FT-Transformer per-scalar embedding with CLS token pooling for surrogate model
- [Phase 03]: --arch CLI overrides --use-residual; sweep runs sequentially for GPU safety
- [Phase 03]: [Phase 03, Plan 04]: CoM velocity at indices 4-5 in 130-dim relative state, all dimension refs use named REL_* constants
- [Phase 03]: vram_target reduced 0.85->0.70 and probe includes denormalization overhead to prevent OOM
- [Phase 03]: --save-dir overrides timestamped run_dir for sweep directory control
- [Phase 03]: DataLoader num_workers=0 for multiprocessing safety; training_state.pt saved each epoch for resume
- [Phase 13]: PhysicsRegularizer uses trapezoidal integration for kinematic constraints with 4 constraint types
- [Phase 13]: NondimScales: L_ref=1.0m, t_ref=0.5s, F_ref=E*I/L^2 (physics-based, not z-score)
- [Phase 13]: Inextensible rod approximation: omit stretching, keep bending + RFT friction for 2D snake
- [Phase quick-260318-h2e]: PINN section uses \mathbf{} (not \bm{}) to match report convention; ansatz params named \avec to avoid action vector conflict
- [Phase quick-260319-3kt]: Markdown report (report/report.md) created as single source of truth for content edits; edit-Markdown-first workflow established
- [Phase 14]: 3x1024 network (scaled up from paper's 3x256) to maximize GPU utilization
- [Phase 14]: SAC switched from direct wandb.init to wandb_utils.setup_run for consistency
- [Phase 14]: Watchdog timeout in run_experiment.py: wall_time + 10min, exit 137/143 = hung
- [Phase 14]: Mock physics backend (_MockRodState) used since DisMech not installed
- [Phase 14]: ParallelEnv workers run on CPU to avoid CUDA context exhaustion
- [Phase 14]: Quick validation used 10K frames due to SAC UTD=4 throughput with single env
- [Phase 14]: step_mdp() for auto-reset MDP state advancement in vectorized SAC
- [Phase 14]: Used quick validation checkpoints for video rollouts since full 1M-frame training ETA ~9 days
- [Phase 14]: PPO learning signal assessed as INCONCLUSIVE (not failure) -- runs too short for batch-based learning
- [Phase 14]: SAC confirmed learning signal for all 4 tasks (21-69% reward improvement)
- [Phase 15]: No entropy bonus in MM-RKHS loss -- KL regularizer handles exploration per CONTEXT.md
- [Phase 15]: Old distribution reconstructed from stored loc/scale via TanhNormal (not from saved policy weights)
- [Phase 15]: Linear-time RBF MMD^2 estimator (O(n) not O(n^2)) using paired samples from TanhNormal distributions
- [Phase 15]: Choi2025MMRKHSConfig mirrors PPO config structure for fair comparison (same network, env, parallelism)
- [Phase 15]: SKIP_GPU_LOCK env var for concurrent multi-GPU training
- [Phase 15]: 100K MM-RKHS validation confirmed learning signal: reward 9->17 on follow_target
- [Phase quick-260320-jsu]: Report restructured to 6 chapters: Intro, Related Work, Surrogate Model, RL-Elastica, RL-DisMech, PINN; physics derivations moved to appendix; issue tracker consolidated to longtable
- [Phase 17]: Mirrored rl-debug skill structure exactly: 4 phases, decision tree format, quick symptom lookup
- [Phase 17]: Full decision tree inline in SKILL.md so Claude can diagnose without reading source code

### Pending Todos

- Optimize PyElastica inner substep loop to reduce Python overhead (area: general)
- Update Dockerfile for advisor macOS deployment (area: tooling)
- Explore pure ODE data generation as alternative to Elastica (area: general)

### Roadmap Evolution

- Phase 3 added: Train surrogate model using supervised learning
- Phase 4 added: Validate surrogate model against Elastica solver trajectories
- Phase 5 added: Train RL agent using surrogate model
- Phase 6 added: Write research report in LaTeX
- Phase 7 added: Foundation model exploration for snake robot dynamics
- Phase 8 added: Train RL baseline directly on Elastica for controlled comparison
- Phase 02.2 added: collect RL-step-only transitions (minimal change from 2.1)
- Phase 03.1 removed: architecture experiments merged into Phase 3 as Plans 03–05
- Phase 9 added: Physics framework comparison experiment — DisMech, Genesis, FEM vs Elastica
- Phase 10 added: Train RL agent for tunnel and pipe navigation — snake traverses building infrastructure
- Phase 11 added: Explore model-based RL using surrogate as world model for planning and policy optimization
- Phase 12 added: Explore Hamiltonian or Lagrangian neural networks for snake dynamics
- Phase 06.1 inserted after Phase 6: Improve report structure and organization (URGENT)
- Phase 02.3 inserted after Phase 2: Collect DisMech snake dynamics data for surrogate training (URGENT)
- Phase 13 added: Implement PINN and DD-PINN surrogate models
- Phase 14 added: Replicate Choi2025 soft robot control paper using ML workflow
- Phase 15 added: Implement Operator-Theoretic Policy Gradient (arXiv:2603.17875) in TorchRL alongside PPO and SAC
- Phase 17 added: Design PINN debugging workflow and diagnostic pipeline

### Blockers/Concerns

None currently.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | add verification for Phase 2.2 and document worker respawn investigation needs | 2026-03-11 | 4facf4a | [1-add-verification-for-phase-2-2-and-docum](./quick/1-add-verification-for-phase-2-2-and-docum/) |
| 2 | write introduction section for research report | 2026-03-11 | 96ff13b | [2-write-the-introduction-section-for-the-p](./quick/2-write-the-introduction-section-for-the-p/) |
| 3 | add structured notation explanations after equations | 2026-03-11 | c705001 | [3-add-structured-notation-explanations-aft](./quick/3-add-structured-notation-explanations-aft/) |
| 4 | restructure section 2.1 with upfront I/O framing | 2026-03-11 | 6239b38 | [4-restructure-section-2-1-to-clearly-prese](./quick/4-restructure-section-2-1-to-clearly-prese/) |
| 260316-l3z | organize ROADMAP.md for cleaner formatting | 2026-03-16 | fac3a02 | [260316-l3z-organize-roadmap-md-for-cleaner-formatti](./quick/260316-l3z-organize-roadmap-md-for-cleaner-formatti/) |
| 260316-s3f | create DisMech surrogate package and write Chapter 3 | 2026-03-16 | 946e78c | [260316-s3f-create-surrogate-model-for-dismech-based](./quick/260316-s3f-create-surrogate-model-for-dismech-based/) |
| 260317-l6p | restructure report sections 2.2 and 2.3 | 2026-03-17 | 1642e0e | [260317-l6p-restructure-report-sections-2-2-and-2-3-](./quick/260317-l6p-restructure-report-sections-2-2-and-2-3-/) |
| 260317-lcq | set all GSD quality profile agents to opus | 2026-03-17 | 5e96373 | [260317-lcq-make-sure-all-the-gsd-agents-use-opus-wi](./quick/260317-lcq-make-sure-all-the-gsd-agents-use-opus-wi/) |
| 260317-lb3 | integrate method-of-lines explanation into report | 2026-03-17 | b323dc9 | [260317-lb3-integrate-explicit-integration-method-of](./quick/260317-lb3-integrate-explicit-integration-method-of/) |
| 260318-h2e | Add PINN and DD-PINN chapter to report using existing codebase content | 2026-03-18 | 21c6170 | [260318-h2e-add-pinn-and-dd-pinn-chapter-to-report-u](./quick/260318-h2e-add-pinn-and-dd-pinn-chapter-to-report-u/) |
| 260319-3rv | Create PDE system framework knowledge document for PINNs | 2026-03-19 | 30b517d | [260319-3rv-create-markdown-document-clean-pde-syste](./quick/260319-3rv-create-markdown-document-clean-pde-syste/) |
| 260319-3kt | Create Markdown report mirroring LaTeX report.tex | 2026-03-19 | 998e78f | [260319-3kt-create-markdown-report-mirroring-latex-r](./quick/260319-3kt-create-markdown-report-mirroring-latex-r/) |
| 260319-snc | Align choi2025 SAC and PPO configs to match paper | 2026-03-19 | b1bd7d1 | [260319-snc-align-choi2025-sac-and-ppo-configs-to-ma](./quick/260319-snc-align-choi2025-sac-and-ppo-configs-to-ma/) |
| 260319-stp | Align PPO config to paper's 256x3 network | 2026-03-19 | b86b5c8 | [260319-stp-align-choi2025-sac-and-ppo-configs-to-ma](./quick/260319-stp-align-choi2025-sac-and-ppo-configs-to-ma/) |
| 260320-jsu | Rewrite LaTeX report based on new structure with concise, structured style | 2026-03-20 | b1fa428 | [260320-jsu-rewrite-latex-report-based-on-new-struct](./quick/260320-jsu-rewrite-latex-report-based-on-new-struct/) |

## Session Continuity

Last session: 2026-03-26T13:25:08.573Z
Stopped at: Completed 17-03-PLAN.md
