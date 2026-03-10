---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Collection running in tmux gsd-collect-rl → data/surrogate_rl_step/ (flat format + forces); stop manually at 10 GB
stopped_at: "Completed 06-03-PLAN.md: Methods section and Discussion physics/pipeline subsections"
last_updated: "2026-03-10T15:41:29.163Z"
last_activity: "2026-03-10 -- Completed 02.2-01: flat collection pipeline, FlatStepDataset TDD, W&B logging"
progress:
  total_phases: 13
  completed_phases: 2
  total_plans: 17
  completed_plans: 9
  percent: 53
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** Train and validate a surrogate model of snake robot dynamics for use in RL training
**Current focus:** Phase 3: Train Surrogate Model — Wave 2 (analysis, plots, model selection)

## Current Position

Phase: 02.2 (Collect RL-Step-Only Data) — COMPLETE (1/1 plans done)
Next Phase: Phase 3 surrogate training (FlatStepDataset ready; wait for ≥10 GB in data/surrogate_rl_step/)
Status: Collection running in tmux gsd-collect-rl → data/surrogate_rl_step/ (flat format + forces); stop manually at 10 GB
Last activity: 2026-03-10 -- Completed 02.2-01: flat collection pipeline, FlatStepDataset TDD, W&B logging

Progress: [█████░░░░░] 53%

## Dataset Summary

### V1 (Phase 1 Output — Archived)

- **Location:** `data/surrogate_v1/` (archived — use for reference only)
- **Size:** 2.3 GB, 28 batch files from 16 workers (w00–w15)
- **Per batch:** states (N, 124), actions (N, 5), serpenoid_times (N), next_states (N, 124), episode_ids, step_indices, forces dict

### V2 (Phase 02.1 Output — Active Collection)

- **Location:** `data/surrogate/` (active — collection running)
- **Target size:** ~25 GB (50M transition pairs)
- **Per batch:** substep_states (N, 5, 124), actions (N, 5), t_start (N,), episode_ids, step_ids
- **Format:** Checkpoint-style — 5 rod boundary states per run (start + 4 post-step)
- **State vector (124-dim):** pos_x(21), pos_y(21), vel_x(21), vel_y(21), yaw(20), omega_z(20)
- **Actions (5-dim):** amplitude, frequency, wave_number, phase_offset, direction_bias
- **Collection config:** Sobol quasi-random, 30% perturbation, 50% random action fraction, perturb_omega_std=1.5 rad/s

## Performance Metrics

- Total plans completed: 2
- Total execution time: ~3.6 hours

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

### Pending Todos

- Update Dockerfile for advisor macOS deployment (area: tooling)
- Establish mathematical formulation of surrogate model approximation (area: general)

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

### Blockers/Concerns

None currently.

## Session Continuity

Last session: 2026-03-10T15:41:29.153Z
Stopped at: Completed 06-03-PLAN.md: Methods section and Discussion physics/pipeline subsections
