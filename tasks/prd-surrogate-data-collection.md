---
name: Surrogate Data Collection — Healthy Run
description: >
  Collect maximum feasible transitions from PyElastica for neural surrogate
  training, with pre-flight smoke test and post-collection data validation.
type: task
status: planned
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, data-collection, elastica, prd]
aliases: []
---

# PRD: Surrogate Data Collection — Healthy Run

## Introduction

Run the surrogate data collection pipeline (`aprx_model_elastica/collect_data.py`) to completion, collecting the maximum feasible number of transitions from PyElastica. The goal is a single clean, monitored run that produces validated training data for the neural surrogate model, with safety rails to prevent data loss or silent failures.

## Goals

- Collect the maximum feasible transitions (target: **10M**) within hardware constraints (45 GB disk, 48 CPUs, 251 GB RAM)
- Run with 16 parallel workers using Sobol quasi-random actions (random-only, no policy)
- No force/torque collection (not needed by surrogate, saves ~4x storage)
- Graceful handling of interrupts — partial data is always saved
- Post-collection spot-check: verify data integrity, shapes, NaN-free, coverage stats

## User Stories

### US-001: Pre-flight validation
**Description:** As a user, I want to verify the pipeline works before committing to a 10-hour run, so that I don't waste time on a broken configuration.

**Acceptance Criteria:**
- [ ] Run a small smoke test: 2 workers, 1000 transitions, to a temp directory
- [ ] Verify output files exist with correct shapes (states: `(N, 124)`, actions: `(N, 5)`)
- [ ] Verify no NaN/Inf values in output tensors
- [ ] Verify episode IDs are non-colliding across workers
- [ ] Clean up temp directory after smoke test
- [ ] Print estimated wall-clock time for the full 10M run based on measured FPS

### US-002: Execute full data collection
**Description:** As a user, I want to run the full collection with maximum throughput and safety, so that I get as much data as possible without risk of data loss.

**Acceptance Criteria:**
- [ ] Run with: 16 workers, Sobol actions, 30% perturbation, no forces, `.pt` format
- [ ] Target: 10M transitions to `data/surrogate/`
- [ ] Disk space check before starting: abort if <15 GB free (10M transitions ≈ 10 GB + safety margin)
- [ ] Progress logging every 50 episodes per worker (already implemented)
- [ ] Ctrl-C gracefully terminates workers and saves partial batches (already implemented)
- [ ] Output directory: `data/surrogate/`

### US-003: Live W&B monitoring during collection
**Description:** As a user, I want a live W&B dashboard showing whether data collection is ahead or behind schedule, so I can monitor remotely and decide whether to intervene.

**Acceptance Criteria:**
- [ ] The smoke test (US-001) establishes a **baseline FPS** from measured throughput
- [ ] `collect_data.py` initializes a W&B run (project: `surrogate-data-collection`) at startup
- [ ] The main process logs aggregate metrics to W&B at a regular interval (every 30–60s), reading from the shared counter
- [ ] Logged metrics per interval:
  - `transitions_collected` — total so far (from shared counter)
  - `fps_current` — transitions per second over the last interval
  - `fps_rolling` — rolling average FPS since start
  - `fps_baseline` — expected FPS from smoke test
  - `schedule_delta_pct` — `(actual - expected) / expected × 100%` (positive = ahead)
  - `pct_complete` — `transitions_collected / target × 100`
  - `eta_hours` — estimated hours remaining at current FPS
  - `disk_used_gb` — size of `data/surrogate/` on disk
  - `disk_free_gb` — remaining free disk space
- [ ] W&B run config logs: num_workers, num_transitions target, baseline_fps, save_dir, sobol, perturbation settings
- [ ] W&B run finishes cleanly on normal completion or Ctrl-C (calls `wandb.finish()`)
- [ ] W&B is optional: if `wandb` is not installed or `WANDB_API_KEY` is not set, collection runs without it (warning printed)

### US-004: Post-collection data validation
**Description:** As a user, I want to spot-check the collected data for integrity and coverage, so that I know the data is suitable for surrogate training.

**Acceptance Criteria:**
- [ ] Load all batch files and report: total transitions, total episodes, total files, total size on disk
- [ ] Verify no NaN or Inf in states, actions, next_states, serpenoid_times
- [ ] Verify state dimensions: states and next_states are `(N, 124)`, actions are `(N, 5)`
- [ ] Verify episode ID uniqueness (no collisions across workers)
- [ ] Report basic coverage stats: min/max/mean/std for each state group (pos_x, pos_y, vel_x, vel_y, yaw, omega_z)
- [ ] Report action distribution stats: min/max/mean per action dimension
- [ ] Report episode length distribution: min/max/mean steps per episode
- [ ] Print summary table to stdout

## Functional Requirements

- FR-1: The smoke test must complete in under 2 minutes and clean up after itself
- FR-2: The full collection must use `--num-transitions 10000000 --num-workers 16 --no-collect-forces --save-dir data/surrogate`
- FR-3: Pre-run disk space check must abort with a clear error if insufficient space
- FR-4: The validation script must work on both `.pt` and `.parquet` batch files
- FR-5: The validation script must exit with code 0 if all checks pass, non-zero otherwise
- FR-6: All batch files must use the existing naming convention: `batch_w{id}_{idx}.pt`
- FR-7: W&B logging runs in the main process only (not in workers) — reads the shared `mp.Value` counter, no IPC beyond the existing atomic counter
- FR-8: W&B is optional — `collect_data.py` must work without wandb installed (graceful degradation with a warning)
- FR-9: The `--baseline-fps` flag sets the expected FPS for schedule delta calculation (from smoke test)

## Non-Goals

- No surrogate model training in this PRD (separate task)
- No force/torque collection
- No on-policy data collection (no trained policy checkpoint)
- No changes to the collection algorithm or physics
- No custom web dashboard (W&B provides the live monitoring UI)
- No automatic restart on failure (manual re-run uses append mode)

## Technical Considerations

- **Storage estimate**: ~1 KB per transition without forces. 10M transitions ≈ 10 GB. 45 GB free disk is ample.
- **FPS measurements**:
  - Single env: **17 FPS** (57 ms/step, measured)
  - 16 envs via TorchRL ParallelEnv (IPC): **51 FPS** (measured)
  - 2 independent workers (no IPC): **29 FPS** (measured, near-linear: 2 × 17)
  - 16 independent workers (no IPC): **~270 FPS** (projected, 16 × 17 — not yet measured; the smoke test will establish the actual baseline)
- **Wall-clock estimate**: Depends on actual FPS. At 51 FPS (worst case, IPC-like): ~54 hours. At 270 FPS (projected): ~10 hours. The smoke test baseline determines which scenario we're in.
- **Memory**: Each worker uses ~50 MB (PyElastica rod + Python overhead). 16 workers ≈ 800 MB. Negligible vs 233 GB available.
- **Append mode**: If the run is interrupted and restarted, `collect_data.py` already detects existing data and appends (batch indices and episode IDs offset automatically).
- **Batch save frequency**: Default `episodes_per_save=100`. At ~500 steps/episode, that's ~50K transitions per batch file. Frequent enough to limit data loss on crash.

## Success Metrics

- Smoke test passes with zero errors
- Full collection completes (or is gracefully interrupted) with >0 valid transitions saved
- Validation reports zero NaN/Inf values
- Validation reports zero episode ID collisions
- Action space coverage: all 5 action dimensions span the full [-1, 1] range (within 0.05 of bounds)
- State space coverage: position, velocity, yaw, and omega groups all have non-trivial std (not collapsed to a single mode)

## Open Questions

- Should we set a wall-clock timeout (e.g., 12 hours max) instead of a fixed transition target, to bound the run duration?
- Should the validation script be a standalone file (`aprx_model_elastica/validate_data.py`) or a subcommand (`python -m aprx_model_elastica validate-data`)?
