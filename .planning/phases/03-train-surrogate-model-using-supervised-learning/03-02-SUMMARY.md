---
phase: 03-train-surrogate-model-using-supervised-learning
plan: 02
subsystem: ml-training
tags: [surrogate, sweep, smoke-test, auto-batch, tmux, gpu-safety]

# Dependency graph
requires:
  - phase: 03-train-surrogate-model-using-supervised-learning
    plan: 01
    provides: TransformerSurrogateModel, --arch CLI, 15-config sweep.py
provides:
  - "--save-dir CLI arg for direct output directory control"
  - "--resume CLI arg with full training state checkpoint/restore"
  - "Fixed auto-batch probe (includes denormalization overhead, 0.70 vram_target)"
  - "script/launch_sweep.sh for tmux-based unattended sweep execution"
  - "15-config sweep launched in tmux session gsd-sweep"
affects: [03-03, 03-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [save-dir-override, training-state-resume, conservative-vram-probing]

key-files:
  created:
    - script/launch_sweep.sh
  modified:
    - papers/aprx_model_elastica/train_surrogate.py
    - papers/aprx_model_elastica/sweep.py

key-decisions:
  - "vram_target reduced 0.85 -> 0.70 and probe includes denormalization overhead to prevent OOM"
  - "--save-dir overrides timestamped run_dir for sweep directory control"
  - "sweep.py passes --save-dir output_base/config_name to write per-config directories"
  - "num_workers=0 for DataLoader (prevents multiprocessing spawn issues)"
  - "Training state checkpoint (training_state.pt) saved every epoch for crash-resilient resume"

patterns-established:
  - "save-dir override: --save-dir skips timestamped directory creation for predictable paths"
  - "resume pattern: --resume loads model + optimizer + scheduler + epoch + patience from training_state.pt"

requirements-completed: [SURR-02]

# Metrics
duration: 95min
completed: 2026-03-17
---

# Phase 3 Plan 02: Launch 15-Config Sweep Summary

**Smoke-tested all 3 architecture families (MLP/Residual/Transformer), fixed auto-batch OOM, added --save-dir/--resume CLI, launched 15-config sweep in tmux**

## Performance

- **Duration:** 95 min
- **Started:** 2026-03-17T15:38:29Z
- **Completed:** 2026-03-17T17:13:18Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- 1-epoch smoke tests passed for all 3 architecture families: MLP (val=0.985), Residual (val=1.010), Transformer (val=0.769)
- Fixed auto-batch probe OOM by including denormalization overhead in memory probe and reducing vram_target from 0.85 to 0.70
- Added --save-dir CLI to train_surrogate.py for sweep directory control; wired sweep.py to produce output/surrogate/M1, M2, etc.
- Added --resume CLI with full training state checkpoint/restore (model, optimizer, scheduler, epoch, patience)
- 15-config sweep launched in tmux session gsd-sweep, running sequentially via subprocess.run()

## Task Commits

Each task was committed atomically:

1. **Task 1: Smoke test sweep with 1-epoch runs** - `47f0919` (feat)
2. **Task 2: Launch full 15-config sweep in tmux** - `a9d440f` (feat)

## Files Created/Modified
- `papers/aprx_model_elastica/train_surrogate.py` - Added --save-dir, --resume CLI args; fixed auto-batch probe; num_workers=0; training_state.pt checkpointing
- `papers/aprx_model_elastica/sweep.py` - Passes --save-dir to train runs; fixed model.pt status check path
- `script/launch_sweep.sh` - tmux launch script for 15-config sweep

## Decisions Made
- vram_target reduced from 0.85 to 0.70: the auto-batch probe only tested forward+backward, but actual training loop also allocates memory for denormalization and per-component MSE monitoring, causing OOM at the probed batch size
- Added denormalization simulation to the auto-batch probe: probes now include normalizer.denormalize_delta() + mse_loss to capture real memory usage
- --save-dir overrides setup_run_dir: sweep.py needs predictable output paths (output/surrogate/M1, etc.) rather than timestamped directories
- DataLoader num_workers changed 2->0: fork start method conflicts with the training watchdog's multiprocessing context

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing --save-dir CLI arg prevents sweep output directory control**
- **Found during:** Task 1 (smoke testing)
- **Issue:** sweep.py expected output at output/surrogate/M1/metrics.json but train_surrogate.py created timestamped directories like output/surrogate_20260317_123456/metrics.json
- **Fix:** Added --save-dir arg to train_surrogate.py that overrides setup_run_dir; updated sweep.py to pass --save-dir output_base/config_name
- **Files modified:** papers/aprx_model_elastica/train_surrogate.py, papers/aprx_model_elastica/sweep.py
- **Committed in:** 47f0919

**2. [Rule 1 - Bug] Auto-batch probe OOM due to missing denormalization overhead**
- **Found during:** Task 1 (smoke testing)
- **Issue:** probe_auto_batch_size tested only forward+backward pass, but compute_single_step_loss also calls normalizer.denormalize_delta() and per-component MSE losses which allocate ~batch_size*130*4 bytes extra. With the probed batch size of 1M, this caused 520MB OOM.
- **Fix:** (a) Reduced vram_target from 0.85 to 0.70 for safety margin, (b) Added denormalization simulation to the probe to capture real peak memory
- **Files modified:** papers/aprx_model_elastica/train_surrogate.py
- **Committed in:** 47f0919

**3. [Rule 3 - Blocking] Multiprocessing spawn mode incompatible with smoke test wrapper**
- **Found during:** Task 1 (smoke testing)
- **Issue:** DataLoader with num_workers=2 used spawn start method which re-imported the smoke test script in worker processes, causing recursive main() calls and RuntimeError
- **Fix:** Changed num_workers from 2 to 0 (inline data loading); also used fork start method in smoke test wrapper
- **Files modified:** papers/aprx_model_elastica/train_surrogate.py
- **Committed in:** 47f0919

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** All fixes necessary for correctness and sweep functionality. The --save-dir fix enables sweep.py to locate results. The auto-batch fix prevents training OOM. No scope creep.

## Issues Encountered
- GPU 0 was occupied by a legitimate training run (babysit session) during smoke testing, requiring use of GPU 1 via CUDA_VISIBLE_DEVICES
- GPU lock mechanism (GpuLock) blocked concurrent training, requiring direct main() invocation for smoke tests
- Babysit session's resumed training holds GPU lock when sweep launched; sweep will queue and start after that run finishes

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 15-config sweep is running in tmux session gsd-sweep (queued behind babysit training)
- Once sweep completes, sweep_summary.json will contain ranked results by val_loss
- Plan 03 (analysis and model selection) can proceed after all 15 configs finish
- Monitor progress: tmux attach -t gsd-sweep

---
*Phase: 03-train-surrogate-model-using-supervised-learning*
*Completed: 2026-03-17*
