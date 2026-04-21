---
phase: 03-train-surrogate-model-using-supervised-learning
plan: 01
subsystem: ml-training
tags: [surrogate, transformer, ft-transformer, rmsnorm, sweep, architecture, wandb]

# Dependency graph
requires:
  - phase: 02.2-collect-rl-step-only-minimal-change-from-2-1
    provides: FlatStepDataset and data/surrogate_rl_step/ flat-format transitions
provides:
  - TransformerSurrogateModel (FT-Transformer with RMSNorm, CLS token, Pre-Norm blocks)
  - RMSNorm normalization module
  - --arch CLI arg for train_surrogate.py (mlp/residual/transformer selection)
  - sweep.py with 15 architecture configs (M1-M5, R1-R3, W1-W3, T1-T4)
  - Expanded W&B metrics (param_count, batch_size, gpu_memory_mb)
  - 19 unit and integration tests
affects: [03-02, 03-03, 03-04]

# Tech tracking
tech-stack:
  added: [RMSNorm, FT-Transformer]
  patterns: [per-scalar-embedding, cls-token-pooling, pre-norm-transformer]

key-files:
  created:
    - papers/aprx_model_elastica/sweep.py
    - tests/test_surrogate_phase3.py
  modified:
    - papers/aprx_model_elastica/model.py
    - papers/aprx_model_elastica/train_config.py
    - papers/aprx_model_elastica/train_surrogate.py

key-decisions:
  - "RMSNorm for all transformer normalization (not LayerNorm)"
  - "FT-Transformer approach: per-scalar embedding with CLS token pooling"
  - "--arch CLI overrides --use-residual for cleaner sweep support"
  - "FlatStepDataset rollout guard: auto-forces rollout_loss_weight=0.0"
  - "Sweep runs sequentially (not parallel) for GPU memory safety"

patterns-established:
  - "arch field on SurrogateModelConfig controls model selection"
  - "Transformer configs pass --n-layers/--n-heads/--d-model via CLI"

requirements-completed: [SURR-01]

# Metrics
duration: 7min
completed: 2026-03-17
---

# Phase 3 Plan 01: Architecture Sweep Infrastructure Summary

**FT-Transformer surrogate model with RMSNorm, 15-config sweep.py, --arch CLI, and expanded W&B metrics for architecture comparison**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-17T15:24:36Z
- **Completed:** 2026-03-17T15:31:50Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- TransformerSurrogateModel with FT-Transformer architecture (per-scalar embeddings, CLS token, Pre-Norm blocks with RMSNorm)
- 15-config sweep.py covering MLP, Residual MLP, Wide/Deep MLP, and FT-Transformer architectures
- --arch CLI arg with transformer-specific --n-layers/--n-heads/--d-model overrides
- gpu_memory_mb and param_count/batch_size W&B init metrics
- 19 passing unit and integration tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Add TransformerSurrogateModel and update configs (TDD)** - `2bba751` (test: RED), `56dcbcd` (feat: GREEN)
2. **Task 2: Wire FlatStepDataset, --arch CLI, W&B metrics, and 15-config sweep** - `4adc71f` (feat)

## Files Created/Modified
- `papers/aprx_model_elastica/model.py` - Added RMSNorm, TransformerSurrogateModel, _TransformerBlock
- `papers/aprx_model_elastica/train_config.py` - Added arch, n_layers, n_heads, d_model fields to SurrogateModelConfig
- `papers/aprx_model_elastica/train_surrogate.py` - Added --arch/--n-layers/--n-heads/--d-model CLI, transformer model branch, W&B metrics, rollout guard
- `papers/aprx_model_elastica/sweep.py` - New 15-config architecture sweep runner
- `tests/test_surrogate_phase3.py` - 19 unit and integration tests

## Decisions Made
- RMSNorm (not LayerNorm) for transformer normalization -- modern standard, better training stability
- FT-Transformer per-scalar embedding -- each of input_dim scalars gets its own Linear(1, d_model) projection
- --arch overrides --use-residual for cleaner sweep CLI
- FlatStepDataset rollout guard auto-forces rollout_loss_weight=0.0 -- Phase 02.2 has no trajectory windows
- Sweep runs sequentially (not parallel) -- safer for single-GPU setup

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Adapted dimensions for 130-dim relative state representation**
- **Found during:** Task 1
- **Issue:** Plan specified 124/131 dimensions (raw state), but codebase uses 130-dim relative representation with input_dim=139
- **Fix:** Tests and model use SurrogateModelConfig defaults which reflect the actual 130/139/130 dimensions
- **Files modified:** tests/test_surrogate_phase3.py
- **Committed in:** 56dcbcd

---

**Total deviations:** 1 auto-fixed (1 blocking -- dimension mismatch between plan and codebase)
**Impact on plan:** Necessary adaptation to match actual codebase state. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All sweep infrastructure ready for Plan 02 (launch 15-config sweep)
- sweep.py --dry-run verified with 15 configs
- TransformerSurrogateModel imports and forward pass verified
- All tests pass

---
*Phase: 03-train-surrogate-model-using-supervised-learning*
*Completed: 2026-03-17*
