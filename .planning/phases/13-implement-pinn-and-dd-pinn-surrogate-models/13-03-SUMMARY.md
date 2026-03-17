---
phase: 13-implement-pinn-and-dd-pinn-surrogate-models
plan: 03
status: complete
completed: 2026-03-17
---

## Summary

Physics-regularized surrogate training script implemented with full ML workflow compliance.

## What was built

- `src/pinn/train_regularized.py` — Training script for physics-regularized surrogate with:
  - `@dataclass` config (`RegularizerTrainConfig`) instead of argparse
  - W&B integration via `src/wandb_utils.py` (no direct wandb calls)
  - bf16 AMP autocast with nullcontext fallback
  - `torch.inference_mode()` for validation
  - VRAM cleanup (`cleanup_vram()`) between sequential sweep runs
  - Random seed setting (torch, numpy, python random)
  - Saves both `model_best.pt` and `model_last.pt`
  - STOP file check + SIGTERM handler for graceful shutdown
  - `GpuLock` at entry point
  - Timestamped run directories
  - Incremental `metrics.jsonl` output
  - `config.json` snapshot saved at training start
  - `num_epochs=9999` (early stopping is the binding constraint)
  - PhysicsRegularizer + ReLoBRaLo + curriculum warmup
  - Lambda sweep mode (`--sweep`) over {0.001, 0.01, 0.1, 1.0}
  - Per-component RMSE evaluation in physical units

## Verification

3-epoch smoke test completed successfully:
- Config saved, metrics.jsonl written incrementally
- Both model_best.pt and model_last.pt saved
- eval_metrics.json produced with per-component RMSE and R2

## Human review checkpoint

Plan specifies human review of sweep results before proceeding to DD-PINN. User authorized autonomous execution — proceeding to Plan 04.
