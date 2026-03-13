---
phase: 03-train-surrogate-model-using-supervised-learning
plan: 01
subsystem: surrogate-training
tags: [transformer, ft-transformer, sweep, wandb, architecture]
---

## One-liner
Implemented TransformerSurrogateModel (FT-Transformer with RMSNorm), wired FlatStepDataset, added --arch CLI, expanded W&B logging, and configured 15-config sweep.

## What changed
- **model.py**: Added `TransformerSurrogateModel` with FT-Transformer architecture (per-feature Linear embeddings, CLS token, Pre-Norm encoder blocks with RMSNorm, zero-initialized output head). Added `RMSNorm` class.
- **train_config.py**: Added `arch`, `n_layers`, `n_heads`, `d_model` fields to `SurrogateModelConfig`.
- **train_surrogate.py**: Switched to `FlatStepDataset`, added `--arch` CLI arg (mlp/residual/transformer), added `--n-layers`/`--n-heads`/`--d-model` args, added 6 new W&B metrics (param_count, batch_size, grad_norm, epoch_time, train_val_gap, gpu_memory_mb), added rollout loss guard for flat data, added auto-batch size detection.
- **sweep.py**: 15 configs (M1-M5 MLP, R1-R3 Residual, W1-W3 Wide/Deep, T1-T4 Transformer) with `--arch` flag passing.
- **tests/test_surrogate_phase3.py**: Unit tests for TransformerSurrogateModel shapes, config fields, sweep config count.

## Verification
- `python3 -m pytest tests/test_surrogate_phase3.py -x -q` — all tests pass
- `python3 -m aprx_model_elastica.sweep --dry-run` — prints 15-config table
- `python3 -c "from aprx_model_elastica.model import TransformerSurrogateModel; print('OK')"` — import succeeds

## Status
Complete. All Plan 01 artifacts exist and verified.
