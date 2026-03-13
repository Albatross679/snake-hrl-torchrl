---
created: "2026-03-10T13:07:13.148Z"
title: Retrain surrogate models in phases 3 and 3.1 on new dataset
area: general
files:
  - aprx_model_elastica/train_surrogate.py
  - aprx_model_elastica/train_config.py
  - .planning/phases/03-train-surrogate-model-using-supervised-learning/03-01-PLAN.md
  - .planning/phases/03.1-surrogate-model-architecture-experiments-rollout-loss-residual-history-window/
---

## Problem

Phase 02.1 will produce a new dataset with per-node CPG phase encoding, replacing the Phase 1 data. When that dataset arrives, the Phase 3 hyperparameter sweep and the Phase 3.1 architecture experiments (rollout loss, residual MLP, history window) must be re-run on the new data to ensure all results are based on the improved dataset.

Any models trained on the old Phase 1 data (e.g., the current 512x3 MLP baseline, val_loss=0.2161, R²=0.784) will be stale and should not be carried forward to Phase 4 validation.

## Solution

When Phase 02.1 data collection completes:
1. Update the data path in `train_config.py` (or sweep config) to point to the new dataset
2. Re-run the Phase 3 hyperparameter sweep (5 configs: LR × model size) from scratch on new data
3. Re-run Phase 3.1 architecture experiments (rollout loss variants, residual MLP, history window) on new data
4. Update `output/surrogate/best/` checkpoint with the winner from the new sweep
5. Archive or clearly label any checkpoints trained on old Phase 1 data to avoid confusion
