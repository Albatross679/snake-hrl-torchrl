# Experiment: finetune_001 - Fresh Training (130-dim relative state)

## Date: 2026-03-17

## Context
- Originally planned to fine-tune from `output/surrogate_20260317_113722` (best val=0.601, R2=0.393)
- **Shape mismatch**: old checkpoint used state_dim=128 (data: `surrogate_rl_step_rel128`), current architecture uses state_dim=130 (data: `surrogate_rl_step`)
- Input dim: 137 (old) vs 139 (new), Output dim: 128 vs 130
- Decision: train from scratch with current architecture

## Run Details
- Run dir: `output/surrogate_20260317_171708`
- W&B run: `cluh1n4v`
- PID: 345316 (on GPU 0)
- Data: `data/surrogate_rl_step` (3,903,214 train / 433,386 val transitions)
- Model: 3,433,602 params, 4x1024 MLP with SiLU + LayerNorm
- Batch size: 212,992 (auto-probed)
- LR: 0.0001, cosine schedule, 5 warmup epochs
- Early stopping patience: 30
- Mixed precision: bf16

## Hyperparameters (default)
- lr: 0.0001
- weight_decay: 0.0001
- hidden_dims: [1024, 1024, 1024, 1024]
- state_noise_std: 0.001
- density weighting: enabled (clip_max=10)

## Progress
- Epoch 1: train=1.669, val=0.999, R2=0.005 (254s/epoch)

## Target
- Previous best: val=0.601, R2=0.393 (at epoch 110, 128-dim architecture)
- Goal: match or exceed with 130-dim architecture

## Notes
- A prior hyperparameter sweep (M1-M5, R1-R3, T1-T3, W1-W3) all failed due to data path issues (relative path from wrong cwd)
- Both GPUs available (2x RTX A4000 16GB), using GPU 0
