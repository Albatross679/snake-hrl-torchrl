# Phase 3: Train Surrogate Model Using Supervised Learning - Context

**Gathered:** 2026-03-10
**Updated:** 2026-03-11 (replanned: 15-config sweep with MLP + Residual + Wide/Deep + FT-Transformer)
**Status:** Ready for planning

<domain>
## Phase Boundary

Train neural surrogate models on the Phase 02.2 dataset to predict one-step Cosserat rod dynamics (state + action → next_state delta). This phase covers:

1. **Code changes**: Wire FlatStepDataset, implement TransformerSurrogateModel with RMSNorm, expand W&B logging
2. **15-config architecture sweep**: MLP, Residual MLP, Wide/Deep MLP, FT-Transformer
3. **Analysis and model selection**: Diagnostic plots, per-component RMSE, human-reviewed selection

Does NOT cover: multi-step trajectory validation (Phase 4), RL training with surrogate.

</domain>

<decisions>
## Implementation Decisions

### 15-config sweep design (LOCKED)
All configs use: Phase 02.2 data via FlatStepDataset, rollout_weight=0.0 (no trajectories), auto-batch for GPU VRAM, W&B logging.

**MLP (5 configs):**
| # | Config | LR | Hidden | LayerNorm |
|---|--------|----|--------|-----------|
| 1 | M1 | 1e-4 | 256×3 | Yes |
| 2 | M2 | 3e-4 | 256×3 | Yes |
| 3 | M3 | 1e-4 | 512×3 | Yes |
| 4 | M4 | 3e-4 | 512×3 | Yes |
| 5 | M5 | 1e-3 | 512×3 | Yes |

**Residual MLP (3 configs):**
| # | Config | LR | Hidden | LayerNorm |
|---|--------|----|--------|-----------|
| 6 | R1 | 3e-4 | 512×3 | Yes |
| 7 | R2 | 1e-3 | 512×3 | Yes |
| 8 | R3 | 3e-4 | 1024×3 | Yes |

**Wide/Deep MLP (3 configs):**
| # | Config | LR | Hidden | LayerNorm |
|---|--------|----|--------|-----------|
| 9 | W1 | 3e-4 | 512×4 | Yes |
| 10 | W2 | 3e-4 | 1024×3 | Yes |
| 11 | W3 | 1e-3 | 1024×3 | Yes |

**FT-Transformer (4 configs):**
| # | Config | LR | Layers | Heads | d_model | RMSNorm |
|---|--------|----|--------|-------|---------|---------|
| 12 | T1 | 3e-4 | 4 | 4 | 128 | Yes |
| 13 | T2 | 3e-4 | 6 | 8 | 256 | Yes |
| 14 | T3 | 1e-4 | 6 | 8 | 256 | Yes |
| 15 | T4 | 1e-4 | 8 | 8 | 512 | Yes |

### TransformerSurrogateModel architecture (LOCKED)
- FT-Transformer (Feature Tokenizer Transformer) approach
- Each of 131 input scalars → learned d_model embedding (131 feature tokens)
- [CLS] token prepended (132 tokens total)
- Pre-Norm transformer encoder blocks:
  - x + MultiHeadAttention(RMSNorm(x)) — residual around attention
  - x + FFN(RMSNorm(x)) — residual around feed-forward
- RMSNorm (NOT LayerNorm) for all transformer normalization
- Standard transformer residual connections in every block
- CLS token output → Linear → 124 (state delta), zero-initialized
- Same forward(state, action, time_encoding) → delta interface as MLP models
- Same predict_next_state() method with normalizer support

### Normalization strategy (LOCKED)
- MLP variants (M1-M5, W1-W3): LayerNorm after each hidden layer (existing)
- Residual MLP variants (R1-R3): LayerNorm in ResidualBlock (existing)
- Transformer variants (T1-T4): RMSNorm (Pre-Norm style, modern standard)
- Input/output normalization: StateNormalizer z-score (all configs)

### Expanded W&B logging (LOCKED — 6 new metrics)
- `grad_norm`: from clip_grad_norm_ return value, logged per epoch
- `param_count`: logged once at init via wandb.config
- `batch_size`: logged once at init via wandb.config (captures auto-batch result)
- `epoch_time`: wall-clock seconds per epoch
- `train_val_gap`: train_loss - val_loss (overfitting signal)
- `gpu_memory_mb`: torch.cuda.max_memory_allocated() / 1e6

### Config changes (LOCKED)
- `train_config.py`: Add `arch` field (str: "mlp"|"residual"|"transformer"), `n_layers`, `n_heads`, `d_model` to SurrogateModelConfig
- `train_surrogate.py`: Add --arch CLI arg, transformer model branch, FlatStepDataset import, new W&B metrics
- `sweep.py`: Replace SWEEP_CONFIGS with 15 new configs, pass --arch flag

### Training curriculum (LOCKED)
- Single-step MSE only (rollout_weight=0.0 for all configs — Phase 02.2 has no trajectories)
- Auto-batch size detection (find_max_batch_size already implemented)
- Density-weighted sampling enabled; input noise std=0.001
- 200 epochs max, early stopping patience=30, gradient clipping norm=1.0
- Dropout=0.0 (not needed with large dataset + LayerNorm/RMSNorm + early stopping)
- Cosine LR schedule with 5-epoch warmup

### Success criteria (LOCKED)
- Primary metric: per-step error in physical units (pos mm, vel mm/s, yaw rad, omega_z rad/s)
- Per-component breakdown: pos_x, pos_y, vel_x, vel_y, yaw, omega_z
- Human reviews diagnostic plots and per-component RMSE before promoting best model
- Phase 4 consumes output/surrogate/best/ (model.pt, normalizer.pt, config.json, selection.json)

</decisions>

<specifics>
## Specific Ideas

- Success metric is the difference between Elastica output and surrogate model output
- Per-step comparison only in Phase 3; multi-step trajectory validation deferred to Phase 4
- Existing training pipeline (train_surrogate.py) has auto-batch, early stopping, density weighting — extend with transformer support and W&B metrics
- Sweep runs sequentially (15 configs × ~200 epochs each) in tmux session

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `train_surrogate.py`: Training loop with single-step MSE, density weighting, cosine LR, W&B, early stopping, checkpointing, auto-batch; needs FlatStepDataset wiring + transformer branch + W&B metrics
- `model.py:SurrogateModel`: Base MLP (131→hidden→124), delta prediction
- `model.py:ResidualSurrogateModel`: Skip connections every 2 hidden layers; asserts uniform dims
- `model.py:ResidualBlock`: Two linear layers + LayerNorm + SiLU + skip connection
- `dataset.py:FlatStepDataset`: Primary dataset for Phase 02.2 flat format
- `sweep.py`: Sweep runner launching configs via subprocess; needs 15-config update
- `train_config.py:SurrogateTrainConfig`: Dataclass config; needs arch/transformer fields
- `state.py:StateNormalizer`: z-score normalization with save/load
- `find_max_batch_size()`: Binary search for max GPU batch size (already in train_surrogate.py)

### Established Patterns
- CLI entry: `python -m aprx_model_elastica.train_surrogate --epochs 200 --lr 3e-4 --wandb`
- Sweep runners launch via subprocess.run(); read metrics.json; write ranked summary JSON
- Delta prediction: next_state = current_state + model(normalized_input)
- Checkpoint: model.pt + normalizer.pt + config.json in save_dir

### Integration Points
- Reads from `data/surrogate_rl_step/` (Phase 02.2 output, flat format via FlatStepDataset)
- Writes model to `output/surrogate/`
- Canonical best checkpoint: `output/surrogate/best/` (Phase 4 input)
- W&B project: snake-hrl-surrogate
- Figures output: `figures/surrogate_training/`

</code_context>

<deferred>
## Deferred Ideas

- Multi-step trajectory validation (50-500 step rollouts) -- Phase 4
- History window models (HistorySurrogateModel K=2) -- needs multi-step trajectory data
- Progressive rollout length curriculum (4 -> 8 -> 16 steps) -- future optimization
- Component-weighted MSE (upweighting omega_z dims) -- if needed after sweep results
- BatchNorm as normalization variant -- if sweep shows overfitting patterns

</deferred>

---

*Phase: 03-train-surrogate-model-using-supervised-learning*
*Context gathered: 2026-03-10, updated: 2026-03-11*
