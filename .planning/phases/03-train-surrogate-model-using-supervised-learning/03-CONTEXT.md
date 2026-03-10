# Phase 3: Train Surrogate Model Using Supervised Learning - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Train a neural surrogate MLP on the Phase 1 dataset (~814K transitions) to predict one-step Cosserat rod dynamics (state + action -> next_state delta). Run a small hyperparameter sweep, select best model by validation loss, and generate basic comparison plots. Does NOT cover: multi-step trajectory validation (Phase 4), RL training with surrogate, or architecture changes beyond MLP.

</domain>

<decisions>
## Implementation Decisions

### Hyperparameter strategy
- Small sweep: 3-5 runs exploring learning rate x model size
- Learning rate: {1e-4, 3e-4, 1e-3}
- Hidden dims: {[256,256,256], [512,512,512], [512,512,512,512]}
- All other hyperparameters kept at defaults (batch=4096, weight_decay=1e-5, etc.)
- W&B logging enabled for all sweep runs (project: snake-hrl-surrogate)
- Best model selected by lowest single-step validation MSE

### Success criteria
- Primary metric: per-step error comparing Elastica ground truth vs surrogate prediction
- Measure in physical units (position mm, velocity mm/s, angle rad, angular velocity rad/s)
- Per-component breakdown: pos_x, pos_y, vel_x, vel_y, yaw, omega_z
- Generate basic plots: per-component error histograms, predicted-vs-actual overlays for key components (CoM position, heading, velocity)
- Plots saved to figures/surrogate_training/

### Fallback if model doesn't converge
- First: expand sweep with wider LR range, bigger models, more regularization (~10 runs)
- If still insufficient after expanded sweep: flag for architecture change (1D-CNN or LSTM per architecture comparison doc)

### Training curriculum
- Keep existing two-phase curriculum: single-step MSE (epochs 1-20) then add 8-step rollout loss (epochs 20+, weight=0.1)
- Density-weighted sampling enabled (inverse-density from 4 summary features)
- Input noise injection: state_noise_std=0.001 (fixed, not swept)
- Full 200 epochs with early stopping (patience=30) for each sweep run

### Claude's Discretion
- Exact sweep grid (whether to do full 3x3=9 or a subset of 3-5 configs)
- Which specific state components to plot in predicted-vs-actual overlays
- Per-component error threshold interpretation (what constitutes "good enough")
- Cosine LR schedule warmup duration (default 5 epochs)
- Number of dataloader workers

</decisions>

<specifics>
## Specific Ideas

- Success metric is the difference between Elastica output and surrogate model output (user's words)
- Per-step comparison only in Phase 3; multi-step trajectory validation deferred to Phase 4
- The existing training pipeline (`train_surrogate.py`) is already complete and should be used as-is, with CLI args for sweep variation

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `train_surrogate.py`: Complete training loop with single-step + multi-step loss, density weighting, cosine LR, W&B, early stopping, checkpointing
- `model.py:SurrogateModel`: MLP with configurable hidden_dims, activation, LayerNorm, dropout, delta prediction
- `dataset.py:SurrogateDataset`: Loads all batch .pt files, episode-level train/val split, density weight computation
- `dataset.py:TrajectoryDataset`: Trajectory windows for multi-step rollout loss
- `train_config.py:SurrogateTrainConfig`: Dataclass config with all hyperparameters, CLI override support
- `state.py:StateNormalizer`: z-score normalization with save/load, per-feature statistics
- `validate.py:_save_plots()`: Matplotlib pattern (Agg backend, dpi=150, bbox_inches="tight")

### Established Patterns
- CLI entry: `python -m aprx_model_elastica.train_surrogate --epochs 200 --lr 3e-4 --wandb`
- Config override via CLI args (argparse) on top of dataclass defaults
- Delta prediction: next_state = current_state + model(normalized_input)
- Phase encoding: sin/cos of omega*t (not raw time) via `action_to_omega_batch` + `encode_phase_batch`
- Checkpoint: model.pt + normalizer.pt + config.json in save_dir

### Integration Points
- Reads from `data/surrogate/` (Phase 1 output, 27 .pt batch files)
- Writes model to `output/surrogate/` (model.pt, normalizer.pt, config.json)
- W&B project: snake-hrl-surrogate (per-epoch train/val loss, per-component losses, LR)
- Figures output: `figures/surrogate_training/` (new directory for Phase 3)
- Phase 4 consumes `output/surrogate/model.pt` + `output/surrogate/normalizer.pt`

</code_context>

<deferred>
## Deferred Ideas

- Multi-step trajectory validation (50-500 step rollouts) -- Phase 4
- Architecture alternatives (1D-CNN, GNN, Transformer) -- only if MLP fails after expanded sweep
- Progressive rollout length curriculum (4 -> 8 -> 16 steps) -- future optimization
- Noise level sweep (state_noise_std as hyperparameter) -- if needed after initial sweep
- Comparing density-weighted vs unweighted sampling -- future ablation

</deferred>

---

*Phase: 03-train-surrogate-model-using-supervised-learning*
*Context gathered: 2026-03-10*
