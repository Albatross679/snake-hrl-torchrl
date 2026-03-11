# Phase 3: Train Surrogate Model Using Supervised Learning - Context

**Gathered:** 2026-03-10
**Updated:** 2026-03-10 (expanded to include architecture experiments, formerly Phase 3.1)
**Status:** In progress (plans 01–04 complete, plan 05 pending sweep completion)

<domain>
## Phase Boundary

Train a neural surrogate MLP on the Phase 02.2 dataset to predict one-step Cosserat rod dynamics (state + action → next_state delta). This phase covers the full surrogate model development pipeline:

1. **Hyperparameter sweep** (Plans 01–02): 5-run LR × model-size sweep, analysis, diagnostic plots
2. **Architecture experiments** (Plans 03–05): Add residual connections and history-window variants, tune rollout loss weight/horizon, run arch sweep (Experiments A+B), human-reviewed architecture selection, copy winner to `output/surrogate/best/`

Does NOT cover: multi-step trajectory validation (Phase 4), RL training with surrogate, or architecture overhauls (1D-CNN, GNN, Transformer).

**Baseline:** `sweep_lr1e3_h512x3` on Phase 02.1 data, val_loss=0.2161, R²=0.784, omega_z R²=0.23 (reference only — Phase 03 re-runs on Phase 02.2 data)

</domain>

<decisions>
## Implementation Decisions

### Hyperparameter sweep (Plans 01–02)
- 5-run sweep: all 3 LRs (1e-4, 3e-4, 1e-3) × all 3 model sizes (256x3, 512x3, 512x4)
- W&B logging enabled for all sweep runs (project: snake-hrl-surrogate)
- Best model selected by lowest single-step validation MSE
- Training data: data/surrogate_rl_step/ via FlatStepDataset (Phase 02.2 flat format)
- Plan 01 includes Task 0: wire FlatStepDataset into train_surrogate.py (serpenoid_time alias, model-based rollout loss)

### Architecture experiments (Plans 03–05)
- Three experiment families: A (rollout loss tuning), B (residual MLP), C (history window K=2, only if A+B insufficient)
- Experiment A configs: rw ∈ {0.0, 0.1(baseline), 0.3, 0.5}, steps ∈ {8, 16}
- Experiment B: ResidualSurrogateModel (1 residual block + extra layer for 512x3)
- Experiment C: HistorySurrogateModel K=2 — only if omega_z R² remains < 0.4 after A+B
- All arch experiments fixed at: lr=1e-3, hidden_dims=512x3 (best from Plan 01)
- `ResidualSurrogateModel` asserts uniform hidden_dims — prevents skip-connection shape mismatches
- `HistoryDataset` extends `TrajectoryDataset(rollout_length=history_k+1)` — reuses window builder

### Success criteria
- Primary metric: per-step error in physical units (position mm, velocity mm/s, angle rad, angular velocity rad/s)
- Per-component breakdown: pos_x, pos_y, vel_x, vel_y, yaw, omega_z
- Architecture selection gate (Plan 05 Task 2): human reviews results before promoting to output/surrogate/best/
- Phase 4 consumes output/surrogate/best/ (model.pt, normalizer.pt, config.json, selection.json)

### Training curriculum
- Primary training objective: single-step MSE on FlatStepDataset transitions from data/surrogate_rl_step/
- Rollout loss uses model-based chaining (compute_rollout_loss() feeds surrogate's own prediction as next input); data-window TrajectoryDataset is NOT used since Phase 02.2 has steps_per_run=1 (independent transitions, no multi-step trajectories)
- Density-weighted sampling enabled; input noise std=0.001 fixed
- 200 epochs max, early stopping patience=30

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
- `train_surrogate.py`: Complete training loop with single-step + multi-step loss, density weighting, cosine LR, W&B, early stopping, checkpointing; CLI args include --rollout-weight, --rollout-steps, --use-residual, --history-k
- `model.py:SurrogateModel`: Base MLP (131→512x3→124), delta prediction, predict_next_state()
- `model.py:ResidualSurrogateModel`: Skip connections every 2 hidden layers; asserts uniform dims
- `model.py:HistorySurrogateModel`: Extended input (131+K×129), K-step history concatenated
- `model.py:ResidualBlock`: Two linear layers + LayerNorm + SiLU + skip connection
- `dataset.py:SurrogateDataset`: Legacy single-step dataset for Phase 02.1 checkpoint format (NOT used for Phase 03 training)
- `dataset.py:FlatStepDataset`: Primary training dataset for Phase 02.2 flat format (states/next_states/t_start/forces); returns `serpenoid_time` as alias for `t_start` for backward compat with training loop
- `dataset.py:TrajectoryDataset`: Trajectory windows for rollout loss training (Phase 02.1 format; NOT used for Phase 03 — Phase 02.2 has no multi-step trajectories)
- `dataset.py:HistoryDataset`: Extends TrajectoryDataset; returns K prior steps + current + delta target (Phase 02.1 format; Phase 03 defers history-window training to data with multi-step trajectories)
- `sweep.py`: 5-run LR×size sweep runner (Plan 01 artifact)
- `arch_sweep.py`: Architecture experiment sweep runner; ARCH_SWEEP_CONFIGS + baseline injection + --dry-run
- `train_config.py:SurrogateTrainConfig`: Dataclass config; rollout_steps, rollout_loss_weight, use_residual, history_k
- `state.py:StateNormalizer`: z-score normalization with save/load
- `validate.py:_save_plots()`: Matplotlib pattern (Agg backend, dpi=150, bbox_inches="tight")

### Established Patterns
- CLI entry: `python -m aprx_model_elastica.train_surrogate --epochs 200 --lr 3e-4 --wandb`
- Sweep runners launch via subprocess.run(); read metrics.json; write ranked summary JSON
- Delta prediction: next_state = current_state + model(normalized_input)
- Phase encoding: sin/cos of omega*t via `action_to_omega_batch` + `encode_phase_batch`
- Checkpoint: model.pt + normalizer.pt + config.json in save_dir
- Best checkpoint at output/surrogate/best/ (selection.json documents provenance)

### Integration Points
- Reads from `data/surrogate_rl_step/` (Phase 02.2 output, flat format via FlatStepDataset)
- Writes model to `output/surrogate/` and `output/surrogate/arch_sweep/`
- Canonical best checkpoint: `output/surrogate/best/` (Phase 4 input)
- W&B project: snake-hrl-surrogate (per-epoch train/val loss, per-component losses, LR)
- Figures output: `figures/surrogate_training/`
- Tests: `tests/test_surrogate_arch.py` (ARCH-01 through ARCH-04, all GREEN)

</code_context>

<deferred>
## Deferred Ideas

- Multi-step trajectory validation (50-500 step rollouts) -- Phase 4
- Architecture alternatives (1D-CNN, GNN, Transformer) -- only if MLP fails after expanded sweep
- History window K > 2 -- K=2 is recommended max; higher K triples inference state and dataset complexity
- Progressive rollout length curriculum (4 -> 8 -> 16 steps) -- future optimization
- Noise level sweep (state_noise_std as hyperparameter) -- if needed after initial sweep
- Component-weighted MSE (upweighting omega_z dims) -- potential Experiment A6 if A+B insufficient
- Comparing density-weighted vs unweighted sampling -- future ablation

</deferred>

---

*Phase: 03-train-surrogate-model-using-supervised-learning*
*Context gathered: 2026-03-10*
