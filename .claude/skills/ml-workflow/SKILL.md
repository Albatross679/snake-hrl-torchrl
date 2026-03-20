---
name: ml-workflow
description: |
  Checklist for verifying ML training pipelines against user's established conventions.
  Covers: config system (hierarchical dataclass configs for neural, tree, and RL pipelines),
  experiment tracking (W&B via WandbLogger), training execution (Lightning Trainer, BatchSizeFinder, bf16),
  monitoring (/loop), hyperparameter strategy (W&B random sweeps, early-stopping-only, config combinations),
  RL/alignment algorithms (PPO, SAC, DQN, DPO, GRPO, CISPO), reward design, training stability,
  codebase structure (LightningModule, LightningDataModule), and documentation expectations.
  Use when: (1) Setting up experiment configurations for any ML framework,
  (2) Creating configs for SL Neural, SL Tree, RL, or PPO training pipelines,
  (3) Setting up or running ML training experiments,
  (4) Creating new config variants or experiment combinations,
  (5) Launching training runs or scheduling batches,
  (6) Monitoring active training,
  (7) Planning hyperparameter searches or ablation studies,
  (8) Scaffolding a new ML project or adding a new model/task,
  (9) Setting up RL alignment training (DPO, GRPO, CISPO) or classic RL (PPO, SAC, DQN),
  (10) Designing reward functions or diagnosing RL training instability.
  Works with: PyTorch Lightning, TorchRL, Hugging Face, XGBoost, LightGBM, CatBoost.
  Always apply these preferences unless the user explicitly overrides them.
---

# ML Training Pipeline Checklist

Structured checklist for building and verifying ML training pipelines. Every training script must satisfy each applicable item before it ships. Items marked **(conditional)** apply only when the stated condition is true.

**Framework**: PyTorch Lightning is the default training framework for all neural network training (SL and RL). Tree-based models (XGBoost, LightGBM, CatBoost) use their native APIs directly.

**Field references** (read on demand for detailed field tables):
- [references/config-system.md](references/config-system.md) — Config hierarchy, base fields, built-in infrastructure, output conventions, framework adaptation
- [references/sl_neural_fields.md](references/sl_neural_fields.md) — SL Neural base + Regression + Classification fields, metrics contract
- [references/sl_neural_tasks.md](references/sl_neural_tasks.md) — Task configs: LSTM, Transformer, CNN, Transformer Seq Clf, Transformer LM, T5 NL-to-SQL
- [references/sl_tree_fields.md](references/sl_tree_fields.md) — SL Tree + Regression + task configs (XGBoost, LightGBM, CatBoost)
- [references/rl_fields.md](references/rl_fields.md) — RL: Classic Control (PPO, SAC, DQN) + LLM Alignment (DPO, GRPO, CISPO), metrics, reward design, stability
- [references/codebase-structure.md](references/codebase-structure.md) — Codebase layout, module responsibilities, key patterns
- [references/known-issues.md](references/known-issues.md) — Common training pitfalls and their fixes

---

## Phase 1: Configuration

- [ ] Config is a `@dataclass` inheriting from the correct level in the hierarchy (Base → SL Neural / SL Tree / RL → task-specific). See [references/config-system.md](references/config-system.md) for the full hierarchy.
- [ ] Config includes all **built-in infrastructure** from Base: output directory, console logging, checkpointing, metrics log, experiment tracking. Each is individually disableable.
- [ ] Config is **serializable** — supports `to_dict()` / `from_dict()` round-trip via JSON.
- [ ] **CLI variant selection** — `--config` flag selects among config classes by name. Per-field CLI overrides apply on top. Priority: dataclass defaults < config class < CLI flags.
- [ ] Time-budget fields use **hours** (not seconds) as the unit (e.g., `max_wall_clock_hours`).
- [ ] `num_epochs` is set **high** (effectively unlimited) so early stopping is the binding constraint — not a hard epoch cap.
- [ ] **(conditional: multiple configs)** Config variants are separate `@dataclass` classes inheriting from a base, overriding only differing fields. Includes: baseline, single-dimension variants, and at least one aggressive multi-dimension variant.

## Phase 2: Codebase Structure (Lightning)

- [ ] Per-part modules follow `partN/` layout: `config.py`, `data.py`, `model.py`, `train.py`. See [references/codebase-structure.md](references/codebase-structure.md).
- [ ] Shared infrastructure lives in `src/` — config hierarchy, W&B integration (`src/wandb_utils.py`), system metrics. Never duplicated per-part.
- [ ] Root entry points are thin wrappers delegating to part-specific `main()`.
- [ ] **(conditional: neural training)** Model inherits `lightning.LightningModule`:
  - `__init__`: store hyperparams via `self.save_hyperparameters()`, build layers
  - `forward()`: pure forward pass (inference)
  - `training_step(batch, batch_idx)`: forward + loss, return loss. Log via `self.log()`
  - `validation_step(batch, batch_idx)`: forward + loss + eval metrics. Log via `self.log()`
  - `configure_optimizers()`: return optimizer and optional LR scheduler dict
- [ ] **(conditional: neural training)** Data pipeline inherits `lightning.LightningDataModule`:
  - `setup(stage)`: load/split datasets
  - `train_dataloader()`: return training DataLoader
  - `val_dataloader()`: return validation DataLoader
  - `test_dataloader()`: return test DataLoader (optional)
  - Uses **dynamic padding** in collate (pad per-batch to longest sequence, not fixed-length in dataset)
- [ ] **(conditional: neural training)** `model.py` additionally exposes: `save_model()`, `load_model_from_checkpoint()` for standalone checkpoint loading outside Lightning.

## Phase 3: W&B Experiment Tracking (Lightning)

- [ ] Every training run logs to W&B via `WandbLogger`:
  ```python
  from lightning.pytorch.loggers import WandbLogger
  logger = WandbLogger(project="my-project", name=cfg.name, log_model=False)
  trainer = Trainer(logger=logger)
  ```
- [ ] Hyperparameters logged automatically via `self.save_hyperparameters()` in `LightningModule.__init__()`.
- [ ] Per-step and per-epoch metrics logged via `self.log()` and `self.log_dict()` inside `training_step` / `validation_step`:
  ```python
  self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
  self.log_dict({"val_loss": val_loss, "val_rmse": rmse}, on_epoch=True)
  ```
- [ ] **Metric key namespacing** followed:
  - *(no prefix)*: per-epoch (`train_loss`, `val_loss`, eval metrics, `lr`)
  - `batch/`: per-step metrics (use `on_step=True`)
  - `timing/`: per-epoch timing breakdown (see Phase 3.5)
  - `tracking/`: per-epoch (`best_{metric}`, `epochs_since_improvement`)
  - `system/`: per-epoch (GPU/CPU/RAM — Lightning logs GPU metrics automatically when `log_every_n_steps` is set)
- [ ] **One-time params** logged via `logger.experiment.config.update()` or `self.log_dict()` at start: `total_params`, `trainable_params`, `num_train_samples`, `num_val_samples`, `gpu_name`.
- [ ] **Model artifact**: best model checkpoint uploaded to W&B as versioned artifact via `WandbLogger(log_model="all")` or manual `wandb.log_artifact()` — once per run at end of training.

## Phase 3.5: Training Loop Profiling

- [ ] **Per-section elapsed time** tracked. Use Lightning's built-in profiler or custom callbacks for timing:
  ```python
  from lightning.pytorch.profilers import SimpleProfiler
  trainer = Trainer(profiler=SimpleProfiler())  # or AdvancedProfiler() for per-function breakdown
  ```
- [ ] Sections to instrument (via custom callback `on_train_batch_start/end`, `on_validation_start/end`):
  - `timing/env_step_seconds` — environment stepping (RL: time inside `env.step()` or `collector.rollout()`)
  - `timing/inference_seconds` — policy forward pass
  - `timing/backward_seconds` — loss computation + backward pass + optimizer step
  - `timing/data_seconds` — data loading, replay buffer sampling
  - `timing/overhead_seconds` — everything else (logging, checkpointing)
- [ ] **Timing metrics logged to W&B** under `timing/` namespace via `self.log()` in callbacks.
- [ ] **(conditional: RL with parallel envs)** `timing/env_step_seconds` measures the full `ParallelEnv.step()` call including inter-process communication.
- [ ] **Timing fraction logged** — compute and log `timing/env_step_pct`, `timing/backward_pct` etc. as percentage of total iteration time.

## Phase 4: Mixed Precision (bf16)

- [ ] **(conditional: neural training)** bf16 enabled via Lightning Trainer:
  ```python
  trainer = Trainer(precision="bf16-mixed")
  ```
  **Not** fp16 — bf16 has same exponent range as fp32, no `GradScaler` needed. Lightning handles AMP context automatically.
- [ ] No manual `torch.amp.autocast()` calls in model code — Lightning wraps `training_step` and `validation_step` automatically.
- [ ] **No GradScaler** in the codebase (bf16 doesn't need it).
- [ ] **(conditional: pre-Ampere GPU)** `precision="32-true"` is set instead.

## Phase 4.5: Async Environment Stepping (RL)

- [ ] **(conditional: RL with num_envs > 1)** Use `ParallelEnv` (not `SerialEnv`) to pipeline CPU env stepping with GPU policy inference:
  ```python
  from torchrl.envs import ParallelEnv
  env = ParallelEnv(
      num_workers=config.num_envs,
      create_env_fn=lambda: make_env(config),
  )
  ```
- [ ] Env creation function must be **picklable** — use a top-level function or `CloudpickleWrapper`, not a lambda capturing local state.
- [ ] `env.close()` called during cleanup to terminate worker processes (prevents zombie processes).
- [ ] **(conditional: CPU-bound physics like DisMech)** `ParallelEnv` is especially critical — GPU sits idle during serial physics stepping.

## Phase 4.6: TorchRL Environment & Collection Safety (RL)

- [ ] **RewardSum transform** applied: `env.append_transform(RewardSum())`. Without it, TorchRL collectors never populate `episode_reward` and monitoring is blind.
- [ ] **All custom observation keys declared in specs** — `SyncDataCollector` silently drops keys not in `observation_spec`. Add `Unbounded` specs for diagnostic fields.
- [ ] **(conditional: vectorized envs)** Use `env.step_and_maybe_reset()` (not `env.step()` + manual `step_mdp()`) — `ParallelEnv.step()` does NOT auto-reset done environments.
- [ ] **(conditional: vectorized envs)** Env `self._device` must be `torch.device` object, not a string — TorchRL's `BatchedEnvBase._reset()` calls `.type` on it.
- [ ] **(conditional: num_envs >= 32)** Set thread-limiting env vars at script top, before imports:
  ```python
  os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
  os.environ.setdefault("MKL_NUM_THREADS", "1")
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  ```
- [ ] **Batch structure for v0.11+**: done/reward/episode_reward are under `batch["next"]`, not batch root.
- [ ] **Import compatibility**: Use try/except for `BoundedTensorSpec`→`Bounded`, `CompositeSpec`→`Composite`, `UnboundedContinuousTensorSpec`→`Unbounded`.
- [ ] **(conditional: PPO with bf16)** Never wrap loss module in bf16 autocast — log-prob and importance ratio math requires f32 precision. Only autocast network forward passes.
- [ ] **(conditional: PPO with target_kl)** Track `actual_updates` counter for metric averaging — KL early stopping means `actual_updates << num_epochs * num_batches`.
- [ ] **(conditional: PPO)** NaN guard: check `isfinite(loss)` before backward, `isfinite(grad_norm)` after clipping, per-batch KL early stopping.
- [ ] **(conditional: SAC)** Verify `clip_grad_norm_()` is actually called in `_update()` for both critic and actor — config field alone is not enough.
- [ ] **(conditional: reproducing paper results)** Compare every physics parameter in config against the paper's parameter tables before training. Rod radius (enters I as r^4) and Young's modulus are common off-by-orders errors.

## Phase 5: VRAM Management (Lightning BatchSizeFinder)

- [ ] **(conditional: neural training)** Auto batch size tuning via Lightning's `BatchSizeFinder` callback:
  ```python
  from lightning.pytorch.callbacks import BatchSizeFinder
  trainer = Trainer(callbacks=[BatchSizeFinder(mode="binsearch")])
  ```
- [ ] `LightningModule` and `LightningDataModule` must expose a `batch_size` attribute that `BatchSizeFinder` can modify. Typically set in config and passed through:
  ```python
  class MyDataModule(L.LightningDataModule):
      def __init__(self, batch_size=32, ...):
          self.batch_size = batch_size
      def train_dataloader(self):
          return DataLoader(self.train_ds, batch_size=self.batch_size, ...)
  ```
- [ ] **No custom `probe_auto_batch_size()` functions** — use `BatchSizeFinder` instead.
- [ ] `gradient_accumulation_steps` available to extend effective batch beyond VRAM ceiling:
  ```python
  trainer = Trainer(accumulate_grad_batches=cfg.gradient_accumulation_steps)
  ```
- [ ] **(conditional: sequential configs)** `cleanup_vram()` called between configs: delete model, trainer → `torch.cuda.empty_cache()` → `gc.collect()`.

## Phase 6: Training Loop Structure (Lightning Trainer)

- [ ] `main()` follows this sequence:
  1. Parse CLI args
  2. Load config + apply CLI overrides
  3. Set random seed via `lightning.seed_everything(cfg.seed)`
  4. Instantiate `LightningDataModule` (handles data loading)
  5. Instantiate `LightningModule` (handles model + optimizer + scheduler)
  6. Configure callbacks list (see below)
  7. Configure `WandbLogger`
  8. Instantiate `Trainer` with all settings
  9. `trainer.fit(model, datamodule=dm)` — handles entire training loop
  10. `trainer.test(model, datamodule=dm, ckpt_path="best")` — final eval on best checkpoint
  11. Cleanup + `wandb.finish()`
- [ ] **Callbacks** configured on `Trainer`:
  ```python
  from lightning.pytorch.callbacks import (
      EarlyStopping, ModelCheckpoint, BatchSizeFinder,
      LearningRateMonitor, RichProgressBar,
  )
  callbacks = [
      EarlyStopping(monitor="val_loss", patience=cfg.patience, mode="min"),
      ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True),
      BatchSizeFinder(mode="binsearch"),
      LearningRateMonitor(logging_interval="epoch"),
  ]
  trainer = Trainer(callbacks=callbacks, ...)
  ```
- [ ] **Early stopping** via `EarlyStopping` callback. Patience measured in **eval cycles**. Optional `min_delta` for minimum improvement threshold.
- [ ] Saves **both** best model (via `ModelCheckpoint(save_top_k=1)`) and last model (via `save_last=True`).
- [ ] **(conditional: multiple configs)** Configs run **sequentially in a single process**, each with its own Trainer + WandbLogger instance.

## Phase 7: Graceful Lifecycle

- [ ] Lightning handles **SIGTERM/SIGINT** automatically — saves checkpoint and exits cleanly.
- [ ] **STOP file**: custom callback checks for `STOP` file between epochs via `on_train_epoch_end`:
  ```python
  class StopFileCallback(L.Callback):
      def on_train_epoch_end(self, trainer, pl_module):
          if Path("STOP").exists():
              trainer.should_stop = True
  ```
- [ ] Both mechanisms handled by setting `trainer.should_stop = True`, which lets Lightning finish the current epoch and run cleanup.
- [ ] **(conditional: sequential training)** **Hung process watchdog**: polls W&B run status, kills process if alive N minutes after run marked FINISHED. Exit codes 137/143 from watchdog treated as success.

## Phase 8: Entry Point & Execution

- [ ] **GPU lock**: entry point wraps `main()` with `GpuLock()` in `if __name__ == "__main__"`:
  ```python
  if __name__ == "__main__":
      from src.utils.gpu_lock import GpuLock
      with GpuLock():
          main()
  ```
  Uses `flock` on `/tmp/gpu-task.lock`. Concurrent GPU tasks queue (not error).
- [ ] **Pre-flight check** before launch: `ps aux | grep -E "python.*train" | grep -v grep` — confirm GPU is free.
- [ ] **(conditional: run > 5 min)** Launch with `nohup` and unbuffered output:
  ```bash
  PYTHONUNBUFFERED=1 nohup <command> > output/<descriptive_log>.txt 2>&1 &
  ```
- [ ] Monitor with `tail -f output/<log>.txt` or `/loop 10m /babysit-training`.

## Phase 9: Output Directory

- [ ] Lightning's default output: `{output.base_dir}/{name}_{YYYYMMDD_HHMMSS}/`
  ```python
  trainer = Trainer(default_root_dir=run_dir)
  ```
- [ ] Contains: `config.json` (full snapshot saved manually), `console.log`, Lightning checkpoint files.
- [ ] `ModelCheckpoint` saves to `{run_dir}/checkpoints/`: `best.ckpt`, `last.ckpt`.
- [ ] Metrics logged to W&B (primary) and optionally to `metrics.jsonl` via custom callback.
- [ ] **(conditional: tree models)** Also includes `metrics.json` (final summary) and `plots/` (feature importance, pred vs actual, SHAP).

## Phase 10: Hyperparameter Search

**(conditional: search space > 3 variants)**

- [ ] Use **W&B random sweeps** (not grid, not Bayesian). Random gives broader cross-architecture coverage.
- [ ] Sweep script lives alongside its part: `partN/sweep.py`.
- [ ] `main_with_config(cfg)` extracted from `main()` so sweeps call the pipeline programmatically.
- [ ] `setup_run()` is **sweep-aware**: detects `wandb.run.sweep_id`, skips `wandb.init()`, updates config instead.
- [ ] Mutually exclusive structural choices (e.g., LoRA vs MLP vs vanilla) encoded as a single `architecture` preset parameter decoded into config fields.
- [ ] **Per-trial stopping by early stopping only** — no per-trial epoch cap or time budget.
- [ ] `--max-hours` bounds **total sweep** wall clock (not per trial).
- [ ] Run sweeps in **tmux** (not nohup): `tmux new -s sweep && python partN/sweep.py --max-hours 12`.
- [ ] OOM recovery: `sweep_train()` catches exceptions explicitly to unwind stack before `cleanup_vram()`. Double `gc.collect()` pass for reference cycles.

## Phase 11: Monitoring

- [ ] After launching a background training run, start babysit monitoring:
  ```
  /loop 10m /babysit-training
  ```
  Covers: process health, metric trending, GPU/system checks, checkpoint integrity, hung process detection, auto-restart from checkpoint, issue documentation.

## Phase 12: Documentation

- [ ] Each code change that modifies behavior gets a log entry in `logs/` immediately (not batched).
- [ ] Training runs get an experiment entry in `experiments/`.
- [ ] Bugs/errors get an issue entry in `issues/` before or alongside the fix.
- [ ] All documentation files include correct frontmatter (`type`, `status`, and type-specific properties).
