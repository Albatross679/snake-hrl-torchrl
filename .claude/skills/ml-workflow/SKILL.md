---
name: ml-workflow
description: |
  Checklist for verifying ML training pipelines against user's established conventions.
  Covers: config system (hierarchical dataclass configs for neural, tree, and RL pipelines),
  experiment tracking (W&B), training execution (sequential auto-batch, VRAM management, bf16),
  monitoring (/loop), hyperparameter strategy (W&B random sweeps, early-stopping-only, config combinations),
  RL/alignment algorithms (PPO, SAC, DQN, DPO, GRPO, CISPO), reward design, training stability,
  codebase structure, and documentation expectations.
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
  Works with: PyTorch, Hugging Face, TensorFlow/Keras, XGBoost, LightGBM, CatBoost, JAX.
  Always apply these preferences unless the user explicitly overrides them.
---

# ML Training Pipeline Checklist

Structured checklist for building and verifying ML training pipelines. Every training script must satisfy each applicable item before it ships. Items marked **(conditional)** apply only when the stated condition is true.

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

## Phase 2: Codebase Structure

- [ ] Per-part modules follow `partN/` layout: `config.py`, `data.py`, `model.py`, `train.py`. See [references/codebase-structure.md](references/codebase-structure.md).
- [ ] Shared infrastructure lives in `src/` — config hierarchy, W&B integration (`src/wandb_utils.py`), system metrics. Never duplicated per-part.
- [ ] Root entry points are thin wrappers delegating to part-specific `main()`.
- [ ] `data.py` uses **dynamic padding** in collate (pad per-batch to longest sequence, not fixed-length in dataset).
- [ ] `model.py` exposes: `initialize_model()`, `save_model()`, `load_model_from_checkpoint()`, `save_training_state()`, `load_training_state()`.

## Phase 3: W&B Experiment Tracking

- [ ] Every training run logs to W&B. No direct `wandb.*` calls in training code — all go through `src/wandb_utils.py` helpers.
- [ ] Training loop calls W&B integration in order: `setup_run(cfg)` → per-epoch `log_epoch_metrics()` → `log_extra_params()` after model init → `end_run()` at finish.
- [ ] Custom metric axes defined via `wandb.define_metric` so batch-level and epoch-level metrics have independent x-axes.
- [ ] **Metric key namespacing** followed:
  - *(no prefix)*: per-epoch (`train_loss`, `dev_loss`, eval metrics, `gradient_norm`, `lr`)
  - `batch/`: per-batch (`loss`, `gradient_norm`, `lr`, step = global batch counter)
  - `timing/`: per-epoch (`epoch_seconds`, `wall_clock_seconds`, `train_epoch_seconds`, `train_tokens_per_sec`) plus per-section breakdown (see Phase 3.5)
  - `tracking/`: per-epoch (`best_{metric}`, `epochs_since_improvement`)
  - `system/`: per-epoch (GPU/CPU/RAM when `log_system_metrics=True`)
- [ ] **One-time params** logged: `total_params`, `trainable_params`, `num_train_samples`, `num_dev_samples`, `gpu_name`.
- [ ] `wandb.finish()` called before starting a new run (critical for auto-batch sequential mode).
- [ ] Resume supported via `wandb.init(resume="allow", id=run_id)`.
- [ ] **Model artifact**: best model checkpoint uploaded to W&B as versioned artifact via `log_model_artifact()` — once per run at end of training (not per improvement).

## Phase 3.5: Training Loop Profiling

- [ ] **Per-section elapsed time** tracked in the training loop. Each iteration measures wall-clock time for each major section using `time.monotonic()` pairs. Sections to instrument:
  - `timing/env_step_seconds` — environment stepping (physics simulation, e.g., DisMech). For RL: time inside `env.step()` or `collector.rollout()`.
  - `timing/inference_seconds` — policy forward pass (actor inference for action selection). For off-policy (SAC): time in `actor(obs)`. For on-policy (PPO): included in rollout collection.
  - `timing/backward_seconds` — loss computation + backward pass + optimizer step. For SAC: critic + actor + alpha updates combined. For PPO: all minibatch epochs combined.
  - `timing/data_seconds` — data loading, replay buffer sampling, or batch preparation.
  - `timing/overhead_seconds` — everything else (logging, checkpointing, metric computation).
- [ ] **Timing metrics logged to W&B** under `timing/` namespace, per batch or per epoch depending on granularity.
- [ ] **(conditional: RL with parallel envs)** `timing/env_step_seconds` measures the full `ParallelEnv.step()` call including inter-process communication — this is the key number for diagnosing CPU-bound bottlenecks.
- [ ] **Timing fraction logged** — compute and log `timing/env_step_pct`, `timing/backward_pct` etc. as percentage of total iteration time. Allows quick identification of bottleneck on W&B dashboard.
  ```python
  total = env_dt + inference_dt + backward_dt + data_dt + overhead_dt
  metrics["timing/env_step_pct"] = env_dt / total * 100
  metrics["timing/backward_pct"] = backward_dt / total * 100
  # etc.
  ```

## Phase 4: Mixed Precision (bf16)

- [ ] **(conditional: neural training)** bf16 autocast enabled by default (`use_amp=True`). **Not** fp16 — bf16 has same exponent range as fp32, no `GradScaler` needed.
- [ ] AMP context manager pattern used:
  ```python
  from contextlib import nullcontext
  def _amp_context(use_amp, device):
      if use_amp and 'cuda' in str(device):
          return torch.amp.autocast('cuda', dtype=torch.bfloat16)
      return nullcontext()
  ```
- [ ] Forward + loss wrapped in AMP context. Backward is **outside** the context.
- [ ] Inference wrapped in `torch.inference_mode()` + AMP context.
- [ ] **No GradScaler** in the codebase (bf16 doesn't need it).
- [ ] **(conditional: pre-Ampere GPU)** `use_amp=False` is set.

## Phase 4.5: Async Environment Stepping (RL)

- [ ] **(conditional: RL with num_envs > 1)** Use `ParallelEnv` (not `SerialEnv`) to pipeline CPU env stepping with GPU policy inference. `SerialEnv` blocks the GPU while each env steps sequentially on CPU.
  ```python
  from torchrl.envs import ParallelEnv
  env = ParallelEnv(
      num_workers=config.num_envs,
      create_env_fn=lambda: make_env(config),
  )
  ```
- [ ] Env creation function must be **picklable** — use a top-level function or `CloudpickleWrapper`, not a lambda capturing local state.
- [ ] `env.close()` called during cleanup to terminate worker processes (prevents zombie processes).
- [ ] **(conditional: CPU-bound physics like DisMech)** `ParallelEnv` is especially critical — GPU sits idle during serial physics stepping. With parallel stepping, GPU can process the previous batch while envs compute the next.

## Phase 5: VRAM Management

- [ ] **(conditional: neural training)** Auto batch size tuning enabled (`auto_batch_size=True`).
- [ ] Probing strategy uses **worst-case measurement**: probe batches built from longest sequences (sorted by length), peak allocation measured via `torch.cuda.max_memory_allocated()` with `reset_peak_memory_stats()`.
- [ ] Two-phase search: coarse (powers of 2) then fine (intermediate between last pass/first fail).
- [ ] Target is **85% of total VRAM** (15% margin for fragmentation).
- [ ] `gradient_accumulation_steps` available to extend effective batch beyond VRAM ceiling.
- [ ] **(conditional: sequential configs)** `cleanup_vram()` called between configs: delete model, optimizer, scheduler → `torch.cuda.empty_cache()` → `gc.collect()`.

## Phase 6: Training Loop Structure

- [ ] `main()` follows this sequence:
  1. Parse CLI args
  2. Load config + apply CLI overrides
  3. Set random seeds
  4. Load data (train/dev/test loaders)
  5. Initialize model + optional wrappers
  6. Auto batch size tuning (if enabled)
  7. Optimizer + scheduler setup
  8. Resume from checkpoint (if configured)
  9. Setup W&B + output directory
  10. Training loop
  11. Reload best checkpoint → final dev eval (synchronous) → test inference
  12. Cleanup + end W&B run
- [ ] **Early stopping** by patience measured in **eval cycles** (not raw epochs). Effective patience = `patience_epochs * eval_every_n_epochs`. Optional `patience_tolerance` for minimum improvement threshold.
- [ ] Saves **both** best model (by primary eval metric) and last model.
- [ ] **(conditional: CPU-bound eval)** Two-phase async evaluation: GPU inference in main thread, CPU-bound metric computation in `ThreadPoolExecutor`. Futures drained before next eval or exit.
- [ ] **(conditional: multiple configs)** Configs run **sequentially in a single process**, each with its own W&B run.

## Phase 7: Graceful Lifecycle

- [ ] **STOP file**: training loop checks for `STOP` file between epochs; current epoch finishes before exit.
- [ ] **SIGTERM handler**: signal handler sets a flag; same graceful drain as STOP file.
- [ ] Both mechanisms drain pending async work, save checkpoints, and finish W&B run before exiting.
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

- [ ] Run directory: `{output.base_dir}/{name}_{YYYYMMDD_HHMMSS}/`
- [ ] Contains: `config.json` (full snapshot), `console.log`, `metrics.jsonl` (one JSON object per epoch).
- [ ] Checkpoints subdirectory: `model_best.*`, `model_last.*`, `training_state.*` (for resume).
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
