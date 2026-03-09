---
name: ml-config-system
description: |
  A framework-agnostic hierarchical configuration specification for ML experiments.
  Defines config levels, built-in infrastructure, output conventions, and metrics contracts.
  Use when: (1) Setting up experiment configurations for any ML framework,
  (2) Creating configs for SL Neural, SL Tree, RL, or PPO training pipelines,
  (3) Extending the hierarchy for new tasks or algorithms.
  Works with: PyTorch, Hugging Face, TensorFlow/Keras, XGBoost, LightGBM, CatBoost, JAX, or any other framework.
---

# ML Configuration System

A framework-agnostic specification for ML experiment configuration. Describes **what** a config should contain — not how to implement it. Generate the appropriate code for whatever framework the user is working with (dataclasses, HF TrainingArguments, Keras callbacks, plain dicts, etc.).

## Design Principles

1. **Hierarchical** — configs inherit from a base level and add fields for their domain
2. **Batteries included** — every config gets output directory, console logging, checkpointing, metrics log, and MLflow tracking by default (all individually disableable)
3. **Serializable** — every config can be saved to JSON and loaded back
4. **Framework-agnostic** — the spec defines semantics; the implementation adapts to the target framework

## User Preferences

- **Human-friendly time units** — use hours (not seconds) for time-budget fields (e.g. `max_wall_clock_hours`). Convert to seconds internally where needed.
- **Config variant selection via CLI** — provide a `--config` flag that selects among Python config modules by name (e.g. `--config config_1` dynamically imports `part1.config_1`). Per-field CLI overrides apply on top. Priority: dataclass defaults < config module < CLI flags. Avoid JSON config files for variant selection.

## Config Hierarchy

Each level adds fields on top of its parent. All levels include the built-in infrastructure from Base.

```
Base                                    (core fields + all infrastructure)
├── SL Neural                           (epoch-based, gradient-based training)
│   ├── SL Neural Regression            (regression loss, eval metrics, figures)
│   │   ├── LSTM Regression             (hidden_size, num_layers, static branch, fusion head)
│   │   ├── Transformer Regression      (d_model, n_heads, n_layers, positional encoding)
│   │   └── CNN Regression              (conv_channels, kernel_sizes, batch norm, pooling)
│   └── SL Neural Classification        (classification loss, num_classes, eval metrics)
│       ├── Transformer Sequence Clf    (d_model, d_internal, attention type, causal mask, per-position output)
│       ├── Transformer Language Model  (multi-head, context_size, perplexity tracking)
│       └── T5 NL-to-SQL               (seq2seq fine-tune/scratch, record F1/EM, beam search)
├── SL Tree                             (boosting-round-based, tree ensemble training)
│   ├── SL Tree Regression              (regression eval, feature importance, SHAP)
│   │   ├── XGBoost Regression          (hist tree method, standard boosting)
│   │   ├── XGBoost Two-Stage           (classifier + regressor for sparse targets)
│   │   ├── LightGBM Regression         (leaf-wise growth, num_leaves)
│   │   └── CatBoost Regression         (ordered boosting, bagging_temperature)
│   └── SL Tree Classification          (extend as needed)
├── Reinforcement Learning (RL)         (timestep-based training)
│   ├── PPO                             (proximal policy optimization)
│   └── (SAC, DQN, etc.)                (extend as needed)
└── (your task inherits from any of these)
```

**Field references** (read on demand):
- `sl_neural_fields.md` — SL Neural + SL Neural Regression + SL Neural Classification fields, metrics contracts, task configs (LSTM, Transformer, CNN regression; Transformer Sequence Clf, Transformer LM classification)
- `sl_tree_fields.md` — SL Tree + SL Tree Regression fields, metrics contract, framework param mapping, task configs (XGBoost, LightGBM, CatBoost, Two-Stage XGBoost)
- `rl_fields.md` — RL + PPO fields

---

## Level 0: Base

Every ML experiment config starts here. All other levels inherit these fields **and** all built-in infrastructure below.

**Core fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| name | string | `"experiment"` | Experiment name; used in run directory naming |
| seed | integer | `42` | Random seed for reproducibility |
| device | string | `"auto"` | Compute device: `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`. For non-PyTorch frameworks, adapt to equivalent |

### Built-in: Output Directory

Always present. Creates a timestamped run directory. Grouped under `output`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| output.base_dir | string | `"output"` | Parent directory for all experiment runs |
| output.save_config | boolean | `true` | Save config snapshot (JSON) to run directory |
| output.timestamp_format | string | `"%Y%m%d_%H%M%S"` | Timestamp format for directory naming |
| output.subdirs | map | `{"checkpoints": "checkpoints", "plots": "plots"}` | Subdirectories to create |

Run directory: `{output.base_dir}/{name}_{timestamp}/`

### Built-in: Console Logging

Always present. Captures stdout/stderr to a log file. Grouped under `console`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| console.enabled | boolean | `true` | Enable console capture |
| console.filename | string | `"console.log"` | Log file name |
| console.separate_streams | boolean | `false` | Split stdout/stderr into separate files |
| console.stdout_filename | string | `"stdout.log"` | Stdout file (when separate_streams is true) |
| console.stderr_filename | string | `"stderr.log"` | Stderr file (when separate_streams is true) |
| console.tee_to_console | boolean | `true` | Also print to terminal |
| console.line_timestamps | boolean | `false` | Prefix each line with a timestamp |
| console.timestamp_format | string | `"%H:%M:%S"` | Timestamp format for line prefixes |
| console.flush_frequency | integer | `1` | Flush to disk every N writes |

### Built-in: Checkpointing

Always present. Controls model saving. Grouped under `checkpointing`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| checkpointing.enabled | boolean | `true` | Enable model checkpointing |
| checkpointing.save_best | boolean | `true` | Save best model by tracked metric |
| checkpointing.save_last | boolean | `true` | Save final model |
| checkpointing.save_every_n | integer | `5` | Save every N epochs/rounds; 0 disables |
| checkpointing.metric | string | `"loss"` | Metric to track for best model |
| checkpointing.mode | string | `"min"` | `"min"` or `"max"` |
| checkpointing.best_filename | string | `"model_best"` | Best model filename (add framework extension: `.pt`, `.h5`, `.json`, etc.) |
| checkpointing.last_filename | string | `"model_last"` | Last model filename |
| checkpointing.periodic_filename_format | string | `"model_epoch_{n}"` | Periodic checkpoint filename template |

### Built-in: Metrics Log

Always present. Appends one JSON object per epoch to a JSON-lines file. Human-readable, zero extra dependencies. Grouped under `metricslog`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| metricslog.enabled | boolean | `true` | Enable metrics logging |
| metricslog.filename | string | `"metrics.jsonl"` | Log file name (relative to run dir) |
| metricslog.flush_every_epoch | boolean | `true` | Flush after every write |

### Built-in: MLflow Tracking

Always present. Logs params, metrics, and artifacts to an MLflow SQLite backend for centralized experiment comparison via the MLflow UI. Grouped under `mlflow`. Implementation lives in `src/mlflow_utils.py`.

**IMPORTANT:** Always use the SQLite backend (`sqlite:///mlflow.db`), never the file-based `mlruns/` store. SQLite supports full UI features and keeps all data in a single file.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| mlflow.enabled | boolean | `true` | Enable MLflow tracking |
| mlflow.tracking_uri | string | `"sqlite:///mlflow.db"` | Tracking URI. **Must** use `sqlite:///` prefix. Never use file-based `mlruns/` |
| mlflow.experiment_name | string | derived from `name` | MLflow experiment name (groups runs). Defaults to the config `name` field if not set |
| mlflow.run_name | string | derived from `name` | MLflow run name (human-readable label within experiment). Defaults to the config `name` field if not set |
| mlflow.log_artifacts | boolean | `true` | Log checkpoint files as MLflow artifacts at end of training |

**Integration contract — training loops must:**

1. **Start of training:** call `setup_mlflow(experiment_name, run_name, params_dict)` to set tracking URI, create/get experiment, start run, and log all config params. Pass `run_id` when resuming a previous run.
2. **Per epoch:** call `log_epoch_metrics(metrics_dict, step=epoch)` for epoch-level metrics (loss, eval metrics, lr, timing, tracking)
3. **Per batch (optional):** call `log_epoch_metrics(metrics_dict, step=global_step)` with `batch/`-prefixed keys for fine-grained monitoring
4. **One-time metadata:** call `log_extra_params(params_dict)` after model init to log static values (model size, dataset size, GPU name)
5. **End of training:** call `end_mlflow_run()` to close the active run
6. **Optionally:** call `log_model_checkpoint(checkpoint_dir)` to log model files as artifacts

**Helper functions** (in `src/mlflow_utils.py`):

```python
setup_mlflow(experiment_name, run_name, params_dict, tracking_uri="sqlite:///mlflow.db", run_id=None)
log_epoch_metrics(metrics_dict, step)
log_extra_params(params_dict)
log_model_checkpoint(checkpoint_dir, artifact_subdir="checkpoints")
end_mlflow_run()
```

**Metric key namespacing convention:**

Metric keys use `/`-separated prefixes to organize into logical groups in the MLflow UI:

| Prefix | Granularity | Contents |
|--------|-------------|----------|
| *(no prefix)* | per epoch | Core metrics: `train_loss`, `dev_loss`, eval metrics, `gradient_norm`, `lr` |
| `timing/` | per epoch | `epoch_seconds`, `wall_clock_seconds`, `train_epoch_seconds`, `train_tokens_per_sec` |
| `tracking/` | per epoch | `best_{metric}`, `epochs_since_improvement` |
| `system/` | per epoch | GPU/CPU/RAM metrics from `collect_system_metrics()` |
| `batch/` | per batch | `loss`, `gradient_norm`, `lr` (step = global batch counter) |

**One-time params** (logged via `log_extra_params`, not `log_epoch_metrics`):

| Param | Purpose |
|-------|---------|
| `total_params` | Total model parameters |
| `trainable_params` | Trainable parameters |
| `num_train_samples` | Training set size |
| `num_dev_samples` | Dev/validation set size |
| `gpu_name` | GPU device name (when CUDA available) |
| `avg_epoch_seconds` | Average epoch duration (logged at end of training) |

**Dual logging:** Metrics are always written to both `metrics.jsonl` (local, human-readable) and MLflow (SQLite, for UI comparison). The two systems are independent — disabling one does not affect the other.

---

## Level Summaries

### SL Neural (Level 1a)

Epoch-based, gradient-based supervised training (LSTM, Transformer, CNN, MLP). Adds: `num_epochs`, `batch_size`, `learning_rate`, `weight_decay`, optimizer/scheduler, regularization (grad clip, dropout), early stopping by patience, metric logging flags.

### SL Neural Regression (Level 2a)

Inherits SL Neural. Adds: `loss_fn`, `eval_metrics`, `figures`, `hparam_metrics`.

**Task configs (Level 3):** LSTM Regression, Transformer Regression, CNN Regression — each adds architecture-specific fields (hidden_size/num_layers, d_model/n_heads/n_layers, conv_channels/kernel_sizes) and data config (seq_length, stride, normalization).

### SL Neural Classification (Level 2b)

Inherits SL Neural. Adds: `loss_fn`, `eval_metrics`. Default eval_metrics: `["record_f1", "record_em", "sql_em", "error_rate"]` (task-specific — override for your use case).

**Task configs (Level 3):** Transformer Sequence Classification (single-head attention, per-position classification with 3 attention variants), Transformer Language Model (multi-head attention, next-token prediction over vocabulary, perplexity tracking), T5 NL-to-SQL (seq2seq fine-tune/scratch, record F1/EM, beam search).

→ Full spec: `sl_neural_fields.md`

### SL Tree (Level 1b)

Boosting-round-based tree ensembles (XGBoost, LightGBM, CatBoost). Adds: `n_estimators`, `learning_rate` (shrinkage), `max_depth`, sampling/regularization, early stopping by rounds, feature importance, SHAP config.

### SL Tree Regression (Level 2b)

Inherits SL Tree. Adds: `objective`, `eval_metric` (framework-native), `eval_metrics` (agnostic), `figures`.

**Task configs (Level 3):** XGBoost Regression, XGBoost Two-Stage (sparse targets), LightGBM Regression, CatBoost Regression — each adds framework-specific fields and data config (lag_hours, rolling_windows, interactions).

→ Full spec: `sl_tree_fields.md`

### RL (Level 1c)

Timestep-based reinforcement learning. Adds: `total_timesteps`, `gamma`, `learning_rate`, `num_envs`, observation/reward normalization.

### PPO (Level 2c)

Inherits RL. Adds: `clip_epsilon`, `gae_lambda`, `entropy_coef`, `value_coef`, `max_grad_norm`, `target_kl`.

→ Full spec: `rl_fields.md`

---

## Output Directory Structure

Every experiment automatically creates a timestamped run directory.

### Tree-Based Models

```
{output.base_dir}/{name}_{YYYYMMDD_HHMMSS}/
├── config.json              # full config snapshot
├── console.log              # captured stdout/stderr
├── metrics.jsonl            # per-epoch metrics (JSON-lines)
├── metrics.json             # final metrics (rmse, mae, r2, mape, n_trees_used, top_features)
├── predictions.parquet      # model predictions on test set
├── checkpoints/
│   └── model_best.json      # best model (native tree format)
└── plots/
    ├── feature_importance.png
    ├── pred_vs_actual.png
    ├── residual_dist.png
    ├── shap_importance.png
    └── shap_summary.png
```

### Neural Models

```
{output.base_dir}/{name}_{YYYYMMDD_HHMMSS}/
├── config.json              # full config snapshot
├── console.log              # captured stdout/stderr
├── metrics.jsonl            # per-epoch metrics (JSON-lines)
├── predictions.parquet      # model predictions on test set
└── checkpoints/
    └── model_best.*         # best model (.pt, .h5, .safetensors, etc.)
```

### Multi-Architecture Experiments

```
{output.base_dir}/{name}_{YYYYMMDD_HHMMSS}/
├── config.json
├── results.json             # array of per-architecture results
└── plots/
    ├── comparison_r2.png
    └── best_per_arch.png
```

---

## How to Create a Task-Specific Config

1. **Pick a parent level** — Neural network? → SL Neural / SL Neural Regression. Tree ensemble? → SL Tree / SL Tree Regression. RL? → RL / PPO. Other? → Base.
2. **Inherit its fields** — your config gets all parent fields + all built-in infrastructure
3. **Add task-specific fields** — model architecture, data paths, task-unique hyperparameters
4. **Override defaults** — change any inherited defaults to suit your experiment
5. **Disable what you don't need** — `checkpointing.enabled = false`, `metricslog.enabled = false`, `mlflow.enabled = false`
6. **Attach additional pieces** — custom composable pieces if needed

## Required Helpers

When generating config code, provide these utilities adapted to the target framework:

1. **save_config** — serialize config to JSON
2. **load_config** — deserialize JSON back into config object, handling nested structures
3. **setup_output_dir** — create timestamped run directory, subdirectories, save config.json
4. **setup_console_logging** — redirect stdout/stderr to log file (with optional tee); return cleanup function
5. **MLflow integration** — use `src/mlflow_utils.py` helpers (`setup_mlflow`, `log_epoch_metrics`, `log_model_checkpoint`, `end_mlflow_run`). Do not call raw `mlflow.*` functions directly; always go through the utils module

## Framework Adaptation Guide

| Framework | Config Format | Checkpoints | Device Handling | Notes |
|-----------|--------------|-------------|-----------------|-------|
| **PyTorch** | Python dataclasses | `.pt` | `torch.device(cfg.device)` | Full manual control; implement all helpers |
| **Hugging Face** | Map to `TrainingArguments` + custom dataclasses | Handled by Trainer | Handled by Trainer | `compute_metrics` for eval_metrics; log to JSON-lines |
| **TensorFlow / Keras** | Dataclasses or dicts | `.h5` / SavedModel | `tf.device` / `tf.distribute` | Keras callbacks for checkpointing, early stopping; log to JSON-lines |
| **XGBoost** | Dicts or dataclasses | `.json` | `tree_method="gpu_hist"` | Map SL Tree fields to native params; see `sl_tree_fields.md` |
| **LightGBM** | Dicts or dataclasses | `.txt` | `device="gpu"` | `colsample_bytree` → `feature_fraction`; see `sl_tree_fields.md` |
| **CatBoost** | Dicts or dataclasses | `.cbm` | `task_type="GPU"` | Different param names; see `sl_tree_fields.md` |
| **JAX / Flax** | Dataclasses or flax.struct | `.msgpack` / `.safetensors` | `jax.devices()` | Orbax for checkpointing |

## Additional Composable Pieces

The five infrastructure groups (output, console, checkpointing, metricslog, mlflow) are built into Base. Create new standalone groups for specialized concerns (e.g., EvalConfig, DistributedConfig). Attach as fields on task configs — do not add to the hierarchy.

## Extending the Hierarchy

- **New algorithm** (SAC, DQN): inherit from appropriate parent, add algorithm-specific fields. Infrastructure inherited automatically.
- **New composable piece** (distributed, custom eval): standalone group, attached as a field where needed.
- **New task type** (SL Tree Classification): inherit from SL Tree, add task-specific fields (`num_classes`, `class_weights`, etc.). SL Neural Classification is already defined — see `sl_neural_fields.md`.

**IMPORTANT:** When adding a new config level or task-specific config to any reference file, always update the Config Hierarchy tree diagram in this SKILL.md to reflect it. The hierarchy diagram is the single source of truth for the full config structure.
