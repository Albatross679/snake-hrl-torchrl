# ML Configuration System

A framework-agnostic specification for ML experiment configuration. Describes **what** a config should contain — not how to implement it. Generate the appropriate code for whatever framework the user is working with (dataclasses, HF TrainingArguments, Keras callbacks, plain dicts, etc.).

## Design Principles

1. **Hierarchical** — configs inherit from a base level and add fields for their domain
2. **Batteries included** — every config gets output directory, console logging, checkpointing, metrics log, and experiment tracking by default (all individually disableable)
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
│   ├── Classic Control                 (environment-based, continuous/discrete action spaces)
│   │   ├── PPO                         (proximal policy optimization, on-policy)
│   │   ├── SAC                         (soft actor-critic, off-policy, continuous actions)
│   │   └── DQN                         (deep Q-network, off-policy, discrete actions)
│   └── LLM Alignment                  (generation-based, preference/reward optimization)
│       ├── DPO                         (direct preference optimization, offline)
│       ├── GRPO                        (group relative policy optimization, online)
│       └── CISPO                       (clipped IS policy optimization, online)
└── (your task inherits from any of these)
```

**Field references** (read on demand):
- [sl_neural_fields.md](sl_neural_fields.md) — SL Neural base + Regression + Classification fields, metrics contract
- [sl_neural_tasks.md](sl_neural_tasks.md) — Task configs: LSTM, Transformer, CNN, Transformer Seq Clf, Transformer LM, T5 NL-to-SQL
- [sl_tree_fields.md](sl_tree_fields.md) — SL Tree + Regression fields, metrics contract, framework param mapping, task configs
- [rl_fields.md](rl_fields.md) — RL: Classic Control (PPO, SAC, DQN) + LLM Alignment (DPO, GRPO, CISPO), metrics, reward design, stability

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
| console.tee_to_console | boolean | `true` | Also print to terminal |

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

### Built-in: Metrics Log

Always present. Appends one JSON object per epoch to a JSON-lines file. Grouped under `metricslog`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| metricslog.enabled | boolean | `true` | Enable metrics logging |
| metricslog.filename | string | `"metrics.jsonl"` | Log file name (relative to run dir) |

### Built-in: Experiment Tracking

Always present. Logs params, metrics, and artifacts to an experiment tracker (W&B, MLflow, etc.). Field names are project-specific — see codebase-structure.md for the current project's implementation.

**Integration contract — training loops must:**

1. **Start of training:** call `setup_run(cfg)` to initialize tracker, log all config params
2. **Per epoch:** call `log_epoch_metrics(metrics_dict, step=epoch)`
3. **One-time metadata:** call `log_extra_params(params_dict)` after model init
4. **End of training:** call `end_run()` to close the active run

---

## Output Directory Structure

### Neural Models

```
{output.base_dir}/{name}_{YYYYMMDD_HHMMSS}/
├── config.json              # full config snapshot
├── console.log              # captured stdout/stderr
├── metrics.jsonl            # per-epoch metrics (JSON-lines)
└── checkpoints/
    └── model_best.*         # best model (.pt, .h5, .safetensors, etc.)
```

### Tree-Based Models

```
{output.base_dir}/{name}_{YYYYMMDD_HHMMSS}/
├── config.json
├── console.log
├── metrics.jsonl
├── metrics.json             # final metrics (rmse, mae, r2, mape, n_trees_used, top_features)
├── checkpoints/
│   └── model_best.json      # best model (native tree format)
└── plots/
    ├── feature_importance.png
    ├── pred_vs_actual.png
    └── shap_summary.png
```

---

## How to Create a Task-Specific Config

1. **Pick a parent level** — Neural network? → SL Neural. Tree ensemble? → SL Tree. RL? → RL/PPO.
2. **Inherit its fields** — your config gets all parent fields + all built-in infrastructure
3. **Add task-specific fields** — model architecture, data paths, task-unique hyperparameters
4. **Override defaults** — change any inherited defaults to suit your experiment
5. **Disable what you don't need** — `checkpointing.enabled = false`, etc.

## Framework Adaptation Guide

| Framework | Config Format | Checkpoints | Device Handling |
|-----------|--------------|-------------|-----------------|
| **PyTorch** | Python dataclasses | `.pt` | `torch.device(cfg.device)` |
| **Hugging Face** | Map to `TrainingArguments` | Handled by Trainer | Handled by Trainer |
| **XGBoost** | Dicts or dataclasses | `.json` | `tree_method="gpu_hist"` |
| **LightGBM** | Dicts or dataclasses | `.txt` | `device="gpu"` |
| **CatBoost** | Dicts or dataclasses | `.cbm` | `task_type="GPU"` |
| **JAX / Flax** | Dataclasses | `.msgpack` | `jax.devices()` |
