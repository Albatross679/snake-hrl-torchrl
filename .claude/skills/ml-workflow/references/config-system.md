# ML Configuration System

A framework-agnostic specification for ML experiment configuration. Describes **what** a config should contain — not how to implement it. Generate the appropriate code for whatever framework the user is working with (dataclasses, HF TrainingArguments, Keras callbacks, plain dicts, etc.).

## Design Principles

1. **Hierarchical** — configs inherit from a base level and add fields for their domain
2. **Batteries included** — every config gets output directory, console logging, checkpointing, metrics log, and experiment tracking by default (all individually disableable)
3. **Serializable** — every config can be saved to JSON and loaded back
4. **Framework-agnostic** — the spec defines semantics; the implementation adapts to the target framework
5. **Lightning-native** — for neural training, configs map directly to Lightning Trainer args and callback params

## User Preferences

- **Human-friendly time units** — use hours (not seconds) for time-budget fields (e.g. `max_wall_clock_hours`). Convert to seconds internally where needed.
- **Config variant selection via CLI** — provide a `--config` flag that selects among Python config modules by name (e.g. `--config config_1` dynamically imports `part1.config_1`). Per-field CLI overrides apply on top. Priority: dataclass defaults < config module < CLI flags. Avoid JSON config files for variant selection.

## Config Hierarchy

Each level adds fields on top of its parent. All levels include the built-in infrastructure from Base.

```
Base                                    (core fields + all infrastructure)
├── SL Neural                           (epoch-based, gradient-based training via Lightning)
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
| seed | integer | `42` | Random seed for reproducibility (use `lightning.seed_everything(seed)`) |
| device | string | `"auto"` | Compute device: `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`. Lightning uses `accelerator` and `devices` args instead |

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

Always present. Controls model saving. For Lightning, maps to `ModelCheckpoint` callback. Grouped under `checkpointing`.

| Field | Type | Default | Lightning Mapping |
|-------|------|---------|-------------------|
| checkpointing.enabled | boolean | `true` | Include `ModelCheckpoint` in callbacks |
| checkpointing.save_best | boolean | `true` | `ModelCheckpoint(save_top_k=1)` |
| checkpointing.save_last | boolean | `true` | `ModelCheckpoint(save_last=True)` |
| checkpointing.save_every_n | integer | `5` | `ModelCheckpoint(every_n_epochs=5)` |
| checkpointing.metric | string | `"loss"` | `ModelCheckpoint(monitor="val_loss")` |
| checkpointing.mode | string | `"min"` | `ModelCheckpoint(mode="min")` |

### Built-in: Metrics Log

Always present. Appends one JSON object per epoch to a JSON-lines file. Grouped under `metricslog`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| metricslog.enabled | boolean | `true` | Enable metrics logging |
| metricslog.filename | string | `"metrics.jsonl"` | Log file name (relative to run dir) |

Note: With Lightning, primary metrics go to W&B via `WandbLogger`. The metrics.jsonl is a local backup via custom callback.

### Built-in: Experiment Tracking

Always present. For Lightning, use `WandbLogger`:

```python
from lightning.pytorch.loggers import WandbLogger
logger = WandbLogger(project=cfg.wandb_project, name=cfg.name, log_model=False)
trainer = Trainer(logger=logger)
```

**Integration contract — LightningModule must:**

1. **`__init__`:** call `self.save_hyperparameters()` to log config to W&B automatically
2. **`training_step` / `validation_step`:** use `self.log()` / `self.log_dict()` for metrics
3. **One-time metadata:** log via `logger.experiment.config.update()` after model init
4. **End of training:** call `wandb.finish()` after `trainer.fit()` completes

---

## Output Directory Structure

### Neural Models (Lightning)

```
{output.base_dir}/{name}_{YYYYMMDD_HHMMSS}/
├── config.json              # full config snapshot (saved manually)
├── console.log              # captured stdout/stderr
├── metrics.jsonl            # per-epoch metrics (optional local backup)
└── checkpoints/
    ├── best.ckpt            # best model by ModelCheckpoint
    └── last.ckpt            # final epoch checkpoint
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
6. **Map to Lightning** — config fields should map cleanly to `Trainer(...)` args and callback params

## Framework Adaptation Guide

| Framework | Config Format | Checkpoints | Device Handling | Training |
|-----------|--------------|-------------|-----------------|----------|
| **PyTorch Lightning** | Python dataclasses → Trainer args | `.ckpt` via `ModelCheckpoint` | `Trainer(accelerator="auto")` | `trainer.fit()` |
| **Hugging Face** | Map to `TrainingArguments` | Handled by Trainer | Handled by Trainer | `Trainer.train()` |
| **XGBoost** | Dicts or dataclasses | `.json` | `tree_method="gpu_hist"` | `xgb.train()` |
| **LightGBM** | Dicts or dataclasses | `.txt` | `device="gpu"` | `lgb.train()` |
| **CatBoost** | Dicts or dataclasses | `.cbm` | `task_type="GPU"` | `model.fit()` |

## Config → Lightning Trainer Mapping

Key config fields and their Lightning equivalents:

| Config Field | Lightning Trainer Arg |
|---|---|
| `num_epochs` | `max_epochs` |
| `gradient_accumulation_steps` | `accumulate_grad_batches` |
| `use_amp` / `precision` | `precision="bf16-mixed"` or `"32-true"` |
| `grad_clip_norm` | `gradient_clip_val` |
| `early_stopping_patience` | `EarlyStopping(patience=N)` callback |
| `checkpointing.*` | `ModelCheckpoint(...)` callback |
| `auto_batch_size` | `BatchSizeFinder(mode="binsearch")` callback |
| `log_every_n_steps` | `log_every_n_steps` |
| `device` | `accelerator="auto"`, `devices="auto"` |
| `num_workers` | Set in `DataLoader` inside `LightningDataModule` |
