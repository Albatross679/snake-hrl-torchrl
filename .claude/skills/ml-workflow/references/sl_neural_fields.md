# SL Neural Fields Reference

Field specifications for SL Neural (Level 1a), SL Neural Regression (Level 2a), and SL Neural Classification (Level 2b). All inherit Base fields and built-in infrastructure (output, console, checkpointing, metricslog, experiment tracking).

For Level 3 task-specific configs (LSTM, Transformer, CNN, T5, etc.), see [sl_neural_tasks.md](sl_neural_tasks.md).

## Table of Contents

- **SL Neural Base**
  - [Training fields](#training-fields) | [Optimizer](#optimizer-and-scheduler) | [Data loading](#data-loading) | [Regularization](#regularization) | [Mixed precision](#mixed-precision) | [Early stopping](#early-stopping) | [Metric logging](#metric-logging-flags)
- **SL Neural Regression** — [Fields](#level-2a-sl-neural-regression)
- **SL Neural Classification** — [Fields](#level-2b-sl-neural-classification)
- **Metrics Contract** — [Core](#sl-neural--core-epoch-metrics) | [Timing](#sl-neural--timing-metrics) | [Tracking](#sl-neural--tracking-metrics) | [System](#sl-neural--system-metrics) | [Per-Batch](#sl-neural--per-batch-metrics) | [One-Time](#sl-neural--one-time-params) | [Task Eval](#sl-neural-classification--task-eval-metrics) | [Recommended](#recommended-additional-metrics)

---

## Level 1a: SL Neural

For gradient-based, epoch-based supervised training (LSTM, Transformer, CNN, MLP, etc.).

### Training fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| num_epochs | integer | `100` | Number of training epochs |
| batch_size | integer | `32` | Training batch size |
| learning_rate | float | `1e-3` | Initial learning rate for optimizer |
| weight_decay | float | `0.0` | L2 regularization strength |
| gradient_accumulation_steps | integer | `1` | Accumulate gradients over N batches before optimizer step. Effective batch size = `batch_size * gradient_accumulation_steps` |

### Optimizer and scheduler

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| optimizer | string | `"Adam"` | Optimizer name: `"Adam"`, `"AdamW"`, `"SGD"` |
| scheduler | string or null | `null` | LR scheduler: `"cosine"`, `"linear"`, `"step"`, or null for none |
| scheduler_min_lr | float | `1e-6` | Minimum learning rate for scheduler |
| num_warmup_epochs | integer | `0` | Number of warmup epochs (linear ramp from 0 to `learning_rate`) |
| warmup_ratio | float or null | `null` | Alternative to `num_warmup_epochs`: warmup as fraction of total steps (more portable across batch sizes). If set, overrides `num_warmup_epochs` |

### Regularization

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| grad_clip_norm | float or null | `null` | Maximum gradient norm; null disables clipping |
| dropout | float | `0.0` | Dropout rate |

### Mixed precision

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| use_amp | boolean | `true` | bf16 autocast via `torch.amp.autocast('cuda', dtype=torch.bfloat16)`. Requires Ampere+ GPU. Set `false` for older GPUs or debugging |

### Data loading

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| num_workers | integer | `2` | DataLoader worker processes for parallel data loading |
| pin_memory | boolean | `true` | Pin host memory for faster GPU transfer |
| prefetch_factor | integer | `2` | Batches to prefetch per worker. Increase if GPU is starved for data |

### Early stopping

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| early_stopping_patience | integer | `0` | Stop after N epochs without improvement; 0 disables |

### Metric logging flags

Control which optional metric groups the training loop should log:

| Field | Type | Default | What to log |
|-------|------|---------|-------------|
| log_system_metrics | boolean | `true` | System/GPU metrics (`system/*`), timing metrics (`timing/*`) per epoch |

Note: Core metrics (`train_loss`, `dev_loss`, `gradient_norm`, `lr`, eval metrics) and per-batch metrics (`batch/*`) are always logged. Tracking metrics (`tracking/*`) are always logged. Only system and timing metrics are controlled by the flag.

---

## Level 2a: SL Neural Regression

Inherits all SL Neural fields. Adds regression-specific configuration.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| loss_fn | string | `"mse"` | Loss function: `"mse"`, `"mae"`, `"huber"`, `"smooth_l1"` |
| eval_metrics | list of strings | `["rmse", "mae", "r2"]` | Metrics to compute on validation set each epoch |
| figures | list of strings | `["pred_vs_actual", "residual_distribution"]` | Figures to generate at the end of training |
| hparam_metrics | list of strings | `["rmse", "mae", "r2"]` | Metrics to record for cross-run hyperparameter comparison |

---

## Level 2b: SL Neural Classification

Inherits all SL Neural fields. Adds classification-specific configuration.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| loss_fn | string | `"cross_entropy"` | Loss function: `"cross_entropy"` (CrossEntropyLoss on raw logits), `"nll"` (NLLLoss on log_softmax output) |
| eval_metrics | list of strings | `["record_f1", "record_em", "sql_em", "error_rate"]` | Metrics to compute on validation set each epoch. Task-specific — override for your use case |

### Weight initialization

Classification models expose explicit weight initialization control (also available to regression configs if overridden):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| weight_init_enabled | boolean | `true` | Apply custom weight initialization |
| weight_init_method | string | `"uniform"` | Initialization method: `"uniform"`, `"xavier_uniform"`, `"xavier_normal"`, `"kaiming_uniform"`, `"kaiming_normal"` |
| weight_init_min | float | `-0.1` | Lower bound (uniform only) |
| weight_init_max | float | `0.1` | Upper bound (uniform only) |

---

## Metrics Contract

All metrics are logged via `log_epoch_metrics(metrics_dict, step)` to the experiment tracker (W&B). Metric keys use `/`-separated prefixes to organize into logical groups. Metrics are also written to `metrics.jsonl` when enabled.

### SL Neural — Core Epoch Metrics

Logged every epoch (step = epoch number):

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `train_loss` | per epoch | Average training loss (token-weighted) |
| `dev_loss` | per epoch | Average dev/validation loss (token-weighted) |
| `gradient_norm` | per epoch | Average gradient L2 norm across batches |
| `lr` | per epoch | Current learning rate |

### SL Neural — Timing Metrics

| Metric Key | Frequency | Controlled By |
|------------|-----------|---------------|
| `timing/epoch_seconds` | per epoch | `log_system_metrics` |
| `timing/wall_clock_seconds` | per epoch | `log_system_metrics` |
| `timing/train_epoch_seconds` | per epoch | `log_system_metrics` |
| `timing/train_tokens_per_sec` | per epoch | `log_system_metrics` |

### SL Neural — Tracking Metrics

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `tracking/best_{metric}` | per epoch | Best value of the tracked metric so far |
| `tracking/epochs_since_improvement` | per epoch | Epochs without improvement (for early stopping monitoring) |

### SL Neural — System Metrics

Logged per epoch when `log_system_metrics = true`. Collected by `src/utils/system_metrics.py:collect_system_metrics()`. All keys prefixed with `system/`:

| Metric Key | Source | Purpose |
|------------|--------|---------|
| `system/gpu_mem_allocated_mb` | `torch.cuda` | Currently allocated GPU memory |
| `system/gpu_mem_reserved_mb` | `torch.cuda` | Reserved GPU memory |
| `system/gpu_mem_peak_mb` | `torch.cuda` | Peak allocated GPU memory |
| `system/gpu_util_pct` | `pynvml` | GPU utilization percentage |
| `system/gpu_temp_c` | `pynvml` | GPU temperature (Celsius) |
| `system/gpu_power_w` | `pynvml` | GPU power draw (watts) |
| `system/ram_used_gb` | `psutil` | System RAM used |
| `system/ram_pct` | `psutil` | System RAM usage percentage |
| `system/process_rss_mb` | `psutil` | Process resident memory |
| `system/cpu_pct` | `psutil` | CPU utilization percentage |

Note: `pynvml` and `psutil` metrics degrade gracefully — missing libraries simply omit those keys.

### SL Neural — Per-Batch Metrics

Logged every batch (step = global batch counter across all epochs):

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `batch/loss` | per batch | Batch loss (not averaged) |
| `batch/gradient_norm` | per batch | Batch gradient L2 norm (before clipping) |
| `batch/lr` | per batch | Current learning rate |

### SL Neural — One-Time Params

Logged once via `log_extra_params()` after model initialization:

| Param Key | Purpose |
|-----------|---------|
| `total_params` | Total model parameters |
| `trainable_params` | Trainable parameters (after freezing) |
| `num_train_samples` | Training dataset size |
| `num_dev_samples` | Dev/validation dataset size |
| `gpu_name` | GPU device name (when CUDA available) |
| `avg_epoch_seconds` | Average epoch duration (logged at end of training) |

### SL Neural Classification — Task Eval Metrics

Logged per epoch in addition to all SL Neural metrics above. The specific metrics depend on the `eval_metrics` list:

| Metric Key | Frequency | Controlled By |
|------------|-----------|---------------|
| Each name in `eval_metrics` | per epoch | `eval_metrics` list |

With NL-to-SQL defaults (`eval_metrics = ["record_f1", "record_em", "sql_em", "error_rate"]`), each epoch logs: `record_f1`, `record_em`, `sql_em`, `error_rate`.

Example epoch log entry:
```json
{"epoch": 5, "train_loss": 0.42, "dev_loss": 0.65, "record_f1": 0.78, "record_em": 0.52, "sql_em": 0.31, "error_rate": 0.05, "gradient_norm": 1.23, "lr": 8.5e-5}
```

### Transformer Language Model Metrics

Same as SL Neural Classification metrics, with these eval_metrics overrides:

| Metric Key | Frequency | Controlled By |
|------------|-----------|---------------|
| `train_perplexity` | per epoch | `eval_metrics` list |
| `dev_perplexity` | per epoch | `eval_metrics` list |

With defaults, each epoch logs: `train_perplexity`, `dev_perplexity`, `gradient_norm`

---

## Recommended Additional Metrics

Metrics not yet implemented but recommended for richer training diagnostics. Add these when setting up new training pipelines or extending existing ones.

### Gradient Health (per epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `gradient_norm_min` | Detect vanishing gradients | `min(batch_grad_norms)` over epoch |
| `gradient_norm_max` | Detect exploding gradients before clipping masks them | `max(batch_grad_norms)` over epoch |
| `gradient_norm_std` | Gradient stability — high std signals noisy optimization | `std(batch_grad_norms)` over epoch |

Average gradient norm alone can mask problems: if some batches explode and others vanish, the average looks normal.

### Effective Update Magnitude (per epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `effective_update` | Actual parameter update magnitude (more informative than LR alone) | `lr * gradient_norm` per epoch |

Shows how much parameters actually move. Useful when LR and grad norm change in opposite directions (scheduler ramp-down + increasing gradients = stable updates, not visible from either metric alone).

### Token-Level Accuracy (per epoch, seq2seq tasks)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `token_accuracy` | Per-token prediction accuracy on dev set | `correct_tokens / total_tokens` (excluding padding) |

Gives signal every epoch even when Record F1 is noisy from SQL execution errors. A model with high token accuracy but low F1 suggests correct SQL syntax with wrong values. Cheap to compute alongside dev_loss.

### Surface-Level Generation Quality (per eval epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `bleu` | Surface similarity of generated vs reference SQL | `sacrebleu.corpus_bleu(predictions, [references])` |
| `chrf` | Character-level F-score — more robust than BLEU for short outputs | `sacrebleu.corpus_chrf(predictions, [references])` |

Independent of SQL execution. Catches generation quality regressions even when evaluation DB is unavailable or slow.

### Inference Performance (per eval epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `inference/tokens_per_sec` | Generation throughput | `total_generated_tokens / generation_wall_time` |
| `inference/avg_output_length` | Average generated sequence length | `mean(output_lengths)` |
| `inference/ms_per_sample` | Latency per example | `generation_wall_time * 1000 / num_samples` |

Catches generation degeneration (infinite loops, excessive beam expansion) and tracks deployment readiness.

### Memory Health (per epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `system/gpu_mem_fragmentation_mb` | Early warning for OOM before it happens | `gpu_mem_reserved_mb - gpu_mem_allocated_mb` |

High fragmentation (large gap between reserved and allocated) means CUDA has memory but can't use it contiguously. Precursor to OOM even when `nvidia-smi` shows free memory.

### Overfitting Detection (per eval epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `overfit_ratio` | Ratio of dev loss to train loss — rising ratio signals overfitting | `dev_loss / train_loss` |

A ratio near 1.0 means good generalization. Rising above 1.5-2.0 signals overfitting. More actionable than watching two separate loss curves diverge visually.
