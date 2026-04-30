# SL Neural Fields Reference

Field specifications for SL Neural (Level 1a), SL Neural Regression (Level 2a), and SL Neural Classification (Level 2b). All inherit Base fields and built-in infrastructure (output, console, checkpointing, metricslog, experiment tracking).

All neural training uses **PyTorch Lightning**. Config fields map to `Trainer` args, `LightningModule` methods, and callback params.

For Level 3 task-specific configs (LSTM, Transformer, CNN, T5, etc.), see [sl_neural_tasks.md](sl_neural_tasks.md).

## Table of Contents

- **SL Neural Base**
  - [Training fields](#training-fields) | [Optimizer](#optimizer-and-scheduler) | [Data loading](#data-loading) | [Regularization](#regularization) | [Mixed precision](#mixed-precision) | [Early stopping](#early-stopping) | [Metric logging](#metric-logging-flags)
- **SL Neural Regression** — [Fields](#level-2a-sl-neural-regression)
- **SL Neural Classification** — [Fields](#level-2b-sl-neural-classification)
- **Metrics Contract** — [Core](#sl-neural--core-epoch-metrics) | [Timing](#sl-neural--timing-metrics) | [Tracking](#sl-neural--tracking-metrics) | [System](#sl-neural--system-metrics) | [Per-Batch](#sl-neural--per-batch-metrics) | [One-Time](#sl-neural--one-time-params) | [Task Eval](#sl-neural-classification--task-eval-metrics) | [Recommended](#recommended-additional-metrics)

---

## Level 1a: SL Neural

For gradient-based, epoch-based supervised training (LSTM, Transformer, CNN, MLP, etc.) using Lightning.

### Training fields

| Field | Type | Default | Lightning Mapping |
|-------|------|---------|-------------------|
| num_epochs | integer | `100` | `Trainer(max_epochs=100)` |
| batch_size | integer | `32` | `LightningDataModule.batch_size` (auto-tuned by `BatchSizeFinder`) |
| learning_rate | float | `1e-3` | `configure_optimizers()` |
| weight_decay | float | `0.0` | `configure_optimizers()` |
| gradient_accumulation_steps | integer | `1` | `Trainer(accumulate_grad_batches=1)` |

### Optimizer and scheduler

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| optimizer | string | `"Adam"` | Optimizer name: `"Adam"`, `"AdamW"`, `"SGD"`. Set in `configure_optimizers()` |
| scheduler | string or null | `null` | LR scheduler: `"cosine"`, `"linear"`, `"step"`, or null. Returned from `configure_optimizers()` |
| scheduler_min_lr | float | `1e-6` | Minimum learning rate for scheduler |
| num_warmup_epochs | integer | `0` | Number of warmup epochs (linear ramp from 0 to `learning_rate`) |
| warmup_ratio | float or null | `null` | Alternative to `num_warmup_epochs`: warmup as fraction of total steps. If set, overrides `num_warmup_epochs` |

Lightning example:
```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate,
                                  weight_decay=self.cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.num_epochs,
                                                            eta_min=self.cfg.scheduler_min_lr)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
```

### Regularization

| Field | Type | Default | Lightning Mapping |
|-------|------|---------|-------------------|
| grad_clip_norm | float or null | `null` | `Trainer(gradient_clip_val=N)` |
| dropout | float | `0.0` | Set in model `__init__` |

### Mixed precision

| Field | Type | Default | Lightning Mapping |
|-------|------|---------|-------------------|
| use_amp | boolean | `true` | `Trainer(precision="bf16-mixed")` if true, `"32-true"` if false |

bf16 autocast. Requires Ampere+ GPU. Lightning handles AMP context wrapping automatically — no manual `torch.amp.autocast()` calls needed.

### Data loading

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| num_workers | integer | `2` | DataLoader worker processes in `LightningDataModule` |
| pin_memory | boolean | `true` | Pin host memory for faster GPU transfer |
| prefetch_factor | integer | `2` | Batches to prefetch per worker |

### Early stopping

| Field | Type | Default | Lightning Mapping |
|-------|------|---------|-------------------|
| early_stopping_patience | integer | `0` | `EarlyStopping(patience=N)` callback; 0 = no callback added |

### Metric logging flags

| Field | Type | Default | What to log |
|-------|------|---------|-------------|
| log_system_metrics | boolean | `true` | System/GPU metrics (`system/*`), timing metrics (`timing/*`) per epoch |

Note: Core metrics are always logged via `self.log()` in `training_step` / `validation_step`. System metrics via custom callback or Lightning's built-in GPU monitoring.

---

## Level 2a: SL Neural Regression

Inherits all SL Neural fields. Adds regression-specific configuration.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| loss_fn | string | `"mse"` | Loss function: `"mse"`, `"mae"`, `"huber"`, `"smooth_l1"` |
| eval_metrics | list of strings | `["rmse", "mae", "r2"]` | Metrics to compute in `validation_step` each epoch |
| figures | list of strings | `["pred_vs_actual", "residual_distribution"]` | Figures to generate at the end of training |
| hparam_metrics | list of strings | `["rmse", "mae", "r2"]` | Metrics to record for cross-run hyperparameter comparison |

---

## Level 2b: SL Neural Classification

Inherits all SL Neural fields. Adds classification-specific configuration.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| loss_fn | string | `"cross_entropy"` | Loss function: `"cross_entropy"` (CrossEntropyLoss on raw logits), `"nll"` (NLLLoss on log_softmax output) |
| eval_metrics | list of strings | `["record_f1", "record_em", "sql_em", "error_rate"]` | Metrics to compute in `validation_step` |

### Weight initialization

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| weight_init_enabled | boolean | `true` | Apply custom weight initialization in `__init__` |
| weight_init_method | string | `"uniform"` | Method: `"uniform"`, `"xavier_uniform"`, `"xavier_normal"`, `"kaiming_uniform"`, `"kaiming_normal"` |
| weight_init_min | float | `-0.1` | Lower bound (uniform only) |
| weight_init_max | float | `0.1` | Upper bound (uniform only) |

---

## Metrics Contract

All metrics logged via `self.log()` / `self.log_dict()` in LightningModule methods, routed to W&B via `WandbLogger`. Metrics are also written to `metrics.jsonl` via custom `MetricsJsonlCallback` when enabled.

### SL Neural — Core Epoch Metrics

Logged every epoch via `self.log()` in `training_step` and `validation_step`:

| Metric Key | Frequency | `self.log()` Args |
|------------|-----------|-------------------|
| `train_loss` | per epoch | `on_step=False, on_epoch=True` |
| `val_loss` | per epoch | `on_epoch=True` |
| `gradient_norm` | per epoch | Custom callback or `on_step=True` in `training_step` |
| `lr` | per epoch | `LearningRateMonitor` callback |

### SL Neural — Timing Metrics

| Metric Key | Frequency | Controlled By |
|------------|-----------|---------------|
| `timing/epoch_seconds` | per epoch | `TimingCallback` |
| `timing/wall_clock_seconds` | per epoch | `TimingCallback` |
| `timing/train_epoch_seconds` | per epoch | `TimingCallback` |
| `timing/train_tokens_per_sec` | per epoch | `TimingCallback` |

### SL Neural — Tracking Metrics

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `tracking/best_{metric}` | per epoch | Best value of tracked metric so far |
| `tracking/epochs_since_improvement` | per epoch | Epochs without improvement |

### SL Neural — System Metrics

Logged per epoch when `log_system_metrics = true`. Collected by custom callback using `src/utils/system_metrics.py`. All keys prefixed with `system/`:

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

### SL Neural — Per-Batch Metrics

Logged every batch via `self.log(..., on_step=True)` in `training_step`:

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `batch/loss` | per batch | Batch loss |
| `batch/gradient_norm` | per batch | Batch gradient L2 norm |
| `batch/lr` | per batch | Current learning rate |

### SL Neural — One-Time Params

Logged once via `logger.experiment.config.update()` after model initialization:

| Param Key | Purpose |
|-----------|---------|
| `total_params` | Total model parameters |
| `trainable_params` | Trainable parameters (after freezing) |
| `num_train_samples` | Training dataset size |
| `num_val_samples` | Validation dataset size |
| `gpu_name` | GPU device name (when CUDA available) |

### SL Neural Classification — Task Eval Metrics

Logged per epoch in `validation_step` via `self.log_dict()`:

| Metric Key | Frequency | Controlled By |
|------------|-----------|---------------|
| Each name in `eval_metrics` | per epoch | `eval_metrics` list |

### Transformer Language Model Metrics

| Metric Key | Frequency | Controlled By |
|------------|-----------|---------------|
| `train_perplexity` | per epoch | `eval_metrics` list |
| `dev_perplexity` | per epoch | `eval_metrics` list |

---

## Recommended Additional Metrics

Metrics not yet implemented but recommended for richer training diagnostics. Add these when setting up new training pipelines.

### Gradient Health (per epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `gradient_norm_min` | Detect vanishing gradients | `min(batch_grad_norms)` over epoch |
| `gradient_norm_max` | Detect exploding gradients before clipping masks them | `max(batch_grad_norms)` over epoch |
| `gradient_norm_std` | Gradient stability | `std(batch_grad_norms)` over epoch |

### Effective Update Magnitude (per epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `effective_update` | Actual parameter update magnitude | `lr * gradient_norm` per epoch |

### Overfitting Detection (per eval epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `overfit_ratio` | Rising ratio signals overfitting | `val_loss / train_loss` |

### Memory Health (per epoch)

| Metric Key | Purpose | How to Compute |
|------------|---------|----------------|
| `system/gpu_mem_fragmentation_mb` | Early OOM warning | `gpu_mem_reserved_mb - gpu_mem_allocated_mb` |
