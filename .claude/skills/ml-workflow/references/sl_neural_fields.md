# SL Neural Fields Reference

Field specifications for SL Neural (Level 1a), SL Neural Regression (Level 2a), SL Neural Classification (Level 2b), and task-specific configs (Level 3). All inherit Base fields and built-in infrastructure (output, console, checkpointing, metricslog, experiment tracking).

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

### Optimizer and scheduler

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| optimizer | string | `"Adam"` | Optimizer name: `"Adam"`, `"AdamW"`, `"SGD"` |
| scheduler | string or null | `null` | LR scheduler: `"cosine"`, `"linear"`, `"step"`, or null for none |
| scheduler_min_lr | float | `1e-6` | Minimum learning rate for scheduler |

### Regularization

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| grad_clip_norm | float or null | `null` | Maximum gradient norm; null disables clipping |
| dropout | float | `0.0` | Dropout rate |

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

## Level 3: Task-Specific Neural Regression Configs

These inherit all SL Neural Regression fields and add architecture-specific parameters. Each groups its task-specific fields under a model params section and a data config section.

### LSTM Regression

Inherits: SL Neural Regression. For sequence-to-value regression using LSTM with optional static feature fusion.

**Recommended parent overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| num_epochs | `100` | LSTMs need more epochs to converge |
| batch_size | `512` | Larger batches for sequence models |
| learning_rate | `1e-3` | Standard for Adam + LSTM |
| weight_decay | `1e-4` | Light regularization |
| scheduler | `"cosine"` | Smooth LR decay |
| early_stopping_patience | `15` | Patient early stopping |
| grad_clip_norm | `1.0` | Prevent exploding gradients (important for RNNs) |

**LSTM architecture fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| hidden_size | integer | `256` | LSTM hidden state dimension |
| num_layers | integer | `3` | Number of stacked LSTM layers |
| dropout_lstm | float | `0.3` | Dropout between LSTM layers |
| bidirectional | boolean | `false` | Use bidirectional LSTM |

**Static feature branch** (for hybrid temporal+static architectures):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| static_embedding_dim | integer | `32` | Static feature embedding size |
| static_hidden_dims | list of integers | `[64]` | MLP hidden layer sizes for static branch |
| dropout_static | float | `0.3` | Dropout in static branch |

**Fusion head** (combines temporal and static outputs):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| head_dims | list of integers | `[128, 64]` | MLP head layer sizes after concatenation |
| dropout_head | float | `0.3` | Dropout in fusion head |

**Data config:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| seq_length | integer | `48` | Input sequence length (in timesteps; e.g., 48 × 15min = 12 hours) |
| stride | integer | `4` | Sliding window stride |
| normalize_features | boolean | `true` | Per-building feature normalization |
| normalize_target | boolean | `true` | Per-building target normalization |
| num_workers | integer | `2` | DataLoader workers |
| pin_memory | boolean | `true` | Pin memory for GPU transfer |
| static_features | list of strings | `["grossarea", "floorsaboveground", "building_age"]` | Static building features to include |

**Checkpoint format:** `.pt` (state_dict + scaler_stats + feature counts + optimizer/scheduler/epoch)

---

### Transformer Regression

Inherits: SL Neural Regression. For sequence-to-value regression using a Transformer encoder with positional encoding.

**Recommended parent overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| num_epochs | `50` | Transformers converge faster than LSTMs |
| batch_size | `256` | Standard for attention models |
| learning_rate | `1e-3` | Standard for Adam + Transformer |
| weight_decay | `1e-5` | Very light regularization |
| scheduler | `"cosine"` | Smooth LR decay |
| early_stopping_patience | `10` | Faster convergence = less patience needed |
| dropout | `0.1` | Light dropout in encoder layers |

**Transformer architecture fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| d_model | integer | `64` | Embedding / model dimension |
| n_heads | integer | `4` | Number of attention heads |
| n_layers | integer | `3` | Number of transformer encoder layers |
| d_ff | integer | `128` | Feedforward hidden dimension |
| activation | string | `"gelu"` | Activation function: `"gelu"`, `"relu"`, `"leaky_relu"` |
| use_positional_encoding | boolean | `true` | Add sinusoidal positional encoding |

**FC head** (after mean-pooled encoder output):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| fc_dims | list of integers | `[64]` | FC head hidden layer sizes |
| dropout_fc | float | `0.3` | Dropout in FC head |

**Data config:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| seq_length | integer | `24` | Input sequence length (in timesteps) |
| stride | integer | `1` | Sliding window stride |
| normalize_features | boolean | `true` | Per-building feature normalization |
| normalize_target | boolean | `true` | Per-building target normalization |
| num_workers | integer | `4` | DataLoader workers |
| pin_memory | boolean | `true` | Pin memory for GPU transfer |

**Checkpoint format:** `.pt` (state_dict + n_features + seq_length)

---

### CNN Regression

Inherits: SL Neural Regression. For sequence-to-value regression using 1D convolutions (Conv1d) with pooling.

**Recommended parent overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| num_epochs | `80` | CNNs converge moderately fast |
| batch_size | `1024` | CNNs handle large batches well |
| learning_rate | `3e-4` | Slightly lower LR for conv models |
| weight_decay | `1e-4` | Light regularization |
| scheduler | `"cosine"` | Smooth LR decay |
| early_stopping_patience | `15` | Moderate patience |

**CNN architecture fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| conv_channels | list of integers | `[64, 128, 256]` | Output channels per Conv1d layer |
| kernel_sizes | list of integers | `[7, 5, 3]` | Kernel size per Conv1d layer (must match length of conv_channels) |
| pool_size | integer | `2` | MaxPool1d kernel size |
| dropout_conv | float | `0.15` | Dropout after each conv block |
| use_batch_norm | boolean | `true` | Apply BatchNorm1d after each conv layer |
| activation | string | `"gelu"` | Activation function: `"gelu"`, `"relu"`, `"leaky_relu"` |

**FC head** (after AdaptiveAvgPool1d → flatten):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| fc_dims | list of integers | `[128, 64]` | FC head hidden layer sizes |
| dropout_fc | float | `0.3` | Dropout in FC head |

**Validation:** `conv_channels` and `kernel_sizes` must have the same length. `seq_length` must be ≥ `2^len(conv_channels)` (enough length to survive pooling).

**Data config:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| seq_length | integer | `96` | Input sequence length (e.g., 96 × 15min = 24 hours) |
| stride | integer | `4` | Sliding window stride |
| normalize_features | boolean | `true` | Per-building feature normalization |
| normalize_target | boolean | `true` | Per-building target normalization |
| num_workers | integer | `4` | DataLoader workers |
| pin_memory | boolean | `true` | Pin memory for GPU transfer |

**Checkpoint format:** `.pt` (state_dict + n_features + seq_length)

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

## Level 3: Task-Specific Neural Classification Configs

These inherit all SL Neural Classification fields and add architecture-specific parameters. Each groups its task-specific fields under model architecture, attention, and data config sections.

### Transformer Sequence Classification

Inherits: SL Neural Classification. For per-position classification using a custom single-head Transformer encoder. Each position in the input sequence produces an independent class prediction (e.g., counting character occurrences at each position).

**Recommended parent overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| num_epochs | `30` | Converges quickly on character-level tasks |
| batch_size | `32` | Standard for sequence classification |
| learning_rate | `1e-3` | Standard for Adam |
| dropout | `0.0` | Small models may not need dropout |
| early_stopping_patience | `5` | Quick convergence = low patience |
| loss_fn | `"nll"` | Model outputs log_softmax directly |

**Transformer architecture fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| vocab_size | integer | `27` | Vocabulary size (e.g., 26 letters + space) |
| num_positions | integer | `20` | Maximum input sequence length |
| d_model | integer | `64` | Embedding and model dimension |
| d_internal | integer | `64` | Attention key/value dimension and FFN intermediate dimension |
| num_layers | integer | `1` | Number of stacked TransformerLayer modules |

**Attention fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| attention_type | string | `"standard"` | Attention mechanism: `"standard"`, `"relative_position"`, `"alibi"` |
| causal_mask | boolean | `true` | Apply upper-triangular causal mask (tokens attend only to past) |
| max_relative_position | integer | `20` | Maximum relative distance for relative position attention (bidirectional: 2×max−1 embeddings) |
| alibi_slope | float | `0.125` | Slope for ALiBi linear bias: `−slope × |i − j|` |
| use_positional_encoding | boolean | `true` | Add learned positional embeddings to token embeddings. Disable when using ALiBi |

**Layer normalization:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| layer_norm_type | string | `"post"` | Layer norm placement: `"pre"` (before sublayer), `"post"` (after residual), `"none"` |
| layer_norm_eps | float | `1e-5` | Epsilon for LayerNorm numerical stability |

**Attention type details:**

| Type | Mechanism | Positional Encoding | Key Parameters |
|------|-----------|---------------------|----------------|
| `standard` | Scaled dot-product `Q·K^T / √d` | Required (learned) | — |
| `relative_position` | Standard + learned relative position bias `Q·R_{j-i}` (Shaw et al.) | Optional (additive) | `max_relative_position` |
| `alibi` | Standard + linear distance penalty `−slope·|i−j|` (Press et al.) | Disable (`false`) | `alibi_slope` |

**Recommended scenario presets:**

| Scenario | num_layers | attention_type | positional_encoding | Notes |
|----------|-----------|----------------|---------------------|-------|
| Baseline | 1 | standard | true | Single-layer standard attention |
| Relative | 1 | relative_position | true | Learned relative position bias |
| ALiBi | 1 | alibi | false | No positional encoding needed |
| Deep standard | 3–4 | standard | true | Multi-layer for complex dependencies |

**Data config:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| seq_length | integer | `20` | Input sequence length (characters) |

**Checkpoint format:** `.pt` (state_dict + architecture config)

**Checkpointing overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| checkpointing.metric | `"dev_accuracy"` | Track classification accuracy |
| checkpointing.mode | `"max"` | Higher accuracy is better |

---

### Transformer Language Model

Inherits: SL Neural Classification. For next-token prediction (character-level language modeling) using a multi-head Transformer with causal masking. Classification is over the full vocabulary at each position.

**Recommended parent overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| num_epochs | `20` | LM converges with fewer epochs on large text |
| batch_size | `64` | Larger batches for language modeling |
| learning_rate | `1e-3` | Standard for Adam |
| dropout | `0.1` | Regularization for multi-head attention |
| early_stopping_patience | `5` | Standard patience |
| loss_fn | `"nll"` | Model outputs log_softmax directly |
| weight_init_method | `"xavier_uniform"` | Better for deeper multi-head models |

**Transformer architecture fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| vocab_size | integer | `27` | Vocabulary size (e.g., 26 letters + space) |
| num_positions | integer | `128` | Maximum sequence length the model can process |
| d_model | integer | `128` | Embedding and model dimension |
| d_internal | integer | `128` | FFN intermediate dimension |
| num_layers | integer | `2` | Number of stacked Transformer layers |
| num_heads | integer | `4` | Number of attention heads (`d_model` must be divisible by `num_heads`; head_dim = `d_model / num_heads`) |

**Attention fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| attention_type | string | `"standard"` | Attention mechanism: `"standard"`, `"relative_position"`, `"alibi"` |
| max_relative_position | integer | `128` | Maximum relative distance for relative position attention |
| use_positional_encoding | boolean | `true` | Add learned positional embeddings (with dropout). Disable for ALiBi |

**Layer normalization:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| layer_norm_type | string | `"pre"` | `"pre"` (Pre-LN: norm before sublayer; requires final LayerNorm) or `"post"` (Post-LN: norm after residual) |
| layer_norm_eps | float | `1e-5` | Epsilon for LayerNorm |

**Data config:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| context_size | integer | `64` | Context window for training sequences (input length per sample) |

**Multi-head attention detail:** Queries, keys, and values are projected to full `d_model` dimension, split into `num_heads` heads of size `head_dim = d_model / num_heads`, attention computed per head, concatenated, then projected back through `W_o`. Per-head ALiBi slopes use `2^(−8·(i+1)/n)` for head `i` of `n` heads. Relative position embeddings are per-head at `head_dim` size.

**Causal masking:** Always enabled. Upper-triangular mask of `−∞` applied before softmax to prevent attending to future tokens.

**Model variants:**

| Variant | Class | Attention | Multi-head | Notes |
|---------|-------|-----------|------------|-------|
| PyTorch standard | `TransformerLM` | `nn.TransformerEncoder` | Yes | Uses `nn.TransformerEncoderLayer`, scales embeddings by `√d_model` |
| Custom | `TransformerLMCustom` | Custom `TransformerLayerLM` | Yes | Supports standard, relative_position, ALiBi; includes final LayerNorm for Pre-LN |

**Checkpoint format:** `.pt` (state_dict + vocab_index + num_positions)

**Checkpointing overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| checkpointing.metric | `"dev_perplexity"` | Track language model quality |
| checkpointing.mode | `"min"` | Lower perplexity is better |

**Eval metrics override:**

| Field | Override | Reason |
|-------|----------|--------|
| eval_metrics | `["perplexity"]` | Primary LM metric |
| hparam_metrics | `["perplexity"]` | For cross-run comparison |

---

### T5 NL-to-SQL (Fine-tune / From-Scratch)

Inherits: SL Neural Classification. For sequence-to-sequence NL-to-SQL translation using T5 (encoder-decoder). Two variants: fine-tuning a pretrained T5-small, or training from scratch with the same architecture.

**Configs:** `part1/config.py` (T5FineTuneConfig), `part2/config.py` (T5ScratchConfig). Both inherit `SLNeuralClsConfig`.

**T5 Fine-tune recommended overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| num_epochs | `20` | Pretrained model converges quickly |
| batch_size | `16` | Memory-efficient for T5-small |
| learning_rate | `1e-4` | Lower LR for fine-tuning |
| weight_decay | `0.01` | Light regularization |
| scheduler | `"cosine"` | Smooth LR decay |
| patience_epochs | `5` | Early stopping |
| loss_fn | `"cross_entropy"` | CrossEntropyLoss over vocabulary |

**T5 From-scratch recommended overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| num_epochs | `50` | Random init needs more epochs |
| learning_rate | `1e-3` | Higher LR for training from scratch |
| patience_epochs | `0` | No early stopping by default |

**T5-specific fields (both variants):**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| model_checkpoint | string | `"google-t5/t5-small"` | HuggingFace model ID (architecture source for both variants) |
| finetune | boolean | `true`/`false` | Load pretrained weights (true) or random init (false) |
| freeze_encoder | boolean | `false` | Freeze entire encoder (fine-tune only) |
| freeze_embeddings | boolean | `false` | Freeze shared embeddings (fine-tune only) |
| unfreeze_last_n_decoder | integer or null | `null` | Freeze all decoder layers except last N (fine-tune only) |
| input_prefix | string | `""` | Prefix prepended to NL input (e.g. `"translate English to SQL: "`) |
| include_schema | boolean | `false` | Prepend database schema to NL input |
| max_new_tokens | integer | `256` | Maximum tokens for generation |
| num_beams | integer | `1` | Beam search width (1 = greedy) |
| resume_run_dir | string or null | `null` | Path to previous run directory to resume training |
| max_wall_clock_hours | float or null | `null` | Stop after this many hours |
| test_batch_size | integer | `16` | Batch size for eval/test (can differ from training) |

**Checkpointing overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| checkpointing.metric | `"record_f1"` | Track NL-to-SQL record-level F1 |
| checkpointing.mode | `"max"` | Higher F1 is better |

**Eval metrics:**

| Field | Override | Reason |
|-------|----------|--------|
| eval_metrics | `["record_f1", "record_em", "sql_em", "error_rate"]` | NL-to-SQL evaluation metrics |

**Checkpoint format:** `.pt` (model state_dict via HuggingFace `save_pretrained`/`from_pretrained`). Training state saved separately for resume (optimizer, scheduler, epoch, best_val, mlflow_run_id).

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
