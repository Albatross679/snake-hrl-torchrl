# SL Neural Task Configs Reference

Level 3 task-specific configs that inherit from SL Neural Regression or SL Neural Classification. For base fields and metrics contract, see [sl_neural_fields.md](sl_neural_fields.md).

## Table of Contents

- **Regression Tasks**: [LSTM](#lstm-regression) | [Transformer](#transformer-regression) | [CNN](#cnn-regression)
- **Classification Tasks**: [Transformer Seq Clf](#transformer-sequence-classification) | [Transformer LM](#transformer-language-model) | [T5 NL-to-SQL](#t5-nl-to-sql-fine-tune--from-scratch)

---

## Task-Specific Neural Regression Configs

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

## Task-Specific Neural Classification Configs

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
| min_new_tokens | integer or null | `null` | Minimum tokens for generation (prevents truncated SQL) |
| length_penalty | float or null | `null` | Beam search length penalty (>1 favors longer, <1 shorter) |
| temperature | float | `1.0` | Sampling temperature for generation. Only used in DPO candidate generation; training/eval uses greedy or beam search (deterministic) |
| top_k | integer or null | `null` | Top-k sampling (limits sampling pool to k highest-probability tokens). Used in DPO candidate generation |
| top_p | float or null | `null` | Nucleus sampling (limits sampling pool to smallest set with cumulative probability >= p). Used in DPO candidate generation |
| resume_run_dir | string or null | `null` | Path to previous run directory to resume training |
| max_wall_clock_hours | float or null | `null` | Stop after this many hours |
| test_batch_size | integer | `16` | Batch size for eval/test (can differ from training) |
| label_smoothing | float | `0.0` | Label smoothing for CrossEntropyLoss |
| schema_mode | string | `"tables"` | Schema format: `"tables"` (table names only) |
| gradient_accumulation_steps | integer | `1` | Accumulate gradients over N batches before optimizer step. Effective batch size = `batch_size * gradient_accumulation_steps`. Use when auto_batch_size hits VRAM ceiling but larger effective batch is desired |
| weight_tying | boolean | `true` | Tie encoder/decoder embedding weights (T5 default). Consider disabling when training from scratch if encoder and decoder need distinct representations |

**LoRA fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| use_lora | boolean | `false` | Enable LoRA (Low-Rank Adaptation) via PEFT |
| lora_r | integer | `16` | LoRA rank (low-rank dimension) |
| lora_alpha | integer | `32` | LoRA scaling factor (effective scale = alpha/r) |
| lora_dropout | float | `0.05` | Dropout on LoRA layers |
| lora_target_modules | list of strings | `["q", "v"]` | Which attention projections to apply LoRA to |

LoRA initialization is handled internally by PEFT (Kaiming uniform for A matrix, zeros for B matrix). No config field to override.

**LoRA presets used in experiments:**

| Preset | `lora_r` | `lora_alpha` | `lora_target_modules` | Use Case |
|--------|----------|--------------|----------------------|----------|
| Standard | 16 | 32 | `["q", "v"]` | Default — fast training, good for fine-tune and DPO |
| Wide | 32 | 64 | `["q", "k", "v", "o"]` | More adapter capacity, all attention projections |
| Frozen encoder | 16 | 32 | `["q", "v"]` | Combined with `freeze_encoder=True` for minimal training |
| Warm-start | 16 | 32 | `["q", "v"]` | Applied on top of a pre-trained FT checkpoint |

**MLP projection head fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| use_mlp_head | boolean | `false` | Add MLP layer after decoder, before restricted vocab projection |
| mlp_dim | integer | `1024` | MLP hidden dimension |
| mlp_dropout | float | `0.1` | MLP dropout |

**DPO/RL alignment fields:** See [rl_fields.md](rl_fields.md) for DPO, GRPO, and CISPO config fields, metrics, reward design, and training stability guidance.

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
