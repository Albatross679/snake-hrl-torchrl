# ML Codebase Structure Reference

The user's preferred code organization for ML experiment projects. This pattern has been refined across multiple projects and should be followed when scaffolding new ML work or extending existing projects.

## Directory Layout

```
project-root/
├── CLAUDE.md                # Project instructions for Claude Code
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # Pip requirements (for submission/portability)
│
├── train_entry.py           # Root entry point — delegates to partN.train.main()
├── evaluate.py              # CLI evaluation script (metrics from predicted vs ground truth)
├── utils.py                 # Shared metrics computation (F1, EM, error rates)
│
├── src/                     # Shared infrastructure (never duplicated per-part)
│   ├── config.py            # Config hierarchy: BaseConfig → SLNeuralConfig → TaskConfig
│   ├── mlflow_utils.py      # Experiment tracking (W&B). Named "mlflow" for legacy reasons
│   └── utils/
│       └── system_metrics.py  # GPU/CPU/RAM collection per epoch
│
├── part1/                   # Approach 1 (e.g. fine-tune pretrained model)
│   ├── config.py            # All config variants as @dataclass classes
│   ├── data.py              # Dataset class, collate functions, DataLoader factory
│   ├── model.py             # Model init, save_model, load_model, save/load training state
│   ├── model_domain.py      # Domain-specific wrapper (e.g. restricted vocab, MLP head)
│   ├── train.py             # Training loop: main(), train(), eval, multi-config batch
│   └── eval_checkpoint.py   # Standalone eval for saved checkpoints
│
├── part2/                   # Approach 2 (e.g. train from scratch)
│   ├── config.py            # Different hyperparameters (higher LR, more epochs)
│   ├── data.py              # Same or similar to part1
│   ├── model.py             # Same interface, different initialization
│   └── train.py             # Same loop structure, different flags
│
├── part3/                   # Approach 3 (e.g. prompting / in-context learning)
│   ├── config.py            # PromptingConfig (model, shot count, prompt type)
│   ├── data.py              # Load prompting data (train examples + dev/test queries)
│   ├── model.py             # LLM loading (Gemma, CodeGemma, etc.)
│   └── train.py             # Prompting pipeline (create prompts, inference, eval)
│
├── data/                    # Raw datasets (DO NOT MODIFY)
├── model/                   # Saved model weights (not committed)
├── output/                  # Run outputs: output/{name}_{timestamp}/
├── records/                 # Pickled evaluation records
├── results/                 # Predicted output files (SQL, text, etc.)
│
├── logs/                    # Code change logs
├── experiments/             # Training run reports
├── issues/                  # Bug/error documentation
├── knowledge/               # Domain knowledge
├── references/              # External citations
├── tasks/                   # PRDs and task specs
├── media/                   # Images, plots
└── script/                  # Standalone utility scripts
```

## Module Responsibilities

### Per-Part Modules (partN/)

**config.py** — All experiment variants as `@dataclass` classes inheriting from a shared base config. Each variant overrides only the fields that differ. Ordered in priority (fastest/simplest first). Example:

```python
@dataclass
class MyConfig_baseline(BaseTaskConfig):
    name: str = "baseline"
    learning_rate: float = 1e-4

@dataclass
class MyConfig_aggressive(BaseTaskConfig):
    name: str = "aggressive"
    learning_rate: float = 5e-4
    label_smoothing: float = 0.1
```

**data.py** — `Dataset` class with tokenization, `collate_fn` with dynamic padding (no fixed-length padding in dataset — pad per-batch in collate), `get_dataloader()` factory, `load_data()` convenience. Shared tokenizer instantiated once at module level.

**model.py** — `initialize_model()` (pretrained or random init, optional layer freezing), `save_model()` (handles special cases like LoRA merge), `load_model_from_checkpoint()`, `save_training_state()` / `load_training_state()` for resume.

**model_domain.py** — Domain-specific wrappers around the base model. Examples: restricted output vocabulary, MLP projection head, custom `generate()` constraints. Wraps the base model as `self.model` and delegates standard methods.

**train.py** — The main training script. Structure:

```
main()
  ├── Parse CLI args (--config selects class, per-field overrides)
  ├── Load config, apply CLI overrides
  ├── Set random seeds
  ├── Load data (train/dev/test loaders)
  ├── Initialize model + optional wrappers
  ├── Auto batch size tuning (optional)
  ├── Optimizer + scheduler setup
  ├── Resume from checkpoint (if configured)
  ├── Setup W&B + output directory
  ├── train() loop
  │   ├── Per epoch: train_epoch → eval_epoch (async SQL) → checkpoint → early stop check
  │   ├── Graceful stop check (SIGTERM / STOP file)
  │   ├── Time budget check (max_wall_clock_hours)
  │   └── Drain pending async futures before exit
  ├── Reload best checkpoint
  ├── Final dev eval (synchronous)
  ├── Test inference
  └── Cleanup + end W&B run
```

**eval_checkpoint.py** — Standalone script that loads a saved checkpoint and runs evaluation. Useful for re-evaluating after code changes without retraining.

### Shared Infrastructure (src/)

**config.py** — Config hierarchy with `@dataclass` inheritance. Three levels:
- `BaseConfig` — seed, device, output dir, console logging, checkpointing, metrics log
- `SLNeuralConfig` — epochs, batch_size, learning_rate, optimizer, scheduler, regularization, early stopping
- Task-specific config — loss function, eval metrics, model architecture fields

All configs support `to_dict()` / `from_dict()` serialization with recursive type resolution. Nested sub-configs (`CheckpointingConfig`, `OutputConfig`, etc.) as embedded dataclasses.

**mlflow_utils.py** — Experiment tracking wrapper (W&B). Legacy filename; these functions:
- `setup_run(cfg)` — create output dir, save config.json, `wandb.init()`, define custom metric axes
- `log_epoch_metrics(metrics_dict, step)` — route `batch/*` keys to global_step axis, all others to epoch axis
- `log_extra_params(params)` — one-time metadata (model size, dataset size, hardware)
- `end_run()` — `wandb.finish()`

**utils/system_metrics.py** — `collect_system_metrics(device)` returns GPU memory/utilization/temperature, CPU/RAM usage. `collect_hardware_info()` returns static system info. Graceful degradation if pynvml/psutil unavailable.

## Key Patterns

### Multi-Config Batch Execution

```python
configs = [Config_v1, Config_v2, Config_v3, ...]  # priority order
for ConfigClass in configs:
    if stop_requested():
        break
    cfg = ConfigClass()
    cleanup_vram()          # del model, torch.cuda.empty_cache(), gc.collect()
    run_training(cfg)       # full train → eval → checkpoint cycle
    end_run()               # finish W&B run
```

### Two-Phase Async Evaluation

Non-last epochs overlap CPU-bound metric computation with the next training epoch:
- **Phase A (GPU):** Generate predictions via `model.generate()`, compute dev loss
- **Phase B (CPU):** Execute SQL queries in thread pool, compute F1/EM metrics
- Phase B runs in a `ThreadPoolExecutor` future; results collected before next eval or exit

### Dynamic Padding Collate

Datasets return variable-length tokenized sequences. The collate function pads per-batch to the longest sequence in that batch — no wasted compute on fixed-length padding.

### Config-Driven CLI

```bash
python partN/train.py --config ConfigClassName --learning_rate 5e-4 --num_epochs 40
```

Priority: dataclass defaults < config class < CLI flags. The `--config` flag selects which dataclass to instantiate; remaining flags override individual fields.

## Output Directory Convention

```
output/{config_name}_{YYYYMMDD_HHMMSS}/
├── config.json            # Full config snapshot (reproducibility)
├── console.log            # Captured stdout/stderr
├── metrics.jsonl          # One JSON object per epoch (loss, metrics, timing)
└── checkpoints/
    ├── model_best.pt      # Best by primary eval metric
    ├── model_last.pt      # Final epoch
    └── training_state.pt  # Optimizer, scheduler, epoch, best_val (for resume)
```
