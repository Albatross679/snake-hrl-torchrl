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
│   ├── wandb_utils.py       # Experiment tracking (W&B) — setup helpers for WandbLogger
│   ├── callbacks.py         # Shared Lightning callbacks (StopFileCallback, MetricsJsonlCallback, TimingCallback)
│   └── utils/
│       └── system_metrics.py  # GPU/CPU/RAM collection per epoch
│
├── part1/                   # Approach 1 (e.g. fine-tune pretrained model)
│   ├── config.py            # All config variants as @dataclass classes
│   ├── data.py              # LightningDataModule: setup(), train_dataloader(), val_dataloader()
│   ├── model.py             # LightningModule: training_step(), validation_step(), configure_optimizers()
│   ├── train.py             # Entry point: main() assembles Trainer + callbacks + logger, calls trainer.fit()
│   └── eval_checkpoint.py   # Standalone eval for saved checkpoints
│
├── part2/                   # Approach 2 (e.g. train from scratch)
│   ├── config.py            # Different hyperparameters (higher LR, more epochs)
│   ├── data.py              # LightningDataModule — same interface, possibly different data
│   ├── model.py             # LightningModule — same interface, different initialization
│   └── train.py             # Same Trainer structure, different config
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

**data.py** — `LightningDataModule` subclass. Handles dataset creation, train/val/test splits, and DataLoader construction. Uses dynamic padding in collate (pad per-batch to longest sequence, not fixed-length in dataset). Example:

```python
class MyDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size  # BatchSizeFinder modifies this

    def setup(self, stage=None):
        # Load and split datasets
        self.train_ds = ...
        self.val_ds = ...

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.cfg.num_workers,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.cfg.num_workers,
                          pin_memory=True)
```

**model.py** — `LightningModule` subclass. Encapsulates model architecture, forward pass, loss computation, optimizer setup, and metric logging. Example:

```python
class MyModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = build_network(cfg)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pred = self(batch["input"])
        loss = self.loss_fn(pred, batch["target"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["input"])
        loss = self.loss_fn(pred, batch["target"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                       lr=self.cfg.learning_rate,
                                       weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.num_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

Additionally exposes: `save_model()`, `load_model_from_checkpoint()` for standalone checkpoint loading outside Lightning.

**train.py** — The main entry point. Assembles Lightning components and calls `trainer.fit()`. Structure:

```
main()
  ├── Parse CLI args (--config selects class, per-field overrides)
  ├── Load config, apply CLI overrides
  ├── lightning.seed_everything(cfg.seed)
  ├── Instantiate LightningDataModule
  ├── Instantiate LightningModule
  ├── Configure callbacks:
  │   ├── ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True)
  │   ├── EarlyStopping(monitor="val_loss", patience=cfg.patience)
  │   ├── BatchSizeFinder(mode="binsearch")
  │   ├── LearningRateMonitor(logging_interval="epoch")
  │   └── StopFileCallback()
  ├── Configure WandbLogger
  ├── Instantiate Trainer(
  │       max_epochs=cfg.num_epochs,
  │       precision="bf16-mixed",
  │       accumulate_grad_batches=cfg.gradient_accumulation_steps,
  │       gradient_clip_val=cfg.grad_clip_norm,
  │       callbacks=callbacks,
  │       logger=logger,
  │       default_root_dir=run_dir,
  │   )
  ├── trainer.fit(model, datamodule=dm)
  ├── trainer.test(model, datamodule=dm, ckpt_path="best")
  └── wandb.finish()
```

**eval_checkpoint.py** — Standalone script that loads a saved checkpoint and runs evaluation. Useful for re-evaluating after code changes without retraining.

### Shared Infrastructure (src/)

**config.py** — Config hierarchy with `@dataclass` inheritance. Three levels:
- `BaseConfig` — seed, device, output dir, console logging, checkpointing, metrics log
- `SLNeuralConfig` — epochs, batch_size, learning_rate, optimizer, scheduler, regularization, early stopping
- Task-specific config — loss function, eval metrics, model architecture fields

All configs support `to_dict()` / `from_dict()` serialization with recursive type resolution. Nested sub-configs (`CheckpointingConfig`, `OutputConfig`, etc.) as embedded dataclasses.

**wandb_utils.py** — Experiment tracking helpers for WandbLogger setup:
- `create_wandb_logger(cfg)` — create `WandbLogger` with project, name, config
- `log_extra_params(logger, params)` — one-time metadata (model size, dataset size, hardware)
- Helper for sweep-aware logger creation (detect `wandb.run.sweep_id`)

**callbacks.py** — Shared Lightning callbacks:
- `StopFileCallback` — checks for `STOP` file between epochs, sets `trainer.should_stop = True`
- `MetricsJsonlCallback` — writes per-epoch metrics to local `metrics.jsonl` file
- `TimingCallback` — tracks per-section wall-clock time and logs to W&B under `timing/`

**utils/system_metrics.py** — `collect_system_metrics(device)` returns GPU memory/utilization/temperature, CPU/RAM usage. `collect_hardware_info()` returns static system info. Graceful degradation if pynvml/psutil unavailable.

## Papers Layout (Alternative to partN/)

For reproducing published results, the project uses `papers/<author_year>/` instead of `partN/`:

```
papers/choi2025/              # Paper reproduction
├── config.py                 # Physics + RL config (dataclass, inherits from src/configs/)
├── env.py                    # TorchRL environment wrapping physics simulator
├── train.py                  # Entry point (approach task training)
├── train_ppo.py              # PPO-specific training with TorchRL
└── train_sac.py              # SAC-specific training (if applicable)
```

Same conventions apply: configs are `@dataclass`, entry points are thin wrappers, shared infra lives in `src/`.

**Key difference:** Paper reproductions include physics simulation parameters that MUST match the paper's tables. Always verify rod radius, Young's modulus, damping constants, and friction ratios against the paper before training.

## Key Patterns

### Multi-Config Batch Execution

```python
configs = [Config_v1, Config_v2, Config_v3, ...]  # priority order
for ConfigClass in configs:
    if stop_requested():
        break
    cfg = ConfigClass()
    cleanup_vram()          # del model, trainer, torch.cuda.empty_cache(), gc.collect()

    # Each config gets its own Trainer + Logger
    dm = MyDataModule(cfg)
    model = MyModel(cfg)
    logger = WandbLogger(project=cfg.project, name=cfg.name)
    trainer = Trainer(max_epochs=cfg.num_epochs, logger=logger, callbacks=...)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")
    wandb.finish()
```

### Dynamic Padding Collate

Datasets return variable-length tokenized sequences. The collate function pads per-batch to the longest sequence in that batch — no wasted compute on fixed-length padding. Implemented inside `LightningDataModule.collate_fn()`.

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
├── metrics.jsonl          # One JSON object per epoch (local backup)
└── checkpoints/
    ├── best.ckpt          # Best by primary eval metric (ModelCheckpoint)
    └── last.ckpt          # Final epoch (ModelCheckpoint save_last=True)
```
