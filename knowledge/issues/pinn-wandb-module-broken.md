---
name: PINN training crash — wandb module has no init attribute
description: wandb was a namespace package stub without actual code, causing AttributeError on wandb.init
type: issue
status: resolved
severity: high
subtype: system
created: 2026-03-18
updated: 2026-03-18
tags: [pinn, wandb, training, crash]
aliases: []
---

## Symptom

`python -m src.pinn.train_regularized` crashed with:
```
AttributeError: module 'wandb' has no attribute 'init'
```

## Root Cause

The `wandb` package was installed as a namespace package stub (no `__file__`, no `__version__`), likely from a partial or corrupted installation. `import wandb` succeeded but the module had no actual code.

## Fix

Reinstalled wandb: `pip install wandb` → installed wandb 0.25.1 with all dependencies.

## Verification

```python
import wandb; print(wandb.__version__)  # 0.25.1
```
