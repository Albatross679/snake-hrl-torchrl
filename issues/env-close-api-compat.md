---
name: SoftManipulatorEnv.close() incompatible with TorchRL API
description: close() missing **kwargs causes TypeError when TorchRL passes raise_if_closed
type: issue
status: resolved
severity: low
subtype: compatibility
created: 2025-03-25
updated: 2025-03-25
tags: [torchrl, api, compatibility]
aliases: []
---

## Problem

TorchRL's `TransformedEnv.close()` passes `raise_if_closed=True` to the base env's `close()` method. `SoftManipulatorEnv.close()` didn't accept keyword arguments:

```
TypeError: SoftManipulatorEnv.close() got an unexpected keyword argument 'raise_if_closed'
```

This caused crashes during collector shutdown (after the last batch).

## Fix

Changed `close(self)` to `close(self, **kwargs)` and forwarded kwargs to `super().close(**kwargs)` in `papers/choi2025/env.py:526`.
