---
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
tags: [torchrl, compatibility, v0.11, bug, env]
type: issue
status: resolved
severity: medium
subtype: compatibility
---

# TorchRL v0.11 close() Signature Change

## Problem

TorchRL v0.11 passes `raise_if_closed` kwarg to `close()`. The custom environment had `close(self)` with no kwargs, causing a TypeError.

**Affected:** `locomotion_elastica/env.py`

## Fix

Changed to `close(self, **kwargs)` and `super().close(**kwargs)`.
