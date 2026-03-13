---
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
description: TorchRL v0.11 renamed spec classes BoundedTensorSpec to Bounded and CompositeSpec to Composite
tags: [torchrl, compatibility, v0.11, bug, imports]
type: issue
status: resolved
severity: high
subtype: compatibility
---

# TorchRL v0.11 Import Renames

## Problem

TorchRL v0.11 renamed several spec classes. Existing imports broke at runtime.

**Affected:** `src/networks/actor.py:12`

- `BoundedTensorSpec` → `Bounded`
- `CompositeSpec` → `Composite`
- `UnboundedContinuousTensorSpec` → `Unbounded`

## Fix

Added try/except import fallback (same pattern already used in `locomotion_elastica/env.py`).
