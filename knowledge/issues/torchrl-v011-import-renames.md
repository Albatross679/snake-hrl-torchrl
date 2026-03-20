---
name: TorchRL v0.11 spec class import renames (BoundedTensorSpec → Bounded, etc.)
description: TorchRL v0.11 renamed BoundedTensorSpec, CompositeSpec, and UnboundedContinuousTensorSpec to Bounded, Composite, and Unbounded
type: issue
status: resolved
severity: high
subtype: compatibility
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
tags: [torchrl, compatibility, v0.11, bug, imports]
aliases: []
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
