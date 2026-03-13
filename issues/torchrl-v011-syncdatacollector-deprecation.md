---
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
description: SyncDataCollector is deprecated in TorchRL v0.11 and will be removed in v0.13
tags: [torchrl, compatibility, v0.11, deprecation, collector]
type: issue
status: open
severity: low
subtype: compatibility
---

# TorchRL v0.11 SyncDataCollector Deprecation

## Problem

`SyncDataCollector` is deprecated in TorchRL v0.11 and will be removed in v0.13. Currently works with a deprecation warning.

**Affected:** `src/trainers/ppo.py`

## Fix

Not yet fixed. Should migrate to `Collector` before upgrading to v0.13.
