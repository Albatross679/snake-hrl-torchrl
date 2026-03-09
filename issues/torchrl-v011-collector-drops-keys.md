---
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
tags: [torchrl, compatibility, v0.11, bug, collector, diagnostics]
type: issue
status: resolved
severity: high
subtype: compatibility
---

# TorchRL v0.11 Collector Drops Unspecced Keys

## Problem

`SyncDataCollector` only preserves keys declared in env specs. Custom diagnostic keys (`v_g`, `dist_to_goal`, `theta_g`, `reward_velocity`, `reward_potential`, `goal_reached`, `starvation`, etc.) were being silently dropped during collection.

**Affected:** `locomotion_elastica/env.py` (diagnostic keys)

## Fix

Added all diagnostic keys to `observation_spec` as `UnboundedContinuousTensorSpec` entries. Also ensured `_reset` and `_step` always include these keys (not conditionally on `done`).
