---
name: Python bytecode cache prevented hotfix from loading
description: After editing collect_data.py to fix stall detection, restarting the collector still ran the old broken code due to cached .pyc files
type: issue
status: resolved
severity: medium
subtype: system
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, data-collection, python, debugging]
aliases: []
---

# Python Bytecode Cache Prevented Hotfix From Loading

## Symptom

After fixing the stall detection grace period in `collect_data.py`, the collector was killed and restarted. The new run exhibited the same false-positive stall behavior — the fix appeared to have no effect.

## Root Cause

Python caches compiled bytecode in `__pycache__/` directories as `.pyc` files. When a module is imported, Python checks the `.pyc` timestamp against the `.py` source. However, in certain conditions (rapid edit-kill-restart cycles, or when the `.pyc` mtime granularity masks the edit), Python may serve the stale cached bytecode instead of recompiling.

The `forkserver` start method compounds this: the server process is forked early and may have already cached module imports before the source was edited.

## Fix

1. Killed all running collector processes (`kill` + verified with `ps aux`)
2. Cleared all bytecode caches:
   ```bash
   find /home/coder/snake-hrl-torchrl -name "__pycache__" -exec rm -rf {} +
   ```
3. Restarted the collector — fix loaded correctly

## Lesson

When hotfixing code in a running multiprocessing pipeline:
1. Kill **all** processes (main + workers + forkserver)
2. Clear `__pycache__/` directories for the edited modules
3. Then restart

Alternatively, set `PYTHONDONTWRITEBYTECODE=1` during development to avoid caching entirely.
