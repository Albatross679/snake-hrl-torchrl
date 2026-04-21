---
name: SAC 500 ParallelEnv workers crash with BrokenPipeError
description: TorchRL ParallelEnv with 500 spawn workers overwhelms multiprocessing pipes
type: issue
status: resolved
severity: high
subtype: system
created: 2026-03-20
updated: 2026-03-20
tags: [sac, parallelenv, torchrl, multiprocessing, choi2025]
aliases: [broken-pipe-500-envs]
---

## Symptom

SAC follow_target training with `--num-envs 500` crashes immediately at 0% progress. Log shows hundreds of `BrokenPipeError` and `ConnectionResetError` from TorchRL's `ParallelEnv` spawn workers. All workers fail during the `child_pipe.send("started")` handshake.

Also produces warning: `2504 leaked semaphore objects to clean up at shutdown`.

## Root Cause

Python's multiprocessing spawn mode cannot reliably handle 500 simultaneous subprocess pipes. The parent process either times out or cannot service all 500 connection handshakes, causing workers to get `BrokenPipeError` when they try to signal readiness. This is not an OOM issue (127Gi RAM free), not a file descriptor limit (ulimit 1M), but a fundamental scaling limitation of `multiprocessing.Pipe` at high worker counts.

The Choi et al. (2025) paper specifies 500 parallel environments, but their implementation likely used a different parallelism strategy (e.g., vectorized envs, GPU-side batching, or a different framework).

## Fix Applied

Restarted with `--num-envs 100` which works reliably. Training runs at ~1030 it/s with 100 envs.

## Files Modified

None — runtime parameter change only (`--num-envs 100` instead of `--num-envs 500`).

## Notes

- Original run dir: `output/fixed_follow_target_sac_lr1e3_500envs_20260320_122730` (empty metrics)
- New run dir: `output/fixed_follow_target_sac_lr1e3_100envs_20260320_124053`
- W&B run: `hk8judy9`
