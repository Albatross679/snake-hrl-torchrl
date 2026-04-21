---
name: Parallel arch sweep — rollout DataLoader thread crash at epoch 21
description: Two of five parallel arch sweep runs (A3, A5) crashed with thread/worker errors when all processes simultaneously spun up their rollout DataLoaders at epoch 21.
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-10
updated: 2026-03-10
tags: [surrogate, arch-sweep, parallel-training, dataloader, threads]
aliases: []
---

## Summary

During parallel arch sweep execution (phase 3.1, plan 03.1-02), two of five simultaneous training runs crashed at epoch 25 (shortly after rollout loss activated at epoch 21).

## Affected Runs

- `arch_A3_rw0.3` (pid 112769) — crashed at epoch 25 evaluation
- `arch_A5_rw0.3_s16` (pid 112771) — crashed at epoch 25 training step

## Error Messages

**A3:**
```
RuntimeError: can't start new thread
  File "torch/utils/data/dataloader.py", line 1652, in _shutdown_workers
    self._worker_result_queue.put((None, None))
  File "multiprocessing/queues.py", line 192, in _start_thread
    self._thread.start()
```

**A5:**
```
RuntimeError: DataLoader worker (pid 128433) exited unexpectedly with exit code 1.
RuntimeError: DataLoader worker (pid(s) 128433) exited unexpectedly
```

## Root Cause

All 5 parallel processes entered the rollout loss phase simultaneously (epoch 21), each spawning a `TrajectoryDataset` + `DataLoader(num_workers=2)`. The simultaneous creation of 10 new DataLoader worker processes under existing thread/process load caused a thread spawn failure in two processes.

System resources were not the bottleneck:
- RAM: 209 GB available (each process uses ~3.8 GB base, ~4.9 GB with rollout dataset)
- System thread limit: 2,060,535 (only 1,062 in use)
- The other 3 runs (A1, A4, B1) survived without issue

The crash was likely a timing/contention race: all processes hit `epoch == rollout_start_epoch` simultaneously and flooded the OS with thread creation requests.

## Resolution

Restarted A3 and A5 as independent background processes. By this point, A1/A4/B1 are well past epoch 21 so the simultaneous contention will not recur.

## Prevention

For future parallel sweeps with rollout loss: stagger `rollout_start_epoch` per run or reduce `num_workers` for the rollout DataLoader from 2 to 1 when running 5+ parallel processes.
