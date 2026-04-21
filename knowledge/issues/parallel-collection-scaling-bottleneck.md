---
name: Parallel Data Collection Scaling Bottleneck
description: L3 cache contention and memory bandwidth saturation limit parallel worker scaling beyond ~16 workers
type: issue
status: resolved
severity: low
subtype: performance
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, data-collection, parallelism, performance, hardware]
aliases: [cache-bottleneck, memory-bandwidth-scaling]
---

# Parallel Data Collection Scaling Bottleneck

## Problem

Parallel surrogate data collection with `multiprocessing.Process` workers scales near-linearly up to ~16 workers but shows diminishing returns beyond that. Adding more workers does not proportionally increase throughput.

## Hardware Specs

**CPU**: 2x Intel Xeon E5-2680 v3 (Haswell-EP), 12 cores / 24 threads each = 48 logical CPUs

| Cache Level | Size | Scope |
|-------------|------|-------|
| L1d | 32 KB per core | Private per core |
| L1i | 32 KB per core | Private per core |
| L2 | 256 KB per core | Private per core |
| L3 | 30 MB per socket (60 MB total) | Shared across 12 cores per socket |

**Memory bandwidth**: DDR4-2133, 4 channels per socket. Theoretical max ~68 GB/s per socket (~136 GB/s total). Practical sustained: ~70-95 GB/s.

**NUMA topology**: 2 sockets, each with its own L3 cache and memory channels.
- Socket 0: CPUs 0-11, 24-35
- Socket 1: CPUs 12-23, 36-47
- Cross-socket access goes through QPI interconnect (higher latency).

## Root Cause Analysis

The bottleneck is **not** CPU cores or L1/L2 cache. Those are private per core and don't contend between workers.

The bottleneck is the **shared L3 cache and memory bus**:

1. **L3 cache contention**: Each socket has 30 MB of L3 shared among 12 cores. PyElastica's Cosserat rod solver performs many small array operations. When 12+ workers per socket are active, their working sets compete for L3 space, causing evictions and cache misses.

2. **Memory bandwidth saturation**: When L3 misses increase, all workers hit main RAM simultaneously. The memory bus has finite bandwidth (~68 GB/s per socket). PyElastica's array operations (position updates, force computation) are memory-bound — the CPU finishes arithmetic faster than it can fetch the next array.

3. **NUMA cross-socket penalty**: If the OS schedules a worker on socket 0 but its memory is allocated on socket 1, access goes through the QPI interconnect at higher latency.

### Scaling behavior

| Workers | Bottleneck | Expected scaling |
|---------|-----------|-----------------|
| 1-8 | None — private L1/L2 sufficient, ample L3 space | Near-linear |
| 8-16 | L3 contention begins on each socket | Slightly sublinear |
| 16-32 | Memory bandwidth saturation on both sockets | Diminishing returns |
| 32-48 | Full saturation — adding workers barely helps | Plateau |

### Per-worker data footprint

Each PyElastica worker's rod state is small (~10 KB for a 20-element rod), well within L2 (256 KB). However, the solver touches many temporary arrays during integration (forces, torques, position updates), expanding the effective working set. At high worker counts, the aggregate working set exceeds L3 capacity.

## Resolution

Default `num_workers=16` in `aprx_model_elastica/collect_config.py` is well-matched to this hardware:
- ~8 workers per socket keeps L3 contention manageable
- Near-linear scaling achieved (smoke test: 2 workers = 1.7x speedup)
- Projected ~270 FPS at 16 workers vs ~17 FPS single-threaded

No code change needed — this is a hardware-imposed scaling limit. The surrogate model approach sidesteps this entirely: collect data slowly on CPU once, then train and run the surrogate on GPU at orders-of-magnitude higher throughput.

## Related

- [surrogate-parallel-data-collection.md](../experiments/surrogate-parallel-data-collection.md) — experiment with design and smoke test results
- `aprx_model_elastica/collect_config.py` — `num_workers` config field
- `aprx_model_elastica/collect_data.py` — multiprocess worker implementation
