---
name: Surrogate data collection overwrites on re-run
description: >
  Running collect_data.py twice to the same save_dir overwrites existing batch
  files instead of appending, causing silent data loss. Episode IDs also collide
  across runs, breaking the train/val episode-level split.
type: issue
status: resolved
severity: high
subtype: system
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, data-collection, data-integrity]
aliases: []
---

## Problem

`collect_data.py` had no awareness of pre-existing data in the output directory:

1. **Batch file overwrite**: Both batch index counters and file names (`batch_w00_0000.pt`,
   `batch_0000.pt`, etc.) always started at 0. A second collection run to the same
   `--save-dir` silently overwrote the first run's files.

2. **Episode ID collision**: Worker episode IDs were computed as
   `worker_id * 10_000_000`, with no offset for existing data. If two runs wrote to the
   same directory, `SurrogateDataset` would see duplicate episode IDs. Since the
   train/val split is by episode, this could leak validation episodes into training.

## Impact

- Users collecting data incrementally (e.g., 500K random → 500K on-policy) would lose
  the first batch entirely.
- Merging datasets required manual file renaming — error-prone and undocumented.

## Fix

Added two helper functions in `collect_data.py`:

- **`_find_next_batch_idx(save_dir, prefix, save_format)`**: Scans for existing batch
  files matching the prefix pattern, returns `max_index + 1` so new files get unique
  names.
- **`_find_max_episode_id(save_dir)`**: Scans all existing `.pt` and `.parquet` batch
  files, returns `max_episode_id + 1` so new episode IDs don't collide.

Both `_single_process_collect` and `_multiprocess_collect` now:
1. Detect existing data and log "Append mode" with the offset.
2. Pass the episode offset to the collection loop / workers.
3. Start batch file indices from where the previous run left off.

## Usage

```bash
# First run: collects 500K random transitions
python3 -m aprx_model_elastica.collect_data \
    --num-transitions 500000 --save-dir data/surrogate

# Second run: appends 500K on-policy transitions (no overwrite)
python3 -m aprx_model_elastica.collect_data \
    --num-transitions 500000 --save-dir data/surrogate \
    --policy-checkpoint output/best.pt
```

Both runs' data is seamlessly loaded by `SurrogateDataset` with correct episode IDs.
