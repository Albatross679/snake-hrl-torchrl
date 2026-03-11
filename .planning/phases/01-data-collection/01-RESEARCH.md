# Phase 1: Health Monitoring and Data Integrity - Research

**Researched:** 2026-03-09
**Domain:** Process health monitoring, data validation, graceful shutdown, alerting for multiprocess Python data collection pipeline
**Confidence:** HIGH

## Summary

Phase 1 adds health monitoring and data integrity to the existing multiprocess data collection pipeline (`aprx_model_elastica/collect_data.py`). The pipeline uses `multiprocessing.Process` with `forkserver` start method, 16 workers each running a PyElastica Cosserat rod simulation, and a monitoring loop in the main process with 30s polling. The current code already has the basic monitoring skeleton (worker `is_alive()` checks, shared counter, W&B metric logging) but lacks: per-worker status tracking, crash detection with exit codes, worker respawning, NaN/Inf filtering, atomic batch saves, signal-based graceful shutdown, W&B alerts, and structured event logging.

All required technologies are already available in the project environment: Python 3.12 stdlib (`multiprocessing`, `signal`, `json`, `tempfile`, `os`), `wandb` 0.25.0 (with `wandb.alert()` and `AlertLevel` enum), `psutil` 7.2.1 (for optional CPU monitoring), `numpy` (for NaN/Inf detection via `np.isfinite()`), and `pytest` 9.0.2. No new dependencies need to be installed.

**Primary recommendation:** Modify the existing `_multiprocess_collect()` loop and `_collection_loop()` / `collect_episode()` functions in-place rather than creating a separate monitor package. Phase 1's scope (health + integrity) is tightly coupled to the collection process itself. The external observer monitor (`python -m aprx_model_elastica.monitor`) should be a thin wrapper that attaches to the same W&B run and reads the JSON event log, not a full monitoring framework.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Monitor architecture: External observer process (`python -m aprx_model_elastica.monitor`) runs alongside the collector
- Communicates via filesystem signals (decided at roadmap creation)
- Polls every 30 seconds (matching existing poll interval in collect_data.py)

### Claude's Discretion
- **NaN/Inf filtering strategy:** Per-transition vs per-episode discard, where filtering happens (worker-side vs save-time), whether to log discarded data for debugging
- **Worker recovery behavior:** Fresh start vs resume on respawn, seed strategy for respawned workers, handling of incomplete batch data from crashed workers
- **Shutdown coordination:** Signal propagation mechanism, atomic batch save strategy, drain vs immediate stop
- **Event log schema:** JSON structure, file location, rotation policy
- **W&B alert thresholds:** What constitutes "high NaN rate", alert rate limiting to avoid overnight spam, severity levels for different event types
- **Stall detection criteria:** How to determine a worker is stalled (zero progress for N consecutive intervals), restart mechanism

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| HLTH-01 | Monitor reports per-worker alive/dead status every poll interval (30s) | Use `Process.is_alive()` + `Process.exitcode` per worker; track via per-worker `mp.Value` shared counters for progress; report in monitor stdout and W&B |
| HLTH-02 | Dead workers auto-detected and respawned with new seed within 60s | Check `exitcode is not None` (died) in monitor loop; spawn new `mp.Process` with fresh seed derived from `config.seed + worker_id * 137 + respawn_count * 7919`; reuse same `shared_counter` |
| HLTH-03 | Stalled workers (alive but zero progress for 2+ consecutive intervals) detected and restarted | Track per-worker transition deltas between polls using per-worker `mp.Value` counters; if delta == 0 for 2+ consecutive intervals (60s+), `terminate()` then respawn |
| HLTH-04 | NaN/Inf episodes discarded before saving, discard count logged | Add `np.all(np.isfinite(next_state))` check in `collect_episode()` after each step; on failure, discard entire episode (return empty); worker increments a per-worker NaN counter; logged to W&B |
| HLTH-05 | Graceful shutdown on SIGINT/SIGTERM preserves all completed batch files | Install signal handler in main process that sets `mp.Event`; workers check event in `_collection_loop`; on signal, workers flush current batch then exit cleanly; main waits with timeout |
| OBSV-01 | W&B alerts for worker death, stall, high NaN rate | Use `wandb.alert(title=..., text=..., level=AlertLevel.ERROR/WARN, wait_duration=300)` from the monitor process; rate limit per-alert-type |
| OBSV-04 | All monitoring events logged to structured JSON event log | Append JSONL (one JSON object per line) to `{save_dir}/events.jsonl`; schema: `{timestamp, event_type, worker_id, details, severity}` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `multiprocessing` | stdlib (Python 3.12) | Worker process management, shared state | Already in use; `Process`, `Value`, `Event` are the correct primitives for this architecture |
| `signal` | stdlib (Python 3.12) | SIGINT/SIGTERM graceful shutdown | Already used in PPOTrainer; same pattern applies to collection |
| `numpy` | >= 2.0.0 (installed) | NaN/Inf detection via `np.isfinite()` | Already used everywhere; `np.all(np.isfinite(arr))` is the standard validation check |
| `wandb` | 0.25.0 (installed) | Alerts and metric logging | Already integrated; `wandb.alert()` API verified working with `AlertLevel.INFO/WARN/ERROR` |
| `json` | stdlib | Structured event log (JSONL format) | No external dependency needed; one JSON object per line is the standard for structured logs |
| `os` / `tempfile` | stdlib | Atomic file writes (tmp + rename) | Pattern already used in `PPOTrainer.save_checkpoint()`; use `os.replace()` for atomicity |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `psutil` | 7.2.1 (installed) | Per-worker CPU monitoring (optional stall heuristic) | Only if zero-progress detection is insufficient; `psutil.Process(pid).cpu_percent()` gives per-process CPU usage |
| `time` | stdlib | `time.monotonic()` for elapsed time, `time.time()` for wall-clock timestamps in event log | Always |
| `datetime` | stdlib | ISO-8601 timestamp formatting for event log | Always |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Per-worker `mp.Value` counters | `mp.Array` for all workers | Array is more compact but `Value` per worker is simpler to manage when respawning workers (new Value per worker) |
| JSONL event log | Python `logging` with `python-json-logger` | `logging` adds framework overhead; direct JSONL append with `json.dumps()` is simpler and sufficient for single-writer append-only log |
| `mp.Event` for shutdown | Signal file on disk | `mp.Event` is faster and avoids filesystem polling; signal file is for the external monitor process, not for worker coordination |
| `os.replace()` for atomic save | `atomicwrites` library | `os.replace()` is stdlib and atomically replaces on same filesystem (Linux); no need for external dependency |

**Installation:**
```bash
# No new packages needed. All dependencies already installed.
```

## Architecture Patterns

### Recommended Project Structure
```
aprx_model_elastica/
  collect_data.py       # MODIFY: add per-worker counters, NaN filtering, atomic saves, signal handling, event logging
  collect_config.py     # MODIFY: add monitor-related config fields (nan_threshold, stall_intervals, alert settings)
  monitor.py            # NEW: external observer process that reads event log and shows status
  __main__.py           # MODIFY: add "monitor" command dispatch
```

### Pattern 1: Per-Worker Shared Counters for Health Tracking
**What:** Each worker gets its own `mp.Value('l', 0)` counter that it increments after each episode. The monitoring loop reads all per-worker counters each poll to compute deltas and detect stalls.
**When to use:** When you need per-worker progress visibility without IPC overhead.
**Example:**
```python
# In _multiprocess_collect():
worker_counters = [mp.Value('l', 0) for _ in range(num_workers)]

# In _worker_fn():
def _worker_fn(worker_id, config, shared_counter, worker_counter, ...):
    # After each episode:
    with worker_counter.get_lock():
        worker_counter.value += n_steps

# In monitoring loop:
for i, (p, wc) in enumerate(zip(workers, worker_counters)):
    current = wc.value
    delta = current - prev_worker_counts[i]
    if delta == 0:
        stall_count[i] += 1
    else:
        stall_count[i] = 0
    prev_worker_counts[i] = current
```

### Pattern 2: NaN/Inf Filtering at Episode Level in Worker
**What:** After `collect_episode()` completes, check all states and next_states with `np.all(np.isfinite(...))`. If any transition in the episode has NaN/Inf, discard the entire episode. This is simpler and safer than per-transition filtering (which could leave partial episodes).
**When to use:** Always -- PyElastica's explicit integrator can diverge, producing NaN that propagates through all subsequent steps.
**Example:**
```python
# In _collection_loop(), after collect_episode():
ep_data = collect_episode(...)

# Validate episode data
states_finite = np.all(np.isfinite(ep_data["states"]))
next_states_finite = np.all(np.isfinite(ep_data["next_states"]))
actions_finite = np.all(np.isfinite(ep_data["actions"]))

if not (states_finite and next_states_finite and actions_finite):
    nan_discard_count += 1
    # Log but don't save -- silently discard
    continue  # skip to next episode
```
**Recommendation:** Per-episode discard (not per-transition) because once NaN appears in a Cosserat rod step, all subsequent steps in that episode are also invalid. Worker-side filtering (in `_collection_loop`) is better than save-time filtering because it prevents NaN data from ever entering batch buffers.

### Pattern 3: Atomic Batch Save with tmp+rename
**What:** Write batch data to a temporary file in the same directory, then atomically rename to the final path. Prevents truncated files from crashes or shutdown during save.
**When to use:** Always -- replaces the current direct `torch.save(data, path)`.
**Example:**
```python
# In _save_batch_pt():
path = save_dir / f"{prefix}_{batch_idx:04d}.pt"
tmp_path = save_dir / f"{prefix}_{batch_idx:04d}.pt.tmp"
torch.save(data, str(tmp_path))
os.replace(str(tmp_path), str(path))  # Atomic on same filesystem
```
**Source:** Same pattern already used in `src/trainers/ppo.py:save_checkpoint()` (lines 501-508).

### Pattern 4: Graceful Shutdown via mp.Event + Signal Handler
**What:** Main process installs SIGINT/SIGTERM handler that sets a shared `mp.Event`. Workers check the event in their collection loop. On shutdown: workers finish current episode, flush any pending batch, then exit.
**When to use:** Always -- prevents data loss on Ctrl+C or `kill`.
**Example:**
```python
# In _multiprocess_collect():
shutdown_event = mp.Event()

def _signal_handler(signum, frame):
    print(f"\n{signal.Signals(signum).name} received, requesting graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# In _collection_loop():
while not _target_reached() and not shutdown_event.is_set():
    ...  # collect episodes

# After loop: save remaining batch
if episodes_in_batch > 0:
    _save_batch(...)
```
**Source:** Reference pattern from `src/trainers/ppo.py:_signal_handler()` (lines 162-166).

### Pattern 5: W&B Alert with Rate Limiting
**What:** Use `wandb.alert()` for critical events with `wait_duration` to prevent spam during overnight runs.
**When to use:** Worker death, worker stall, high NaN rate.
**Example:**
```python
from wandb import AlertLevel

# Worker death alert
wandb.alert(
    title="Worker Died",
    text=f"Worker {worker_id} (PID {pid}) exited with code {exitcode}. Respawning.",
    level=AlertLevel.ERROR,
    wait_duration=300,  # 5 min between alerts of same title
)

# High NaN rate alert
wandb.alert(
    title="High NaN Rate",
    text=f"NaN discard rate {nan_rate:.1%} exceeds threshold {threshold:.1%}",
    level=AlertLevel.WARN,
    wait_duration=600,  # 10 min between alerts
)
```
**Verified:** `wandb.alert()` signature is `(title: str, text: str, level: str | AlertLevel | None = None, wait_duration: int | float | timedelta | None = None)`. AlertLevel enum values: `INFO`, `WARN`, `ERROR`. Title must be < 64 characters.

### Pattern 6: JSONL Event Log
**What:** Append one JSON object per line to `{save_dir}/events.jsonl`. Each event has a fixed schema. The external monitor process reads this file to show status.
**When to use:** All monitoring events (worker death, respawn, stall, NaN discard, shutdown, alert sent).
**Schema:**
```python
import json, datetime

def log_event(event_log_path, event_type, severity, worker_id=None, details=None):
    event = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_type": event_type,  # "worker_died", "worker_respawned", "worker_stalled", "nan_discard", "shutdown", "alert_sent"
        "severity": severity,      # "info", "warn", "error"
        "worker_id": worker_id,    # int or None
        "details": details or {},  # arbitrary dict
    }
    with open(event_log_path, "a") as f:
        f.write(json.dumps(event) + "\n")
```

### Anti-Patterns to Avoid
- **Embedding heavy monitoring in workers:** Workers should only increment counters and filter NaN. All alerting, event logging, and status reporting happens in the main process monitoring loop.
- **Using `multiprocessing.Pool` for worker management:** The project already uses raw `Process` objects with `forkserver`. Pool would complicate respawning with custom state (per-worker seeds, Sobol samplers).
- **Catching bare `Exception` around batch saves:** The current code silently swallows W&B errors with `except Exception: pass`. For batch saves, failures must be logged and reported, not swallowed.
- **Checking `is_alive()` without `exitcode`:** A process can be not-alive but have `exitcode=None` if it was never started. Always check `exitcode is not None` to confirm a process actually died vs was never launched.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NaN/Inf detection | Custom per-element loop | `np.all(np.isfinite(array))` | Vectorized, handles NaN, Inf, -Inf in one call; 100x faster than Python loop on 124-dim state arrays |
| Atomic file write | Custom fsync+rename | `tempfile.NamedTemporaryFile` + `os.replace()` | `os.replace()` is POSIX-guaranteed atomic on same filesystem; already proven in `PPOTrainer.save_checkpoint()` |
| W&B alerting | Custom Slack/email integration | `wandb.alert(title, text, level, wait_duration)` | Built-in rate limiting, Slack/email routing, already configured in project |
| Process health check | Custom `/proc` parsing | `Process.is_alive()` + `Process.exitcode` + optionally `psutil.Process(pid).cpu_percent()` | Stdlib is sufficient; psutil only needed if CPU-based stall detection is wanted |
| Structured logging | `logging` module with custom formatter | Direct `json.dumps()` to JSONL file | Single writer, append-only, no rotation needed for <24h runs; `logging` framework is overkill |
| Signal handling | Custom signal-file polling | `signal.signal()` + `mp.Event()` | In-process event is instant; file polling has latency and filesystem overhead |

**Key insight:** Every component of Phase 1 has a stdlib or already-installed solution. No new dependencies are needed. The complexity is in the coordination logic (when to respawn, when to alert, how to drain), not in the individual primitives.

## Common Pitfalls

### Pitfall 1: Worker Respawn Race Condition
**What goes wrong:** When respawning a dead worker, the new process gets the same `worker_id` and `shared_counter` reference, but the old process's batch file prefix (`batch_w{id:02d}`) may still have an in-progress `.tmp` file or the batch index scanner picks up the old process's files.
**Why it happens:** The respawned worker calls `_find_next_batch_idx()` which scans the directory. If the dead worker was mid-save, a `.tmp` file exists but the final `.pt` does not, so the index is correct. But if the dead worker completed a save but died before incrementing its counter, the new worker and the monitoring loop disagree on the count.
**How to avoid:** Have the respawned worker call `_find_next_batch_idx()` at startup (already done in `_collection_loop`). Clean up any `.tmp` files for that worker's prefix before respawning. The batch index scanner only looks at final `.pt` files, so `.tmp` files do not interfere.
**Warning signs:** Duplicate batch file names (overwrite), or gaps in batch indices.

### Pitfall 2: Signal Handler in Forked Children
**What goes wrong:** With `forkserver`, child processes inherit a clean state but signal handlers installed in the parent are NOT inherited (unlike `fork`). However, SIGINT is still delivered to the entire process group when Ctrl+C is pressed in a terminal, causing all children to receive KeyboardInterrupt independently.
**Why it happens:** Python's default SIGINT handler raises `KeyboardInterrupt`. With `forkserver`, children get the default handler, not the parent's custom one.
**How to avoid:** Use `mp.Event` as the shutdown signal, not signals. In workers, catch `KeyboardInterrupt` and treat it as a shutdown request. The parent's signal handler sets the `mp.Event` which workers also check. Optionally, set `signal.signal(signal.SIGINT, signal.SIG_IGN)` in workers so only the parent handles the signal, then use the Event for coordination.
**Warning signs:** Workers crashing on Ctrl+C instead of draining their current batch.

### Pitfall 3: Shared Counter Torn Reads
**What goes wrong:** Reading `shared_counter.value` without the lock can give a torn read on some architectures (though unlikely on x86-64 for a 64-bit long).
**Why it happens:** The current code only locks on writes (`with shared_counter.get_lock(): shared_counter.value += n_steps`) but reads without lock.
**How to avoid:** For the monitoring loop's read-only access to per-worker counters, torn reads on a `ctypes.c_long` are not a practical concern on x86-64 Linux. However, for correctness, wrap reads in `with counter.get_lock():` when precision matters (e.g., the final count after shutdown).

### Pitfall 4: W&B Alert Title Length Limit
**What goes wrong:** `wandb.alert()` requires title < 64 characters. Including long details like full paths or PIDs in the title causes a silent failure or truncation.
**Why it happens:** W&B API limitation.
**How to avoid:** Keep titles short and categorical (e.g., "Worker Died", "Worker Stalled", "High NaN Rate"). Put details in the `text` parameter which has no meaningful length limit.

### Pitfall 5: Event Log File Corruption on Crash
**What goes wrong:** If the main process crashes while writing to `events.jsonl`, the last line may be a partial JSON string, causing JSON parse errors when reading the log.
**Why it happens:** Python's `file.write()` is not atomic for multi-byte writes.
**How to avoid:** Write each event as a complete line ending with `\n`. Readers should skip lines that fail to parse (`json.loads()` in a try/except). Since events are append-only and each is a complete JSON object on one line, at most one event is lost on crash.

## Code Examples

### Worker Respawn Implementation
```python
# Source: Derived from existing _multiprocess_collect() pattern
def _respawn_worker(worker_id, config, shared_counter, worker_counter,
                    shutdown_event, existing_episode_offset, respawn_count,
                    save_dir):
    """Respawn a dead worker with a new seed."""
    # Clean up any .tmp files from the dead worker
    prefix = f"batch_w{worker_id:02d}"
    for tmp in save_dir.glob(f"{prefix}_*.tmp"):
        tmp.unlink(missing_ok=True)

    # New seed ensures different exploration from the dead worker
    new_seed_offset = respawn_count * 7919  # large prime to avoid seed collision

    p = mp.Process(
        target=_worker_fn,
        args=(worker_id, config, shared_counter, worker_counter,
              shutdown_event, existing_episode_offset, new_seed_offset),
        name=f"collector-{worker_id}-r{respawn_count}",
    )
    p.start()
    return p
```

### NaN Filtering in Collection Loop
```python
# Source: Integration into existing _collection_loop()
ep_data = collect_episode(env, policy=policy, use_random=use_random, ...)

# Validate: discard entire episode if any NaN/Inf
if not (np.all(np.isfinite(ep_data["states"])) and
        np.all(np.isfinite(ep_data["next_states"])) and
        np.all(np.isfinite(ep_data["actions"]))):
    nan_discard_count += 1
    if shared_nan_counter is not None:
        with shared_nan_counter.get_lock():
            shared_nan_counter.value += 1
    continue  # Skip to next episode; do NOT increment transition counters
```

### Graceful Shutdown Signal Handler
```python
# Source: Adapted from src/trainers/ppo.py signal handling pattern
shutdown_event = mp.Event()
original_sigint = signal.getsignal(signal.SIGINT)
original_sigterm = signal.getsignal(signal.SIGTERM)

def _shutdown_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    print(f"\n{sig_name} received, requesting graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)

# In monitoring loop:
try:
    while not shutdown_event.is_set():
        alive = [p for p in workers if p.is_alive()]
        if not alive:
            break
        # ... poll workers, check health ...
        shutdown_event.wait(timeout=poll_interval)  # replaces time.sleep()
finally:
    signal.signal(signal.SIGINT, original_sigint)
    signal.signal(signal.SIGTERM, original_sigterm)
```

### External Monitor Process (thin reader)
```python
# aprx_model_elastica/monitor.py
"""External monitor: reads event log and shows per-worker status."""
import json, time, sys
from pathlib import Path

def tail_events(save_dir: str, poll_interval: float = 30.0):
    """Tail the events.jsonl file and display per-worker status."""
    event_log = Path(save_dir) / "events.jsonl"
    worker_status = {}  # worker_id -> {"status": alive/dead/stalled, "last_event": ...}

    pos = 0
    while True:
        if event_log.exists():
            with open(event_log) as f:
                f.seek(pos)
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        _update_status(worker_status, event)
                    except json.JSONDecodeError:
                        pass  # Skip partial lines
                pos = f.tell()

        # Display status table
        _print_status_table(worker_status)
        time.sleep(poll_interval)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `wandb.log()` only for metrics | `wandb.alert()` for critical events | wandb 0.14.0+ (stable since 2023) | Enables proactive Slack/email notifications without custom integration |
| `multiprocessing.Pool` for workers | Raw `Process` with manual management | Project design choice | Better control over per-worker state, seeds, and respawn logic |
| `torch.save()` direct to path | `tempfile` + `os.replace()` atomic pattern | Python best practice | Prevents truncated batch files on crash/shutdown |
| Print-based logging | JSONL structured event log | Industry standard 2023+ | Machine-parseable, enables external monitor process and post-hoc analysis |

**Deprecated/outdated:**
- `multiprocessing.Pool` automatic worker restart: exists but undocumented and does not preserve custom per-worker state
- `atomicwrites` library: maintainer deprecated it; `os.replace()` is the stdlib solution
- `signal.SIGCHLD` handler for child death detection: unnecessary complexity when polling `is_alive()` every 30s suffices

## Open Questions

1. **W&B Run Sharing Between Collector and Monitor**
   - What we know: The collector already creates a W&B run. The external monitor process needs to log metrics and send alerts.
   - What's unclear: Whether the monitor should resume the same W&B run (via `wandb.init(resume="must", id=run_id)`) or create a linked run in the same project.
   - Recommendation: Write the W&B run ID to a file (`{save_dir}/.wandb_run_id`). The monitor reads it and resumes the same run. This avoids duplicate runs in the dashboard. If resume fails, monitor operates in local-only mode.

2. **Stall Detection Threshold Sensitivity**
   - What we know: A worker running PyElastica at ~57 FPS produces ~57 transitions/second. An episode of 500 steps takes ~8.8 seconds. With 30s polling, a healthy worker should produce ~1,700 transitions between polls.
   - What's unclear: What "zero progress for 2+ intervals" means precisely -- should this be literally zero delta, or below some minimum threshold?
   - Recommendation: Use literally zero delta (exactly 0 new transitions in 60s). A slow-but-alive worker producing even 1 transition/minute is not stalled. If the per-worker counter is 0 for 2 consecutive polls and `is_alive()` returns True, that is a true stall (deadlock or infinite loop in PyElastica).

3. **NaN Discard vs. Episode Truncation**
   - What we know: NaN typically appears when perturbation pushes the rod into an unstable configuration. The NaN propagates through all subsequent steps.
   - What's unclear: Whether it is worth saving the pre-NaN transitions from the same episode (steps before divergence).
   - Recommendation: Discard the entire episode. Pre-NaN transitions from a diverging simulation may already be in a physically unreliable regime, and partial episodes complicate episode ID tracking. Simplicity wins for Phase 1.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (testpaths = ["tests"]) |
| Quick run command | `python3 -m pytest tests/test_monitor.py -x -v` |
| Full suite command | `python3 -m pytest tests/ -x -v --timeout=60` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HLTH-01 | Per-worker alive/dead status reported each poll | unit | `python3 -m pytest tests/test_monitor.py::test_per_worker_status -x` | Wave 0 |
| HLTH-02 | Dead worker detected and respawned within 60s | integration | `python3 -m pytest tests/test_monitor.py::test_worker_respawn -x` | Wave 0 |
| HLTH-03 | Stalled worker detected after 2+ zero-progress intervals | unit | `python3 -m pytest tests/test_monitor.py::test_stall_detection -x` | Wave 0 |
| HLTH-04 | NaN/Inf episodes discarded, count tracked | unit | `python3 -m pytest tests/test_monitor.py::test_nan_filtering -x` | Wave 0 |
| HLTH-05 | Graceful shutdown preserves batch files | integration | `python3 -m pytest tests/test_monitor.py::test_graceful_shutdown -x` | Wave 0 |
| OBSV-01 | W&B alerts fire for death/stall/NaN events | unit (mock) | `python3 -m pytest tests/test_monitor.py::test_wandb_alerts -x` | Wave 0 |
| OBSV-04 | Structured JSON event log written for all events | unit | `python3 -m pytest tests/test_monitor.py::test_event_log -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/test_monitor.py -x -v`
- **Per wave merge:** `python3 -m pytest tests/ -x -v --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_monitor.py` -- new file covering all 7 requirements above
- [ ] `tests/conftest.py` -- may need shared fixtures for mock W&B, temp save directories, fake worker processes
- [ ] No framework install needed (pytest 9.0.2 already installed)

## Sources

### Primary (HIGH confidence)
- **Codebase analysis:** `aprx_model_elastica/collect_data.py` (993 lines) -- existing monitoring loop, worker management, batch save functions
- **Codebase analysis:** `src/trainers/ppo.py` (570 lines) -- signal handler pattern, atomic save pattern
- **Codebase analysis:** `aprx_model_elastica/collect_config.py` -- dataclass config pattern to extend
- **Python 3.12 docs:** `multiprocessing.Process.exitcode`, `multiprocessing.Event`, `signal.signal()`, `os.replace()`
- **Verified locally:** `wandb.alert()` signature: `(title: str, text: str, level: str | AlertLevel | None, wait_duration: int | float | timedelta | None)`. AlertLevel: `INFO`, `WARN`, `ERROR`. Title max 64 chars.
- **Verified locally:** `psutil` 7.2.1 installed, `pytest` 9.0.2 installed, numpy `np.isfinite()` handles NaN, Inf, -Inf

### Secondary (MEDIUM confidence)
- [W&B Alert Documentation](https://docs.wandb.ai/guides/runs/alert) -- alert API, rate limiting via `wait_duration`
- [Python multiprocessing docs](https://docs.python.org/3/library/multiprocessing.html) -- `Process.is_alive()`, `Process.exitcode`, `Event`, `Value`
- [Python signal docs](https://docs.python.org/3/library/signal.html) -- signal handling in multiprocess contexts
- [Atomic file writes best practices](https://python.plainenglish.io/simple-safe-atomic-writes-in-python3-44b98830a013) -- tmp+rename pattern
- [JSONL for structured logging](https://ndjson.com/use-cases/log-processing/) -- one JSON object per line standard
- [Graceful shutdown with multiprocessing](https://the-fonz.gitlab.io/posts/python-multiprocessing/) -- Event-based shutdown coordination

### Tertiary (LOW confidence)
- [Python bug #38084](https://bugs.python.org/issue38084) -- multiprocessing cannot recover from crashed worker (confirms manual respawn needed)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and verified working; no new dependencies
- Architecture: HIGH -- patterns derived from existing codebase (`PPOTrainer` signal handling, `_multiprocess_collect` monitoring loop)
- Pitfalls: HIGH -- identified from codebase analysis and existing project pitfalls research (`.planning/research/PITFALLS.md`)
- Validation: HIGH -- pytest infrastructure exists; test patterns are straightforward (mock processes, temp directories)

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable domain; stdlib + wandb API unlikely to change)
