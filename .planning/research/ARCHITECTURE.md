# Architecture Research

**Domain:** Autonomous data collection monitoring for neural surrogate model training
**Researched:** 2026-03-09
**Confidence:** HIGH

## System Overview

```
+===========================================================================+
|                     ORCHESTRATION LAYER                                     |
|  +---------------------+   +---------------------+   +------------------+  |
|  | Monitor Scheduler   |   | Stop Condition      |   | Issue Tracker /  |  |
|  | (periodic timer)    |   | Evaluator           |   | Doc Writer       |  |
|  +----------+----------+   +----------+----------+   +--------+---------+  |
|             |                         |                        |           |
+=============|=========================|========================|===========+
              |                         |                        |
+-------------|=========================|========================|-----------+
|             v          HEALTH CHECK LAYER                      v           |
|  +---------------------+   +---------------------+   +------------------+ |
|  | Process Health      |   | Data Quality         |   | Resource Monitor | |
|  | Checker             |   | Checker              |   | (disk, CPU, mem) | |
|  +----------+----------+   +----------+----------+   +--------+---------+ |
|             |                         |                        |           |
+=============|=========================|========================|===========+
              |                         |                        |
+-------------|=========================|========================|-----------+
|             v          METRICS LAYER                           v           |
|  +---------------------+   +---------------------+   +------------------+ |
|  | Throughput Tracker   |   | Coverage Analyzer   |   | W&B Logger       | |
|  | (FPS, transitions)  |   | (state-action grid) |   | (remote metrics) | |
|  +----------+----------+   +----------+----------+   +--------+---------+ |
|             |                         |                        |           |
+=============|=========================|========================|===========+
              |                         |                        |
+-------------|=========================|========================|-----------+
|             v          DATA LAYER                              v           |
|  +---------------------+   +---------------------+   +------------------+ |
|  | Batch Files (.pt)   |   | Health State File   |   | Monitoring Logs  | |
|  | data/surrogate/     |   | (.json checkpoint)  |   | (issues/, logs/) | |
|  +---------------------+   +---------------------+   +------------------+ |
+===========================================================================+

              EXISTING COLLECTION PIPELINE (unchanged)
+===========================================================================+
|  +---------------------+                                                   |
|  | collect_data.py     |----> Worker Processes (16x) ---> Batch Files     |
|  | (main process +     |      (1 env per worker)          (.pt/.parquet)  |
|  |  monitoring loop)   |                                                   |
|  +---------------------+                                                   |
+===========================================================================+
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Monitor Scheduler | Periodically trigger health checks on a configurable interval | Python timer/sleep loop or cron-like scheduling in a standalone script |
| Stop Condition Evaluator | Decide if collection should stop based on multi-criteria (min time, min samples, coverage target) | Pure function that reads metrics and returns CONTINUE/STOP |
| Process Health Checker | Verify worker processes are alive, detect crashes/hangs, measure FPS degradation | Read shared counter + check `Process.is_alive()` + FPS delta analysis |
| Data Quality Checker | Validate batch files for NaN/Inf, shape consistency, value range sanity | Load recent batch files, run statistical checks |
| Coverage Analyzer | Track state-action space coverage on a discretized grid, report fill percentage | Histogram binning over summary features (4D state + 5D action), compute occupancy |
| Resource Monitor | Check disk space, memory usage, CPU utilization | `shutil.disk_usage()`, `psutil` for memory/CPU |
| Throughput Tracker | Track instantaneous and rolling FPS, compare against baseline | Already exists in `_multiprocess_collect()` monitoring loop |
| W&B Logger | Push health metrics to Weights & Biases for remote dashboard | Already exists; extend with coverage and health metrics |
| Health State File | Persist monitoring state between checks for crash recovery | JSON file updated each monitoring cycle |
| Issue Tracker / Doc Writer | Create issue/log Markdown files when problems are detected or fixed | Write to `issues/` and `logs/` directories following project conventions |

## Recommended Project Structure

```
aprx_model_elastica/
|-- collect_data.py           # Existing: data collection with worker processes
|-- collect_config.py         # Existing: collection configuration dataclass
|-- monitor/                  # NEW: autonomous monitoring package
|   |-- __init__.py
|   |-- runner.py             # Monitor scheduler: periodic check loop
|   |-- health.py             # Process health checker + resource monitor
|   |-- coverage.py           # State-action coverage analyzer
|   |-- quality.py            # Data quality validation (NaN, range, shape)
|   |-- stop_conditions.py    # Multi-criteria stop condition evaluator
|   |-- diagnostics.py        # Auto-diagnosis and fix routines
|   |-- reporter.py           # W&B metrics push + stdout summary
|   |-- state.py              # Health state persistence (JSON checkpoint)
|   +-- docs.py               # Markdown documentation writer (issues/logs)
|-- dataset.py                # Existing: dataset loading + density weighting
|-- state.py                  # Existing: rod state utilities
|-- model.py                  # Existing: surrogate MLP
|-- validate.py               # Existing: post-collection validation
+-- ...
```

### Structure Rationale

- **`monitor/` as a subpackage:** Keeps monitoring concerns isolated from existing collection code. The collection pipeline (`collect_data.py`) should not need modification -- the monitor observes it externally by reading shared state and batch files.
- **One file per concern:** Each monitoring responsibility (health, coverage, quality, stop conditions) gets its own module. This makes the build order explicit: you can implement and test `health.py` before `coverage.py` because they have no mutual dependencies.
- **`runner.py` as the entry point:** A single coordinator that imports and calls the checkers. Simpler than a framework-based approach (Airflow, Prefect) which would be overkill for one monitoring loop.
- **`state.py` for persistence:** The monitor must survive restarts. A JSON checkpoint file stores the last known good state (last check time, cumulative metrics, coverage grid). On restart, the monitor resumes from the checkpoint rather than recomputing from scratch.

## Architectural Patterns

### Pattern 1: External Observer (not Embedded Monitor)

**What:** The monitoring system runs as a separate process that observes the collection pipeline from the outside, rather than being embedded within the collection loop.

**When to use:** When the existing pipeline already works and you want to add monitoring without modifying it. When the monitor might crash and you don't want to take down the collector.

**Trade-offs:**
- Pro: Collection and monitoring are decoupled. A monitor crash does not affect data collection.
- Pro: The existing `_multiprocess_collect()` monitoring loop already provides basic FPS/progress tracking. The external monitor adds coverage analysis and stop conditions on top.
- Con: Must communicate through shared filesystem (batch files, health state file) rather than in-process data structures. Slightly higher latency for detecting issues.

**Example:**
```python
# runner.py -- external observer pattern
class MonitorRunner:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.state = HealthState.load_or_create(config.state_file)

    def run(self, interval_seconds: int = 300):
        """Run periodic health checks until collection stops."""
        while True:
            report = self.check_health()
            self.update_coverage()
            self.log_to_wandb(report)

            if self.should_stop(report):
                self.signal_stop()
                break

            self.state.save(self.config.state_file)
            time.sleep(interval_seconds)

    def check_health(self) -> HealthReport:
        process_health = health.check_processes(self.config.save_dir)
        resource_health = health.check_resources(self.config.save_dir)
        data_health = quality.check_recent_batches(self.config.save_dir)
        return HealthReport(process_health, resource_health, data_health)
```

**This is the recommended pattern for this project** because the collection pipeline is already mature and functional. The monitor should not risk destabilizing it.

### Pattern 2: Multi-Criteria Stop Condition

**What:** Collection stops only when ALL of several conditions are satisfied simultaneously: minimum runtime, minimum sample count, and minimum coverage threshold.

**When to use:** When you need both quantity and quality guarantees from a data collection run. A time-only stop might produce too little data; a count-only stop might miss coverage gaps.

**Trade-offs:**
- Pro: Ensures the dataset meets quality requirements before stopping.
- Pro: Coverage-based stopping avoids collecting redundant data in already-covered regions.
- Con: Coverage computation adds overhead. Must be fast enough for periodic checks (sampling-based, not exhaustive).

**Example:**
```python
# stop_conditions.py
@dataclass
class StopCriteria:
    min_runtime_hours: float = 8.0
    min_transitions: int = 500_000
    min_coverage_pct: float = 70.0   # % of state-action grid cells occupied
    max_runtime_hours: float = 24.0  # hard ceiling regardless of coverage

def should_stop(
    elapsed_hours: float,
    total_transitions: int,
    coverage_pct: float,
    criteria: StopCriteria,
) -> tuple[bool, str]:
    """Return (should_stop, reason)."""
    if elapsed_hours >= criteria.max_runtime_hours:
        return True, "max_runtime_exceeded"

    time_ok = elapsed_hours >= criteria.min_runtime_hours
    count_ok = total_transitions >= criteria.min_transitions
    coverage_ok = coverage_pct >= criteria.min_coverage_pct

    if time_ok and count_ok and coverage_ok:
        return True, "all_criteria_met"

    return False, f"waiting: time={time_ok}, count={count_ok}, coverage={coverage_ok}"
```

### Pattern 3: Checkpoint-Resumable Monitoring State

**What:** The monitor persists its state (coverage grid, last check timestamp, cumulative metrics) to a JSON file after each check cycle. On restart, it loads this state instead of recomputing from all batch files.

**When to use:** When the monitor might be restarted (crash recovery, manual restart, machine reboot) during a multi-hour collection run.

**Trade-offs:**
- Pro: Fast restart without re-scanning all batch files.
- Pro: Coverage grid accumulates across checks without re-reading old data.
- Con: State file can become stale if batch files are manually deleted. Must handle this gracefully.

**Example:**
```python
# state.py
@dataclass
class HealthState:
    last_check_time: float = 0.0
    last_batch_idx: int = -1       # highest batch file index already processed
    total_transitions_seen: int = 0
    coverage_grid: dict = field(default_factory=dict)  # bin_key -> count
    issues_logged: list = field(default_factory=list)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f)

    @classmethod
    def load_or_create(cls, path: str) -> "HealthState":
        if Path(path).exists():
            with open(path) as f:
                return cls(**json.load(f))
        return cls()
```

## Data Flow

### Monitoring Data Flow

```
[Collection Workers]
    | (write batch files to disk)
    v
[data/surrogate/batch_w*.pt]
    | (read by monitor)
    v
[Monitor Runner] ---> [Process Health Checker] ---> dead worker detection
    |                       |
    |                  [Resource Monitor] ---> disk/mem alerts
    |                       |
    +---> [Coverage Analyzer] ---> state-action grid occupancy %
    |         | (read new batch files since last check)
    |         v
    |    [coverage_grid updated in HealthState]
    |
    +---> [Data Quality Checker] ---> NaN/Inf/range alerts
    |         | (sample recent batch files)
    |         v
    |    [quality report]
    |
    +---> [Stop Condition Evaluator]
    |         | (time AND count AND coverage)
    |         v
    |    [CONTINUE or STOP decision]
    |
    +---> [W&B Logger] ---> push metrics to remote dashboard
    |
    +---> [Doc Writer] ---> issues/*.md, logs/*.md
    |
    +---> [Health State File] ---> .monitor_state.json (for crash recovery)
```

### Stop Signal Flow

```
[Monitor] determines should_stop=True
    |
    v
[Signal File] write "STOP" to data/surrogate/.stop_signal
    |
    v
[collect_data.py monitoring loop] reads signal file
    |  (check in _target_reached() or monitoring loop)
    v
[Workers] observe shared_counter >= target OR signal
    |
    v
[Graceful shutdown]
```

### Key Data Flows

1. **Batch file -> Coverage grid:** The coverage analyzer reads new batch files (those with index > `last_batch_idx`), bins their state-action pairs into a discretized grid, and updates the running coverage count. This is incremental -- old files are not re-read.

2. **Health metrics -> W&B:** Each monitoring cycle pushes a standard set of metrics to W&B: `coverage_pct`, `transitions_total`, `fps_rolling`, `disk_free_gb`, `worker_alive_count`, `data_quality_score`. These augment the existing metrics already logged by `_multiprocess_collect()`.

3. **Problem detection -> Documentation:** When the monitor detects an issue (dead worker, NaN data, FPS drop below threshold), it creates a Markdown file in `issues/` following the project's documentation conventions. This provides an audit trail of what happened during the overnight run.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1M transitions (~1 hour) | Current architecture is sufficient. Coverage check reads ~10 batch files per cycle. |
| 10M transitions (~10 hours) | Coverage grid in memory is fine (4D state x 5D action, 20 bins each = manageable). Incremental batch reading prevents re-scanning. Monitor check interval of 5 minutes means ~120 checks total. |
| 100M+ transitions (multi-day) | Coverage grid might need coarser binning or sampling. Health state file keeps growing; consider periodic compaction. Multiple collection runs can share a single coverage grid via checkpoint. |

### Scaling Priorities

1. **First bottleneck: Coverage computation time.** If batch files accumulate faster than the monitor can read them, use sampling (read every Nth batch) rather than every file. At 16 workers x 100 episodes/save x 500 steps/episode, that is ~50K transitions per batch save. With 270 FPS aggregate, a new batch appears roughly every 3 minutes. A 5-minute check interval means ~1-2 new batch files per check -- very manageable.

2. **Second bottleneck: Disk I/O from monitoring reads.** The monitor reads batch files that workers are actively writing. Use read-only access and handle partial/corrupt reads gracefully (skip files that fail to load). Workers write atomically (save complete, then rename) so this should not be an issue.

## Anti-Patterns

### Anti-Pattern 1: Embedding Monitor Logic in the Collection Loop

**What people do:** Add coverage checking, stop condition evaluation, and documentation writing directly into `_multiprocess_collect()` or `_collection_loop()`.

**Why it's wrong:** The collection code is already complex (multiprocess coordination, batch saving, W&B logging). Adding monitoring logic creates tight coupling. A bug in coverage analysis could crash the data collection process, losing hours of work. The existing monitoring loop in `_multiprocess_collect()` (lines 780-860) does exactly the right thing: basic FPS/progress tracking that cannot fail.

**Do this instead:** Run the monitor as a separate process or script. It reads shared state (batch files, shared counter) but cannot crash the collector.

### Anti-Pattern 2: Re-reading All Batch Files Every Check

**What people do:** On each monitoring cycle, load every batch file to recompute coverage from scratch.

**Why it's wrong:** After 8 hours of collection at 270 FPS, there are ~7.8M transitions across ~150+ batch files. Loading all of them every 5 minutes wastes I/O bandwidth and CPU time that the collection workers need.

**Do this instead:** Track `last_batch_idx` in health state. Each cycle only reads new batch files. Coverage grid accumulates incrementally.

### Anti-Pattern 3: Polling Shared Counter at High Frequency

**What people do:** Check `shared_counter.value` every second to detect stalls or compute FPS.

**Why it's wrong:** The shared counter uses a lock (`shared_counter.get_lock()`). High-frequency polling from an external monitor process contends with the 16 worker processes that increment it. At 270 FPS aggregate, that is 270 lock acquisitions per second from workers. Adding another 1/s from the monitor is negligible, but the principle matters: don't touch the shared counter more than necessary.

**Do this instead:** Use the batch file filesystem as the primary signal. Count transitions by scanning new batch file headers (read episode/step counts from file metadata) rather than polling the shared counter. Check the counter at most once per monitoring cycle (every 5 minutes).

### Anti-Pattern 4: Using the Monitor to Fix Issues by Modifying Worker State

**What people do:** When a dead worker is detected, the monitor tries to restart it by directly manipulating multiprocessing state or injecting new processes.

**Why it's wrong:** The multiprocessing.Process objects, shared counter, and forkserver state are owned by the parent process in `_multiprocess_collect()`. An external monitor process cannot safely inject new workers.

**Do this instead:** The monitor should signal the parent process to handle restarts (via a signal file or IPC). Or, more practically: for an overnight run where 1 of 16 workers dies, the remaining 15 workers are sufficient. The monitor documents the dead worker in `issues/` and logs the throughput impact. Restarting workers is a stretch goal, not a first-phase requirement.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| W&B (Weights & Biases) | `wandb.log()` from monitor process via `wandb.init()` with same run ID or a linked monitoring run | Already integrated in `collect_data.py`. Monitor can either share the W&B run (by resuming it) or create a linked "monitoring" run. Creating a separate "monitoring" run is safer (avoids metric key collisions). |
| Filesystem (batch files) | Read-only access to `data/surrogate/batch_*.pt` | Workers write atomically. Monitor should catch `torch.load()` failures gracefully. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Monitor <-> Collection Process | Shared filesystem (batch files, stop signal file) | No direct IPC. Monitor reads files; writes stop signal. Collection reads stop signal. |
| Monitor <-> Coverage Analyzer | In-process function call | Coverage analyzer is a library module called by the monitor runner, not a separate process. |
| Monitor <-> W&B | Network API (HTTP) | Uses `wandb` Python SDK. Must handle network failures gracefully (catch exceptions, don't crash monitor). |
| Monitor <-> Documentation Writer | In-process function call -> filesystem write | Writes Markdown files to `issues/` and `logs/` directories. |

## Build Order (Dependencies Between Components)

The components should be built in this order, reflecting their dependency chain:

```
Phase 1: Foundation
  health.py          (no deps beyond stdlib + psutil)
  state.py           (no deps, just JSON serialization)
  reporter.py        (depends on wandb, already available)

Phase 2: Analysis
  quality.py         (depends on torch for batch file loading)
  coverage.py        (depends on torch + numpy; reads batch files)

Phase 3: Decision
  stop_conditions.py (depends on coverage.py output + health.py output)
  diagnostics.py     (depends on health.py for problem detection)

Phase 4: Orchestration
  runner.py          (imports all above; the main loop)
  docs.py            (depends on health reports for content)

Phase 5: Integration
  Modify collect_data.py to read stop signal file
  Wire up W&B run linking between collector and monitor
```

**Why this order:**
- Health checking and state persistence are standalone utilities with no project-specific dependencies -- build and test them first.
- Coverage and quality analysis require loading batch files, which means they need working data. Build them after the foundation so you can test with actual (or mock) batch files.
- Stop conditions aggregate signals from health, coverage, and quality. They cannot be built until those signals exist.
- The runner orchestrates everything. Build it last when all components are tested individually.
- Integration with the existing `collect_data.py` (stop signal reading) is a minimal change to proven code, so defer it until the monitor is working end-to-end.

## Sources

- Existing codebase: `aprx_model_elastica/collect_data.py`, `collect_config.py`, `dataset.py`, `state.py`, `model.py`, `env.py`, `validate.py`
- Project knowledge: `experiments/surrogate-parallel-data-collection.md`, `experiments/surrogate-data-coverage-improvements.md`
- Project issues: `issues/parallel-collection-scaling-bottleneck.md`, `issues/surrogate-data-collection-no-append.md`, `issues/surrogate-model-data-imbalance.md`
- [Data Pipeline Monitoring - Monte Carlo](https://www.montecarlodata.com/blog-data-pipeline-monitoring/)
- [W&B Run Metrics Notifications](https://wandb.ai/wandb_fc/product-announcements-fc/reports/Introducing-run-metrics-notifications-for-W-B-Models--VmlldzoxMjEwMTQxNA)
- [W&B Programmatic Run API](https://docs.wandb.ai/models/runs)
- [AI Agent Design Patterns - Microsoft](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [Long-Running Agent Workflows](https://palospublishing.com/designing-long-running-llm-agent-workflows/)
- [Python Watchdog Thread Pattern](https://superfastpython.com/watchdog-thread-in-python/)
- [Data Quality Framework Guide](https://www.montecarlodata.com/blog-data-quality-framework/)
- [Building Surrogate Models with RL Trajectories](https://arxiv.org/abs/2509.01285)

---
*Architecture research for: Autonomous data collection monitoring for neural surrogate model training*
*Researched: 2026-03-09*
