# Stack Research

**Domain:** Autonomous data collection monitoring, coverage-based stopping, overnight ML pipeline management
**Researched:** 2026-03-09
**Confidence:** HIGH

## Executive Summary

This project already has a mature data collection pipeline (`aprx_model_elastica/collect_data.py`) with multiprocess workers, W&B logging, Sobol quasi-random sampling, and post-collection validation. The monitoring/autonomy layer does NOT need heavy infrastructure (no Airflow, no Celery, no Kubernetes). The right stack is a lightweight watchdog loop built from libraries already installed or available in the Python standard library, plus targeted additions for coverage tracking and alerting.

The key insight: this is a single-machine, single-run, overnight job -- not a distributed pipeline. The monitoring agent is a Python script that periodically inspects the collection process, checks coverage metrics, and decides whether to continue or stop. Heavyweight orchestration tools would add complexity without benefit.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Python stdlib `multiprocessing` | 3.12 (installed) | Process management, shared state | Already used by collect_data.py. `mp.Value` for shared counters, `mp.Event` for graceful shutdown signals. Zero additional deps. | HIGH |
| Python stdlib `signal` | 3.12 (installed) | Graceful shutdown on SIGTERM/SIGINT | Standard pattern for overnight jobs. Register handlers in main process, propagate to workers via `mp.Event`. Already partially implemented (KeyboardInterrupt handler in collect_data.py). | HIGH |
| Python stdlib `threading.Timer` / `time.sleep` loop | 3.12 (installed) | Periodic monitoring checks | The monitoring loop in `_multiprocess_collect` already polls every 30s. Extend this pattern rather than adding a scheduler library. Simple, debuggable, no deps. | HIGH |
| `wandb` | 0.25.0 (installed) | Metric logging + alert notifications | Already integrated. Use `wandb.alert()` for Slack/email notifications on completion, stalls, crashes, and coverage milestones. Use `wandb.log()` for coverage metrics dashboard. | HIGH |
| `psutil` | 7.2.1 (installed) | System resource monitoring | Already listed as optional dep in pyproject.toml. Monitor CPU, memory, disk I/O during collection. Detect worker crashes, zombie processes, resource exhaustion. | HIGH |
| `numpy` | 2.4.0 (installed) | Coverage grid computation | `np.histogramdd` for multi-dimensional state-action binning. Already used throughout. Coverage fraction = occupied_bins / total_bins. | HIGH |
| `scipy.stats.qmc.discrepancy` | 1.17.1 (installed) | Coverage quality metric | Quantifies how uniformly samples fill the state-action hypercube. Lower discrepancy = better coverage. Already available -- scipy is installed. Complements grid-based coverage with a continuous metric. | HIGH |

### Supporting Libraries

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| `filelock` | 3.25.0 (installed, torch dep) | Cross-process file locking | Lock coverage checkpoint files when monitor and workers both need to read/write coverage state. Already installed as torch dependency. | HIGH |
| `torch.quasirandom.SobolEngine` | 2.10.0 (installed) | Quasi-random action sampling | Already used in `SobolActionSampler`. No changes needed. | HIGH |
| `json` (stdlib) | 3.12 | Coverage state serialization | Save/load coverage grid state as JSON checkpoint for monitor to read. Simpler than pickle, human-readable for debugging. | HIGH |
| `dataclasses` (stdlib) | 3.12 | Configuration for monitor | Extend the existing `DataCollectionConfig` with monitoring/stopping fields. Already the pattern used throughout. | HIGH |
| `pathlib` (stdlib) | 3.12 | File system operations | Already used. Monitor reads batch files, checks disk space, manages coverage checkpoints. | HIGH |
| `datetime` (stdlib) | 3.12 | Wall-clock time tracking | Track collection start time, compute elapsed hours for time-based stop criteria. | HIGH |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `systemd` (Linux) | Process supervision for overnight runs | Create a `.service` unit file with `Restart=on-failure`, `RestartSec=30`. Ensures collection restarts if the process dies unexpectedly. The append mode in collect_data.py already handles resumption. |
| `tmux` / `screen` | Terminal session persistence | Alternative to systemd for ad-hoc runs. `tmux new -d -s collect 'python -m aprx_model_elastica.collect_data ...'`. Already available on the machine. |
| W&B Dashboard | Visual monitoring | Create a custom dashboard with coverage%, FPS, disk usage, ETA panels. Already set up for basic metrics. |

## Installation

```bash
# Everything needed is already installed. Verify:
python3 -c "import wandb, psutil, numpy, scipy, torch, filelock; print('All deps OK')"

# The only new pip install (optional, for nicer scheduling syntax):
# pip install schedule  # NOT recommended -- see "What NOT to Use" below

# For systemd service (no pip needed):
# sudo cp script/surrogate-collect.service /etc/systemd/system/
# sudo systemctl enable surrogate-collect
```

## Architecture: How the Monitoring Stack Fits Together

```
                    +---------------------------+
                    |  systemd / tmux           |
                    |  (process supervision)    |
                    +---------------------------+
                              |
                    +---------------------------+
                    |  collect_data.py main()   |
                    |  - Spawns workers         |
                    |  - Runs monitor loop      |
                    |  - Checks stop criteria   |
                    +---------------------------+
                     /        |        \
            +--------+  +--------+  +--------+
            |Worker 0|  |Worker 1|  |Worker N|    (mp.Process)
            |1 env   |  |1 env   |  |1 env   |
            +--------+  +--------+  +--------+
                 \         |         /
                  +--------+--------+
                  | Shared State     |
                  | - mp.Value(count)|
                  | - mp.Event(stop) |
                  | - coverage.json  |
                  +------------------+
                           |
                  +------------------+
                  | Monitor Loop     |
                  | (in main proc)   |
                  | - psutil checks  |
                  | - coverage grid  |
                  | - wandb.log()    |
                  | - wandb.alert()  |
                  | - stop decision  |
                  +------------------+
```

The monitor loop runs IN the main process of collect_data.py (extending the existing 30s poll loop in `_multiprocess_collect`). It does NOT need to be a separate script or daemon. This is the simplest correct architecture because:

1. The main process already has access to `shared_counter` and all worker `Process` objects
2. Adding a separate monitor process would require IPC to share state
3. The existing poll loop already does FPS + disk monitoring -- just extend it

## Stop Criteria Stack

The multi-criteria stop condition uses only stdlib + numpy:

```python
# All three must be true to stop:
stop_conditions = {
    "min_time":       elapsed_hours >= config.min_hours,        # 8.0 hours default
    "min_samples":    shared_counter.value >= config.min_transitions,  # 1_000_000
    "coverage_target": coverage_fraction >= config.target_coverage,    # 0.85
}
should_stop = all(stop_conditions.values())
```

Coverage fraction is computed from `np.histogramdd`:

```python
# Project to summary features (4D: com_x, com_y, vel_mag, mean_omega)
# + action dimensions (5D: amplitude, frequency, wave_number, phase, turn_bias)
# = 9D grid, but binned sparsely (e.g., 5 bins per dim = 5^9 ~2M bins -- too many)
#
# Better: use SEPARATE grids for state-space (4D) and action-space (5D)
# State: 4D x 10 bins = 10,000 cells
# Action: 5D x 8 bins = 32,768 cells
# Coverage = fraction of cells with >= min_samples_per_bin (e.g., 3)

state_hist, _ = np.histogramdd(state_features, bins=10)
action_hist, _ = np.histogramdd(actions, bins=8)
state_coverage = (state_hist >= min_per_bin).sum() / state_hist.size
action_coverage = (action_hist >= min_per_bin).sum() / action_hist.size
coverage_fraction = min(state_coverage, action_coverage)
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Extend existing poll loop in collect_data.py | Separate monitor daemon script | Only if you need to monitor MULTIPLE concurrent collection runs, or if the collection script is not under your control |
| `wandb.alert()` for notifications | Slack webhooks / email via smtplib | Only if W&B is not available or if you need custom notification formatting beyond alert title+text |
| `np.histogramdd` for coverage grid | `scipy.stats.qmc.discrepancy` alone | Discrepancy is a single float (good for tracking trend) but doesn't tell you WHICH regions are undersampled. Use both: histogramdd for coverage map, discrepancy for quality score. |
| `psutil` for resource monitoring | `/proc` filesystem parsing | Only on Linux and only if psutil is unavailable. psutil is cross-platform and much cleaner API. |
| `systemd` service for overnight supervision | Docker with `restart: unless-stopped` | If running in Docker (Dockerfile exists in repo). systemd is simpler for bare-metal. Both work with append mode. |
| `mp.Event` for graceful stop signal | Writing a sentinel file | Sentinel files work but are race-prone. `mp.Event` is purpose-built for inter-process signaling. |
| `filelock` for coverage checkpoint | `mp.Lock` | `mp.Lock` only works between parent/child processes. `filelock` works across independent processes AND survives process crashes. |
| `json` for coverage state files | `torch.save` / pickle | JSON is human-readable (debug overnight without loading Python), smaller, and doesn't require torch import to inspect. |
| `time.sleep` poll loop | `schedule` library | `schedule` adds readable syntax (`schedule.every(30).seconds.do(check)`) but is single-threaded and not worth the dependency for a simple loop. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Apache Airflow / Dagster / Prefect | Massive overkill for a single-machine, single-script overnight run. These are for multi-stage, multi-service DAG pipelines. Installation alone would take longer than writing the monitor loop. | Extend the existing poll loop in collect_data.py |
| Celery | Requires a message broker (Redis/RabbitMQ). Designed for distributed task queues, not process monitoring. | `multiprocessing` (already used) |
| APScheduler | Adds persistence, job stores, cron triggers -- none needed for a simple "check every 30 seconds" loop. Would be the 3rd scheduler in the codebase (after the existing poll loop and the collection loop). | `time.sleep(30)` in the existing loop |
| `schedule` library | Zero-dependency claim is nice, but it's still an unnecessary dependency for a trivial loop. Single-threaded design means a slow check blocks the next check. | `time.sleep(30)` in a loop |
| MLflow | Good for experiment tracking but W&B is already integrated and working. Adding a second experiment tracker creates confusion. | `wandb` (already integrated) |
| Prometheus + Grafana | Production monitoring stack for web services. Setting up a Prometheus server to monitor one Python script is absurd. | `wandb` dashboard + `psutil` |
| Kubernetes CronJob | Container orchestration for a single overnight script on one machine. The Dockerfile exists but K8s is not set up and not needed. | `systemd` service or `tmux` |
| `subprocess.Popen` for workers | Less control than `multiprocessing.Process`. Can't share `mp.Value` counters or `mp.Event` signals. Harder to pass config. | `multiprocessing.Process` (already used) |
| SQLite for coverage state | Adds complexity. Coverage state is a small JSON (<1KB). No concurrent write contention if only the monitor writes it. | JSON file with `filelock` |
| Redis / memcached | External service dependency for sharing state between processes. `mp.Value` and files already work. | `mp.Value` + JSON checkpoint |

## Stack Patterns by Variant

**If running as a one-off overnight job:**
- Use `tmux` or `nohup` to keep the session alive
- The monitor loop handles everything within the script
- Coverage checkpoint saved to `data/surrogate/coverage_state.json`

**If running as a recurring scheduled job:**
- Use `systemd` service with `Restart=on-failure`, `RestartSec=30`
- Append mode in collect_data.py handles resumption automatically
- W&B creates a new run per restart, preserving history

**If running in Docker:**
- Use `docker-compose.yml` with `restart: unless-stopped`
- Mount `data/` as a volume for persistence
- The existing Dockerfile + docker-compose.yml can be extended

## Version Compatibility

| Package | Version (Installed) | Compatible With | Notes |
|---------|---------------------|-----------------|-------|
| wandb 0.25.0 | torch 2.10.0 | W&B alert API stable since 0.14.0. `AlertLevel.INFO/WARN/ERROR` enum available. |
| psutil 7.2.1 | Python 3.12 | Fully compatible. `Process`, `cpu_percent`, `virtual_memory`, `disk_usage` all available. |
| numpy 2.4.0 | scipy 1.17.1, torch 2.10.0 | `np.histogramdd` signature stable. All array operations compatible. |
| scipy 1.17.1 | numpy 2.4.0 | `scipy.stats.qmc.discrepancy` available since scipy 1.7.0. Current version fully compatible. |
| filelock 3.25.0 | Python 3.12 | Installed as torch dependency. API is `FileLock(path)` context manager. Stable. |
| multiprocessing (stdlib) | Python 3.12 | torch 2.10.0 | `forkserver` start method already configured in collect_data.py. |

## Key API Patterns

### W&B Alerts (for notifications)

```python
import wandb
from wandb import AlertLevel

# In the monitor loop:
wandb.alert(
    title="Collection Complete",
    text=f"Collected {total:,} transitions. Coverage: {coverage:.1%}",
    level=AlertLevel.INFO,
    wait_duration=300,  # throttle: max 1 alert per 5 min
)

# Alert levels: INFO (milestones), WARN (stalls, low disk), ERROR (crashes)
```

### Coverage Grid (numpy.histogramdd)

```python
import numpy as np

# 4D state features: [com_x, com_y, vel_mag, mean_omega]
features = extract_summary_features(states)  # (N, 4)
hist, edges = np.histogramdd(features, bins=10, range=expected_ranges)
occupied = (hist >= min_samples_per_bin).sum()
total_bins = hist.size  # 10^4 = 10,000
coverage = occupied / total_bins
```

### Discrepancy Score (scipy)

```python
from scipy.stats import qmc

# Normalize samples to [0, 1]^d
normalized = (features - lo) / (hi - lo)
disc = qmc.discrepancy(normalized, method='CD')  # centered discrepancy
# Lower = better coverage. Track trend over time.
```

### Process Health (psutil)

```python
import psutil

# Check worker processes are alive and consuming CPU
for pid in worker_pids:
    try:
        proc = psutil.Process(pid)
        cpu = proc.cpu_percent(interval=0.1)
        mem_mb = proc.memory_info().rss / 1024**2
        if cpu < 1.0:
            # Worker might be stalled
            wandb.alert(title="Worker Stall", text=f"PID {pid} CPU={cpu}%")
    except psutil.NoSuchProcess:
        # Worker crashed
        wandb.alert(title="Worker Crash", text=f"PID {pid} died", level=AlertLevel.ERROR)
```

### Graceful Stop Signal (multiprocessing.Event)

```python
import multiprocessing as mp

stop_event = mp.Event()

# In workers: check periodically
while not stop_event.is_set():
    ep_data = collect_episode(...)
    ...

# In monitor: set when criteria met
if all(stop_conditions.values()):
    stop_event.set()  # all workers exit their loops
```

## Sources

- [W&B Alert Documentation](https://docs.wandb.ai/models/runs/alert) -- verified wandb.alert() API, AlertLevel enum, wait_duration parameter (HIGH confidence)
- [psutil Documentation](https://psutil.readthedocs.io/en/latest/) -- verified Process monitoring, cpu_percent, memory_info, disk_usage APIs (HIGH confidence)
- [NumPy histogramdd](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html) -- verified multi-dimensional histogram API for coverage grid (HIGH confidence)
- [SciPy QMC discrepancy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.discrepancy.html) -- verified discrepancy metric for coverage quality assessment (HIGH confidence)
- [PyTorch SobolEngine](https://docs.pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html) -- verified quasi-random sampling (already in use) (HIGH confidence)
- [filelock Documentation](https://py-filelock.readthedocs.io/en/latest/) -- verified cross-process file locking API (HIGH confidence)
- [systemd service documentation](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html) -- verified Restart=on-failure for process supervision (HIGH confidence)

---
*Stack research for: Autonomous data collection monitoring for neural surrogate models*
*Researched: 2026-03-09*
