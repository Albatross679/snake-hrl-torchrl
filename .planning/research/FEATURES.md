# Feature Research

**Domain:** Autonomous data collection monitoring for neural surrogate models (physics simulation)
**Researched:** 2026-03-09
**Confidence:** HIGH

## Context

This feature research is for an autonomous monitoring and coverage-based collection system that wraps an existing data collection pipeline (`aprx_model_elastica/collect_data.py`). The pipeline collects state-action-next_state transitions from a PyElastica Cosserat rod snake robot simulation using multiprocess workers. The monitoring system must run overnight (8+ hours) without human intervention.

Key constraints from the existing system:
- 16 parallel workers, each running 1 PyElastica env (~270 FPS combined)
- Data saved as `.pt` or `.parquet` batch files to disk
- W&B already integrated for basic metric logging (FPS, progress, ETA, disk usage)
- Sobol quasi-random action sampling + state perturbation already implemented
- Density-weighted training already handles some coverage issues downstream
- 5D action space, 124D state space (6 physical groups: pos_x, pos_y, vel_x, vel_y, yaw, omega_z)

## Feature Landscape

### Table Stakes (Users Expect These)

Features that any autonomous overnight collection system must have. Without these, the system requires babysitting and defeats its purpose.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Worker health monitoring** | Dead/hung workers silently reduce throughput; overnight run finishes with half the expected data | LOW | Check `Process.is_alive()` and FPS-per-worker. Already have `shared_counter` for global progress. Add per-worker heartbeat timestamps via `multiprocessing.Value`. |
| **Crash detection and auto-restart** | PyElastica can segfault or throw unhandled exceptions in edge cases; a dead worker at hour 1 wastes 7 hours of that CPU core | MEDIUM | Wrap worker spawning in retry logic in the monitoring loop. Respawn with same config but new seed. Track restart count per worker. |
| **Stall detection** | A worker can hang (deadlock in BLAS, infinite loop in physics) without crashing. `is_alive()` returns True but no progress is made | LOW | Compare per-worker transition count at each poll interval. If delta is zero for 2+ consecutive intervals (60+ seconds), flag as stalled. |
| **Multi-criteria stop condition** | "Collect N transitions" is not enough -- need minimum runtime AND minimum samples AND coverage targets to ensure data quality | MEDIUM | Composite predicate: `elapsed >= min_hours AND total_transitions >= min_count AND coverage_score >= threshold`. All three must be true. Continue collecting until satisfied. |
| **Disk space monitoring with early warning** | Running out of disk mid-collection corrupts the last batch file and wastes hours of work | LOW | Already have disk check at startup. Add periodic check in monitoring loop (every poll interval). Alert + graceful stop if free space drops below 2 GB. |
| **Graceful shutdown with data preservation** | Ctrl-C or stop signal must not corrupt in-progress batch files | LOW | Already partially implemented (KeyboardInterrupt handler). Extend to flush current batch buffers before terminating workers. |
| **Progress logging to stdout and W&B** | Must be able to check collection status remotely (via W&B dashboard or `tail -f` on log file) without SSH-ing in | LOW | Already implemented: 30-second poll loop logs to stdout and W&B. Enhance with per-worker breakdown and health status. |
| **NaN/Inf detection during collection** | Bad physics states (NaN positions, Inf velocities) produce garbage training data; must detect and discard before saving | LOW | Check each episode's arrays after `collect_episode()` returns. Discard episodes with NaN/Inf, log the discard, increment a counter. Already have NaN checks in post-validation; move to runtime. |

### Differentiators (Competitive Advantage)

Features that elevate this from "a script that runs overnight" to "a system that produces high-quality datasets autonomously."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **State-action coverage grid tracking** | Quantifies *what* the dataset covers, not just *how much*. Identifies gaps in the 5D action hypercube and key state dimensions before training reveals them as model errors | MEDIUM | Bin the 5D action space into a coarse grid (e.g., 5 bins per dim = 3125 cells). Track fill fraction. Also bin 4 summary state features (CoM_x, CoM_y, vel_mag, mean_omega) into a separate grid. Log coverage percentage to W&B. |
| **Coverage-based stop condition** | Prevents stopping collection when throughput is high but coverage is poor (e.g., all samples from the same state region due to a stuck worker or degenerate sampling) | MEDIUM | Define coverage target as "X% of action-space grid cells have >= Y samples." Integrate into multi-criteria stop condition. Requires coverage grid tracking feature. |
| **W&B alerts for critical events** | Get notified on phone/Slack when something goes wrong overnight instead of discovering it in the morning | LOW | Use `wandb.alert()` with `AlertLevel.WARN` for worker deaths, stalls, low disk, NaN episodes. Set `wait_duration=300` to avoid spam. Already have W&B run initialized. |
| **Per-worker FPS tracking and degradation detection** | Detects gradual performance issues (memory leaks, thermal throttling, OS scheduling changes) that reduce throughput without causing crashes | LOW | Store FPS per worker in shared array. Compare rolling average to baseline FPS (from smoke test). Alert if any worker drops below 50% of baseline for sustained period. |
| **Automated documentation of monitoring events** | Creates a complete audit trail of what happened during collection -- which workers crashed, when, what was the coverage at each checkpoint | LOW | Write monitoring events to a JSON log file alongside the data. Optionally create a summary markdown file at end of collection. |
| **Data quality scoring per batch** | Catches systematic issues like all-zero states, constant actions, or degenerate episodes early, rather than discovering them during training | MEDIUM | After saving each batch, compute quick quality metrics: action variance, state delta variance, episode length distribution. Flag batches with suspiciously low variance. |
| **Checkpoint-based coverage snapshots** | Periodically save coverage grid state so you can analyze how coverage evolved over time and decide whether to extend collection | LOW | Every N minutes, serialize the coverage grid counters to a JSON or .pt file. Enables post-hoc analysis of coverage growth rate. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem useful but add complexity without proportional value for this specific use case.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Real-time web dashboard** | "I want to see live heatmaps of coverage" | Building a custom web UI is a full project in itself. W&B already provides dashboards. Custom visualizations are better done post-hoc on saved data. | Log scalar coverage metrics to W&B. Generate static heatmap plots at coverage snapshot intervals. |
| **Adaptive sampling (active learning)** | "Automatically focus collection on sparse regions" | Requires a feedback loop where the monitor modifies worker behavior at runtime. Cross-process communication of coverage state to workers adds IPC overhead and complexity. Sobol sequences already provide good space-filling without coordination. | Collect uniformly with Sobol, use density weighting at training time. If specific gaps persist, do a targeted follow-up collection run with adjusted perturbation parameters. |
| **Automatic worker count scaling** | "Add more workers when CPU is idle, reduce when memory is tight" | Dynamic process creation/destruction complicates episode ID management, batch file naming, and shared counter logic. The hardware has a known scaling ceiling at ~16 workers. | Set worker count once at launch based on hardware. Use fixed 16 workers for this machine. |
| **Distributed collection across multiple machines** | "Use all available GPUs/machines" | This is a CPU-bound pipeline (PyElastica runs on CPU). Adding network communication, shared storage, and distributed coordination vastly increases complexity. Single machine with 48 CPUs is sufficient. | Run independent collection jobs on separate machines to different directories. Merge datasets afterwards with the existing append-mode logic. |
| **GPU-accelerated physics for faster collection** | "Run PyElastica on GPU for 100x speedup" | PyElastica is CPU-only. Porting to GPU requires a different simulator (MuJoCo, Genesis, etc.), which changes the physics and invalidates the surrogate model. | Accept CPU throughput ceiling. The surrogate model itself eliminates the need for fast simulation once trained. |
| **Automatic surrogate model training trigger** | "When collection is done, automatically start training" | Tight coupling between collection and training creates a fragile pipeline. Better to validate data quality first, then manually decide to train. | Output a summary report at end of collection. Human reviews coverage and quality metrics, then launches training. |
| **Email/SMS notifications** | "Email me when collection is done" | W&B alerts already provide Slack and email. Adding separate email/SMS infrastructure is redundant. | Use `wandb.alert()` for all notifications. |

## Feature Dependencies

```
[Worker health monitoring]
    |-- requires --> [Per-worker heartbeat timestamps]
    |-- enables --> [Crash detection and auto-restart]
    |-- enables --> [Stall detection]

[State-action coverage grid tracking]
    |-- enables --> [Coverage-based stop condition]
    |-- enables --> [Checkpoint-based coverage snapshots]
    |-- enhances --> [Multi-criteria stop condition]

[Multi-criteria stop condition]
    |-- requires --> [State-action coverage grid tracking] (for coverage criterion)
    |-- requires --> [Worker health monitoring] (to know actual vs expected throughput)

[NaN/Inf detection during collection]
    |-- enhances --> [Data quality scoring per batch]

[W&B alerts for critical events]
    |-- requires --> [Worker health monitoring] (to know what to alert on)
    |-- requires --> [Disk space monitoring] (to know when space is low)

[Automated documentation of monitoring events]
    |-- enhances --> [W&B alerts] (log what was alerted)
    |-- enhances --> [Crash detection] (log restart history)
```

### Dependency Notes

- **Coverage grid tracking enables the stop condition:** Without a coverage metric, the stop condition can only use time and count -- which are necessary but not sufficient for dataset quality.
- **Worker health is foundational:** Every other monitoring feature depends on knowing which workers are alive, stalled, or dead. This must be built first.
- **NaN detection feeds quality scoring:** Runtime NaN detection is a simpler version of batch quality scoring. Build NaN detection first as a boolean check, then extend to richer quality metrics.
- **W&B alerts depend on health monitoring:** You can only alert on conditions you can detect. Build detection first, alerting second.

## MVP Definition

### Launch With (v1)

Minimum viable monitoring system -- enough to run overnight with confidence.

- [ ] **Worker health monitoring** -- per-worker heartbeat + alive check. Foundational for everything else.
- [ ] **Crash detection and auto-restart** -- respawn dead workers so overnight runs don't lose CPU cores.
- [ ] **Stall detection** -- detect hung workers (alive but not progressing) and restart them.
- [ ] **NaN/Inf detection during collection** -- discard bad episodes at source, not after 8 hours.
- [ ] **Multi-criteria stop condition (time + count)** -- stop only when both minimum time and minimum transitions are met. Coverage criterion deferred to v1.x.
- [ ] **Disk space periodic monitoring** -- alert and stop gracefully before disk fills up.
- [ ] **W&B alerts for critical events** -- get notified about crashes, stalls, disk issues.
- [ ] **Graceful shutdown with data preservation** -- ensure no data corruption on stop.

### Add After Validation (v1.x)

Features to add once core monitoring is working reliably.

- [ ] **State-action coverage grid tracking** -- add once basic monitoring proves stable. Needed for coverage-based stop condition.
- [ ] **Coverage-based stop condition** -- integrate into multi-criteria predicate once coverage tracking works.
- [ ] **Per-worker FPS tracking and degradation detection** -- useful for diagnosing throughput issues.
- [ ] **Data quality scoring per batch** -- beyond NaN detection, catch subtler quality issues.
- [ ] **Checkpoint-based coverage snapshots** -- save coverage state for post-hoc analysis.

### Future Consideration (v2+)

Features to defer until the pipeline has run multiple times and patterns emerge.

- [ ] **Automated documentation of monitoring events** -- useful for reproducibility but not blocking.
- [ ] **Coverage gap analysis report** -- generate end-of-run report identifying which state-action regions still need more data for a follow-up collection.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Worker health monitoring | HIGH | LOW | P1 |
| Crash detection and auto-restart | HIGH | MEDIUM | P1 |
| Stall detection | HIGH | LOW | P1 |
| NaN/Inf detection during collection | HIGH | LOW | P1 |
| Multi-criteria stop condition (time + count) | HIGH | LOW | P1 |
| Disk space periodic monitoring | MEDIUM | LOW | P1 |
| W&B alerts for critical events | MEDIUM | LOW | P1 |
| Graceful shutdown with data preservation | MEDIUM | LOW | P1 |
| State-action coverage grid tracking | HIGH | MEDIUM | P2 |
| Coverage-based stop condition | HIGH | MEDIUM | P2 |
| Per-worker FPS tracking | MEDIUM | LOW | P2 |
| Data quality scoring per batch | MEDIUM | MEDIUM | P2 |
| Coverage snapshots | LOW | LOW | P2 |
| Automated event documentation | LOW | LOW | P3 |
| Coverage gap analysis report | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Must have for first overnight run (v1)
- P2: Should have, add after v1 proves stable
- P3: Nice to have, add based on observed needs

## Competitor Feature Analysis

This is not a competitive product -- it is internal tooling. However, analogous systems provide useful reference points.

| Feature | W&B Built-in | Ray Tune | Custom Scripts | Our Approach |
|---------|-------------|----------|----------------|--------------|
| Basic metric logging | Native | Native | Manual | W&B (existing) |
| Worker health | N/A (single process focus) | Native (Ray manages workers) | Manual heartbeat | Custom heartbeat + monitor loop |
| Auto-restart | N/A | Native | systemd/supervisord | In-process respawn in monitor loop |
| Coverage tracking | Not domain-aware | Not domain-aware | Manual | Custom grid binning of state-action space |
| Programmatic alerts | `wandb.alert()` | Callback-based | Email/Slack API | `wandb.alert()` |
| Stop conditions | Max steps only | Configurable schedulers | Manual | Custom multi-criteria predicate |
| Data quality checks | Artifacts metadata | N/A | Manual validation | Runtime NaN/Inf + batch quality scoring |

Our approach leverages W&B for logging/alerting (already integrated) and adds domain-specific features (coverage tracking, physics-aware quality checks) that general-purpose tools cannot provide.

## Sources

- Existing codebase: `aprx_model_elastica/collect_data.py` (current monitoring loop, lines 774-867)
- Existing codebase: `script/validate_surrogate_data.py` (post-collection validation pattern)
- Existing codebase: `script/smoke_test_collect.py` (pre-flight validation pattern)
- Existing codebase: `aprx_model_elastica/dataset.py` (density weighting for coverage compensation)
- [W&B Alert Documentation](https://docs.wandb.ai/models/runs/alert) -- `wandb.alert()` API for programmatic notifications
- [Self-Healing Data Pipelines](https://switchboard-software.com/post/self-healing-data-pipelines-how-ai-automation-saves-millions/) -- patterns for auto-recovery in data pipelines
- [Google ML Production Monitoring](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring) -- monitoring pipelines best practices
- [Active Learning for Surrogate Model Improvement](https://link.springer.com/article/10.1007/s00158-024-03816-9) -- adaptive sampling approaches (informed anti-feature decision)
- [Surrogate Modeling Reporting Standards](https://arxiv.org/pdf/2502.06753) -- coverage and evaluation metrics for surrogate models
- [Python Watchdog Thread Patterns](https://superfastpython.com/watchdog-thread-in-python/) -- heartbeat-based monitoring in Python
- [healthcheck-python](https://github.com/cagdasbas/healthcheck-python) -- health check patterns for multiprocessing apps

---
*Feature research for: Autonomous data collection monitoring for neural surrogate models*
*Researched: 2026-03-09*
