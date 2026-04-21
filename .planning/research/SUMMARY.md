# Project Research Summary

**Project:** Surrogate Model Data Collection
**Domain:** Autonomous data collection monitoring for neural surrogate model training (physics simulation)
**Researched:** 2026-03-09
**Confidence:** HIGH

## Executive Summary

This project builds an autonomous monitoring layer on top of an already-functional data collection pipeline for a neural surrogate model of a PyElastica Cosserat rod snake robot. The collection pipeline (`aprx_model_elastica/collect_data.py`) already handles multiprocess workers, Sobol quasi-random action sampling, W&B logging, and batch file persistence. The monitoring system is NOT a new data pipeline -- it is a lightweight watchdog that periodically inspects the running collection, validates data quality, tracks state-action space coverage, and decides when to stop. Every library needed is already installed (Python stdlib, wandb, psutil, numpy, scipy, filelock). No new infrastructure (Airflow, Celery, Kubernetes) is warranted.

The recommended architecture is an **external observer** pattern: a separate monitor process reads batch files and shared state from the filesystem, computes coverage metrics, checks worker health, and signals the collector to stop via a sentinel file. This decoupling means a monitor crash cannot kill data collection, and the existing collection code requires only minimal modification (reading a stop signal). The monitor itself is a Python package (`aprx_model_elastica/monitor/`) with one module per concern: health checking, coverage analysis, data quality validation, stop condition evaluation, and reporting.

The primary risks are: (1) silent worker death going undetected for hours, wasting CPU capacity; (2) NaN/Inf physics states from perturbation-induced instability poisoning the dataset; (3) coverage metrics giving false confidence by measuring marginal bin occupancy rather than meaningful joint coverage; and (4) disk space exhaustion mid-run due to underestimated per-transition storage. All four are preventable with straightforward engineering described in the pitfalls research. The mitigation pattern is consistent: detect early (per-worker heartbeats, runtime NaN checks, minimum-samples-per-bin coverage, periodic disk monitoring), alert via W&B, and degrade gracefully (discard bad data, stop before corruption).

## Key Findings

### Recommended Stack

The entire stack is already installed. No new pip packages are required. The core technologies are Python stdlib modules (multiprocessing, signal, threading, json, pathlib, dataclasses, datetime) augmented by four libraries already in the environment: wandb (alerts and dashboards), psutil (process and resource monitoring), numpy (coverage grid via `histogramdd`), and scipy (coverage quality via `qmc.discrepancy`). File locking for coverage checkpoints uses `filelock`, which is already installed as a torch dependency.

**Core technologies:**
- **Python `multiprocessing` (stdlib):** Process management, shared state via `mp.Value` and `mp.Event` -- already used by collect_data.py
- **`wandb` 0.25.0:** Metric logging, remote dashboard, and `wandb.alert()` for Slack/email notifications on crashes, stalls, and completion
- **`psutil` 7.2.1:** Worker process health monitoring (CPU, memory), disk usage tracking, zombie process detection
- **`numpy` `histogramdd`:** Multi-dimensional binning for state-action coverage grid (4D state + 5D action, computed separately)
- **`scipy.stats.qmc.discrepancy`:** Continuous coverage quality score complementing the discrete grid metric
- **`filelock` 3.25.0:** Cross-process locking for coverage checkpoint files (already installed as torch dependency)

### Expected Features

**Must have (table stakes) -- v1:**
- Worker health monitoring with per-worker heartbeats and crash/stall detection
- Crash detection with automatic worker restart (respawn dead workers to preserve throughput)
- NaN/Inf detection during collection (discard bad episodes at source, not after 8 hours)
- Multi-criteria stop condition (minimum runtime AND minimum transitions AND coverage target)
- Disk space periodic monitoring with early warning and graceful stop
- W&B alerts for critical events (crashes, stalls, disk, completion)
- Graceful shutdown with data preservation (no corrupt batch files)
- Progress logging to stdout and W&B with per-worker breakdown

**Should have (differentiators) -- v1.x:**
- State-action coverage grid tracking (quantifies dataset quality, not just quantity)
- Coverage-based stop condition (prevents stopping with high count but poor diversity)
- Per-worker FPS tracking and degradation detection (catches memory leaks, thermal throttling)
- Data quality scoring per batch (beyond NaN -- catches degenerate episodes)
- Checkpoint-based coverage snapshots (enables post-hoc coverage growth analysis)

**Defer (v2+):**
- Automated documentation of monitoring events (audit trail -- not blocking)
- Coverage gap analysis report (end-of-run identification of regions needing follow-up collection)
- Adaptive sampling / active learning (complexity without proportional value given Sobol + density weighting)
- Real-time web dashboard (W&B already provides dashboards)
- Distributed collection across machines (single machine with 48 CPUs is sufficient)

### Architecture Approach

The system follows an **external observer** pattern: the monitor runs as a separate process that reads shared state (batch files, shared counters, heartbeat timestamps) from the filesystem without modifying the collection pipeline. Components are organized as a `monitor/` subpackage within `aprx_model_elastica/`, with one file per concern (`health.py`, `coverage.py`, `quality.py`, `stop_conditions.py`, `reporter.py`, `state.py`, `runner.py`, `diagnostics.py`, `docs.py`). Communication between monitor and collector happens through filesystem signals (stop signal file, coverage checkpoint JSON), not direct IPC.

**Major components:**
1. **Monitor Runner (`runner.py`)** -- Periodic check loop that orchestrates all monitoring activities on a configurable interval (default 5 minutes)
2. **Process Health Checker (`health.py`)** -- Per-worker alive/stall/crash detection via heartbeats, exit codes, and FPS delta analysis
3. **Coverage Analyzer (`coverage.py`)** -- Incremental state-action grid binning using `numpy.histogramdd`, tracking minimum samples per bin
4. **Data Quality Checker (`quality.py`)** -- Runtime NaN/Inf validation, state magnitude bounds, batch statistical checks
5. **Stop Condition Evaluator (`stop_conditions.py`)** -- Multi-criteria predicate (time AND count AND coverage AND max ceiling)
6. **Health State (`state.py`)** -- JSON-serialized checkpoint for crash recovery (last batch index, coverage grid, cumulative metrics)
7. **Reporter (`reporter.py`)** -- W&B metric push with local JSON fallback for resilience against network failures

### Critical Pitfalls

1. **Silent worker death** -- Workers crash without the monitor detecting it. Prevent by checking `p.exitcode` per worker each poll, maintaining per-worker heartbeat timestamps, and alerting within 60 seconds of death. Implement atomic batch writes (tmp + rename) to prevent truncated files from dead workers.

2. **NaN/Inf transitions poisoning training** -- PyElastica's explicit integrator diverges on extreme perturbations. Prevent with per-episode NaN/Inf validation inside `collect_episode()`, state magnitude bounds checking, and W&B logging of NaN discard rate. If NaN rate exceeds 5%, automatically reduce perturbation parameters.

3. **Coverage metrics giving false confidence** -- Marginal bin occupancy looks excellent while joint coverage is sparse. Prevent by using minimum-samples-per-bin (50+) instead of binary occupancy, tracking coverage in task-relevant subspaces, and logging coverage heatmaps. Monitor the max/min ratio of `compute_density_weights()` as a proxy for imbalance.

4. **Disk space exhaustion mid-run** -- Per-transition size estimate (1 KB) underestimates by 2.5x when forces are collected. Prevent by computing actual byte sizes from tensor dimensions (~2.5 KB/transition with forces), checking disk every poll interval, and stopping gracefully when free space drops below 2 GB. Implement atomic writes to prevent corruption.

5. **W&B connection failure blocking the monitor loop** -- `wandb.log()` retries on network failure, blocking the single-threaded monitor from checking worker health. Prevent by moving W&B logging to a separate thread with a 5-second timeout, maintaining a local JSON fallback log, and tracking consecutive W&B failures.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Foundation -- Health Monitoring and Data Integrity

**Rationale:** Worker health monitoring is the foundational capability. Every other monitoring feature depends on knowing which workers are alive, stalled, or dead. Data integrity (atomic writes, NaN detection) must be in place before any overnight run to prevent hours of wasted compute.

**Delivers:** A monitoring system that detects worker crashes, stalls, and NaN data in real-time, alerts via W&B, and ensures no corrupt batch files.

**Addresses features:** Worker health monitoring, crash detection, stall detection, NaN/Inf detection, disk space monitoring, graceful shutdown, W&B alerts

**Avoids pitfalls:** Silent worker death (Pitfall 1), NaN transitions (Pitfall 2), disk exhaustion (Pitfall 4), W&B blocking (Pitfall 5)

**Stack elements:** `multiprocessing`, `psutil`, `wandb.alert()`, `filelock`, `json` (health state checkpoint)

**Architecture components:** `health.py`, `state.py`, `reporter.py`, `quality.py` (NaN only), `runner.py` (basic loop)

### Phase 2: Coverage Tracking and Smart Stop Conditions

**Rationale:** Once health monitoring proves stable, add the coverage layer. Coverage tracking requires loading batch files (the same mechanism health monitoring uses), and the stop condition evaluator depends on coverage metrics. This phase transforms the system from "runs until count is reached" to "runs until data quality is sufficient."

**Delivers:** State-action coverage grid, coverage-based stop condition integrated with time and count criteria, coverage checkpoint snapshots for post-hoc analysis.

**Addresses features:** Coverage grid tracking, coverage-based stop condition, multi-criteria stop (adding coverage criterion), coverage snapshots

**Avoids pitfalls:** Coverage false confidence (Pitfall 3), stale metrics (Pitfall 6)

**Stack elements:** `numpy.histogramdd`, `scipy.stats.qmc.discrepancy`, `json` (coverage snapshots)

**Architecture components:** `coverage.py`, `stop_conditions.py`, `runner.py` (extended with coverage checks)

### Phase 3: Advanced Quality and Diagnostics

**Rationale:** With health monitoring and coverage in place, add deeper quality analysis: per-batch quality scoring, per-worker FPS degradation detection, episode ID integrity verification, and auto-diagnosis routines. These features catch subtle issues that basic NaN detection and coverage tracking miss.

**Delivers:** Batch-level quality scoring (action variance, state delta variance, episode length distribution), FPS degradation alerts, episode ID collision detection, auto-diagnosis with recommended actions.

**Addresses features:** Data quality scoring per batch, per-worker FPS tracking, automated event documentation

**Avoids pitfalls:** Episode ID overflow (Pitfall 7), perturbation distribution shift (Pitfall 8), stale metrics (Pitfall 6)

**Stack elements:** `torch` (batch file loading), `numpy` (statistical analysis), `wandb` (quality dashboards)

**Architecture components:** `quality.py` (full implementation), `diagnostics.py`, `docs.py`

### Phase 4: Integration and Hardening

**Rationale:** The final phase wires the monitor into the existing collection pipeline. This is deliberately last because it involves modifying proven code (`collect_data.py`), and all monitor components should be tested independently first. Also includes systemd service setup for production overnight runs.

**Delivers:** Stop signal integration with `collect_data.py`, W&B run linking between collector and monitor, systemd service file for process supervision, end-to-end integration test.

**Addresses features:** Complete autonomous operation, production deployment readiness

**Avoids pitfalls:** All -- integration testing validates the full pitfall-prevention stack

**Architecture components:** Modifications to `collect_data.py` (stop signal reading), systemd service configuration

### Phase Ordering Rationale

- **Health before coverage:** Coverage tracking reads batch files -- the same mechanism that health monitoring uses to detect worker issues. Build the file-reading infrastructure once in Phase 1, extend it for coverage in Phase 2.
- **NaN detection before coverage analysis:** If batch files contain NaN transitions, coverage analysis produces misleading results. Clean data is a prerequisite for meaningful coverage metrics.
- **Coverage before advanced quality:** Per-batch quality scoring and diagnostics are refinements. They matter less than knowing whether the dataset covers the state-action space adequately.
- **Integration last:** The existing `collect_data.py` is proven working code. Modify it only after the monitor is fully functional and tested independently. A bug in early integration could break data collection itself.
- **Dependency chain mirrors architecture build order:** health.py and state.py (no deps) -> coverage.py and quality.py (depend on batch file reading) -> stop_conditions.py (depends on coverage and health outputs) -> runner.py (orchestrates all) -> integration with collect_data.py.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Coverage Tracking):** The right binning strategy (number of bins, which dimensions, minimum samples per bin threshold) requires experimentation. The 4D state summary features may not capture the dimensions that matter most for surrogate accuracy. Consider running a pilot collection and analyzing which state dimensions have the highest surrogate error variance.
- **Phase 3 (Quality/Diagnostics):** Perturbation distribution shift (Pitfall 8) is a subtle issue. Quantifying whether perturbed data helps or hurts surrogate accuracy requires comparing surrogate performance on perturbed vs. unperturbed validation sets -- this analysis does not exist yet.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Worker health monitoring, NaN detection, disk checks, and W&B alerts are well-documented patterns with clear APIs. psutil, multiprocessing, and wandb all have excellent documentation. No ambiguity in implementation.
- **Phase 4 (Integration):** Reading a stop signal file from the collection loop is trivial. systemd service configuration is standard. No research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Every library is already installed and verified. No new dependencies. API patterns confirmed against official documentation. |
| Features | HIGH | Feature list derived from codebase analysis of the existing pipeline. Dependencies between features are clear and validated against the architecture. |
| Architecture | HIGH | External observer pattern is proven for monitoring systems. Component boundaries align with the existing code structure. Build order validated against dependency chain. |
| Pitfalls | HIGH | Pitfalls identified from actual codebase analysis (specific line numbers, specific bugs), known project issues, and domain literature. Recovery strategies are concrete. |

**Overall confidence:** HIGH

All four research files drew from the actual codebase, installed package versions, and official documentation. The project is operating on well-trodden ground: multiprocess monitoring with coverage tracking is a standard pattern in ML data pipelines. The main uncertainty is in the coverage binning strategy (Phase 2), which requires empirical tuning.

### Gaps to Address

- **Optimal coverage binning dimensions and resolution:** The 4D state summary features (CoM_x, CoM_y, vel_mag, mean_omega) are inherited from the existing density weighting code but may not be the dimensions that matter most for surrogate accuracy. Validate during Phase 2 by correlating coverage gaps with surrogate prediction errors.
- **Perturbation quality impact:** No existing analysis compares surrogate accuracy on perturbed vs. unperturbed test data. This gap means we cannot yet determine if the 30% perturbation fraction is helping or hurting. Address during Phase 3 with a stratified evaluation.
- **Worker restart safety:** The architecture research recommends signaling the parent process to restart workers rather than having the external monitor inject new processes. The exact IPC mechanism (signal file vs. Unix signal vs. shared memory flag) needs to be decided during Phase 1 implementation.
- **W&B run linking strategy:** Should the monitor share the collector's W&B run (by resuming it) or create a separate linked "monitoring" run? Creating a separate run is safer (avoids metric key collisions) but loses the single-dashboard view. Decide during Phase 4 integration.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `aprx_model_elastica/collect_data.py`, `collect_config.py`, `dataset.py`, `state.py`, `validate.py`, `env.py`, `model.py`
- Existing scripts: `script/validate_surrogate_data.py`, `script/smoke_test_collect.py`
- [W&B Alert Documentation](https://docs.wandb.ai/models/runs/alert) -- alert API, AlertLevel enum, wait_duration
- [psutil Documentation](https://psutil.readthedocs.io/en/latest/) -- Process monitoring, cpu_percent, memory_info, disk_usage
- [NumPy histogramdd](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html) -- multi-dimensional histogram for coverage grid
- [SciPy QMC discrepancy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.discrepancy.html) -- coverage quality metric
- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html) -- worker crash behavior, shared state
- [filelock Documentation](https://py-filelock.readthedocs.io/en/latest/) -- cross-process file locking
- [systemd service documentation](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html) -- Restart=on-failure

### Secondary (MEDIUM confidence)
- Project issues: `surrogate-data-collection-no-append.md`, `surrogate-model-data-imbalance.md`, `parallel-collection-scaling-bottleneck.md`, `surrogate-spatial-structure-analysis.md`
- Project experiments: `surrogate-parallel-data-collection.md`, `surrogate-data-coverage-improvements.md`
- [W&B GitHub issue #10025](https://github.com/wandb/wandb/issues/10025) -- ConnectTimeout in long-running jobs
- [Stable Baselines3: Dealing with NaNs](https://stable-baselines3.readthedocs.io/en/master/guide/checking_nan.html) -- NaN detection patterns
- [Data Pipeline Monitoring - Monte Carlo](https://www.montecarlodata.com/blog-data-pipeline-monitoring/) -- monitoring patterns
- [Data Quality Framework Guide](https://www.montecarlodata.com/blog-data-quality-framework/) -- quality metrics
- [GROMACS: Managing long simulations](https://manual.gromacs.org/current/user-guide/managing-simulations.html) -- checkpointing best practices

### Tertiary (LOW confidence)
- [Active Learning for Surrogate Model Improvement](https://link.springer.com/article/10.1007/s00158-024-03816-9) -- informed anti-feature decision (adaptive sampling deferred)
- [AI Agent Design Patterns - Microsoft](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns) -- general agent architecture patterns
- [Building Surrogate Models with RL Trajectories](https://arxiv.org/abs/2509.01285) -- surrogate training context

---
*Research completed: 2026-03-09*
*Ready for roadmap: yes*
