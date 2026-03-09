# Roadmap: Surrogate Data Collection Monitoring

## Overview

Transform the existing data collection pipeline into an autonomous overnight system by layering health monitoring (detect and recover from worker failures), coverage-aware stop conditions (stop when data quality is sufficient, not just when a count is reached), and quality analysis with reporting (understand what was collected). Each phase delivers an independently valuable capability: Phase 1 ensures the pipeline stays alive and produces clean data, Phase 2 ensures it collects the right data and stops at the right time, Phase 3 ensures you understand what you got.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Health Monitoring and Data Integrity** - Detect worker crashes/stalls, respawn dead workers, discard bad data, alert via W&B, and preserve data on shutdown
- [ ] **Phase 2: Coverage Tracking and Smart Stop** - Track state-action space coverage, enforce multi-criteria stop conditions, checkpoint coverage state, and monitor disk space
- [ ] **Phase 3: Quality Analysis and Reporting** - Compute per-batch quality metrics, flag anomalies, and generate end-of-collection summary report

## Phase Details

### Phase 1: Health Monitoring and Data Integrity
**Goal**: The monitoring system keeps the collection pipeline alive and producing clean data throughout an overnight run, with full visibility into what happened
**Depends on**: Nothing (first phase)
**Requirements**: HLTH-01, HLTH-02, HLTH-03, HLTH-04, HLTH-05, OBSV-01, OBSV-04
**Success Criteria** (what must be TRUE):
  1. Running `python -m aprx_model_elastica monitor` alongside the collector shows per-worker alive/dead/stalled status every poll interval
  2. When a worker process is manually killed, the monitor detects it and the worker is respawned within 60 seconds with collection resuming
  3. Episodes containing NaN or Inf values are silently discarded (never written to batch files), and the discard count appears in W&B logs
  4. Sending SIGINT or SIGTERM to the collection process results in a clean shutdown with no truncated or corrupt batch files
  5. W&B alerts fire for worker death, worker stall, and high NaN rate events, and all monitoring events are recorded in a structured JSON event log
**Plans:** 2 plans

Plans:
- [ ] 01-01-PLAN.md — Health infrastructure: per-worker counters, NaN filtering, atomic saves, graceful shutdown, event logging
- [ ] 01-02-PLAN.md — Worker lifecycle: crash/stall detection, respawning, W&B alerts, external monitor process

### Phase 2: Coverage Tracking and Smart Stop
**Goal**: The collection system tracks how well the dataset covers the state-action space and stops only when quantity, quality, and time criteria are all met
**Depends on**: Phase 1
**Requirements**: STOP-01, STOP-02, STOP-03, STOP-04
**Success Criteria** (what must be TRUE):
  1. Collection does not stop until all three conditions are met: elapsed time >= min_hours, total transitions >= min_count, and coverage score >= threshold
  2. If disk space drops below 2 GB, the monitor alerts via W&B and triggers a graceful stop before any data corruption occurs
  3. State-action coverage fill fraction is logged to W&B every poll interval, showing coverage growth over time as a live chart
  4. Coverage grid state is saved as a JSON checkpoint at configurable intervals, and these checkpoints can be loaded for post-hoc analysis
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: Quality Analysis and Reporting
**Goal**: After collection completes, the user has a clear picture of dataset quality with per-batch metrics, flagged anomalies, and a comprehensive summary report
**Depends on**: Phase 2
**Requirements**: OBSV-02, OBSV-03, OBSV-05
**Success Criteria** (what must be TRUE):
  1. After each batch save, quality metrics (action variance, state delta variance, episode length distribution) are computed and logged to W&B
  2. Batches with suspiciously low variance or anomalous statistics are flagged in both the console log and W&B with clear descriptions of what is abnormal
  3. When collection ends, a markdown summary report is generated containing coverage statistics, an event timeline, data quality summary, and any flagged anomalies
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Health Monitoring and Data Integrity | 0/2 | Planning complete | - |
| 2. Coverage Tracking and Smart Stop | 0/2 | Not started | - |
| 3. Quality Analysis and Reporting | 0/2 | Not started | - |
