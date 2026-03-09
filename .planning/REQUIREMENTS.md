# Requirements: Surrogate Data Collection Monitoring

**Defined:** 2026-03-09
**Core Value:** Produce a high-quality, well-covered dataset of snake robot dynamics transitions without manual babysitting

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Health & Reliability

- [ ] **HLTH-01**: Monitor reports per-worker alive/dead status every poll interval (30s)
- [ ] **HLTH-02**: Dead workers are automatically detected and respawned with new seed within 60 seconds
- [ ] **HLTH-03**: Stalled workers (alive but zero progress for 2+ consecutive intervals) are detected and restarted
- [ ] **HLTH-04**: Episodes containing NaN or Inf values are discarded before saving, with discard count logged
- [ ] **HLTH-05**: Graceful shutdown on SIGINT/SIGTERM preserves all completed batch files without corruption

### Stop Conditions & Coverage

- [ ] **STOP-01**: Collection continues until all three criteria are met: elapsed >= min_hours AND total_transitions >= min_count AND coverage_score >= threshold
- [ ] **STOP-02**: Disk space checked every poll interval; alert and graceful stop if free space drops below 2 GB
- [ ] **STOP-03**: State-action coverage tracked via binned grid (separate state 4D and action 5D grids) with fill fraction logged to W&B
- [ ] **STOP-04**: Coverage grid state saved as checkpoint every N minutes for post-hoc analysis

### Observability & Alerting

- [ ] **OBSV-01**: W&B alerts sent for worker death, worker stall, low disk, high NaN rate (using wandb.alert with rate limiting)
- [ ] **OBSV-02**: Per-batch quality metrics computed after each save: action variance, state delta variance, episode length distribution
- [ ] **OBSV-03**: Batches with suspiciously low variance or anomalous metrics flagged in logs and W&B
- [ ] **OBSV-04**: All monitoring events (crashes, restarts, stalls, alerts, coverage milestones) logged to a structured JSON event log
- [ ] **OBSV-05**: End-of-collection summary markdown report generated with coverage stats, event timeline, and data quality summary

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Monitoring

- **ADVN-01**: Per-worker FPS tracking with degradation detection (alert if worker drops below 50% baseline)
- **ADVN-02**: Coverage gap analysis report identifying which state-action regions need follow-up collection
- **ADVN-03**: Automatic perturbation parameter adjustment based on coverage gaps

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time web dashboard | W&B already provides dashboards; custom UI is a full project |
| Adaptive sampling / active learning | IPC overhead and complexity not justified; Sobol + density weighting sufficient |
| Automatic worker count scaling | Hardware has known ceiling at ~16 workers; dynamic scaling complicates ID management |
| Distributed multi-machine collection | CPU-bound pipeline; single machine with 48 CPUs sufficient |
| GPU-accelerated physics | PyElastica is CPU-only; changing simulator invalidates surrogate model |
| Automatic training trigger | Better to validate data quality first, then manually launch training |
| Email/SMS notifications | W&B alerts already provide Slack/email |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| HLTH-01 | Phase 1 | Pending |
| HLTH-02 | Phase 1 | Pending |
| HLTH-03 | Phase 1 | Pending |
| HLTH-04 | Phase 1 | Pending |
| HLTH-05 | Phase 1 | Pending |
| STOP-01 | Phase 2 | Pending |
| STOP-02 | Phase 2 | Pending |
| STOP-03 | Phase 2 | Pending |
| STOP-04 | Phase 2 | Pending |
| OBSV-01 | Phase 1 | Pending |
| OBSV-02 | Phase 3 | Pending |
| OBSV-03 | Phase 3 | Pending |
| OBSV-04 | Phase 1 | Pending |
| OBSV-05 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-09 after roadmap creation*
