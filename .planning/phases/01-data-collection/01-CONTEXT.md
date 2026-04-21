# Phase 1: Health Monitoring and Data Integrity - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Keep the data collection pipeline alive and producing clean data throughout an overnight run, with full visibility into what happened. Covers: worker crash/stall detection, worker respawning, NaN/Inf data filtering, graceful shutdown, W&B alerts, and structured JSON event logging. Does NOT cover: coverage tracking (Phase 2), quality analysis (Phase 3), or stop conditions beyond basic health.

</domain>

<decisions>
## Implementation Decisions

### Monitor architecture
- External observer process (`python -m aprx_model_elastica.monitor`) runs alongside the collector
- Communicates via filesystem signals (decided at roadmap creation)
- Polls every 30 seconds (matching existing poll interval in collect_data.py)

### Claude's Discretion
- **NaN/Inf filtering strategy:** Per-transition vs per-episode discard, where filtering happens (worker-side vs save-time), whether to log discarded data for debugging
- **Worker recovery behavior:** Fresh start vs resume on respawn, seed strategy for respawned workers, handling of incomplete batch data from crashed workers
- **Shutdown coordination:** Signal propagation mechanism, atomic batch save strategy, drain vs immediate stop
- **Event log schema:** JSON structure, file location, rotation policy
- **W&B alert thresholds:** What constitutes "high NaN rate", alert rate limiting to avoid overnight spam, severity levels for different event types
- **Stall detection criteria:** How to determine a worker is stalled (zero progress for N consecutive intervals), restart mechanism

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. The success criteria in ROADMAP.md are precise enough to guide implementation:
1. Monitor shows per-worker alive/dead/stalled status every poll interval
2. Killed worker detected and respawned within 60 seconds
3. NaN/Inf episodes silently discarded, discard count in W&B
4. SIGINT/SIGTERM produces clean shutdown with no corrupt batch files
5. W&B alerts for worker death, stall, high NaN rate; structured JSON event log

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `collect_data.py:_multiprocess_collect()`: Existing monitoring loop with 30s polling, `is_alive()` checks, W&B metric logging — foundation for health monitor
- `collect_data.py:_worker_fn()`: Worker process pattern with per-worker seeds, per-worker batch prefixes (`batch_w{id:02d}`)
- `collect_data.py:_save_batch()` / `_save_batch_pt()` / `_save_batch_parquet()`: Batch save functions (need atomicity wrapper)
- `collect_data.py:_find_next_batch_idx()`: Resume-safe batch indexing (already handles append mode)
- `collect_config.py:DataCollectionConfig`: Dataclass config pattern — extend for monitor settings
- `src/trainers/ppo.py:PPOTrainer`: Has SIGINT/SIGTERM signal handler and graceful shutdown pattern to reference

### Established Patterns
- Dataclass-based hierarchical configuration (`DataCollectionConfig` extends with `LocomotionElasticaEnvConfig`)
- `multiprocessing.Process` with `forkserver` start method for workers
- `mp.Value` shared counter for cross-process progress tracking
- W&B integration: `wandb.init()`, `wandb.log()` for metrics (need to add `wandb.alert()`)
- OpenBLAS/OMP/MKL thread limiting in workers

### Integration Points
- Monitor reads filesystem signals written by collector workers
- Monitor writes to same W&B run (or its own run in same project)
- Event log written alongside batch files in save_dir
- New `aprx_model_elastica/monitor.py` module invoked via `__main__.py` dispatcher

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-health-monitoring-and-data-integrity*
*Context gathered: 2026-03-09*
