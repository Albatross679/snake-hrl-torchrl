# Requirements: Surrogate Data Collection & Validation

**Defined:** 2026-03-09
**Updated:** 2026-03-10
**Core Value:** Produce a high-quality, well-covered dataset of snake robot dynamics transitions ready for surrogate model training

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Collection (Phase 1 — COMPLETE)

- [x] **DCOL-01**: Parallel data collection from PyElastica with 16 workers
- [x] **DCOL-02**: Sobol quasi-random actions for better 5D coverage
- [x] **DCOL-03**: State perturbation (30% of episodes) for diverse initial conditions
- [x] **DCOL-04**: NaN/Inf filtering, atomic saves, graceful shutdown
- [x] **DCOL-05**: Health monitoring with crash/stall detection and W&B alerts

### Data Validation (Phase 2)

- [x] **DVAL-01**: Per-dimension distribution analysis — histograms for all 124 state dims and 5 action dims showing uniformity or skew
- [x] **DVAL-02**: Data quality checks — NaN/Inf count, duplicate transitions, constant/near-constant features, outlier detection (>5 sigma)
- [x] **DVAL-03**: Temporal analysis — episode length distribution, step index bias (early vs late timestep over-representation)
- [x] **DVAL-04**: Action space coverage — 5D coverage metric (binned fill fraction), identification of under-sampled action regions
- [x] **DVAL-05**: Summary report with pass/fail assessment per metric and actionable recommendations for surrogate training readiness

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Analysis

- **ADVN-01**: State-action joint coverage heatmaps (PCA-reduced 2D projections)
- **ADVN-02**: Coverage gap analysis identifying specific regions needing follow-up collection
- **ADVN-03**: Automatic perturbation parameter adjustment based on coverage gaps

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time web dashboard | W&B already provides dashboards |
| Automatic recollection | Validate first, then decide manually |
| Distributed multi-machine | Single machine with 48 CPUs sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DCOL-01 | Phase 1 | Complete |
| DCOL-02 | Phase 1 | Complete |
| DCOL-03 | Phase 1 | Complete |
| DCOL-04 | Phase 1 | Complete |
| DCOL-05 | Phase 1 | Complete |
| DVAL-01 | Phase 2 | Complete |
| DVAL-02 | Phase 2 | Complete |
| DVAL-03 | Phase 2 | Complete |
| DVAL-04 | Phase 2 | Complete |
| DVAL-05 | Phase 2 | Complete |

**Coverage:**
- v1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-10 — restructured for data collection (complete) + validation*
