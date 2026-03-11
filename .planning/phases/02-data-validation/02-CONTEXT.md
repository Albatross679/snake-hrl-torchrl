# Phase 2: Data Validation - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Analyze the Phase 2.2 surrogate dataset (~4.3M transitions, ~43K .pt batch files in data/surrogate_rl_step/, 124-dim states, 5-dim actions) for distribution evenness, data quality, anomalies, and coverage gaps. Produce a summary report with structured pass/fail assessment and actionable prose recommendations for surrogate training readiness. Does NOT cover: automatic recollection, config generation for gaps, or surrogate model training.

</domain>

<decisions>
## Implementation Decisions

### Visualization depth
- Summary features only: 4 summary features (CoM_x, CoM_y, velocity magnitude, mean angular velocity) + 5 action dimension histograms
- Matches the features already used by `compute_density_weights()` in dataset.py
- No per-dimension histograms for all 124 state dims (most node-level dims would look similar)

### Action space coverage visualization
- 1D marginal histograms only (one per action dim: amplitude, frequency, wave_number, phase_offset, direction_bias)
- No pairwise 2D heatmaps

### Temporal analysis
- Statistics only: episode length distribution histogram, step index distribution, early-vs-late bias chart
- No individual trajectory plots

### Figure output
- Save to `figures/data_validation_rl_step/` (follows existing pattern from `figures/surrogate_validation/`)

### Actionable recommendations
- Prose recommendations in the summary report (e.g., "Action dim 3 under-sampled in range [0.5, 0.8] — consider targeted recollection")
- No automatic recollection config generation

### Report format
- Structured pass/fail rubric: table with each metric (NaN rate, coverage fill, outlier %, episode length variance, etc.) and PASS/WARN/FAIL status per metric
- Clear go/no-go decision for surrogate training readiness

### Output destinations
- Summary report: `data/surrogate_rl_step/validation_report.md` (co-located with dataset)
- Figures: `figures/data_validation_rl_step/`
- No W&B logging (one-shot analysis, not a running process)

### Claude's Discretion
- Specific pass/fail threshold values for each metric
- Which outlier detection method to use (>5 sigma per requirements, but algorithm details flexible)
- Report markdown formatting and section ordering
- Number of histogram bins for each plot

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. The requirements (DVAL-01 through DVAL-05) are precise enough to guide implementation.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `dataset.py:SurrogateDataset`: Already loads and concatenates all batch files with episode-level train/val split — can be used to load the full dataset
- `dataset.py:compute_density_weights()`: Computes inverse-density weights from 4 summary features (CoM_x, CoM_y, vel_mag, mean_omega) — same features for validation histograms
- `state.py`: Named slices (POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z) and constants (STATE_DIM=124, ACTION_DIM=5, NUM_NODES=21, NUM_ELEMENTS=20)
- `health.py:validate_episode_finite()`: NaN/Inf checking utility
- `validate.py:_save_plots()`: Matplotlib plotting pattern with Agg backend, dpi=150, bbox_inches="tight"

### Established Patterns
- `.pt` batch files with keys: states, actions, serpenoid_times, next_states, episode_ids, step_indices, forces
- State vector layout: pos_x(21), pos_y(21), vel_x(21), vel_y(21), yaw(20), omega_z(20)
- Actions: amplitude, frequency, wave_number, phase_offset, direction_bias (all normalized [-1, 1])
- Figures saved to `figures/` subdirectories with matplotlib Agg backend
- CLI entry point pattern: `python -m aprx_model_elastica.<module>` with argparse

### Integration Points
- Reads from `data/surrogate_rl_step/` (Phase 2.2 output)
- Report written to `data/surrogate_rl_step/validation_report.md`
- Figures written to `figures/data_validation_rl_step/`
- New module: `aprx_model_elastica/validate_data.py` (separate from existing `validate.py` which is surrogate model validation)

</code_context>

<deferred>
## Deferred Ideas

- Pairwise 2D action heatmaps for joint coverage analysis — v2 (ADVN-01)
- State-action joint coverage heatmaps with PCA projections — v2 (ADVN-01)
- Coverage gap analysis for targeted recollection configs — v2 (ADVN-02, ADVN-03)
- W&B logging for comparing validation across collection rounds — future if needed
- Per-episode trajectory visualization (CoM paths) — future if needed

</deferred>

---

*Phase: 02-data-validation*
*Context gathered: 2026-03-10*
