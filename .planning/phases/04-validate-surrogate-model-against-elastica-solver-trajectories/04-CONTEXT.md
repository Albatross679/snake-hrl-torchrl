# Phase 4: Validate Surrogate Model Against Elastica Solver Trajectories - Context

**Gathered:** 2026-03-10
**Updated:** 2026-03-11 (architecture-aware model loading for Phase 3's multi-architecture sweep)
**Status:** Ready for planning

<domain>
## Phase Boundary

Compare trained surrogate model predictions against ground-truth PyElastica trajectories across multiple action regimes to determine if the surrogate is accurate enough for RL training. Produce a structured validation report with per-scenario pass/fail assessment, diagnostic figures, and actionable recommendations. Does NOT cover: surrogate model retraining, data recollection, or RL training.

</domain>

<decisions>
## Implementation Decisions

### Validation scenarios
- **4 scenario types**, 10 episodes each (40 total):
  1. **Random actions** — uniform random in [-1, 1] across all 5 action dims (matches training distribution)
  2. **Forward crawling** — fixed params: amplitude ~2.5, frequency ~1.12 Hz, wave_number ~1.0 (known best locomotion params)
  3. **Slow vs fast gaits** — low-frequency sweep (0.5 Hz) and high-frequency sweep (3.0 Hz), 5 episodes each
  4. **Trained policy rollouts** — use Session 10 best PPO policy (`output/locomotion_elastica_forward_20260307_021959/checkpoints/best.pt`) to generate actions, then replay through surrogate
- Each episode runs up to 500 steps (full episode length)

### Pass/fail criteria
- **Primary horizon:** Step 50 (50-step accuracy is proxy for TD-based RL learning)
- **Overall RMSE threshold** at step 50: Claude sets data-driven thresholds (PASS/WARN/FAIL) based on actual validation results
- **CoM drift threshold** at step 50: relative to real displacement (PASS < 10%, WARN 10-25%, FAIL > 25%) — Claude sets exact numbers
- **Per-component flagging:** report any component (pos_x, pos_y, vel_x, vel_y, yaw, omega_z) that's >2x worse than average RMSE — diagnostic only, not separate pass/fail
- **Per-scenario grading:** each of the 4 scenarios gets its own PASS/WARN/FAIL verdict

### Failure tiers
- **FAIL-SOFT:** surrogate is reasonable but below threshold — recommend tuning (more training epochs, adjust rollout loss weight, etc.)
- **FAIL-HARD:** surrogate diverges badly or produces NaN — fundamental issue, likely needs more/better data or architecture change

### Verdict logic
- **Overall verdict** is "PASS with caveats" if some scenarios PASS and others FAIL — lists which passed/failed
- **WARN is advisory** — does not block proceeding to RL training, only FAIL blocks
- Report includes a **Diagnosis section** when any scenario fails: which scenarios/components failed, likely causes, specific recommendation (more data, bigger model, longer training, etc.)

### Report & output
- **Report path:** `output/surrogate/validation_report.md` (co-located with model checkpoint)
- **Figures path:** `figures/surrogate_validation/` (existing directory)
- **No W&B logging** — one-shot validation, results saved locally
- **CLI entry point:** `python -m aprx_model_elastica.validate --surrogate-checkpoint output/surrogate` (existing pattern)

### Figures
- **Existing plots** (already in validate.py): RMSE over time, CoM drift over time
- **New: Trajectory overlays** — real vs surrogate CoM xy-path, 3 per scenario (best/median/worst by RMSE@50) = 12 plots
- **New: Per-component error heatmap** — component vs time step, colored by RMSE, shows which components diverge fastest
- **New: Scenario comparison bars** — bar chart of RMSE@50 across the 4 scenarios

### Architecture-aware model loading (LOCKED)
- Read `arch` field from `config.json` in checkpoint directory to dispatch model class
- Dispatch map: `"mlp"` → `SurrogateModel`, `"residual"` → `ResidualSurrogateModel`, `"transformer"` → `TransformerSurrogateModel`
- **Error out** if `config.json` is missing or lacks `arch` field — no fallback to MLP
- Validate only `output/surrogate/best/` (best model selected in Phase 3)
- Same validation logic for all architectures — `predict_next_state(state, action, time_enc, normalizer)` interface is unified across all 3 model classes

### Claude's Discretion
- Exact PASS/WARN/FAIL threshold values (data-driven from actual results)
- Internal refactoring of validate.py to support multiple scenario types
- How to load and run the trained PPO policy for action generation
- Specific outlier handling if surrogate produces NaN predictions
- Figure styling, colors, and layout details
- Whether to use separate figures or subplots for trajectory overlays

</decisions>

<specifics>
## Specific Ideas

- Use the Session 10 best.pt policy at `output/locomotion_elastica_forward_20260307_021959/checkpoints/best.pt` for the policy scenario
- Forward crawling params come from verified working config: amp ~2.5, freq ~1.12 Hz, wave_number ~1.0
- Trajectory overlay plots show best/median/worst episode per scenario — most intuitive visual for "does the surrogate move right"
- Follow Phase 2's pass/fail rubric pattern: structured table with per-metric PASS/WARN/FAIL grades

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `validate.py`: Complete validation pipeline — `collect_real_trajectory()`, `rollout_surrogate()`, `compute_errors()`, `_save_plots()`. Needs extension for multiple scenarios, architecture dispatch, and new figures.
- `model.py`: Three model classes with unified `predict_next_state()` interface:
  - `SurrogateModel` (MLP), `ResidualSurrogateModel` (skip connections), `TransformerSurrogateModel` (FT-Transformer with RMSNorm)
  - All share: `forward(state, action, time_encoding)` → delta, `predict_next_state(state, action, time_encoding, normalizer)` → next_state
- `state.py:StateNormalizer`: Load/save normalizer, normalize/denormalize state and delta
- `state.py:RodState2D`: Pack state from PyElastica rod, named slices (POS_X, POS_Y, etc.)
- `train_config.py:SurrogateModelConfig`: Config loading pattern (config.json in checkpoint dir)
- `locomotion_elastica/env.py:LocomotionElasticaEnv`: Real environment for ground-truth trajectories

### Established Patterns
- CLI entry point: `python -m aprx_model_elastica.validate` with argparse
- Matplotlib Agg backend, dpi=150, bbox_inches="tight" for figures
- State vector: 124-dim (pos_x(21), pos_y(21), vel_x(21), vel_y(21), yaw(20), omega_z(20))
- Actions: 5-dim (amplitude, frequency, wave_number, phase_offset, direction_bias) in [-1, 1]
- Error metrics: per-step RMSE, component RMSE, CoM drift, heading drift at horizons [10, 50, 100, 200, 500]
- PPO checkpoint format: TorchRL state dict with actor/critic networks

### Integration Points
- Reads surrogate model from `output/surrogate/` (Phase 3 output)
- Reads trained policy from `output/locomotion_elastica_forward_20260307_021959/checkpoints/best.pt`
- Creates PyElastica env instances for ground-truth rollouts
- Writes report to `output/surrogate/validation_report.md`
- Writes figures to `figures/surrogate_validation/`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-validate-surrogate-model-against-elastica-solver-trajectories*
*Context gathered: 2026-03-10, updated: 2026-03-11*
