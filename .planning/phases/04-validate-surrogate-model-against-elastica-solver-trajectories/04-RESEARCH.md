# Phase 4: Validate Surrogate Model Against Elastica Solver Trajectories - Research

**Researched:** 2026-03-11
**Domain:** Surrogate model multi-step trajectory validation, PyElastica ground-truth comparison
**Confidence:** HIGH

## Summary

Phase 4 extends the existing single-scenario random-action validation (`aprx_model_elastica/validate.py`) into a comprehensive 4-scenario validation suite with pass/fail grading, trajectory overlay figures, and a structured markdown report. The core infrastructure already exists: `collect_real_trajectory()`, `rollout_surrogate()`, `compute_errors()`, and basic plotting. The main work is (1) adding scenario-based action generation (random, forward crawl, slow/fast gaits, trained PPO policy), (2) architecture-aware model dispatch using `config.json`'s `arch` field, (3) new figure types (trajectory overlays, per-component heatmaps, scenario comparison bars), and (4) a structured validation report with per-scenario PASS/WARN/FAIL verdicts.

A critical design constraint from the user: the number of surrogate models from Phase 3 is NOT yet determined. However, CONTEXT.md locks validation to `output/surrogate/best/` (best model selected in Phase 3). This directory does not exist yet -- Phase 3 model selection (SURR-03) must create it before Phase 4 runs. The validation script should validate the single best model at `output/surrogate/best/` with architecture dispatch based on `config.json`.

**Primary recommendation:** Refactor `validate.py` to support pluggable action generators (one per scenario), architecture-aware model loading with strict `arch` field requirement, and structured report generation with per-scenario verdicts -- all building on existing `collect_real_trajectory()`, `rollout_surrogate()`, and `compute_errors()` functions.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **4 scenario types**, 10 episodes each (40 total): random actions, forward crawling, slow/fast gaits, trained policy rollouts
- Each episode runs up to 500 steps
- **Primary horizon:** Step 50 for pass/fail assessment
- CoM drift threshold at step 50: relative to real displacement (PASS < 10%, WARN 10-25%, FAIL > 25%)
- Per-component flagging: any component >2x worse than average RMSE (diagnostic only)
- Per-scenario grading: each scenario gets PASS/WARN/FAIL
- Failure tiers: FAIL-SOFT (reasonable but below threshold) vs FAIL-HARD (diverges/NaN)
- Overall verdict: "PASS with caveats" if mixed results
- WARN is advisory only -- does not block RL training
- Report path: `output/surrogate/validation_report.md`
- Figures path: `figures/surrogate_validation/`
- No W&B logging
- CLI entry point: `python -m aprx_model_elastica.validate --surrogate-checkpoint output/surrogate`
- Architecture-aware model loading: read `arch` from `config.json`, dispatch to correct model class
- Dispatch map: `"mlp"` -> `SurrogateModel`, `"residual"` -> `ResidualSurrogateModel`, `"transformer"` -> `TransformerSurrogateModel`
- **Error out** if `config.json` is missing or lacks `arch` field
- Validate only `output/surrogate/best/` (best model from Phase 3)
- Unified `predict_next_state(state, action, time_enc, normalizer)` interface across all architectures
- Existing plots kept: RMSE over time, CoM drift over time
- New plots: trajectory overlays (3 per scenario = 12), per-component error heatmap, scenario comparison bars
- Forward crawling params: amplitude ~2.5, frequency ~1.12 Hz, wave_number ~1.0
- Trained policy from `output/locomotion_elastica_forward_20260307_021959/checkpoints/best.pt`

### Claude's Discretion
- Exact PASS/WARN/FAIL threshold values (data-driven from actual results)
- Internal refactoring of validate.py to support multiple scenario types
- How to load and run the trained PPO policy for action generation
- Specific outlier handling if surrogate produces NaN predictions
- Figure styling, colors, and layout details
- Whether to use separate figures or subplots for trajectory overlays

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.6+ | Model loading, tensor operations | Already used throughout |
| matplotlib | 3.9+ | All validation figures | Already used in validate.py |
| numpy | 2.0+ | Numerical computations, error metrics | Already used throughout |
| PyElastica | 0.3+ | Ground-truth trajectory generation | Project physics backend |
| TorchRL | 0.11.1 | PPO policy loading (ProbabilisticActor) | Project RL framework |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| seaborn | (optional) | Heatmap colormap for per-component error | Only if available, fallback to matplotlib imshow |
| tensordict | 0.7+ | TensorDict for env step interface | Already a dependency |

**Installation:** No new dependencies needed. All libraries already in requirements.txt.

## Architecture Patterns

### Recommended Project Structure
```
aprx_model_elastica/
├── validate.py          # Main validation script (MODIFY)
├── model.py             # Model classes (READ ONLY)
├── state.py             # State utilities (READ ONLY)
└── train_config.py      # Config classes (READ ONLY)

figures/surrogate_validation/
├── rmse_over_time.png           # Existing
├── com_drift.png                # Existing
├── trajectory_overlay_*.png     # NEW: 12 plots (3 per scenario)
├── component_error_heatmap.png  # NEW
└── scenario_comparison_bars.png # NEW

output/surrogate/
├── best/                        # Phase 3 output (model to validate)
│   ├── config.json
│   ├── model.pt
│   └── normalizer.pt
└── validation_report.md         # NEW: structured report
```

### Pattern 1: Architecture-Aware Model Dispatch
**What:** Load correct model class based on `config.json`'s `arch` field.
**When to use:** Any time loading a surrogate model from checkpoint.
**Example:**
```python
# Source: existing pattern in train_surrogate.py lines 441-447
import json
from pathlib import Path
from aprx_model_elastica.model import (
    SurrogateModel, ResidualSurrogateModel, TransformerSurrogateModel
)
from aprx_model_elastica.train_config import SurrogateModelConfig

ARCH_DISPATCH = {
    "mlp": SurrogateModel,
    "residual": ResidualSurrogateModel,
    "transformer": TransformerSurrogateModel,
}

def load_surrogate(ckpt_dir: Path, device: str = "cpu"):
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {ckpt_dir}")

    with open(config_path) as f:
        config_dict = json.load(f)

    if "arch" not in config_dict:
        raise ValueError(f"config.json in {ckpt_dir} lacks 'arch' field")

    model_config = SurrogateModelConfig(**config_dict)
    model_cls = ARCH_DISPATCH[model_config.arch]
    model = model_cls(model_config).to(device)
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    )
    normalizer = StateNormalizer.load(str(ckpt_dir / "normalizer.pt"), device=device)
    return model, normalizer, model_config
```

### Pattern 2: Scenario-Based Action Generator
**What:** Pluggable action generators for different validation scenarios.
**When to use:** Each scenario type needs a different action strategy.
**Example:**
```python
class ActionGenerator:
    """Base class for scenario-specific action generation."""
    def get_action(self, step: int, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

class RandomActionGenerator(ActionGenerator):
    def get_action(self, step, state, rng):
        return rng.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)

class FixedGaitGenerator(ActionGenerator):
    """Fixed serpenoid parameters for specific gaits."""
    def __init__(self, amplitude=0.0, frequency=0.0, wave_number=0.0,
                 phase_offset=0.0, direction_bias=0.0):
        # Store as normalized [-1, 1] values
        self.action = np.array([amplitude, frequency, wave_number,
                                phase_offset, direction_bias], dtype=np.float32)

    def get_action(self, step, state, rng):
        return self.action.copy()

class PolicyActionGenerator(ActionGenerator):
    """Use trained PPO policy for action generation."""
    def __init__(self, policy_path: str, obs_dim: int = 14, device: str = "cpu"):
        # Load actor network from checkpoint
        ...

    def get_action(self, step, state, rng):
        # Map 124-dim rod state to 14-dim observation, run policy
        ...
```

### Pattern 3: PPO Policy Loading for Action Generation
**What:** Load the Session 10 PPO actor to generate trained-policy actions.
**When to use:** Scenario 4 (trained policy rollouts).
**Critical details:**
- PPO checkpoint format: `{"actor_state_dict": ..., "critic_state_dict": ..., "config": LocomotionElasticaConfig, ...}`
- Must use `weights_only=False` due to pickled config object
- Obs dim: 14 (from `LocomotionElasticaEnv.OBS_DIM`)
- Action dim: 5
- Actor hidden dims: [256, 256, 256], activation: tanh
- The 14-dim observation is NOT the 124-dim rod state -- need to reconstruct observation from rod state using the env's `_get_obs()` method
- **Best approach:** Instantiate a minimal `LocomotionElasticaEnv`, set rod state, call `_get_obs()` to get 14-dim obs, then run actor
- **Alternative:** Inspect `_get_obs()` and replicate the 124->14 mapping directly (avoids env dependency in validation loop)
```python
# Source: src/trainers/ppo.py line 520, locomotion_elastica/env.py line 119
checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
actor_state = checkpoint["actor_state_dict"]
config = checkpoint["config"]
# Reconstruct ActorNetwork with obs_dim=14, action_dim=5, hidden_dims=[256,256,256]
```

### Pattern 4: Generalized collect_real_trajectory with Action Generator
**What:** Modify `collect_real_trajectory()` to accept an `ActionGenerator` instead of hardcoded random actions.
**When to use:** All 4 scenarios.
```python
def collect_real_trajectory(
    env: LocomotionElasticaEnv,
    action_gen: ActionGenerator,
    rng: np.random.Generator,
    max_steps: int = 500,
) -> dict:
    td = env.reset()
    states = [RodState2D.pack_from_rod(env._rod)]
    actions = []
    serp_times = [env._serpenoid._time]

    done = False
    step = 0
    while not done and step < max_steps:
        state = states[-1]
        action = action_gen.get_action(step, state, rng)
        actions.append(action)
        # ... rest same as existing
```

### Anti-Patterns to Avoid
- **Hardcoding model class to SurrogateModel:** The existing validate.py does `model = SurrogateModel(model_config)` -- must use dispatch map instead.
- **Assuming config.json always has `arch`:** Older checkpoints (sweep runs, arch_sweep/arch_A1_rw0.0, arch_B1_residual) lack the `arch` field. Validation MUST error out on these per CONTEXT.md, not silently default to MLP.
- **Using 124-dim state as PPO observation:** The PPO policy expects 14-dim observations, not raw rod state. Must transform through the env's observation extraction.
- **Running env.step() instead of env._step():** The existing code calls `env._step()` directly (bypasses TorchRL wrapper overhead). Keep this pattern.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CoM computation | Custom averaging | `real[:, POS_X].mean(axis=1)` (existing) | Already correct in compute_errors() |
| Phase encoding | Raw sin/cos | `action_to_omega()` + `encode_phase()` from state.py | Handles frequency denormalization correctly |
| Heading computation | Manual atan2 | Existing heading_drift in compute_errors() | Already handles wraparound |
| Figure DPI/sizing | Custom defaults | dpi=150, bbox_inches="tight" (project standard) | Consistent with all other figures |
| Action denormalization | Manual scaling | Use existing env config ranges | Avoids drift from actual training ranges |

**Key insight:** The existing `validate.py` already has correct error computation. The refactoring is about adding scenarios and report structure, not rewriting metrics.

## Common Pitfalls

### Pitfall 1: PPO Checkpoint Loading Requires weights_only=False
**What goes wrong:** `torch.load(..., weights_only=True)` fails because the PPO checkpoint contains a pickled `LocomotionElasticaConfig` object.
**Why it happens:** PPO trainer saves the full config object, not just state dicts.
**How to avoid:** Use `weights_only=False` for PPO checkpoint only. Surrogate model.pt uses `weights_only=True` (it's a pure state dict).
**Warning signs:** `_pickle.UnpicklingError: Weights only load failed` with `GLOBAL locomotion_elastica.config.LocomotionElasticaConfig`

### Pitfall 2: Observation Dimension Mismatch (124 vs 14)
**What goes wrong:** Feeding 124-dim rod state directly to PPO actor produces wrong actions or crashes.
**Why it happens:** `LocomotionElasticaEnv.OBS_DIM = 14`. The env extracts a compact 14-dim observation from the full rod state (distances, angles, velocities relative to goal).
**How to avoid:** Either use the env's `_get_obs()` method or replicate the observation extraction logic. The env needs goal position and rod state to compute observations.
**Warning signs:** Shape mismatch errors, nonsensical policy actions.

### Pitfall 3: Serpenoid Time Accumulation in Surrogate Rollout
**What goes wrong:** Time encoding becomes incorrect if serpenoid time is not properly accumulated during surrogate-only rollouts.
**Why it happens:** In real env, `env._serpenoid._time` accumulates. During surrogate rollout, must manually track time: `t += dt` where dt depends on physics substeps.
**How to avoid:** Use the serpenoid times captured during real trajectory collection (existing pattern in `collect_real_trajectory()`).

### Pitfall 4: NaN Propagation in Autoregressive Rollout
**What goes wrong:** Single NaN prediction cascades through entire remaining trajectory.
**Why it happens:** `next_state = state + delta` -- if delta contains NaN, all subsequent predictions are NaN.
**How to avoid:** Add NaN detection per step in `rollout_surrogate()`. If NaN detected, mark episode as FAIL-HARD and truncate. Report the step at which NaN first appeared.

### Pitfall 5: Missing `arch` Field in Older Checkpoints
**What goes wrong:** Validation script crashes on checkpoint without `arch` field.
**Why it happens:** Phase 3 Plan 01 sweep runs and some arch_sweep runs saved config.json without `arch` field (verified: `arch_A1_rw0.0/config.json` and `arch_B1_residual/config.json` both lack it).
**How to avoid:** Per CONTEXT.md decision, error out with clear message. Phase 3's best model selection (SURR-03) must ensure `output/surrogate/best/config.json` includes `arch`. This is a Phase 3 responsibility, not Phase 4.

### Pitfall 6: Forward Crawling Action Normalization
**What goes wrong:** Raw physics values (amp=2.5, freq=1.12 Hz, wave_number=1.0) fed directly as actions when they should be in normalized [-1, 1] space.
**Why it happens:** Actions in the env are normalized. Amplitude range is [0, 5], frequency range is [0.5, 3.0], wave_number range derived from denorm.
**How to avoid:** Convert physical parameters to normalized [-1, 1] actions:
- amplitude=2.5 -> `(2.5/5.0)*2 - 1 = 0.0`
- frequency=1.12 Hz -> `((1.12-0.5)/(3.0-0.5))*2 - 1 = -0.504`
- wave_number=1.0 -> `((1.0-0.5)/3.0)*2 - 1 = -0.667`

## Code Examples

### Loading Architecture-Aware Surrogate Model
```python
# Source: existing pattern in train_surrogate.py + CONTEXT.md decision
import json
import torch
from pathlib import Path

def load_surrogate_model(ckpt_dir: Path, device: str = "cpu"):
    """Load surrogate model with architecture dispatch."""
    from aprx_model_elastica.model import (
        SurrogateModel, ResidualSurrogateModel, TransformerSurrogateModel
    )
    from aprx_model_elastica.train_config import SurrogateModelConfig
    from aprx_model_elastica.state import StateNormalizer

    DISPATCH = {
        "mlp": SurrogateModel,
        "residual": ResidualSurrogateModel,
        "transformer": TransformerSurrogateModel,
    }

    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json in {ckpt_dir}. Cannot determine model architecture."
        )

    with open(config_path) as f:
        cfg = json.load(f)

    if "arch" not in cfg:
        raise ValueError(
            f"config.json in {ckpt_dir} missing 'arch' field. "
            "Ensure Phase 3 model selection wrote arch to config.json."
        )

    arch = cfg["arch"]
    if arch not in DISPATCH:
        raise ValueError(f"Unknown arch '{arch}'. Expected one of {list(DISPATCH.keys())}")

    model_config = SurrogateModelConfig(**cfg)
    model = DISPATCH[arch](model_config).to(device)
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()

    normalizer = StateNormalizer.load(str(ckpt_dir / "normalizer.pt"), device=device)
    return model, normalizer, model_config
```

### NaN-Safe Surrogate Rollout
```python
# Source: extends existing rollout_surrogate() in validate.py
def rollout_surrogate(model, normalizer, initial_state, actions, serpenoid_times, device="cpu"):
    """Autoregressively unroll surrogate. Detects NaN and truncates."""
    states = [initial_state.copy()]
    state = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0)
    nan_step = None

    with torch.no_grad():
        for t in range(len(actions)):
            action = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
            omega = action_to_omega(actions[t])
            phase = omega * serpenoid_times[t]
            time_enc = torch.tensor(
                encode_phase(phase), dtype=torch.float32, device=device
            ).unsqueeze(0)

            state = model.predict_next_state(state, action, time_enc, normalizer)

            if torch.isnan(state).any():
                nan_step = t + 1
                break

            states.append(state.squeeze(0).cpu().numpy())

    return np.stack(states), nan_step
```

### Forward Crawling Action (Normalized)
```python
# Source: CLAUDE.md verified params + state.py denorm ranges
def make_forward_crawl_action():
    """Fixed forward crawl: amp~2.5, freq~1.12 Hz, wave_number~1.0."""
    # Amplitude: [0, 5] -> 2.5 normalized = (2.5/5)*2-1 = 0.0
    # Frequency: [0.5, 3.0] -> 1.12 normalized = ((1.12-0.5)/(3.0-0.5))*2-1 = -0.504
    # Wave number: [0.5, 3.5] -> 1.0 normalized = ((1.0-0.5)/3.0)*2-1 = -0.667
    # Phase offset: 0.0 (centered)
    # Direction bias: 0.0 (straight)
    return np.array([0.0, -0.504, -0.667, 0.0, 0.0], dtype=np.float32)
```

### Structured Report Generation
```python
# Source: follows Phase 2 validation report pattern (validate_data.py)
def generate_report(scenario_results: dict, report_path: Path):
    """Write structured validation report with per-scenario verdicts."""
    lines = [
        "# Surrogate Model Validation Report",
        f"\n**Generated:** {datetime.datetime.now().isoformat()[:19]}",
        f"**Checkpoint:** {ckpt_dir}",
        f"**Architecture:** {model_config.arch}",
        "",
        "## Summary",
        "",
        f"| Scenario | RMSE@50 | CoM Drift@50 | Verdict |",
        f"|----------|---------|--------------|---------|",
    ]
    for name, result in scenario_results.items():
        lines.append(
            f"| {name} | {result['rmse_50']:.4f} | {result['com_drift_50']:.4f}m | "
            f"{result['verdict']} |"
        )
    # ... per-scenario details, diagnosis section, figures list
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single scenario (random only) | 4 scenarios with typed action generators | Phase 4 | Covers training-relevant action distributions |
| Hardcoded SurrogateModel | Architecture dispatch via config.json `arch` | Phase 3 Plan 03 | Supports MLP, Residual, Transformer |
| Global phase encoding (2-dim) | Per-element phase (60-dim) available | Phase 02.1 | Model input_dim may be 131 or 189 depending on config |
| Console-only output | Structured markdown report with PASS/WARN/FAIL | Phase 4 | Machine-parseable validation results |

**Important version note:** Models trained on Phase 02.1 data use `input_dim=189` (per-element phase), while models trained on V1 data use `input_dim=131` (global phase). The `config.json` stores this, so the dispatch handles it automatically.

## Open Questions

1. **PPO observation extraction from rod state**
   - What we know: PPO obs_dim=14, rod state is 124-dim. Env has `_get_obs()` that computes distance/angle to goal.
   - What's unclear: For policy rollouts, we need goal position. The validation env needs a goal to compute observations.
   - Recommendation: Instantiate full `LocomotionElasticaEnv` for policy scenarios (already done for ground-truth). Use env's `_get_obs()` after setting rod state. This keeps the env as the single source of truth for observation computation.

2. **Data-driven threshold setting**
   - What we know: CONTEXT.md says Claude sets thresholds based on actual results. CoM drift has hard ranges (PASS <10%, WARN 10-25%, FAIL >25% of real displacement).
   - What's unclear: RMSE thresholds cannot be set before seeing data.
   - Recommendation: Run validation, compute median/P25/P75 of RMSE@50 across random scenario first. Set PASS threshold at ~2x median (generous), FAIL at ~5x median. Print thresholds in report for transparency.

3. **Phase 3 best model directory**
   - What we know: CONTEXT.md says validate `output/surrogate/best/`. This directory does not exist yet.
   - What's unclear: Whether Phase 3's SURR-03 will create it as a symlink or copy.
   - Recommendation: Phase 4 should simply require `output/surrogate/best/` to exist with `config.json`, `model.pt`, and `normalizer.pt`. How Phase 3 creates it is Phase 3's concern.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python3 -m pytest tests/test_validate_phase4.py -x` |
| Full suite command | `python3 -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| (no formal IDs) | Architecture dispatch loads correct model class | unit | `pytest tests/test_validate_phase4.py::test_arch_dispatch -x` | No - Wave 0 |
| (no formal IDs) | Strict error on missing arch field | unit | `pytest tests/test_validate_phase4.py::test_missing_arch_errors -x` | No - Wave 0 |
| (no formal IDs) | Forward crawl action normalization correct | unit | `pytest tests/test_validate_phase4.py::test_forward_crawl_normalization -x` | No - Wave 0 |
| (no formal IDs) | NaN detection in rollout truncates correctly | unit | `pytest tests/test_validate_phase4.py::test_nan_detection -x` | No - Wave 0 |
| (no formal IDs) | Report generation produces valid markdown | unit | `pytest tests/test_validate_phase4.py::test_report_generation -x` | No - Wave 0 |
| (no formal IDs) | End-to-end validation with mock model | integration | `pytest tests/test_validate_phase4.py::test_e2e_mock -x` | No - Wave 0 |

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/test_validate_phase4.py -x`
- **Per wave merge:** `python3 -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_validate_phase4.py` -- covers arch dispatch, action normalization, NaN handling, report format
- [ ] No new framework install needed (pytest already configured)

## Sources

### Primary (HIGH confidence)
- **Existing codebase** -- `aprx_model_elastica/validate.py` (current validation implementation, 316 lines)
- **Existing codebase** -- `aprx_model_elastica/model.py` (all 3 model classes with unified interface)
- **Existing codebase** -- `aprx_model_elastica/state.py` (StateNormalizer, phase encoding, state slices)
- **Existing codebase** -- `aprx_model_elastica/train_surrogate.py` (architecture dispatch pattern at lines 441-454)
- **Existing codebase** -- `locomotion_elastica/env.py` (OBS_DIM=14, `_get_obs()`, env setup)
- **Existing codebase** -- `src/trainers/ppo.py` (PPO checkpoint save/load format)
- **Existing codebase** -- `output/surrogate/*/config.json` (verified format with and without `arch` field)
- **Existing codebase** -- PPO checkpoint at `output/locomotion_elastica_forward_20260307_021959/checkpoints/best.pt` (verified format: actor_state_dict, config object)

### Secondary (MEDIUM confidence)
- **CONTEXT.md** -- User decisions on scenarios, thresholds, report format (locked decisions)
- **STATE.md** -- Phase 3 best model is `sweep_lr1e3_h512x3` (lr=1e-3, 512x3, val_loss=0.2161, R^2=0.784)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new dependencies
- Architecture: HIGH -- extending existing validate.py with well-understood patterns from codebase
- Pitfalls: HIGH -- verified by inspecting actual checkpoint files and code

**Research date:** 2026-03-11
**Valid until:** 2026-04-11 (stable -- codebase-specific, not library-dependent)
