---
name: "Elastica phantom physics step eliminated"
description: "The Elastica config had a three-level time stepping hierarchy with a phantom middle layer (physics.dt) that confused the actual two-level structure (substep, RL step)"
type: issue
status: resolved
severity: medium
subtype: physics
created: 2026-03-09
updated: 2026-03-09
tags: [elastica, config, refactor, time-stepping]
aliases: []
---

## Problem

The Elastica time stepping config had three levels:

1. `elastica_substeps = 50` subdivisions of `dt`
2. `physics.dt = 0.05s` (phantom middle layer)
3. `substeps_per_action = 10` repetitions of `dt` per RL step

The actual integration loop in `locomotion_elastica/env.py` already flattened levels 1-2 into a single loop:

```python
total_substeps = substeps_per_action * elastica_substeps  # 10 * 50 = 500
dt_sub = physics.dt / elastica_substeps                    # 0.05 / 50 = 0.001s
for _ in range(total_substeps):
    do_step(..., dt_sub)
```

So `physics.dt = 0.05s` was never used as an actual timestep — it was only an intermediate value used to derive the real substep (`dt / elastica_substeps = 0.001s`) and the RL step duration (`dt * substeps_per_action = 0.5s`).

This caused confusion because `physics.dt` means different things across backends:
- **DisMech**: `dt = 0.05s` IS the substep (implicit solver step). Two-level hierarchy is clean.
- **Elastica**: `dt = 0.05s` is a phantom — the real substep is `dt / elastica_substeps`.
- **naughton2021**: `dt = 2.5e-5` IS the substep, `elastica_substeps = 100` = substeps per RL step.

## Resolution

Added `dt_substep` property to `ElasticaConfig` and `LocomotionElasticaPhysicsConfig`:

```python
@property
def dt_substep(self) -> float:
    """Elastica integration timestep (seconds) = dt / elastica_substeps."""
    return self.dt / self.elastica_substeps
```

Changed `SerpenoidControlConfig.substeps_per_action` from `10` to `500` — now represents the total number of Elastica substeps per RL action, matching the actual loop.

Updated all code to use the two-level hierarchy directly:
- `dt_substep` = 0.001s (Elastica integration timestep)
- `substeps_per_action` = 500 (substeps per RL step)
- `dt_rl = dt_substep * substeps_per_action` = 0.5s

### Files changed

- `src/configs/physics.py`: Added `dt_substep` property to `ElasticaConfig`, updated docstrings
- `locomotion_elastica/config.py`: Added `dt_substep` property to `LocomotionElasticaPhysicsConfig`, `substeps_per_action` 10→500
- `locomotion_elastica/env.py`: Replaced `dt / elastica_substeps` and `dt * substeps_per_action` with `dt_substep` and `substeps_per_action`
- `src/physics/elastica_snake_robot.py`: Use `self.config.dt_substep` instead of `self.config.dt / self.config.elastica_substeps`
- `aprx_model_elastica/env.py`: Use `physics.dt_substep * substeps_per_action` for `dt_rl`

### Not changed (no impact)

- `locomotion/` (DisMech): Already uses clean two-level hierarchy (`dt` is the substep)
- `naughton2021/`, `schaffer2024/`: Don't use `dt_substep`; their `dt * elastica_substeps` patterns still work
- `src/physics/cpg/action_wrapper.py`: Only used with DisMech envs

### Backward compatibility

`dt` and `elastica_substeps` fields are preserved on the base `ElasticaConfig`. Existing code that uses them directly (e.g., naughton2021) is unaffected. The new `dt_substep` property is purely additive.
