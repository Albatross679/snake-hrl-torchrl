---
created: 2026-03-10T16:08:48.457Z
title: Optimize PyElastica inner substep loop to reduce Python overhead
area: general
files:
  - locomotion_elastica/env.py:77
  - locomotion_elastica/env.py:448
  - locomotion_elastica/env.py:541
  - locomotion_elastica/config.py:127
---

## Problem

Data collection runs at ~3 FPS per worker (500 substeps/RL step, 20-element rod). cProfile shows the bottleneck is Python interpreter overhead in the inner substep loop called 500x per RL step, not cache or memory bandwidth.

**Per-substep breakdown (~640 μs total):**
- `apply_forces` (AnisotropicRFTForce, `env.py:77`): **161 μs** — pure Python+NumPy RFT friction, biggest single cost
- `compute_internal_forces_and_torques`: 116 μs — Elastica Numba internals (hard to touch)
- `_apply_curvature_to_elastica` (`env.py:448`): **90 μs** — Python for-loop writing `rest_kappa[0,i]` element-by-element
- `dampen_rates_protocol`: 68 μs — Elastica damping Numba
- Elastica dispatch overhead: ~100 μs — `operator_group.__iter__`, `synchronize`, `do_step`

CPU utilization per worker is only ~30% despite running compute — low IPC from Python interpreter dispatch on tiny 20-element arrays.

## Solution

Two quick wins in code we own (~39% of substep time):

1. **`_apply_curvature_to_elastica` (`env.py:448`)** — replace the `for i in range(n)` Python loop with a single NumPy slice: `self._rod.rest_kappa[0, :n] = curvatures[:n]`. Eliminates Python loop overhead at ~90 μs/substep.

2. **`AnisotropicRFTForce.apply_forces` (`env.py:77`)** — currently pure Python+NumPy called 500x. Numba-JIT this method (add `@numba.njit` or restructure as a compiled kernel). Targets 161 μs/substep.

Also worth investigating: reducing `substeps_per_action` from 500 to a smaller value (e.g. 200) and checking if simulation stability holds — would give a near-linear FPS improvement.
