---
name: PyElastica substep loop optimization benchmark
description: Benchmarks two targeted optimizations to the PyElastica inner substep loop to reduce Python interpreter overhead in data collection workers.
type: experiment
status: complete
created: 2026-03-10
updated: 2026-03-10
tags: [elastica, performance, numba, data-collection, surrogate]
aliases: []
---

## Motivation

Data collection workers (`gsd-collect-rl` session, 16 workers) run at ~3 FPS per worker. cProfile of `env._step()` identified that **Python interpreter overhead in the 500-substep inner loop** is the primary bottleneck, not cache, memory bandwidth, or NUMA topology. Two optimizations were proposed and tested in-memory via monkey-patching, without modifying source files.

## Setup

- **Script**: `script/benchmark_substep_opts.py`
- **Rod**: 20 elements, `LocomotionElasticaEnvConfig` defaults
- **Substeps/RL step**: 500 (`substeps_per_action = 500`)
- **Measurement**: 15 RL steps (each = `env.reset()` + `env._step()`), wall-clock mean
- **Machine**: Tesla V100 node, 48-core Xeon (2× NUMA), CPUs running at ~52% max freq

## Optimizations Tested

### Opt 1 — Curvature loop → NumPy slice

`_apply_curvature_to_elastica` (`env.py:448`) currently uses a Python for-loop:

```python
for i in range(n):
    self._rod.rest_kappa[0, i] = curvatures[i]
```

Replaced with a single NumPy slice:

```python
self._rod.rest_kappa[0, :n] = curvatures[:n]
```

### Opt 2 — Numba RFT kernel

`AnisotropicRFTForce.apply_forces` (`env.py:77`) is called 500× per RL step. Although already vectorized NumPy, each call incurs Python function dispatch overhead and allocates 5 temporary arrays (`vel_elem`, `v_t_scalar`, `v_t`, `v_n`, `f_elem`) on 20-element arrays — where allocation cost dominates compute.

Replaced with a `@numba.njit(fastmath=False)` element-wise kernel that operates in-place on `system.external_forces`, eliminating all temporaries.

**Correctness**: Verified bit-identical rod positions (`max diff = 0.00e+00`) after 5 full RL steps vs. NumPy baseline, using fixed initial heading to control for RNG state. `fastmath=False` preserves IEEE 754 semantics.

## Results

| Configuration | ms/step | FPS/worker | Speedup |
|---|---|---|---|
| Baseline (current) | 299 ms | 3.34 | 1.00× |
| Opt 1: curvature slice | 260 ms | 3.84 | **1.15×** |
| Opt 1 + 2: + Numba RFT | 186 ms | 5.37 | **1.61×** |

Per-substep breakdown:

| | μs/substep | Saved vs baseline |
|---|---|---|
| Baseline | 598 μs | — |
| Opt 1 | 521 μs | 77 μs |
| Opt 1+2 | 372 μs | 225 μs |

## Projected Impact on Data Collection

16 workers, 50M transition target:

| Configuration | Transitions/s | ETA |
|---|---|---|
| Baseline | 54 | ~260 h |
| Opt 1 only | 61 | ~226 h (−34 h) |
| Opt 1 + 2 | 86 | **~162 h (−98 h)** |

## Interpretation

- **Curvature fix (+15%)**: Eliminating the Python for-loop over 20 elements saves ~77 μs/substep. Simple one-line change with no risk.
- **Numba RFT (+61% total)**: The dominant gain. Eliminates 5 temporary NumPy array allocations and Python dispatch overhead per substep. The remaining ~372 μs/substep is Elastica's own Numba internals (`compute_internal_forces_and_torques`, `dampen_rates_protocol`, operator group dispatch) which cannot be changed without modifying the library.
- The **~98-hour ETA reduction** (260 h → 162 h) is available with two localized changes to `locomotion_elastica/env.py`, neither of which affects the currently running collection job.

## Next Steps

- Apply both optimizations to `locomotion_elastica/env.py` in a new collection run or as a patch to the existing one
- Consider reducing `substeps_per_action` from 500 as a further orthogonal speedup (needs stability check)
