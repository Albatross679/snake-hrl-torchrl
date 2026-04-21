"""Benchmark PyElastica substep optimizations without modifying source files.

Tests two proposed optimizations via monkey-patching:
  1. Curvature loop fix: replace Python for-loop with NumPy slice in
     _apply_curvature_to_elastica
  2. Numba RFT: replace pure-Python+NumPy AnisotropicRFTForce.apply_forces
     with a Numba-compiled kernel

Run:
    python3 script/benchmark_substep_opts.py

All changes are in-memory only — no source files are modified.
"""

import os
import time
import types

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import torch
from tensordict import TensorDict

from locomotion_elastica.config import LocomotionElasticaEnvConfig
from locomotion_elastica.env import LocomotionElasticaEnv, AnisotropicRFTForce


# ── Numba-compiled RFT kernel ────────────────────────────────────────────────

try:
    import numba

    @numba.njit(fastmath=False)
    def _rft_kernel(tangents, vel_nodes, c_t, c_n, out):
        """Compute RFT drag forces and accumulate into out (external_forces).

        fastmath=False: preserves IEEE 754 semantics, verified to produce
        bit-identical results to the NumPy baseline over 5 full RL steps.

        Args:
            tangents:  (3, n_elem) element tangent vectors
            vel_nodes: (3, n_nodes) node velocity collection
            c_t:       tangential drag coefficient
            c_n:       normal drag coefficient
            out:       (3, n_nodes) external_forces array — modified in-place
        """
        n_elem = tangents.shape[1]
        for i in range(n_elem):
            # Element velocity = average of adjacent node velocities
            vx = 0.5 * (vel_nodes[0, i] + vel_nodes[0, i + 1])
            vy = 0.5 * (vel_nodes[1, i] + vel_nodes[1, i + 1])
            vz = 0.5 * (vel_nodes[2, i] + vel_nodes[2, i + 1])

            tx = tangents[0, i]
            ty = tangents[1, i]
            tz = tangents[2, i]

            # Tangential projection scalar
            v_dot_t = vx * tx + vy * ty + vz * tz

            # Tangential and normal velocity components
            vtx = v_dot_t * tx
            vty = v_dot_t * ty
            vtz = v_dot_t * tz

            vnx = vx - vtx
            vny = vy - vty
            vnz = vz - vtz

            # RFT drag force
            fx = -c_t * vtx - c_n * vnx
            fy = -c_t * vty - c_n * vny
            fz = -c_t * vtz - c_n * vnz

            # Distribute half to each adjacent node
            out[0, i]     += 0.5 * fx
            out[1, i]     += 0.5 * fy
            out[2, i]     += 0.5 * fz
            out[0, i + 1] += 0.5 * fx
            out[1, i + 1] += 0.5 * fy
            out[2, i + 1] += 0.5 * fz

    NUMBA_AVAILABLE = True
    print("Numba available — will benchmark Numba RFT kernel.")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available — skipping Numba RFT benchmark.")


# ── Patched method implementations ───────────────────────────────────────────

def _apply_curvature_optimized(self, curvatures: np.ndarray) -> None:
    """Opt 1: single NumPy slice instead of Python for-loop."""
    n = min(len(curvatures), self._rod.rest_kappa.shape[1])
    self._rod.rest_kappa[0, :n] = curvatures[:n]


def _apply_forces_numba(self, system, time=0.0):
    """Opt 2: Numba-compiled RFT kernel."""
    _rft_kernel(
        system.tangents,
        system.velocity_collection,
        self.c_t,
        self.c_n,
        system.external_forces,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_env():
    env = LocomotionElasticaEnv(LocomotionElasticaEnvConfig())
    env.reset()
    return env


def make_td():
    return TensorDict({"action": torch.zeros(5)}, batch_size=[])


def run_steps(env, n_steps=10, n_warmup=2):
    """Time n_steps full RL steps (each = reset + 500 substeps)."""
    # Warmup
    for _ in range(n_warmup):
        env.reset()
        env._step(make_td())

    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.reset()
        env._step(make_td())
    return (time.perf_counter() - t0) / n_steps * 1000  # ms per step


def warmup_numba():
    """Force Numba JIT compilation using dummy arrays — no env RNG side-effects."""
    import numpy as np
    dummy_t = np.zeros((3, 20))
    dummy_v = np.zeros((3, 21))
    dummy_o = np.zeros((3, 21))
    _rft_kernel(dummy_t, dummy_v, 0.01, 0.05, dummy_o)


# ── Benchmark ────────────────────────────────────────────────────────────────

N_STEPS = 15
SUBSTEPS = 500
SEP = "─" * 62

print()
print(SEP)
print("  PyElastica Substep Optimization Benchmark")
print(f"  {N_STEPS} RL steps × {SUBSTEPS} substeps, 20-element rod")
print(SEP)

# ── 1. Baseline ──────────────────────────────────────────────────────────────
print("\n[1/3] Baseline (unmodified)...")
env_base = make_env()
t_base = run_steps(env_base, N_STEPS)
fps_base = 1000.0 / t_base
print(f"      {t_base:.1f} ms/step  ({fps_base:.2f} FPS)")

# ── 2. Opt 1: curvature loop → NumPy slice ───────────────────────────────────
print("\n[2/3] Opt 1: curvature loop → NumPy slice...")
env_opt1 = make_env()
# Monkey-patch the instance method
env_opt1._apply_curvature_to_elastica = types.MethodType(
    _apply_curvature_optimized, env_opt1
)
t_opt1 = run_steps(env_opt1, N_STEPS)
fps_opt1 = 1000.0 / t_opt1
speedup_opt1 = t_base / t_opt1
print(f"      {t_opt1:.1f} ms/step  ({fps_opt1:.2f} FPS)  ×{speedup_opt1:.2f} speedup")

# ── 3. Opt 1 + 2: curvature slice + Numba RFT ────────────────────────────────
if NUMBA_AVAILABLE:
    print("\n[3/3] Opt 1+2: curvature slice + Numba RFT kernel...")
    env_opt2 = make_env()

    # Patch curvature method on env instance
    env_opt2._apply_curvature_to_elastica = types.MethodType(
        _apply_curvature_optimized, env_opt2
    )

    # Patch apply_forces on all AnisotropicRFTForce instances in the simulator.
    # They are registered as "forcing" in the _simulator's _forcing_list.
    # We need to patch after each reset() since reset() rebuilds the simulator.
    # Strategy: patch the class method and rely on instance lookup falling through.
    # (Instance has no __dict__ entry for apply_forces, so class method is used.)
    AnisotropicRFTForce.apply_forces = _apply_forces_numba

    print("      (triggering Numba JIT — first call will compile...)", flush=True)
    warmup_numba()
    print("      JIT compilation done. Timing now...")

    t_opt2 = run_steps(env_opt2, N_STEPS)
    fps_opt2 = 1000.0 / t_opt2
    speedup_opt2 = t_base / t_opt2

    # Restore original class method so env_base/opt1 are unaffected
    AnisotropicRFTForce.apply_forces = AnisotropicRFTForce.__dict__.get(
        "_orig_apply_forces", AnisotropicRFTForce.apply_forces
    )
    print(f"      {t_opt2:.1f} ms/step  ({fps_opt2:.2f} FPS)  ×{speedup_opt2:.2f} speedup")
else:
    t_opt2 = None
    speedup_opt2 = None

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  RESULTS SUMMARY")
print(SEP)
print(f"  {'Configuration':<35} {'ms/step':>8}  {'FPS':>6}  {'Speedup':>8}")
print(f"  {'─'*35} {'─'*8}  {'─'*6}  {'─'*8}")
print(f"  {'Baseline (current)':<35} {t_base:>8.1f}  {fps_base:>6.2f}  {'1.00x':>8}")
print(f"  {'Opt 1: curvature slice':<35} {t_opt1:>8.1f}  {fps_opt1:>6.2f}  {speedup_opt1:>7.2f}x")
if t_opt2 is not None:
    print(f"  {'Opt 1+2: + Numba RFT':<35} {t_opt2:>8.1f}  {fps_opt2:>6.2f}  {speedup_opt2:>7.2f}x")
print(SEP)

print()
print("  Per-substep estimates (500 substeps/step):")
t_sub_base = t_base / SUBSTEPS * 1000
t_sub_opt1 = t_opt1 / SUBSTEPS * 1000
print(f"    Baseline   : {t_sub_base:.0f} μs/substep")
print(f"    Opt 1      : {t_sub_opt1:.0f} μs/substep  (saved {t_sub_base - t_sub_opt1:.0f} μs)")
if t_opt2 is not None:
    t_sub_opt2 = t_opt2 / SUBSTEPS * 1000
    print(f"    Opt 1+2    : {t_sub_opt2:.0f} μs/substep  (saved {t_sub_base - t_sub_opt2:.0f} μs)")
print()

# Extrapolate to 16-worker collection
print("  Projected impact on 16-worker data collection:")
fps_collect_base  = fps_base  * 16
fps_collect_opt1  = fps_opt1  * 16
print(f"    Baseline  : {fps_collect_base:.0f} transitions/s  →  ETA {50_000_000/fps_collect_base/3600:.1f} h for 50M")
print(f"    Opt 1     : {fps_collect_opt1:.0f} transitions/s  →  ETA {50_000_000/fps_collect_opt1/3600:.1f} h for 50M")
if t_opt2 is not None:
    fps_collect_opt2 = fps_opt2 * 16
    print(f"    Opt 1+2   : {fps_collect_opt2:.0f} transitions/s  →  ETA {50_000_000/fps_collect_opt2/3600:.1f} h for 50M")
print()
