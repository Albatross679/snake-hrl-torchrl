#!/usr/bin/env bash
# Phase 02.2: RL-step-only surrogate data collection
#
# Collects flat (state, action, next_state, forces) transitions — 1 RL step per run.
# Data format is consistent with Phase 1: flat arrays per transition, with forces dict.
#
# Key differences from Phase 1:
#   - perturb_omega_std=1.5 (vs Phase 1's 0.05) — more aggressive perturbation
#   - Saved in .pt format (vs Phase 1's parquet)
#   - Fresh perturbation per run (vs Phase 1's full episodes)
#
# --steps-per-run 1 + --collect-forces auto-enables flat output in collect_data.py
#
# Output: data/surrogate_rl_step/  (separate from Phase 02.1's data/surrogate/)
#
# Data size target:
#   Each transition ~2 KB (states 124x4 + next_states 124x4 + actions 5x4 + forces ~984 bytes = ~2016 bytes)
#   10 GB minimum = ~5,000,000 transitions
#   --num-transitions 50000000 targets ~100 GB — 10x safety margin above the 10 GB minimum.
#   Collection can be stopped early once >=10 GB is confirmed on disk.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 -m aprx_model_elastica collect \
  --steps-per-run 1 \
  --collect-forces \
  --save-dir data/surrogate_rl_step \
  --num-workers 16 \
  --num-transitions 50000000 \
  --episodes-per-save 100 \
  --perturbation-fraction 0.3 \
  --perturb-omega-std 1.5 \
  --sobol \
  --seed 43 \
  "$@"
