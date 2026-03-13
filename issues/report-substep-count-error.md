---
name: Report substep count error — 50 should be 500
description: The LaTeX report incorrectly stated 50 substeps and 0.05s per RL control step instead of 500 substeps and 0.5s
type: issue
status: resolved
severity: high
subtype: physics
created: 2026-03-11
updated: 2026-03-11
tags: [report, physics, time-stepping]
aliases: []
---

# Report substep count error — 50 should be 500

## Problem

The report (`report/report.tex`) consistently stated that each RL control step uses 50 Position Verlet substeps over 0.05s. The correct values are **500 substeps** over **0.5s**.

## Root Cause

The config has a two-level time stepping hierarchy:

1. **Base Elastica interval** (`ElasticaConfig`): `dt = 0.05`, `elastica_substeps = 50` → `dt_substep = 0.001s`. This is an internal PyElastica integration concept.
2. **RL control step** (`SerpenoidControlConfig`): `substeps_per_action = 500` → 500 × 0.001s = **0.5s**.

The report was written using `dt` and `elastica_substeps` from the physics config, mistaking the base Elastica integration interval (0.05s, 50 substeps) for the RL control step. The actual RL step calls 10 of these base intervals (500 total substeps = 0.5s).

## Fix

Changed all 8 occurrences in `report/report.tex`:
- `0.05` → `0.5` for $\Delta t_{\text{ctrl}}$
- `50` → `500` for $n_{\text{sub}}$ and substep counts
- Affected sections: Introduction (lines 170, 179), Section 2.1.2 Time Integration (lines 309, 335, 339, 347), Section 2.3 PyElastica Simulator (lines 434, 438)
