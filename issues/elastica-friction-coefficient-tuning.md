---
id: cddf410f-8889-4137-a3e8-321a0f3d5bff
name: elastica-friction-coefficient-tuning
description: RFT friction 10:1 ratio caused wild lateral drift, tuned to 5:1
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [bug, locomotion, elastica, physics, friction]
aliases: []
status: resolved
severity: medium
subtype: physics
---

# Elastica Friction Coefficient Tuning

## Problem

Original friction coefficients `rft_ct=0.01, rft_cn=0.1` (10:1 ratio) were too extreme, causing wild lateral drift.

Also, `amplitude_range` was set to (0, 2.0) but amplitudes > 0.15 caused chaotic rod dynamics (with the old small rod radius).

## Fix

Best found via grid search: `rft_ct=0.01, rft_cn=0.05` (5:1 ratio) — gives 94:1 forward/lateral ratio at optimal parameters.

Reduced `amplitude_range` from (0, 2.0) to (0, 0.15) (later changed to (0, 5.0) after rod radius fix).

## Files Modified

- `locomotion_elastica/config.py`
