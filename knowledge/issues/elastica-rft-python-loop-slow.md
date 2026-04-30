---
name: elastica-rft-python-loop-slow
description: AnisotropicRFTForce used Python for-loop instead of vectorized NumPy
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [performance, physics, locomotion, elastica]
aliases: []
status: resolved
severity: medium
subtype: performance
---

# Elastica RFT Python For-Loop Slow

## Problem

The `AnisotropicRFTForce.apply_forces()` iterated over elements in a Python for-loop, adding ~10ms per call × 500 calls per RL step.

## Fix

Replaced with vectorized NumPy operations for the entire force computation.

## Files Modified

- `locomotion_elastica/env.py`
