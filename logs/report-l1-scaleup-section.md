---
name: report-l1-scaleup-section
description: Added L1 scale-up run results and reward curve analysis to report Section E7
type: log
status: complete
subtype: research
created: 2026-04-05
updated: 2026-04-05
tags: [report, ablation, l1-scaleup, reward-curve, steepness]
aliases: []
---

# Added L1 Scale-Up Results to Report (E7)

Added three tables and analysis to `report/report.tex` Section 5.7 (Steepness and Environment Scaling, E7):

1. **Tab. scaleup-l1**: Quartile progression over 32.5M frames showing monotonic distance improvement (0.589 → 0.443)
2. **Tab. scaleup-l1-components**: Reward component breakdown (dist, smooth, PBRS) by quartile
3. **Tab. scaleup-comparison**: L1 vs C5 vs S4 baseline comparison

Key finding: L1 breaks the d≈0.54 plateau (reaching 0.443), demonstrating the plateau is partly a sample budget limitation. Updated the section's concluding paragraph to reflect this nuance.
