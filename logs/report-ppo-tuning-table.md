---
name: report-ppo-tuning-table
description: Added PPO hyperparameter tuning results (T1, T3) to report Section 5.8
type: log
status: complete
subtype: research
created: 2026-04-04
updated: 2026-04-04
tags: [report, ppo-tuning, ablation, reward-design]
aliases: []
---

# Added PPO Tuning Results to Report

Added Table 34 (PPO hyperparameter tuning) to Section 5.8 of `report/report.tex`, between the environment count effect discussion and Section 5.9 (E8).

## Content Added

- Table comparing C5 baseline, T1 (GAE lambda=0.99), and T3 (lambda=0.99 + entropy=0.02)
- Analysis paragraph explaining why neither variant improves on C5
- Summary paragraph tying steepness, env scaling, and PPO tuning into a single plateau verdict

## Key Numbers

| Run | Dist (Q4) | EV | Grad Norm |
|-----|-----------|-----|-----------|
| C5 (baseline) | 0.543 | 0.73 | 5.2 |
| T1 (lambda=0.99) | 0.558 | 0.54 | 7.3 |
| T3 (both) | 0.612 | 0.37 | 12.2 |
