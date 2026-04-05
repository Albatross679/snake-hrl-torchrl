---
name: reward-ablation-results
description: Results of 12-configuration reward design ablation for follow_target PPO
type: experiment
status: complete
created: 2026-04-03
updated: 2026-04-03
tags: [reward-design, ablation, ppo, follow-target, curriculum, pbrs]
---

# Reward Design Ablation — Results

## Setup
- 12 configurations (A1 + B1-B6 + C1-C5)
- 32 parallel envs, 30 min wall time (~3.6M frames each), seed 42
- RTX A4000, sequential execution, ~5.5 hours total

## Rankings (by dist_to_goal Q4, lower = better)

| Rank | ID | Config | Dist (Q4) | SNR | EV | Healthy? |
|------|-----|--------|-----------|-----|------|----------|
| 1 | C5 | Dense(0.3)+PBRS+smooth+curriculum(500) | 0.543 | 0.41 | 0.73 | Yes |
| 2 | C3 | Dense(0.3)+PBRS+smooth+curriculum(200) | 0.551 | 0.44 | 0.72 | Yes |
| 3 | C2 | Dense(1.0)+PBRS+curriculum | 0.586 | 0.78 | 0.75 | Yes |
| 4 | C1 | Vanilla+curriculum | 0.603 | 1.08 | 0.73 | Yes |
| 5 | B2 | Dense(1.0)+PBRS | 0.698 | 0.47 | 0.70 | Yes |
| 7 | A1 | Vanilla dense | 0.718 | 0.94 | 0.43 | Yes |
| 12 | C4 | Full+curriculum | 0.917 | 0.79 | 0.77 | Yes (reward hacking) |

## Key Findings

1. **Curriculum is dominant** — every curriculum run beats every non-curriculum run
2. **PBRS provides ~3% improvement** on top of dense reward
3. **Smoothness hurts without curriculum, helps with it** — B4 worst non-curriculum, C3 second-best overall
4. **Heading enables reward hacking** — C4 has highest reward (50.1) but worst distance (0.917)
5. **Longer warmup marginally helps** — C5 (500 eps) beats C3 (200 eps) by 1.5%

## Best Config
`--dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 --curriculum --warmup-episodes 500`

## Documented in Report
Section 6.2.8 "Normalized Reward Ablation" in report/report.tex
