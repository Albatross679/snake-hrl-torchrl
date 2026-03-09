---
date_created: 2026-03-09
date_modified: 2026-03-09
tags: [experiment, evaluation, locomotion, elastica, ppo]
status: complete
---

# Locomotion Elastica Evaluation — Session 10 Best Checkpoint

## Context

Evaluated the best checkpoint from Session 10 training (step 155,648, training reward=156.66) to verify whether the trained snake actually locomotes. Used deterministic actions (distribution mode) over 20 episodes with randomized initial headings.

## Configuration

| Parameter | Value |
|-----------|-------|
| Checkpoint | `output/locomotion_elastica_forward_20260307_021959/checkpoints/best.pt` |
| Training step | 155,648 |
| Training best reward | 156.66 |
| Evaluation episodes | 20 |
| Action mode | Deterministic (distribution mode) |
| Max episode steps | 500 |
| Goal distance | 2.0m |
| Goal radius | 0.3m |
| Initial heading | Randomized |

## Results

| Metric | Value |
|--------|-------|
| Mean reward | 64.90 ± 1.44 |
| Mean episode length | 500.0 ± 0.0 (all hit limit) |
| Mean displacement | **1.495m** |
| Mean final dist to goal | 0.506m |
| Goal reached | 0/20 (0%) |
| Starvation | 0/20 (0%) |
| Truncated | 20/20 (100%) |

### Per-Episode Data

| Episode | Reward | Displacement | Final Dist | Direction (x,y) |
|---------|--------|-------------|------------|-----------------|
| 1 | 65.64 | 1.568m | 0.433m | (0.26, -1.55) |
| 2 | 66.46 | 1.648m | 0.353m | (-1.53, 0.61) |
| 3 | 64.16 | 1.416m | 0.584m | (0.89, -1.10) |
| 4 | 66.40 | 1.641m | 0.359m | (-0.52, -1.56) |
| 5 | 62.43 | 1.250m | 0.751m | (1.02, 0.73) |
| 10 | 66.39 | 1.665m | 0.342m | (-1.60, 0.45) |
| 17 | 66.69 | 1.670m | 0.331m | (-1.57, -0.56) |
| 20 | 66.66 | 1.668m | 0.333m | (-1.12, -1.23) |

## Analysis

### Locomotion Confirmed

The snake displaces **1.495m on average** per episode — this is real locomotion, not random jitter. The displacement is consistent (std ~0.13m) across all 20 episodes with random initial headings.

### Speed: ~0.003m per step

At 500 steps per episode, the snake averages 0.003m/step (0.006 m/s at dt=0.5s per RL step). This is steady but insufficient to reach the 2.0m goal within the 500-step budget.

### Why 0% Goal Reached

The snake needs ~667 steps to cover 2.0m at current speed. The 500-step limit truncates it at ~1.5m (75% of the way). Solutions:
1. Increase `max_episode_steps` to 700+
2. Increase locomotion speed via reward shaping (e.g., higher `c_dist`)
3. Train with a closer goal (e.g., 1.0m or 1.5m)

### Training vs Eval Reward Gap

Training best reward (156.66) vs eval mean reward (64.90) — 2.4× gap. Likely causes:
- Training used stochastic actions (exploration bonus via entropy)
- Training episodes sometimes reached the goal (100-point bonus)
- Batch rewards during training include episodes with favorable headings

### Heading Adaptation

The snake adapts well to all heading directions. Displacement vectors show movement in many different directions matching the random initial headings, confirming the policy generalizes across orientations.

## Conclusion

**Locomotion is verified.** The trained PPO policy produces reliable serpentine locomotion covering 1.5m per 500-step episode. The main limitation is speed — the snake moves steadily but slowly, running out of steps before reaching the 2.0m goal. This is a good foundation for further optimization.
