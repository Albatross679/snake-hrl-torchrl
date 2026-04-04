# Snake HRL — Hierarchical Reinforcement Learning for Snake Robot Predation

Hierarchical Reinforcement Learning (HRL) for snake robot predation using TorchRL.
The snake robot learns to approach and coil around prey using a two-level hierarchy:
an approach policy and a coil policy, coordinated by a meta-controller.

## Best Checkpoint Evaluation — Follow Target (Choi & Tong, 2025)

The best-performing checkpoint is a PPO agent trained with 100 parallel environments,
vanilla dense reward (`exp(-5d)`), and curriculum learning (100-episode warmup).

**Training best reward: 90.20** | **Eval reward: 8.68 +/- 6.40** (20 episodes, deterministic)

| Metric | Mean +/- Std | Range |
|--------|-------------|-------|
| Episode reward | 8.68 +/- 6.40 | [2.18, 28.13] |
| Mean tip-target distance | 0.795 +/- 0.133 m | [0.520, 1.059] |
| Final tip-target distance | 0.658 +/- 0.213 m | [0.316, 1.092] |

### Evaluation Visualization

![Best PPO Follow Target Evaluation](media/choi2025/best_ppo_follow_target.gif)

*3 episodes of deterministic rollout from the best PPO checkpoint. The manipulator (blue)
deflects toward the moving target (green star) but plateaus at ~0.7m distance, unable to
achieve close tracking. The tip trail (red) shows oscillatory behavior without convergence.*

### Key Finding: Train-Eval Gap

The 10x gap between training best (90.20) and evaluation mean (8.68) reveals that:

1. **Curriculum confound** — training ramps target speed from 20% to 100% over 100 episodes;
   early near-stationary targets inflate batch rewards
2. **Sparse reward plateau** — at d=0.8m, `exp(-5*0.8) = 0.018`; the agent receives
   near-zero gradient signal for fine approach
3. **Stochastic vs deterministic** — training exploration noise occasionally reaches
   high-reward states that the learned mean cannot reproduce

The distance plateau at d ~= 0.6-0.8m is the fundamental performance ceiling for this
reward formulation and physics backend. See Section 5.8 of the report for full analysis.

## Project Structure

```
src/           — source modules (configs, envs, networks, physics, rewards, trainers)
papers/        — paper reimplementations (choi2025/)
report/        — LaTeX report
media/         — figures and animations
output/        — training checkpoints and logs
experiments/   — experiment documentation
```

## Reproducing the Best Checkpoint

```bash
# Train
PYTHONPATH=.:papers python3 -m choi2025.train_ppo \
  --num-envs 100 --total-frames 5000000 \
  --curriculum --warmup-episodes 100

# Evaluate
PYTHONPATH=.:papers python3 -m choi2025.record \
  --checkpoint output/<run>/checkpoints/best.pt \
  --task follow_target --algo ppo \
  --num-episodes 3 --output media/choi2025/best_ppo_follow_target.gif
```
