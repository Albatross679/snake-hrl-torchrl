---
date_created: 2026-02-16
date_modified: 2026-02-16
tags: [reward-design, potential-fields, path-planning, reinforcement-learning]
---

# Artificial Potential Fields as RL Reward Shaping

## Background

The attractive and repulsive fields used in Liu 2021 and Liu 2023's reward functions come from **Artificial Potential Field (APF)** theory, a classical robotics path planning technique (Khatib, 1986).

## The Physical Analogy

Imagine the robot as a charged particle moving through a force field:

- **The goal is a magnet** — it generates an **attractive potential** $U_{\text{att}} = \frac{1}{2}k_{\text{att}}\|\mathbf{p} - \mathbf{p}_g\|^2$ that creates a "valley" centered on the goal. The gradient $F_{\text{att}} = -k_{\text{att}}(\mathbf{p} - \mathbf{p}_g)$ always points toward the goal, pulling the robot in.

- **Each obstacle is a repelling charge** — it generates a **repulsive potential** $U_{\text{rep}}$ that creates a "hill" around the obstacle. Within influence radius $\rho_0$, the force $F_{\text{rep}}$ pushes the robot away. Beyond $\rho_0$, the obstacle is invisible.

The total field is the superposition: $F_{\text{total}} = F_{\text{att}} + F_{\text{rep}}$. A classical planner would just follow this gradient directly.

## Why Use APF as a Reward (Not a Controller)?

Liu 2021/2023 don't use APF for direct control — they use it as a **reward signal**:

$$R_{\text{att}} = \mathbf{v} \cdot F_{\text{att}}, \qquad R_{\text{rep}} = \mathbf{v} \cdot F_{\text{rep}}$$

The dot product $\mathbf{v} \cdot F$ means: **the reward is highest when the robot's actual velocity aligns with the field's desired direction**. If the robot moves along the attractive gradient (toward the goal), it gets positive reward. If it moves against the repulsive gradient (into an obstacle), it gets negative reward.

This is better than using APF directly because:

1. **Classical APF suffers from local minima** — the robot can get permanently stuck in places where $F_{\text{att}} + F_{\text{rep}} = 0$ (e.g., directly behind an obstacle relative to the goal). RL can learn to *explore past* these local minima through trial and error.

2. **APF doesn't know the robot's body** — a potential field treats the robot as a point. The snake robot has a long articulated body that must coordinate 4 links through narrow gaps. RL learns body-aware behaviors (like pushing off obstacles) that a point-mass APF could never produce.

3. **APF gives a dense gradient everywhere** — unlike a sparse "did you reach the goal?" reward, the potential field provides a continuous signal at every position in the workspace, telling the robot *which direction* to move. This makes RL training much easier.

## Why Liu 2021 Has Both, but Liu 2023 Drops Repulsive

- **Liu 2021** operates in **cluttered environments** (obstacle grids). Without $R_{\text{rep}}$, the robot would learn to crash through obstacles since only $R_{\text{att}}$ pulls it toward the goal. The repulsive field creates a natural "stay away" signal around each obstacle.

- **Liu 2023** operates in **open fields** (no obstacles). There's nothing to repel from, so the repulsive term is dropped. Instead, they add a direct velocity term $c_v v_g$ to more explicitly reward speed toward the goal.

## The Curriculum Rings Are Separate

The third term $R_{\text{goal}} = \cos(\theta_g) \sum_k \frac{1}{r_k}\mathbf{I}(\rho_g < r_k)$ is **not** part of APF — it's a separate curriculum mechanism. It provides discrete bonus rewards at expanding distance thresholds $r_k$ around the goal, bridging the gap between the everywhere-dense APF signal and the sparse "reached the goal" event. The $\cos(\theta_g)$ factor ensures the bonus only activates when the robot is actually facing the goal, not just passing nearby.

## Symbol Reference

| Symbol | Meaning |
|---|---|
| $\mathbf{p}$ | Robot's current position |
| $\mathbf{p}_g$ | Goal position |
| $\mathbf{v}$ | Robot's velocity vector |
| $k_{\text{att}}$ | Attractive potential gain |
| $k_{\text{rep}}$ | Repulsive potential gain |
| $\rho_i$ | Distance from robot to obstacle $i$ |
| $\rho_0$ | Influence radius (repulsive force is zero beyond this) |
| $\theta_g$ | Angle between robot heading and goal direction |
| $\rho_g$ | Distance from robot to goal |
| $r_k$ | Radius of the $k$-th curriculum ring |
| $\mathbf{I}(\cdot)$ | Indicator function (1 if true, 0 otherwise) |
| $v_g$ | Velocity component along the goal direction (scalar) |

## References

- Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots. *International Journal of Robotics Research*, 5(1), 90-98.
- Liu et al. (2021). Learning Contact-aware CPG-based Locomotion in a Soft Snake Robot. *CoRL*.
- Liu et al. (2023). Reinforcement Learning of CPG-regulated Locomotion Controller for a Soft Snake Robot. *Soft Robotics*.
