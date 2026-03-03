---
date_created: 2026-02-16
date_modified: 2026-02-16
tags: [reproduction, mujoco, ppo, underwater, locomotion, curriculum-learning]
---

# Zheng, Li & Hayashibe (2022) — Reproduction

Reproduction of: **"An Optimization-Based Approach to Locomotion of Underwater Snake Robots with Series Elastic Actuators"** by Zheng, Li & Hayashibe (Frontiers in Robotics and AI, 2022).

## Paper Summary

A 7-link rigid snake robot learns to swim underwater using PPO with curriculum learning. The paper studies the effect of joint stiffness (series elastic actuators) and fluid properties on swimming efficiency.

### Key Features

- **Snake**: 7 capsule links, 6 hinge joints, torque-controlled motors
- **Fluid**: MuJoCo built-in fluid model (density + viscosity)
- **RL**: PPO-Clip with two-phase curriculum reward
- **Curriculum**: Phase 1 (maximize speed) -> Phase 2 (match target velocity with low power)

## Quick Start

```bash
# From project root
cd "Zheng, Li & Hayashibe (2022)"

# Run tests
python -m pytest tests/test_env.py -v

# Train (short run)
python train.py --epochs 100

# Train with stiffness
python train.py --stiffness 2.0 --epochs 5000

# Evaluate
python evaluate.py checkpoints/zheng2022/best.pt

# Stiffness sweep
python sweep_stiffness.py --sweep stiffness --epochs 5000

# Visualize (requires display)
python visualize.py checkpoints/zheng2022/best.pt --mode live
```

## Files

| File | Description |
|------|-------------|
| `configs.py` | All hyperparameters as a dataclass |
| `env.py` | TorchRL MuJoCo environment (MJCF generation + step logic) |
| `reward.py` | Two-phase curriculum reward function |
| `train.py` | PPO training loop with curriculum and separate LRs |
| `evaluate.py` | Evaluation: velocity, power, efficiency, gait plots |
| `visualize.py` | MuJoCo viewer (live or video) |
| `sweep_stiffness.py` | Stiffness and fluid property sweep experiments |
| `tests/test_env.py` | 18 smoke tests (MJCF, env, reward) |

## Paper Specifications

### Snake Robot
- 7 rigid capsule links connected by 6 hinge joints (1 DOF each)
- Link length: 0.1 m, diameter: 0.02 m
- Total mass: 0.25 kg (density = 1000 kg/m^3, matching water for neutral buoyancy)
- Joint rotation range: [-90, 90] degrees
- Motor force range: [-1, 1] N with gear ratio 0.1 (effective torque: [-0.1, 0.1] Nm)

### Observation Space (16D)
- Head angular velocity (z-axis): 1D
- Joint angular velocities: 6D
- Head rotation angle (yaw): 1D
- Joint rotation angles: 6D
- Head linear velocity (vx, vy): 2D

### Action Space (6D)
- Torque commands in [-1, 1] for each of the 6 motor actuators

### Reward (Curriculum)
- **Phase 1** (epochs 0-2000): `r = 200*v_h - P_hat` (maximize speed)
- **Phase 2** (epochs 2000+): `r = r_v * r_P` (match target velocity, minimize power)
  - Target velocity decreases by 0.02 m/s every 1000 epochs

### PPO Hyperparameters
- Discount: gamma = 0.99
- Clip ratio: 0.2
- Policy LR: 0.003, Value LR: 0.001
- GAE lambda: 0.97
- Network: 2-layer MLP, 256 ReLU units per layer

### Simulation
- MuJoCo with fluid density = 1000 kg/m^3, viscosity = 0.0009 Pa-s (water)
- Gravity = 0 (neutral buoyancy)
- Sim frequency: 100 Hz, Control frequency: 25 Hz
