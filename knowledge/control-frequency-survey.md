---
name: control-frequency-survey
description: Survey of control and RL decision frequencies across soft robot replication papers and project environments
type: knowledge
created: 2026-03-09T10:51:28
updated: 2026-03-09T10:51:28
tags: [control-frequency, rl-frequency, soft-robots, survey, replication]
aliases: [rl-frequency-survey, control-dt-survey]
---

# Control / RL Decision Frequency Survey

Survey of control frequencies and simulation timesteps across all paper replication folders and project environments. Covers soft robot RL papers using MuJoCo, PyElastica, DisMech, and kinematic simulators.

## Summary Table

### Paper Replications

| Paper | RL/Control Freq | Physics dt | Substeps | Control dt | Simulator | Robot Type |
|-------|----------------|------------|----------|------------|-----------|------------|
| Choi 2025 | 100 Hz | 0.01 s | 1 | 0.01 s | Custom implicit | 3D soft manipulator (Cosserat rod) |
| Jiang 2024 | **0.5 Hz** | 0.001 s | 20 | 2.0 s | MuJoCo | Snake with CPG |
| Licher 2025 | 70 Hz | 1e-4 s | — | ~0.014 s | Custom | Pneumatic actuator (MPC, not RL) |
| Liu 2021 | **50 Hz** | 0.02 s | 1 | 0.02 s | MuJoCo | 4-link soft snake |
| Liu 2022 | **10 Hz** | 0.01 s | 10 | 0.1 s | MuJoCo | 9-link wheeled snake |
| Liu 2023 | **20 Hz** | 0.002 s | 25 | 0.05 s | MuJoCo | 4-link soft snake |
| Naughton 2021 | ~400 Hz | 2.5e-5 s | 100 | 0.0025 s | PyElastica | Cosserat rod |
| Schaffer 2024 | **200 Hz** | 2.5e-5 s | 200 | 0.005 s | PyElastica | Biohybrid lattice worm |
| Shi 2020 | **0.25–0.5 Hz** | Kinematic | — | 2–4 s | None | 3-link snake |
| Zheng 2022 | **25 Hz** | 0.01 s | 4 | 0.04 s | MuJoCo | 7-link underwater snake |

### Project Environments

| Environment | RL/Control Freq | Physics dt | Substeps | Control dt | Simulator | Robot Type |
|-------------|----------------|------------|----------|------------|-----------|------------|
| locomotion_elastica | **2 Hz** | 0.05 s | 10 | 0.5 s | PyElastica (50 internal substeps → 0.001 s) | Cosserat rod snake |
| locomotion | **2 Hz** | 0.05 s | 10 | 0.5 s | DisMech | Cosserat rod snake |

**Timing breakdown for `locomotion_elastica`:**
```
RL step = substeps_per_action × physics.dt = 10 × 0.05s = 0.5s (2 Hz)
Each physics step: 50 PyElastica internal substeps → dt_internal = 0.05/50 = 0.001s
Total internal physics steps per RL action: 10 × 50 = 500
```

**Timing breakdown for `locomotion`:**
```
RL step = substeps_per_action × physics.dt = 10 × 0.05s = 0.5s (2 Hz)
DisMech handles its own internal integration within each 0.05s step
```

## Source Files

| Paper | Config File | Env File |
|-------|-------------|----------|
| Choi 2025 | `choi2025/config.py` | — |
| Jiang 2024 | `jiang2024/configs_jiang2024.py` | `jiang2024/env_jiang2024.py` |
| Licher 2025 | `licher2025/configs_licher2025.py` | — |
| Liu 2021 | — | `liu2021/env_liu2021.py` |
| Liu 2022 | `liu2022/configs_liu2022.py` | — |
| Liu 2023 | `liu2023/configs_liu2023.py` | `liu2023/env_liu2023.py` |
| Naughton 2021 | `naughton2021/configs_naughton2021.py` | — |
| Schaffer 2024 | `schaffer2024/configs_schaffer2024.py` | — |
| Shi 2020 | `shi2020/configs_shi2020.py` | — |
| Zheng 2022 | `zheng2022/configs_zheng2022.py` | — |
| locomotion_elastica | `locomotion_elastica/config.py` | `locomotion_elastica/env.py` |
| locomotion | `locomotion/config.py` | `locomotion/env.py` |

## Observations

### By Simulator

- **MuJoCo-based** (Jiang, Liu 2021/2022/2023, Zheng): RL frequency ranges **0.5–50 Hz**. Physics timesteps 0.001–0.02 s with frame skips of 4–25.
- **PyElastica-based** (Naughton, Schaffer): Much higher RL frequency (**200–400 Hz**) due to very fine physics timesteps (2.5e-5 s) with 100–200 substeps per control step. Our `locomotion_elastica` uses a coarser physics dt (0.05 s) with 50 internal PyElastica substeps, yielding **2 Hz** RL frequency.
- **DisMech-based** (`locomotion`): Same 2 Hz RL frequency as `locomotion_elastica` (0.05 s physics dt × 10 substeps = 0.5 s control dt).
- **Kinematic** (Shi 2020): Very low frequency (**0.25–0.5 Hz**), no physics simulation, discrete geometric actions over 2–4 second intervals.
- **Custom implicit** (Choi 2025): 100 Hz with single-step implicit integration (dt = 0.01 s).

### By Control Architecture

- **Direct RL** (no CPG): Typically 10–100 Hz. The agent outputs joint torques or positions every control step.
- **CPG + RL**: Wide variation. Liu 2021/2023 run CPG at physics rate with RL at 20–50 Hz (RL modulates CPG parameters). Jiang 2024 is an extreme case: CPG runs for 100 steps between RL decisions, giving only 0.5 Hz RL frequency.
- **MPC** (Licher 2025): 70 Hz control, but this is model-predictive control rather than RL.

### Practical Range for Soft Robot RL

The most common RL decision frequencies for soft robots fall in the **10–50 Hz** range. This balances:
- Enough temporal resolution to react to dynamics
- Not so fast that the agent must make trivially small adjustments each step
- Sufficient time for soft body deformations to propagate between decisions

### Physics-to-Control Timescale Separation

| Paper | Ratio (physics steps per RL step) |
|-------|-----------------------------------|
| Jiang 2024 | 2000× (most extreme) |
| Liu 2023 | 25× |
| Liu 2022 | 10× |
| Schaffer 2024 | 200× |
| Naughton 2021 | 100× |
| Zheng 2022 | 4× |
| Choi 2025 | 1× (implicit, so stable at large dt) |
| locomotion_elastica | 500× (50 internal × 10 substeps) |
| locomotion | 10× (DisMech handles internal integration) |

Papers using compliant rod simulators (PyElastica) require the largest separation factors because the fine-grained physics timesteps needed for numerical stability are orders of magnitude smaller than useful RL decision timescales.

## Implications for This Project

Our project environments (`locomotion_elastica` and `locomotion`) both run at **2 Hz** RL frequency. Compared to the literature:
- **Much slower than other PyElastica papers**: Naughton (400 Hz) and Schaffer (200 Hz) use 100–200× higher RL frequencies. However, those papers use dt = 2.5e-5 s while ours uses dt = 0.05 s (2000× coarser physics step, offset by 50 internal substeps).
- **Comparable to CPG-based papers**: Jiang 2024 also uses 0.5 Hz with a CPG layer. Our serpenoid wave transform plays a similar role to a CPG, making 2 Hz reasonable.
- **Slower than direct-RL papers**: Liu 2021–2023 (20–50 Hz) and Zheng (25 Hz) use higher frequencies for direct joint control.
- If finer control is needed, reducing `substeps_per_action` from 10 to 2–5 would give **4–10 Hz** RL frequency while keeping the same physics dt.
