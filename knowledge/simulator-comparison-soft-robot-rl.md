---
name: simulator-comparison-soft-robot-rl
description: Deep research comparing Genesis, PyElastica, MuJoCo/MJX, SOFA, DiffTaichi, NVIDIA Warp/Newton, and other simulators for soft robot RL training, with hard performance numbers and citations
type: knowledge
created: 2026-03-09T01:44:27
updated: 2026-03-09T01:44:27
tags: [knowledge, simulator, comparison, genesis, pyelastica, mujoco, mjx, sofa, difftaichi, warp, newton, soft-robot, rl, performance, benchmark, gpu, fem, mpm, cosserat]
aliases: [simulator-benchmark, physics-engine-comparison-2025]
---

# Simulator Comparison for Soft Robot RL Training

Comprehensive comparison of physics simulators for training reinforcement learning policies on soft/continuum robots, with emphasis on snake-like locomotion. Data gathered March 2026.

## Executive Summary

No single simulator dominates all axes. The choice depends on the tradeoff between **physics fidelity** (Cosserat rod / FEM accuracy) and **training throughput** (GPU-parallelized steps/second). Key findings:

1. **PyElastica** provides accurate Cosserat rod physics but is CPU-bound (~57 FPS with 16 parallel envs in our setup). No GPU backend exists. JAX acceleration is aspirational only.
2. **Genesis** claims 43M FPS but this is for trivial rigid-body scenes with disabled collisions. Independent analysis shows 3-10x *slower* than ManiSkill/IsaacGym for realistic manipulation. Soft body (MPM/FEM) performance is undocumented.
3. **MuJoCo/MJX** is mature and well-supported. MJX achieves ~400K-750K steps/sec for simple tasks on A100. **MuJoCo-Warp (MJWarp)** is 152-313x faster than MJX. But MuJoCo models soft robots as rigid-body chains, not continuum rods.
4. **SOFA/SofaGym** provides high-fidelity FEM but is extremely slow: 500K timesteps takes 7-55 days on CPU. No GPU parallelization.
5. **DiffTaichi** is 188x faster than TensorFlow for differentiable elastic simulation, matches CUDA performance, and supports gradient-based optimization that converges in tens of iterations.
6. **Newton/MuJoCo-Warp** (NVIDIA + DeepMind + Disney, 2025) is the most promising emerging option: GPU-native, 70-313x speedup over MJX, supports deformable bodies via custom solvers.

**For this project:** The practical path forward is either (a) approximate the snake as a rigid-body chain in MuJoCo/MJX for 1000x throughput gain, accepting reduced physics fidelity, or (b) wait for Genesis MPM/FEM RL benchmarks to mature, or (c) build a JAX-based Cosserat rod solver (no existing implementation found).

---

## 1. PyElastica

### Overview
- **Physics model:** Cosserat rod theory (1D continuum)
- **Language:** Pure Python + NumPy, optional Numba JIT
- **GPU support:** None. JAX backend is a stated goal (GSoC 2023 proposal) but not implemented.
- **Parallelization:** None native. Must use multiprocessing (ParallelEnv wrapper).
- **Repo:** https://github.com/GazzolaLab/PyElastica

### Performance
| Metric | Value | Source |
|---|---|---|
| Base Python (no Numba) | Very slow (educational only) | PyElastica docs |
| With Numba JIT | ~8x speedup over base | PyElastica v0.1.0 release notes |
| Our project (16 parallel envs, V100) | ~57 FPS | Memory note (Session 10) |
| Elastica-RL-control training time | 3-24 hours per case | Naughton et al. 2021, GitHub |
| Training convergence | ~10M policy evaluations | Naughton et al. 2021 |

### Known Bottlenecks
1. **Pure Python / NumPy core:** All physics computed on CPU. Numba JIT provides only 8x improvement.
2. **No vectorized batching:** Each environment is a separate process. IPC overhead dominates beyond ~16 envs.
3. **Small timestep required:** dt ~ 0.001s with 50+ substeps per action, so wall-clock cost per env step is high.
4. **No GPU offload:** Physics, reward computation, and policy inference all on CPU (except policy network).

### RL Ecosystem
- **gym-softrobot** (https://github.com/skim0119/gym-softrobot): OpenAI Gym wrapper for PyElastica. Tested with Stable Baselines3. No published FPS benchmarks.
- **Elastica-RL-control** (https://github.com/GazzolaLab/Elastica-RL-control): 4 benchmark cases (tracking, reaching, obstacle navigation). PPO, SAC, TD3, TRPO, DDPG tested. Training: 3-24 hours/case on CPU.

### GPU/JAX Status
- GSoC 2023 project proposed JAX or pybind11 acceleration. No merged implementation found.
- Related: `jax-spcs-kinematics` provides JAX-based kinematics for HSA robots, but not full dynamics.
- **No JAX Cosserat rod dynamics implementation exists anywhere** (as of March 2026).

---

## 2. Genesis

### Overview
- **Physics solvers:** Rigid body, MPM, SPH, FEM, PBD, Stable Fluid (unified engine)
- **Language:** Pure Python, compiled via Taichi to GPU kernels
- **GPU support:** CUDA, Metal
- **Parallelization:** Batched environments (4096+ parallel)
- **Repo:** https://github.com/Genesis-Embodied-AI/Genesis

### Claimed Performance
| Metric | Claimed Value | Source |
|---|---|---|
| Franka arm (rigid body) | 43M FPS (RTX 4090) | Genesis paper |
| vs Isaac Gym | 10-80x faster | Genesis paper |
| Locomotion policy training | 26 seconds (Go2, RTX 4090) | Genesis docs |

### Actual Performance (Independent Analysis)
| Metric | Actual Value | Source |
|---|---|---|
| Franka arm (realistic settings) | ~290K FPS (150x lower than claimed) | Stone Tao analysis |
| vs ManiSkill (cube picking) | 3-10x *slower* | Stone Tao analysis |
| 20 cubes in fluid (SPH) | <1 FPS (RTX 3080) | GitHub issue #412 |
| 100 cubes free fall | <10 FPS (RTX 3080) | GitHub issue #412 |
| Complex scene (UR5e + liquid) | <1 FPS (Tesla T4) | GitHub issue #1881 |
| Camera rendering | 10x realtime (vs 1000x for Isaac/ManiSkill) | Stone Tao analysis |

### Benchmark Methodology Issues (GitHub issue #181)
StoneT2000 identified four problems with Genesis benchmarks:
1. Physics substeps=1 (tutorials use 2-4)
2. One action followed by 999 idle steps (solver early-exit optimization)
3. Robot self-collisions disabled by default
4. Object hibernation inflates FPS on inactive scenes

Genesis team acknowledged issues and released corrected benchmarks (Jan 2025), but no independent verification of soft body performance.

### Soft Body Capabilities
- MPM muscle simulation (worm example)
- FEM muscle simulation (soft gripper)
- Hybrid rigid-soft robots (MPM only)
- **No published RL training benchmarks for soft body tasks**
- Material models: neo-Hookean (MPM), stable neo-Hookean (FEM)

### Assessment
Genesis is promising but immature for soft body RL. The 43M FPS headline number is misleading. Realistic rigid-body performance is ~290K FPS. Soft body (MPM/FEM) performance is completely undocumented and likely orders of magnitude slower based on the SPH/fluid reports. The quadratic complexity of global constraint solvers (vs PhysX's linear local solvers) is a fundamental architectural limitation for contact-heavy scenes.

---

## 3. MuJoCo / MJX / MuJoCo-Warp

### Overview
- **Physics model:** Rigid-body multibody dynamics with contacts
- **Soft body approach:** Composite bodies (chains of rigid links + springs), tendons, flexcomp
- **MJX:** JAX-based GPU port (runs on GPU/TPU via XLA)
- **MJWarp:** NVIDIA Warp-based GPU port (CUDA-optimized)
- **Repo:** https://github.com/google-deepmind/mujoco

### Performance (MJX on A100)
| Task | Steps/sec | Source |
|---|---|---|
| Acrobot Swingup (PPO) | 752,092 | MuJoCo Playground |
| CartPole Balance (PPO) | 718,626 | MuJoCo Playground |
| Cheetah Run (PPO) | 435,162 | MuJoCo Playground |
| Humanoid Walk (PPO) | ~91,900 | MuJoCo Playground |
| Go1 Joystick Flat | 417,451 | MuJoCo Playground |
| Berkeley Humanoid | 120,145 | MuJoCo Playground |

### Performance (MJWarp vs MJX on RTX 4090)
| Metric | Value | Source |
|---|---|---|
| Locomotion speedup | 152x faster than MJX | MuJoCo docs |
| Manipulation speedup | 313x faster than MJX | MuJoCo docs |
| Humanoid speedup | 70x over MJX | NVIDIA blog |

### MJX Limitations
- Single-scene MJX is 10x *slower* than CPU MuJoCo (amortized over thousands of parallel envs)
- Contact cost scales with *possible* contacts, not *active* contacts (JAX static shapes)
- Performance degrades beyond ~60 DoF per scene (MJWarp limitation)
- **No native FEM/Cosserat rod support** -- soft robots must be approximated as rigid chains

### Snake Robot in MuJoCo
- **Bing et al. (IJCAI 2019):** Rigid-body snake in MuJoCo, PPO training. Sim-to-real transfer achieved for energy-efficient slithering gaits.
- **Li et al. (2022):** Soft vs rigid snake comparison in MuJoCo. Soft snake (compliant joints) consumed less energy at same velocity.
- **Bing et al. (2022, IEEE RA-M):** "Simulation to Real" -- successfully transferred MuJoCo-learned slithering gaits to physical snake robot.

### Modeling Soft Robots in MuJoCo
- **Composite bodies:** Auto-generate chains of rigid links with joints (rope, cloth, grid macros)
- **Tendons:** Model 3D tendon geometry with wrapping and via-point constraints. Used for tendon-driven continuum robots.
- **Flexcomp (MuJoCo 3+):** FEM-like deformable bodies (shells, volumes). Still limited.
- **Spring-loaded joints:** Approximate continuum compliance. Used by SoMo/SoMoGym.

### SoMo / SoMoGym (Rigid-Link Approximation)
- **Approach:** Continuum manipulators approximated as rigid links + spring-loaded joints in PyBullet
- **Speed:** Fast enough for RL (rigid-body sim)
- **Accuracy:** "Satisfactory for hyper-redundant arms, poorly approximate elastic structures" (review paper)
- **Sim-to-real:** Claimed high-fidelity transfer to physical systems
- **Repo:** https://github.com/GrauleM/somogym

---

## 4. SOFA / SofaGym

### Overview
- **Physics model:** FEM (volumetric meshes), high-fidelity continuum mechanics
- **Language:** C++ core, Python bindings
- **GPU support:** Limited. Primarily CPU-based.
- **Parallelization:** Can run parallel CPU environments, no GPU batching
- **Repo:** https://github.com/SofaDefrost/SofaGym

### Performance
| Metric | Value | Source |
|---|---|---|
| MultiGait training (500K steps, 24 CPU cores) | ~7 days (simplified) / ~55 days (complex) | Protopapa et al. 2023 |
| Domain randomization inference | ~48 hours (20 CPU cores) | Protopapa et al. 2023 |
| CatchTheObject task training | ~23 hours | SofaGym paper |
| Simplified vs complex model speedup | 7.9x | Protopapa et al. 2023 |

### Key Characteristics
- **11 benchmark environments:** CartStem, TrunkReach, Gripper, CatchTheObject, etc.
- **RL integration:** Stable Baselines3, rlberry
- **Domain randomization:** sofa-dr-rl extension for sim-to-real transfer
- **Physics accuracy:** High-fidelity FEM, but computational cost "often renders policy learning infeasible"
- **No GPU batching:** The fundamental bottleneck. FEM on CPU cannot compete with GPU-parallelized rigid body sims.

### Assessment
SOFA provides the highest physics fidelity for soft robots but is impractical for RL at scale. Training a single policy takes days to weeks. The sofa-dr-rl work shows that using simplified models with domain randomization can reduce training time by ~8x while maintaining transfer quality. This suggests the high-fidelity FEM may not be necessary during RL training if domain randomization is used.

---

## 5. DiffTaichi / Taichi-Based Simulation

### Overview
- **Physics model:** Differentiable MPM, elastic bodies, fluids
- **Language:** Python DSL, JIT-compiled to CUDA/CPU
- **GPU support:** Full CUDA support
- **Differentiability:** End-to-end backprop through physics
- **Repo:** https://github.com/taichi-dev/difftaichi

### Performance
| Metric | Value | Source |
|---|---|---|
| ChainQueen (elastic body) vs TensorFlow | 188x faster | DiffTaichi paper (ICLR 2020) |
| ChainQueen vs hand-written CUDA | Equal speed, 4.2x shorter code | DiffTaichi paper |
| Soft robot locomotion optimization | Converges in tens of iterations | DiffTaichi paper |

### Key Characteristics
- **Gradient-based optimization:** Instead of RL (sample-inefficient), uses differentiable physics to compute exact gradients. Converges in 10s of iterations vs millions for RL.
- **PlasticineLab (ICLR 2021):** Soft-body manipulation benchmark built on DiffTaichi/ChainQueen. RL (PPO/SAC/TD3) struggles; gradient-based methods achieve 0.90 vs 0.69 scores.
- **Megakernel fusion:** Preserves GPU arithmetic intensity through source code transformation.

### Limitations
- Primarily designed for gradient-based optimization, not RL
- No Gymnasium/Gym API integration
- No massively parallel env batching (designed for single-env gradient computation)
- PlasticineLab showed RL methods are inefficient in these environments

### Assessment
DiffTaichi is transformative for gradient-based control optimization of soft robots, but it is not designed for RL training loops. If the control problem can be formulated as trajectory optimization rather than RL, DiffTaichi is orders of magnitude more efficient. Genesis internally uses Taichi for kernel compilation.

---

## 6. NVIDIA Warp / Newton

### Overview
- **Newton:** Open-source GPU physics engine (NVIDIA + DeepMind + Disney, 2025)
- **Built on:** NVIDIA Warp (Python framework for GPU spatial computing)
- **Physics:** Rigid body, deformable (cloth, soft body via VBD solver), custom solvers
- **Differentiability:** Full (via Warp autodiff)
- **Repo:** https://github.com/newton-physics/newton

### Performance
| Metric | Value | Source |
|---|---|---|
| MuJoCo-Warp locomotion vs MJX | 152x speedup (RTX 4090) | MuJoCo docs |
| MuJoCo-Warp manipulation vs MJX | 313x speedup (RTX 4090) | MuJoCo docs |
| Humanoid simulation acceleration | 70x over MJX | NVIDIA blog |
| vs PhysX (dexterous manipulation) | 65% faster | NVIDIA blog |

### Deformable Body Support
- VBD (Variational-Based Dynamics) solver for cloth and thin-shell simulation
- Custom solver integration for arbitrary physics
- Compatible with Isaac Lab for RL training
- **Warp FEM module:** Arbitrary-order displacement fields on meshes

### Taccel (Related, Built on Warp)
- GPU tactile robotics simulator (PKU + UCLA, 2025)
- IPC (Incremental Potential Contact) + ABD (Affine Body Dynamics)
- 915 FPS total with 4096 parallel envs on H100 (peg insertion)
- 12.67 FPS with 256 envs (dexterous grasping with full-hand tactile)
- NeurIPS 2025 spotlight

### Assessment
Newton/MuJoCo-Warp is the most promising emerging platform. It combines MuJoCo's mature API with NVIDIA's GPU optimization, and supports deformable bodies through extensible solvers. However, it is very new (2025) and lacks established soft robot RL benchmarks. The 152-313x speedup over MJX is significant. For snake robot simulation, one could potentially implement a Cosserat rod solver as a custom Newton solver.

---

## 7. Other Relevant Simulators

### JAX-FEM
- Differentiable GPU-accelerated 3D FEM solver
- 10x speedup over commercial FEM (7.7M DoF problem)
- Could theoretically be used for soft robot RL, but no Gym integration
- **Repo:** https://github.com/deepmodeling/jax-fem

### Brax (Google)
- JAX-based **rigid body** simulator (not soft body)
- Ant locomotion in 10 seconds (vs 3 hours standard PPO)
- Millions of steps/sec on TPU
- 100-1000x speedup over CPU RL pipelines
- Now largely superseded by MJX for robotics
- **Repo:** https://github.com/google/brax

### JaxSim
- Differentiable physics engine in JAX for robot learning
- CPU/GPU/TPU support, JIT compilation, auto-vectorization
- Rigid-body focused
- **Repo:** https://github.com/ami-iit/jaxsim

### Isaac Gym / Isaac Lab (NVIDIA)
- PhysX 5 backend: rigid body + FEM soft body + PBD
- 10^3 - 10^4 concurrent GPU environments
- FEM deformable bodies: 3x faster than CPU alternatives
- Deformable object support exists but RL benchmarks focus on rigid tasks
- Isaac Lab integrating Newton for future deformable body RL

---

## 8. Comparison Table

| Simulator | Physics Model | Soft Body | GPU Parallel | RL Steps/sec | Training Time (locomotion) | Differentiable | Maturity |
|---|---|---|---|---|---|---|---|
| **PyElastica** | Cosserat rod | Native | No | ~57 FPS (16 envs) | 3-24 hours/case | No | Mature |
| **Genesis** | Multi-solver | MPM/FEM | Yes | ~290K (rigid, corrected) | 26 sec (rigid, Go2) | Partial (MPM) | Early |
| **MuJoCo CPU** | Rigid body | Composite/flex | No | ~10K (single-thread) | Hours | No | Mature |
| **MJX (JAX)** | Rigid body | Composite/flex | Yes | 90K-750K (A100) | 5-30 min | Yes | Mature |
| **MJWarp** | Rigid body | Composite/flex | Yes (CUDA) | ~10M-100M+ (est.) | <5 min (est.) | Yes | New (2025) |
| **SOFA/SofaGym** | FEM | Native | No | ~1-10 | 7-55 days | No | Mature |
| **DiffTaichi** | MPM/elastic | Native | Yes | N/A (gradient-based) | Seconds-minutes | Yes | Research |
| **Newton/Warp** | Extensible | VBD/custom | Yes (CUDA) | 152-313x MJX | Minutes (est.) | Yes | New (2025) |
| **Brax** | Rigid body | No | Yes (JAX) | Millions | 10 sec (Ant) | Yes | Mature |
| **Isaac Lab** | PhysX 5 | FEM/PBD | Yes (CUDA) | 100K+ | Minutes | Partial | Mature |
| **SoMo/SoMoGym** | Rigid approx. | Rigid-link chain | No (PyBullet) | ~100-1K (est.) | Hours | No | Research |

---

## 9. Sim-to-Real Transfer for Snake Robots

### Successful Transfers

1. **Bing et al. (2019, 2022):** MuJoCo rigid-body snake -> physical snake robot. PPO-trained energy-efficient slithering gaits transferred successfully. Key: model snake body dynamics closely to real hardware.

2. **CPG-RL for Soft Snake (2022):** FEM-based soft snake simulator -> real pneumatic soft snake. Used CPG (Matsuoka oscillators) + RL. Free-response oscillation constraints improved transferability. Sim runs in real-time on GPU.

3. **Soft Robot Crawlers (2023, Autonomous Robots):** FEM with model order reduction -> real soft legged robots. Zero-shot transfer achieved. Domain randomization improved robustness. Key insight: reduced-order FEM models are accurate enough for transfer while being fast enough for RL.

### Key Insights for Transfer

| Approach | Physics Gap | Training Speed | Transfer Quality |
|---|---|---|---|
| High-fidelity FEM (SOFA) | Small | Very slow | High (if feasible) |
| Rigid-body chain (MuJoCo) | Large | Very fast | Good (with domain rand.) |
| Reduced-order FEM | Medium | Moderate | Good (zero-shot reported) |
| Simplified + Domain Rand. | Large | 7.9x faster | Good (sofa-dr-rl) |

### The Fidelity-Speed Tradeoff
The sofa-dr-rl work (Protopapa et al. 2023) provides the most actionable finding: **using drastically simplified models with domain randomization achieves comparable transfer quality to high-fidelity FEM, at 7.9x lower training cost**. This suggests that for RL training, physics fidelity matters less than robust domain randomization.

---

## 10. Recommendations for This Project

### Option A: Stay with PyElastica (Current)
- **Pro:** Accurate Cosserat rod physics, proven working
- **Con:** ~57 FPS, training takes hours-days, no GPU path
- **When:** If physics accuracy is paramount and training time is acceptable

### Option B: MuJoCo/MJX Rigid-Body Chain
- **Pro:** 750K+ steps/sec (MJX), 100M+ steps/sec (MJWarp). Proven sim-to-real for snake robots (Bing et al.)
- **Con:** Loses continuous rod physics. Must tune spring stiffness to approximate Cosserat behavior.
- **When:** If training speed is the bottleneck and domain randomization can bridge the physics gap
- **Speedup vs current:** ~10,000x (MJX) to ~1,000,000x (MJWarp)

### Option C: Genesis MPM/FEM
- **Pro:** GPU-parallelized soft body simulation, unified engine
- **Con:** Immature, no soft body RL benchmarks, benchmark credibility issues, soft body FPS likely <1000
- **When:** If Genesis matures significantly in the next 6-12 months

### Option D: Build JAX Cosserat Rod Solver
- **Pro:** Would combine accurate Cosserat physics with GPU parallelization
- **Con:** Significant engineering effort. No existing implementation to build on.
- **When:** If this project needs to advance the state of the art in soft robot RL infrastructure

### Option E: Newton + Custom Cosserat Solver
- **Pro:** GPU-native, extensible architecture designed for custom solvers, differentiable
- **Con:** Very new (2025), would require implementing Cosserat rod solver as a Warp kernel
- **When:** If willing to invest in framework development for long-term payoff

### Recommended Path
**Option B (MuJoCo/MJX)** for immediate speedup, validated by Bing et al.'s successful sim-to-real transfer with a MuJoCo rigid-body snake. Use domain randomization over joint stiffness, friction, and body parameters to bridge the gap to Cosserat rod physics. This gives a 10,000x+ training throughput improvement while maintaining a proven path to sim-to-real transfer.

---

## Sources

### PyElastica
- [PyElastica GitHub](https://github.com/GazzolaLab/PyElastica)
- [Elastica-RL-control GitHub](https://github.com/GazzolaLab/Elastica-RL-control)
- [gym-softrobot GitHub](https://github.com/skim0119/gym-softrobot)
- [PyElastica GSoC 2023 Discussion](https://github.com/GazzolaLab/PyElastica/discussions/225)
- Naughton et al. "Elastica: A compliant mechanics environment for soft robotic control." IEEE RA-L 2021.

### Genesis
- [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis)
- [Genesis Benchmark Issue #181](https://github.com/Genesis-Embodied-AI/Genesis/issues/181)
- [Genesis Performance Issue #412](https://github.com/Genesis-Embodied-AI/Genesis/issues/412)
- [Genesis Slow Simulation Issue #1881](https://github.com/Genesis-Embodied-AI/Genesis/issues/1881)
- [Stone Tao: "How fast is the new hyped Genesis simulator?"](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)
- [Genesis Speed Benchmark Repo](https://github.com/zhouxian/genesis-speed-benchmark)
- [MuJoCo Discussion #2303: Genesis claims vs MJX](https://github.com/google-deepmind/mujoco/discussions/2303)

### MuJoCo / MJX / MJWarp
- [MuJoCo Playground paper](https://arxiv.org/html/2502.08844v1)
- [MJWarp Documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [MJX Documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [MuJoCo-Warp GitHub](https://github.com/google-deepmind/mujoco_warp)
- [Simulator Comparison (Simulately)](https://simulately.wiki/docs/comparison/)

### SOFA / SofaGym
- [SofaGym GitHub](https://github.com/SofaDefrost/SofaGym)
- [sofa-dr-rl GitHub](https://github.com/andreaprotopapa/sofa-dr-rl)
- [SofaGym paper (Soft Robotics, 2023)](https://www.liebertpub.com/doi/10.1089/soro.2021.0123)
- [Domain Randomization for Soft Robots](https://arxiv.org/html/2303.04136)

### DiffTaichi / PlasticineLab
- [DiffTaichi GitHub](https://github.com/taichi-dev/difftaichi)
- Hu et al. "DiffTaichi: Differentiable Programming for Physical Simulation." ICLR 2020.
- [PlasticineLab](https://plasticinelab.csail.mit.edu/) (ICLR 2021)

### NVIDIA Newton / Warp
- [Newton GitHub](https://github.com/newton-physics/newton)
- [Newton Announcement (NVIDIA Blog)](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/)
- [Warp FEM Documentation](https://nvidia.github.io/warp/modules/fem.html)
- [Taccel: GPU Tactile Robotics](https://arxiv.org/abs/2504.12908) (NeurIPS 2025)

### Snake Robot RL / Sim-to-Real
- Bing et al. "Energy-Efficient Slithering Gait Exploration for a Snake-like Robot based on Reinforcement Learning." IJCAI 2019.
- Bing et al. "Simulation to Real: Learning Energy-Efficient Slithering Gaits for a Snake-Like Robot." IEEE RA-M 2022.
- [RL_Snake GitHub](https://github.com/zhenshan-bing/RL_Snake)
- "Reinforcement Learning of CPG-regulated Locomotion Controller for a Soft Snake Robot." arXiv:2207.04899.
- "Sim-to-real transfer of co-optimized soft robot crawlers." Autonomous Robots, 2023.
- [SoMoGym GitHub](https://github.com/GrauleM/somogym)

### Other
- [JAX-FEM GitHub](https://github.com/deepmodeling/jax-fem)
- [Brax GitHub](https://github.com/google/brax)
- [JaxSim GitHub](https://github.com/ami-iit/jaxsim)
- "Rod models in continuum and soft robot control: a review." arXiv:2407.05886.
