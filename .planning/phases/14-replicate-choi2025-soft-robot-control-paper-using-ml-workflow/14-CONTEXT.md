# Phase 14: Replicate Choi2025 Soft Robot Control Paper Using ML Workflow - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Replicate the Choi & Tong (2025) paper's soft manipulator control experiments using our ML workflow. Train SAC and PPO policies across all 4 manipulation tasks (follow_target, inverse_kinematics, tight_obstacles, random_obstacles) using the existing `papers/choi2025/` scaffolding and DisMech implicit time-stepping. The existing code provides env, config, rewards, control, evaluation, and recording — this phase wires it into our training pipeline, runs experiments, and validates learning signal.

</domain>

<decisions>
## Implementation Decisions

### Task scope
- Replicate all 4 tasks: follow_target, inverse_kinematics, tight_obstacles, random_obstacles
- Two algorithms: SAC (paper's choice) and PPO (our addition)
- 1 seed only — not the paper's 5 seeds
- Phased approach: quick validation first (all 4 tasks x 100K frames each), then scale to full 1M frames
- Custom RL algorithm deferred to a future phase

### Training setup
- 32 parallel envs on single GPU (match CPU count; DisMech is CPU-bound)
- Hardware: 2x RTX A4000 (16GB each), 32 CPUs — train on 1 GPU, keep 2nd free
- W&B project: `choi2025-replication`
- SAC hyperparams from paper: lr=0.001, batch=2048, buffer=2M, UTD=4, 3×256 ReLU MLP
- PPO uses standard defaults (clip=0.2, epochs=10, minibatch=64) with same 3×256 network
- Experiment matrix: 4 tasks × 2 algorithms = 8 runs (after validation passes)

### Replication folder
- Training outputs go to `output/{run_name}/`
- Run naming encodes: physical setup (fixed-end), task, algorithm, key hyperparams, timestamp
  - Example: `fixed_follow_target_sac_lr1e3_32envs_20260319/`
- No custom comparison plots — use W&B dashboard for all visualization and comparison

### Validation criteria
- Success = reward improves over training for all 4 tasks with both algorithms (learning signal)
- Not trying to exactly match paper's numbers (different hardware, 1 seed vs 5)
- Record video rollouts from best SAC and PPO checkpoints using existing `record.py`
  - 1-2 episodes per task, saved to `media/choi2025/`

### Claude's Discretion
- Exact PPO hyperparameter values within standard ranges
- How to structure the PPO config dataclass (extend existing `PPOConfig` pattern)
- num_envs tuning if 32 causes issues
- Wall-time limits per run

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Paper
- `papers/choi2025/Choi and Tong - 2025 - Rapidly Learning Soft Robot Control via Implicit Time-Stepping.pdf` — Source paper with SAC hyperparams (Table A.1), simulation params (Table A.2), material params (Table A.3), reward formulations (Sec D), and training curves (Figure 3)

### Existing implementation
- `papers/choi2025/config.py` — Hierarchical dataclass configs: Choi2025Config(SACConfig), Choi2025EnvConfig, Choi2025PhysicsConfig(DismechConfig), Choi2025NetworkConfig
- `papers/choi2025/env.py` — TorchRL EnvBase wrapper for DisMech soft manipulator with 4 task types
- `papers/choi2025/train.py` — SAC training entry point using SACTrainer
- `papers/choi2025/evaluate.py` — Evaluation script with checkpoint loading
- `papers/choi2025/record.py` — Video recording with matplotlib 3D animation
- `papers/choi2025/rewards.py` — Task-specific reward functions (follow_target, IK, obstacle)
- `papers/choi2025/control.py` — Delta curvature controller with Voronoi smoothing
- `papers/choi2025/tasks.py` — Target generation and obstacle management

### ML infrastructure
- `src/trainers/sac.py` — Existing SAC trainer used by the paper scaffolding
- `src/trainers/ppo.py` — Existing PPO trainer to be wired up for the PPO experiments
- `src/configs/training.py` — SACConfig and PPOConfig base classes
- `src/configs/network.py` — ActorConfig, CriticConfig, NetworkConfig base classes
- `src/networks/actor.py` — Actor network factory (create_actor)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `papers/choi2025/` package: Complete env + config + rewards + control + tasks + evaluate + record — ready to use
- `src/trainers/sac.py`: SAC trainer already wired in `train.py`
- `src/trainers/ppo.py`: PPO trainer exists, needs config + wiring for this paper's env
- `src/configs/training.py`: Both `SACConfig` and PPO config base classes available
- `src/configs/run_dir.py` + `setup_run_dir()`: Consolidated run directory setup

### Established Patterns
- Dataclass config hierarchy: project-specific config inherits from algorithm base config
- `Choi2025Config(SACConfig)` already shows the pattern — need analogous `Choi2025PPOConfig(PPOConfig)`
- TorchRL `EnvBase` → `SerialEnv` for parallelism (already in `train.py`)
- `ConsoleLogger` wrapping for structured output

### Integration Points
- `train.py` currently only supports SAC — needs PPO path or separate `train_ppo.py`
- `evaluate.py` loads SACTrainer — needs to support PPO evaluation too
- `record.py` loads actor directly — algorithm-agnostic, should work for both
- Config `__post_init__` sets name from task — extend to include algorithm

</code_context>

<specifics>
## Specific Ideas

- Physical setup is "fixed-end" (clamped manipulator) — encode this in run naming to distinguish from future free-moving snake experiments
- Run naming should include key hyperparams for at-a-glance identification: `fixed_{task}_{algo}_{lr}_{envs}_{date}`
- The paper highlights that DisMech is 6-40x faster than Elastica for parallel stepping — we're only using DisMech

</specifics>

<deferred>
## Deferred Ideas

- Custom RL algorithm (user mentioned adding a third algorithm later — may be its own phase)
- Multi-seed runs (5 seeds per config) for statistical significance — defer until single-seed results look good
- Sim-to-sim comparison (DisMech vs Elastica) as in paper's Table 2 — requires Elastica setup
- Elastica baseline comparison (Phase 8 in roadmap already covers this)

</deferred>

---

*Phase: 14-replicate-choi2025-soft-robot-control-paper-using-ml-workflow*
*Context gathered: 2026-03-19*
