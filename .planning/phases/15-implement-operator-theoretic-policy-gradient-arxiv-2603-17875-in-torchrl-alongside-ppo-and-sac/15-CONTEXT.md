# Phase 15: Implement Operator-Theoretic Policy Gradient (arXiv:2603.17875) - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Add OTPG (Operator-Theoretic Policy Gradient) as a third RL trainer alongside PPO and SAC, implementing the MM-RKHS algorithm from Gupta & Mahajan (2026) adapted for continuous action spaces with neural network function approximation. Benchmark on the Choi2025 4-task suite for direct comparison with Phase 14's PPO/SAC results. This is a learning-signal validation (100K frames), not a full training campaign.

</domain>

<decisions>
## Implementation Decisions

### Loss function design
- MMD surrogate loss: `L = -E[ratio * A] + beta * MMD²(pi_new, pi_old) + (1/eta) * KL`
- Directly implements the paper's majorization bound (Eq 6.1) with MMD as the integral probability metric
- No PPO-style clipping — trust region enforced entirely by MMD penalty + KL regularizer
- No entropy bonus — the KL regularizer and MMD penalty handle exploration; no entropy_coef term
- Combined loss: `total = policy_loss + value_coef * critic_loss`

### Architecture
- Single V(s) critic with GAE advantage estimation (paper's framework uses advantage A_pi(s,a) with value baseline)
- Shared single Adam optimizer for actor and critic (PPO pattern, not SAC's separate optimizers)
- Policy outputs TanhNormal distribution over continuous actions (reuse `create_actor()`)
- Critic uses `create_critic()` ValueOperator wrapper

### Action space
- Continuous actions — keep Choi2025's native 5-dim delta-curvature in [-1, 1]
- No discretization — adapt MM-RKHS via MMD surrogate loss for continuous distributions

### Evaluation setup
- Benchmark on Choi2025 4-task suite only: follow_target, inverse_kinematics, tight_obstacles, random_obstacles
- Quick validation: 100K frames per task (learning signal check, not full training)
- 1 seed per configuration (4 runs total)
- Compare against Phase 14's existing PPO/SAC results (no fresh controls needed)
- W&B project: same `choi2025-replication` or new `otpg-validation` — Claude's discretion

### MMD computation
- RBF (Gaussian) kernel: k(x,y) = exp(-||x-y||² / 2σ²)
- Linear-time unbiased estimator (Gretton et al., 2012) — O(n) not O(n²)
- 16 action samples per state for MMD estimation (configurable via `mmd_num_samples`)

### Hyperparameters
- Novel params: beta=1.0 (trust region), eta=1.0 (mirror descent step size), fixed defaults with manual tuning
- Shared params from PPO defaults: lr=3e-4, epochs=10, mini_batch=256, GAE lambda=0.95, gamma=0.99
- Diagnosis via logged MMD/KL values to determine if trust region is too tight or too loose

### W&B logging
- Log OTPG-specific metrics: mmd_penalty, kl_divergence, surr_advantage, critic_loss (as separate scalars)
- Also log policy_entropy and grad_norm for diagnostics
- Standard reward/episode metrics via existing logging infrastructure

### Claude's Discretion
- W&B project naming (reuse choi2025-replication vs new project)
- Network size (3x1024 like Phase 14 or smaller for faster iteration)
- Whether to re-run PPO/SAC controls alongside OTPG
- RBF kernel bandwidth (sigma) default value
- Exact checkpoint save frequency
- Run naming convention

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Algorithm
- `media/rl_algorithms_pseudocode.pdf` — RL algorithms pseudocode reference (if relevant OTPG content)
- arXiv:2603.17875 — Source paper: MM-RKHS algorithm (Eq 7.1-7.2), majorization bound (Eq 6.1), IPM framework (Section 6)

### Existing trainers (follow these patterns)
- `src/trainers/ppo.py` — PPOTrainer class: `__init__`/`train()`/`_update()` pattern, GAE, SyncDataCollector, checkpointing, bf16 AMP
- `src/trainers/sac.py` — SACTrainer class: alternative trainer pattern for reference
- `src/trainers/__init__.py` — Trainer exports (add OTPGTrainer here)

### Config hierarchy
- `src/configs/training.py` — RLConfig base → PPOConfig/SACConfig hierarchy. OTPGConfig extends RLConfig.
- `src/configs/network.py` — NetworkConfig, ActorConfig, CriticConfig

### Network factories
- `src/networks/actor.py` — `create_actor()`: ProbabilisticActor with TanhNormal
- `src/networks/critic.py` — `create_critic()`: ValueOperator wrapper

### Choi2025 benchmark
- `papers/choi2025/config.py` — Choi2025Config hierarchy, task types, physics/env/control configs
- `papers/choi2025/env.py` — TorchRL EnvBase wrapper, action spec (5-dim continuous [-1,1])
- `papers/choi2025/train.py` — SAC training entry point (pattern for OTPG train script)
- `papers/choi2025/train_ppo.py` — PPO training entry point (pattern for OTPG train script)

### Phase 15 research
- `.planning/phases/15-implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac/15-RESEARCH.md` — Detailed architecture patterns, code examples, pitfalls, MMD implementation

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PPOTrainer` (`src/trainers/ppo.py`): On-policy trainer with GAE, SyncDataCollector, mini-batch updates — OTPG reuses ~90% of this
- `create_actor()` (`src/networks/actor.py`): ProbabilisticActor with TanhNormal distribution — reuse directly
- `create_critic()` (`src/networks/critic.py`): ValueOperator wrapper — reuse directly
- `GAE` (`torchrl.objectives.value`): Generalized Advantage Estimation — reuse directly
- `SyncDataCollector` (`torchrl.collectors`): On-policy data collection — reuse directly
- `wandb_utils` (`src/wandb_utils`): W&B setup/logging utilities — reuse directly
- `setup_run_dir()` (`src/configs/run_dir`): Run directory setup — reuse directly
- `logging_utils` (`src/trainers/logging_utils`): System metrics, grad norm computation

### Established Patterns
- Trainer `__init__` → `train()` → `_update()` → `_log_metrics()` lifecycle
- Dataclass config hierarchy: algorithm-specific config extends `RLConfig`
- bf16 mixed precision via `torch.amp.autocast` with `_amp_context()` helper
- STOP file check for graceful shutdown
- Checkpoint save/load with atomic writes
- `ConsoleLogger` for structured terminal output

### Integration Points
- `src/trainers/__init__.py` — add `OTPGTrainer` export
- `src/configs/training.py` — add `OTPGConfig` dataclass
- `papers/choi2025/` — new `train_otpg.py` entry point (mirrors `train_ppo.py`)
- `papers/choi2025/config.py` — add `Choi2025OTPGConfig(OTPGConfig)` (mirrors `Choi2025PPOConfig`)

</code_context>

<specifics>
## Specific Ideas

- The only genuinely novel code is the loss function in `_update()` — everything else (data collection, networks, GAE, logging, checkpointing) should be copied from PPOTrainer with minimal modifications
- The paper's cost-minimization convention must be negated for reward-maximization: where paper writes "minimize A_pi", we write "maximize -A_pi"
- Clamp exponential weights to [-20, 20] range for numerical stability (log-sum-exp trick)
- Normalize advantages before any exponential operations to prevent NaN/Inf

</specifics>

<deferred>
## Deferred Ideas

- Full 1M-frame training runs — defer until 100K quick validation shows learning signal
- Multi-seed runs (3-5 seeds) for statistical significance
- Adaptive beta/eta scheduling — start with fixed, upgrade if needed
- W&B sweep over beta/eta — defer until manual tuning establishes reasonable ranges
- Continuous action space: alternative adaptation approaches (weighted regression, hybrid PPO clipping)
- Gymnasium baselines (HalfCheetah, Ant) for broader algorithm validation
- Elastica snake environment benchmark

</deferred>

---

*Phase: 15-implement-operator-theoretic-policy-gradient-arxiv-2603-17875-in-torchrl-alongside-ppo-and-sac*
*Context gathered: 2026-03-19*
