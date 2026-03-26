# Phase 15: Implement Operator-Theoretic Policy Gradient (arXiv:2603.17875) - Research

**Researched:** 2026-03-19
**Domain:** Reinforcement learning algorithm implementation (policy gradient, operator theory)
**Confidence:** MEDIUM

## Summary

The paper "Operator-Theoretic Foundations and Policy Gradient Methods for General MDPs with Unbounded Costs" (Gupta & Mahajan, 2026) proposes a theoretical framework viewing MDPs through linear operator theory over Banach spaces. The key computational algorithm is called **MM-RKHS** (Majorization-Minimization in Reproducing Kernel Hilbert Spaces), which the paper demonstrates outperforms standard PPO on finite discrete MDPs (GARNET environments). The paper is primarily theoretical -- it provides mathematical foundations for general state/action space MDPs and derives several known algorithms (PPO, TRPO, policy mirror descent) as special cases of their framework.

**Critical finding:** The paper's concrete implementable algorithm (MM-RKHS, Section 7, Equation 7.2) operates on **finite discrete state-action spaces** where the policy is a probability vector over actions at each state. The policy update rule is a closed-form exponential-weighted update indexed by a positive definite matrix R, which replaces the KL-divergence regularizer in TRPO. This algorithm does NOT directly apply to continuous action spaces with neural network function approximation (the standard deep RL setting used in this project). The paper acknowledges this gap: "further research needs to be done" for problems with infinite-dimensional state/action spaces (Section 8, Conclusion).

**Primary recommendation:** Implement a continuous-action adaptation of the MM-RKHS algorithm by using the paper's majorization bound with Maximum Mean Discrepancy (MMD) as the integral probability metric, adapting the closed-form update into a loss function amenable to gradient-based optimization with neural network policies. This requires approximating the per-state advantage-weighted exponential update as a regression target or surrogate objective. The implementation should follow the existing trainer pattern (PPOTrainer/SACTrainer) for seamless integration.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torchrl | 0.11.1 | RL environment/data infrastructure | Already used by project |
| torch | 2.10.0 | Neural network training | Already used by project |
| tensordict | (bundled) | TensorDict data containers | Already used by project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| wandb | (installed) | Experiment tracking | Via src.wandb_utils |
| numpy | (installed) | Numerical computation | Advantage computation |

No additional libraries are needed. The MM-RKHS algorithm is implemented from scratch using the existing PyTorch/TorchRL infrastructure.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── configs/
│   └── training.py        # Add MMRKHSConfig dataclass (extends RLConfig)
├── trainers/
│   ├── __init__.py         # Export MMRKHSTrainer
│   ├── mmrkhs.py             # New: MMRKHSTrainer class
│   └── logging_utils.py    # Reuse existing
├── networks/
│   ├── actor.py            # Reuse create_actor()
│   └── critic.py           # Reuse create_critic()
```

### Pattern 1: Trainer Class Structure (from existing codebase)
**What:** All trainers follow the same pattern: `__init__` creates networks/optimizer/collector, `train()` runs the main loop, `_update()` performs per-batch gradient updates, `_log_metrics()` handles W&B/console output.
**When to use:** Always -- this is the project's established pattern.
**Example:**
```python
# Follow PPOTrainer pattern exactly:
class MMRKHSTrainer:
    def __init__(self, env, config, network_config, device, run_dir):
        # 1. Create actor (ProbabilisticActor via create_actor())
        # 2. Create critic (ValueOperator via create_critic())
        # 3. Create optimizer (Adam)
        # 4. Create GAE advantage module
        # 5. Create SyncDataCollector
        # 6. Setup W&B, metrics, checkpoints

    def train(self, callback=None) -> dict:
        # Batch loop over collector
        # Compute advantages with GAE
        # Call _update(batch)
        # Log metrics

    def _update(self, batch) -> dict:
        # MM-RKHS loss computation (novel part)
        # Multiple epochs over mini-batches
        # Return loss metrics
```

### Pattern 2: Config Dataclass Hierarchy
**What:** Configs inherit from RLConfig, adding algorithm-specific fields.
**When to use:** The MMRKHSConfig should extend RLConfig, adding MM-RKHS-specific hyperparameters.
**Example:**
```python
@dataclass
class MMRKHSConfig(RLConfig):
    """OTPG (Operator-Theoretic Policy Gradient) configuration.

    Based on MM-RKHS algorithm from Gupta & Mahajan (2026).
    """
    # MM-RKHS hyperparameters
    beta: float = 1.0          # Majorization bound coefficient
    eta: float = 1.0           # Mirror descent step size (eta_k in paper)
    mmd_kernel: str = "rbf"    # Kernel for MMD computation
    mmd_bandwidth: float = 1.0 # RBF kernel bandwidth

    # GAE (shared with PPO)
    gae_lambda: float = 0.95
    normalize_advantage: bool = True

    # Learning rate schedule
    lr_schedule: str = "linear"
    lr_end: float = 1e-5

    # Early stopping
    patience_batches: int = 200
```

### Pattern 3: On-Policy Data Collection (PPO-style)
**What:** MM-RKHS is an on-policy algorithm like PPO/TRPO. Use SyncDataCollector with frames_per_batch rollouts, compute GAE advantages, then update policy.
**When to use:** Always for this algorithm.

### Pattern 4: The MM-RKHS Loss Function (Novel)
**What:** The paper's Equation 7.1-7.2 defines a policy update for finite MDPs. For continuous actions with neural network policies, we adapt this to a surrogate loss:
```python
def compute_otpg_loss(self, batch):
    """Compute MM-RKHS surrogate loss for continuous action spaces.

    The paper's update rule (Eq 7.2):
        pi_{k+1}(s,a) = (1/Z) * pi_k(s,a) * exp(-eta * (A(s,a) - beta*R^T*pi_k(s)))

    For continuous actions with neural networks, we minimize:
        L = E_s[ E_{a~pi_new}[ A_pi_old(s,a) ] + beta * MMD^2(pi_new(.|s), pi_old(.|s)) ]

    This is the majorization bound from Eq 6.1 with MMD as the IPM,
    plus a KL regularizer (mirror descent term from Eq 7.1).
    """
    # Get advantage estimates from batch
    advantage = batch["advantage"]

    # Compute new policy log-probs
    dist = self.actor.get_dist(batch)
    log_prob_new = dist.log_prob(batch["action"])
    log_prob_old = batch["sample_log_prob"]

    # Importance ratio
    ratio = (log_prob_new - log_prob_old).exp()

    # Surrogate advantage term (like PPO but without clipping)
    # The majorization bound provides trust region via MMD penalty
    surr_advantage = ratio * advantage

    # MMD^2 penalty between old and new policy
    # Approximated via samples from both policies
    mmd_penalty = self._compute_mmd_penalty(batch)

    # Mirror descent KL regularizer (Eq 7.1)
    kl_penalty = (log_prob_new - log_prob_old).mean()

    # Total loss: minimize advantage + beta*MMD^2 + (1/eta)*KL
    loss = -surr_advantage.mean() + self.config.beta * mmd_penalty + (1/self.config.eta) * kl_penalty

    return loss
```

### Anti-Patterns to Avoid
- **Implementing the discrete finite-MDP algorithm directly:** The paper's Eq 7.2 gives a closed-form update for finite |S|x|A| policy tables. Neural network policies require a surrogate loss instead.
- **Using the paper's IPM framework with total variation norm:** TV norm is hard to compute for continuous distributions. Use MMD (Maximum Mean Discrepancy) which the paper explicitly suggests as a tractable IPM (Section 6.1, Remark 6.2).
- **Skipping GAE:** The advantage function A_pi is essential. Reuse the exact same GAE module as PPO.
- **Building a separate data collection pipeline:** Reuse SyncDataCollector from TorchRL exactly as PPOTrainer does.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Advantage estimation | Custom GAE | `torchrl.objectives.value.GAE` | Battle-tested, already used by PPO |
| Data collection | Custom rollout loop | `torchrl.collectors.SyncDataCollector` | Handles env stepping, resets, device transfer |
| Actor network | New policy network | `src.networks.actor.create_actor()` | ProbabilisticActor with TanhNormal, already tested |
| Critic network | New value network | `src.networks.critic.create_critic()` | ValueOperator wrapper, already tested |
| W&B logging | Direct wandb calls | `src.wandb_utils` | Standardized across all trainers |
| Run directory setup | Manual path construction | `src.configs.run_dir.setup_run_dir()` | Consistent naming convention |
| Replay buffer | N/A (on-policy) | No replay buffer needed | MM-RKHS is on-policy like PPO |
| Checkpoint save | Custom save | Follow PPOTrainer.save_checkpoint() pattern | Atomic saves, backup files |

**Key insight:** The only genuinely novel code is the loss function in `_update()`. Everything else -- data collection, network creation, advantage estimation, logging, checkpointing -- should be copied from PPOTrainer with minimal modifications.

## Common Pitfalls

### Pitfall 1: Treating the Paper as Providing a Ready-to-Implement Deep RL Algorithm
**What goes wrong:** The paper's MM-RKHS algorithm (Section 7) is defined for finite state-action MDPs with tabular policies. Directly translating Equation 7.2 to neural networks is not possible because (a) the policy is not a probability table, and (b) the positive definite matrix R has no direct neural network analogue.
**Why it happens:** The paper's title mentions "general MDPs" and "policy gradient methods," suggesting deep RL applicability. But the computational algorithm is finite-MDP only; the general MDP results are theoretical.
**How to avoid:** Implement the *spirit* of the algorithm: the majorization bound (Eq 6.1) with MMD penalty + KL regularizer (Eq 7.1), adapted as a differentiable loss function for policy gradient with neural networks.
**Warning signs:** If you find yourself building state-action lookup tables or computing matrix inverses indexed by states, you're implementing the wrong thing.

### Pitfall 2: MMD Computation Cost
**What goes wrong:** Naive kernel MMD computation between policy samples is O(n^2) in the number of samples, which can be expensive during training.
**Why it happens:** MMD requires pairwise kernel evaluations between samples from the old and new policy.
**How to avoid:** Use the unbiased linear-time MMD estimator (Gretton et al., 2012) which is O(n). Alternatively, use a small fixed number of action samples (e.g., 16-32) per state for MMD estimation. Another practical option: replace MMD with a simpler divergence metric (squared difference in means/variances) for the initial implementation, then upgrade.
**Warning signs:** Training throughput drops significantly compared to PPO.

### Pitfall 3: Hyperparameter Sensitivity (beta, eta)
**What goes wrong:** The majorization bound coefficient beta and mirror descent step size eta interact non-trivially. Too large beta over-constrains the policy update; too small makes the bound loose and training unstable.
**Why it happens:** The paper sets beta = ||c||_inf / (1-gamma) and uses eta_k -> infinity. These theoretical values don't translate directly to practical settings.
**How to avoid:** Start with beta=1.0, eta=1.0 as defaults. Add adaptive scheduling: increase eta over training (start conservative, become more aggressive). Log MMD values to diagnose whether the constraint is too tight or too loose.
**Warning signs:** Policy changes become negligible (beta too high) or training diverges (beta too low).

### Pitfall 4: Mixing Up Cost vs Reward Convention
**What goes wrong:** The paper uses a **cost minimization** convention (minimize J_pi), while the existing codebase and TorchRL use **reward maximization** (maximize expected return).
**Why it happens:** The mathematical development in the paper defines c(s,a) as a cost function to be minimized.
**How to avoid:** Negate the advantage in the loss: where the paper writes "minimize A_pi", we write "maximize -A_pi" or equivalently minimize -advantage. The existing PPO loss already handles this (ClipPPOLoss minimizes negative of clipped surrogate).
**Warning signs:** Policy starts choosing worse actions over time.

### Pitfall 5: NaN/Inf in Exponential Weights
**What goes wrong:** The MM-RKHS update involves exp(-eta * A(s,a)), which can overflow or underflow for large advantages.
**Why it happens:** Advantage values can span several orders of magnitude, especially early in training.
**How to avoid:** Normalize advantages before exponentiating (subtract max for numerical stability, like log-sum-exp trick). Clamp the exponent to [-20, 20] range.
**Warning signs:** NaN losses, zero probabilities, policy collapse.

## Code Examples

### Example 1: MMRKHSConfig Dataclass
```python
# Source: project convention from src/configs/training.py
@dataclass
class MMRKHSConfig(RLConfig):
    """Operator-Theoretic Policy Gradient configuration.

    Based on MM-RKHS algorithm (Gupta & Mahajan, 2026, arXiv:2603.17875).
    Adapts the majorization-minimization framework to continuous action
    spaces with neural network function approximation.
    """
    # Majorization bound coefficient (beta in paper Eq 6.1)
    # Controls trust region size via MMD penalty
    beta: float = 1.0

    # Mirror descent step size (eta_k in paper Eq 7.1)
    # Larger eta = more aggressive policy updates
    eta: float = 1.0
    eta_schedule: str = "constant"  # constant, linear_increase
    eta_end: float = 10.0

    # MMD kernel configuration
    mmd_kernel: str = "rbf"       # rbf, linear, polynomial
    mmd_bandwidth: float = 1.0    # sigma for RBF kernel
    mmd_num_samples: int = 16     # action samples per state for MMD

    # PPO-compatible settings (shared infrastructure)
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    entropy_coef: float = 0.01    # Additional entropy bonus
    value_coef: float = 0.5       # Critic loss weight

    # Learning rate schedule
    lr_schedule: str = "linear"
    lr_end: float = 1e-5

    # Early stopping
    patience_batches: int = 200
```

### Example 2: MMD Penalty Computation
```python
def _compute_mmd_penalty(self, obs: torch.Tensor, old_dist, new_dist,
                          num_samples: int = 16) -> torch.Tensor:
    """Compute MMD^2 between old and new policy distributions.

    Uses RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
    Linear-time unbiased estimator.

    Args:
        obs: Observations [batch_size, obs_dim]
        old_dist: Distribution from old policy
        new_dist: Distribution from new policy
        num_samples: Number of action samples per state

    Returns:
        Scalar MMD^2 estimate
    """
    sigma = self.config.mmd_bandwidth

    # Sample actions from both distributions
    with torch.no_grad():
        x = old_dist.sample((num_samples,))  # [num_samples, batch, act_dim]
    y = new_dist.rsample((num_samples,))      # [num_samples, batch, act_dim]

    # Compute kernel values using linear-time estimator
    # Pair up samples: (x_1,x_2), (y_1,y_2), (x_1,y_1)
    n = num_samples // 2
    x1, x2 = x[:n], x[n:2*n]
    y1, y2 = y[:n], y[n:2*n]

    def rbf_kernel(a, b):
        diff = (a - b).pow(2).sum(-1)  # [n, batch]
        return (-diff / (2 * sigma**2)).exp()

    k_xx = rbf_kernel(x1, x2)
    k_yy = rbf_kernel(y1, y2)
    k_xy = rbf_kernel(x1, y1)

    mmd2 = (k_xx + k_yy - 2 * k_xy).mean()
    return mmd2
```

### Example 3: Core Loss Function
```python
def _update(self, batch: TensorDict) -> Dict[str, float]:
    """Perform MM-RKHS update on batch.

    Loss = -E[ratio * A] + beta * MMD^2 + (1/eta) * KL + value_coef * critic_loss
    """
    metrics = {"loss_policy": 0.0, "loss_critic": 0.0, "mmd_penalty": 0.0,
               "kl_divergence": 0.0, "grad_norm": 0.0}
    actual_updates = 0

    for epoch in range(self.config.num_epochs):
        indices = torch.randperm(batch.numel())
        num_batches = max(1, batch.numel() // self.config.mini_batch_size)

        for i in range(num_batches):
            mb_indices = indices[i*self.config.mini_batch_size : (i+1)*self.config.mini_batch_size]
            mb = batch[mb_indices]

            # Forward pass through actor
            dist = self.actor.get_dist(mb)
            log_prob_new = dist.log_prob(mb["action"])
            log_prob_old = mb["sample_log_prob"]

            # Importance ratio
            ratio = (log_prob_new - log_prob_old).exp()
            advantage = mb["advantage"]

            # Surrogate advantage (no clipping -- trust region via MMD)
            surr_advantage = (ratio * advantage).mean()

            # MMD penalty
            old_dist = self.actor.get_dist(mb.clone())  # detached
            mmd = self._compute_mmd_penalty(mb["observation"], old_dist, dist)

            # KL regularizer
            kl = (log_prob_new - log_prob_old).mean()

            # Critic loss
            value_pred = self.critic(mb)["state_value"]
            value_target = mb["value_target"]
            critic_loss = F.mse_loss(value_pred, value_target)

            # Total loss
            policy_loss = -surr_advantage + self.config.beta * mmd + (1/self.config.eta) * kl
            total_loss = policy_loss + self.config.value_coef * critic_loss

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.loss_params, self.config.max_grad_norm)
            self.optimizer.step()
            actual_updates += 1

            metrics["loss_policy"] += policy_loss.item()
            metrics["loss_critic"] += critic_loss.item()
            metrics["mmd_penalty"] += mmd.item()
            metrics["kl_divergence"] += kl.item()

    for key in metrics:
        metrics[key] /= max(1, actual_updates)
    return metrics
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TRPO (KL constraint) | PPO (clipped surrogate) | 2017 | Simpler, nearly as good |
| PPO (first-order) | MM-RKHS (approximate second-order) | 2026 (this paper) | Faster convergence on tabular MDPs; untested for deep RL |
| KL divergence metric | IPM (MMD, Wasserstein) | 2026 (this paper) | Avoids KL derivative computation; MMD easier in high-dim |

**Key novelty of the paper:**
- Uses integral probability metrics (IPMs) instead of KL divergence for the trust region
- Derives a majorization-minimization algorithm that approximates Newton's method
- Shows this recovers TRPO/PPO as special cases in finite MDPs
- MM-RKHS avoids the KL-divergence second derivative computation needed by TRPO

**Deprecated/outdated:**
- Nothing in the existing codebase needs to be deprecated. MM-RKHS is an *addition* alongside PPO and SAC.

## Open Questions

1. **Continuous Action Space Adaptation**
   - What we know: The paper's closed-form update (Eq 7.2) works for discrete actions. The majorization bound (Eq 6.1) and mirror descent framework (Eq 7.1) are general.
   - What's unclear: The best way to translate the exponential policy update into a neural network loss. Options include: (a) weighted regression, (b) surrogate loss with MMD penalty, (c) hybrid with PPO clipping.
   - Recommendation: Start with option (b) -- surrogate loss with MMD penalty -- as it most directly implements the paper's theoretical contribution. If MMD computation is too expensive, fall back to a simpler penalty (e.g., squared mean-action difference) and iterate.

2. **Choice of Positive Definite Matrix R**
   - What we know: In the finite MDP case (Section 7), R indexes a family of RKHS algorithms. R = I (identity) recovers a specific case.
   - What's unclear: How R should be chosen for continuous action spaces with neural network policies.
   - Recommendation: Start with R = I (identity), which makes the RKHS kernel simply the standard RBF kernel. This is the simplest choice and likely sufficient for initial experiments.

3. **Comparative Performance**
   - What we know: On GARNET (discrete, tabular), MM-RKHS converges faster than PPO.
   - What's unclear: Whether the advantage carries over to deep RL with continuous actions and function approximation.
   - Recommendation: Implement first, then run comparative experiments on the same Choi2025 tasks as Phase 14 (follow_target, inverse_kinematics, etc.).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none (pytest defaults) |
| Quick run command | `python -m pytest tests/test_mmrkhs.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OTPG-01 | MMRKHSConfig dataclass validates and inherits RLConfig | unit | `python -m pytest tests/test_mmrkhs.py::test_config -x` | Wave 0 |
| OTPG-02 | MMRKHSTrainer initializes with env/config | unit | `python -m pytest tests/test_mmrkhs.py::test_trainer_init -x` | Wave 0 |
| OTPG-03 | MMD penalty computes without NaN | unit | `python -m pytest tests/test_mmrkhs.py::test_mmd_penalty -x` | Wave 0 |
| OTPG-04 | _update() produces finite loss values | unit | `python -m pytest tests/test_mmrkhs.py::test_update_step -x` | Wave 0 |
| OTPG-05 | Short training run completes without crash | smoke | `python -m pytest tests/test_mmrkhs.py::test_short_training -x` | Wave 0 |
| OTPG-06 | Trainer exports/imports checkpoint | unit | `python -m pytest tests/test_mmrkhs.py::test_checkpoint -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_mmrkhs.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_mmrkhs.py` -- covers OTPG-01 through OTPG-06
- [ ] Framework install: pytest already available

## Sources

### Primary (HIGH confidence)
- arXiv:2603.17875 PDF -- full paper read (25 pages), all equations and algorithms extracted
- Project codebase: `src/trainers/ppo.py`, `src/trainers/sac.py`, `src/trainers/ddpg.py` -- architecture patterns
- Project codebase: `src/configs/training.py` -- config hierarchy
- `pip show torchrl` -- version 0.11.1 confirmed

### Secondary (MEDIUM confidence)
- Paper's experimental validation on GARNET environments (Section 7.1) -- only tabular experiments
- MMD linear-time estimator approach (Gretton et al., 2012) -- well-established technique

### Tertiary (LOW confidence)
- Continuous action space adaptation of MM-RKHS -- no existing implementation or precedent; this is novel engineering
- Expected performance relative to PPO on continuous control -- completely unknown, paper only shows tabular results

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- using existing project libraries, no new dependencies
- Architecture: HIGH -- follows exact patterns from PPOTrainer/SACTrainer
- Algorithm translation (discrete to continuous): LOW -- novel adaptation, no prior implementations exist
- Pitfalls: MEDIUM -- identified key risks, but untested in practice

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (stable; paper is static, no library version changes expected)
