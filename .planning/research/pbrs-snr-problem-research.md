# PBRS Per-Step Signal-to-Noise Ratio Problem - Research

**Researched:** 2026-04-02
**Domain:** Potential-based reward shaping noise, value function stability, PPO training dynamics
**Confidence:** HIGH (problem diagnosis) / MEDIUM (specific mitigations) / LOW (some novel combinations)

## Summary

The per-step SNR problem in PBRS is a **real but under-documented issue** in the RL literature. While PBRS has strong theoretical guarantees (policy invariance, optimal policy preservation), the practical implementation with function approximation exposes a structural noise problem: the potential difference F = gamma * Phi(s') - Phi(s) is inherently high-variance step-to-step because consecutive states have similar potentials, making the difference small relative to noise. This is mathematically analogous to numerical differentiation being noisy -- taking the difference of two similar numbers amplifies relative error.

The literature does not name this exact problem as "PBRS SNR," but the symptoms are well-documented across multiple research threads: (1) critic instability under shaped rewards, (2) the need for reward normalization in PPO, (3) variance-aware reward smoothing techniques, and (4) the practical finding that PBRS works best alongside a base reward rather than alone.

**Primary recommendation:** Never use PBRS-only rewards. Always combine PBRS with a non-PBRS base reward (e.g., `exp(-5*dist)`) that provides a smooth, low-noise learning signal. Use reward normalization (running std), higher GAE lambda (0.97-0.99), and gradient clipping (0.5) to further stabilize training. If PBRS noise remains problematic, apply temporal smoothing of potentials via EMA before computing differences.

---

## Research Question 1: Is This a Known/Documented Problem?

**Confidence: HIGH**

### The Problem Is Real But Not Named

The per-step noise in PBRS is a **structural mathematical property** that follows directly from how PBRS works:

1. **PBRS computes differences of similar values.** If Phi(s) = -dist/R and the snake moves a small amount per step, then Phi(s') is very close to Phi(s). The difference gamma*Phi(s') - Phi(s) is small in magnitude but noisy relative to that magnitude.

2. **This is the "numerical differentiation" problem.** Computing derivatives via finite differences (f(x+h) - f(x))/h is notoriously noisy. PBRS is essentially computing a finite difference of the potential function along the trajectory. The step-to-step signal is tiny; the noise from state estimation, discretization, and physics simulation is comparatively large.

3. **Telescoping rescues the episode level.** Over an episode, F sums to gamma^T * Phi(s_T) - Phi(s_0), which has the full dynamic range of the potential. This is why episode-level SNR (3.3 in the observed case) is acceptable while per-step SNR is poor.

### Literature Support

The problem is documented indirectly in several threads:

- **Romoff et al. (2018), "Reward Estimation for Variance Reduction in Deep RL"**: Showed that using learned reward estimators reduces variance in policy gradient updates. Directly addresses the problem that per-step rewards (including shaped rewards) are noisy and that this noise propagates through advantage estimation.

- **Variance Aware Reward Smoothing (Neurocomputing, 2021)**: Proposed smoothing the reward trajectory directly to reduce variance of both rewards and TD-targets, without changing the value function. Targets the exact problem of high per-step reward variance destabilizing training.

- **Hussenot et al. (2502.01307, 2025), "Improving Effectiveness of PBRS"**: Showed that PBRS effectiveness depends critically on the interaction between the potential function, the external reward, and initial Q-values. When the PBRS signal doesn't dominate the base reward properly, convergence fails. This indirectly documents that PBRS alone creates brittle training dynamics.

- **Andrychowicz et al. (2021), "What Matters for On-Policy Deep AC"**: Large-scale study finding that reward normalization (dividing by running std) is critical for PPO stability. This is a generic fix for noisy rewards but directly applicable to PBRS noise.

- **The 37 Implementation Details of PPO (ICLR Blog Track, 2022)**: Documents that continuous-control PPO uses reward scaling (dividing by rolling discounted return std) as a standard practice, with Engstrom et al. finding it "significantly affects performance."

### The Specific Failure Mode Observed

The gradient escalation pattern (1.1 -> 15.5) in PBRS-only training is consistent with this causal chain:

```
Step 1: Per-step PBRS signal is noisy (small delta, large relative noise)
Step 2: Critic tries to fit noisy per-step values -> overfits to noise
Step 3: Overfit critic produces sharp value landscape
Step 4: GAE advantage estimates inherit the sharp landscape -> noisy advantages
Step 5: Noisy advantages -> large policy gradients
Step 6: Large gradients -> larger policy updates -> more extreme value predictions
Step 7: Feedback loop: steps 2-6 reinforce each other
```

This is essentially the **critic overfitting to noise** problem, exacerbated by PBRS's structural noise properties.

---

## Research Question 2: Established Solutions

**Confidence: MEDIUM-HIGH**

### Solution 1: Always Use a Base Reward Alongside PBRS (CRITICAL)

**Confidence: HIGH** -- This is the most important finding.

PBRS was designed as an *additive supplement* to a base reward, not as a standalone reward signal. The original Ng et al. (1999) formulation is:

```
R'(s, a, s') = R(s, a, s') + F(s, a, s')
```

where R is the base reward and F = gamma*Phi(s') - Phi(s) is the shaping reward. The policy invariance guarantee says the shaped MDP has the same optimal policy as the base MDP -- but this only makes sense when there IS a base reward.

**Using PBRS-only rewards removes the base signal that the critic can reliably learn from.** The base reward (e.g., `exp(-5*dist)`) is a smooth function of state that the critic can fit accurately. PBRS adds per-step shaping ON TOP of this learnable signal.

**Implementation for the snake robot:**
```python
# WRONG: PBRS-only (what caused the gradient explosion)
total = pbrs_dist + pbrs_head  # Noisy, no stable base signal

# RIGHT: Base reward + PBRS (what the current code already supports)
total = dist_weight * exp(-5*dist) + heading_weight * heading_reward + pbrs_dist + pbrs_head
```

The existing `compute_follow_target_reward` in `papers/choi2025/rewards.py` already has this architecture -- `dist_weight=1.0` provides `exp(-5*dist)` as a base. The problem only manifests when `dist_weight=0, pbrs_gamma>0`.

### Solution 2: Reward Normalization via Running Standard Deviation

**Confidence: HIGH** -- Standard PPO practice, well-validated.

Divide rewards by a running estimate of the reward standard deviation. This prevents reward scale from drifting as PBRS magnitudes change during training.

```python
# Standard implementation (as in Stable-Baselines3 VecNormalize)
class RunningRewardNormalizer:
    def __init__(self, gamma=0.99, epsilon=1e-8):
        self.ret_rms = RunningMeanStd()  # Tracks discounted return variance
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = 0.0
    
    def normalize(self, reward):
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)
        return reward / (self.ret_rms.std + self.epsilon)
        # NOTE: No mean subtraction -- only scale normalization
```

**Key detail from the 37 PPO Details paper:** PPO uses "discount-based scaling" where rewards are divided by the std of a rolling discounted sum, WITHOUT mean subtraction. This preserves reward sign while normalizing scale.

In TorchRL, this maps to using `RewardScaling` or implementing a custom transform.

### Solution 3: Higher GAE Lambda for PBRS Environments

**Confidence: MEDIUM-HIGH** -- Well-grounded in theory, specific values need tuning.

GAE lambda controls the bias-variance tradeoff in advantage estimation:
- Lambda=0 (TD(0)): High bias, low variance -- uses only 1-step TD error
- Lambda=1 (Monte Carlo): Low bias, high variance -- uses full episode returns

When per-step rewards are noisy (as with PBRS), **higher lambda helps** because it averages over more steps, letting the telescoping property of PBRS do its work. With lambda close to 1, GAE essentially computes n-step returns where n approaches the episode length, and PBRS telescopes cleanly over long horizons.

**Current setting:** `gae_lambda=0.95` (project default)
**Recommended for PBRS:** `gae_lambda=0.97-0.99`

The tradeoff: higher lambda increases variance from stochastic returns but reduces the bias from noisy per-step PBRS signals. For PBRS specifically, the variance increase is partially offset by the telescoping property -- the long-horizon sum of PBRS is actually LESS noisy than the short-horizon sum.

### Solution 4: Gradient Clipping (Already Implemented)

**Confidence: HIGH** -- Standard practice, already in the codebase.

The project already clips gradients at `max_grad_norm=0.5`. This is the standard value from the 37 PPO details. It should prevent the most extreme gradient explosions but won't fix the underlying noise problem -- it's a symptom-level treatment.

**Additional consideration:** Monitor grad norm over training. If it consistently hits the clip threshold, the underlying signal is too noisy. The 1.1->15.5 escalation suggests the clip threshold was either not applied or was set too high.

### Solution 5: Temporal Smoothing of Potentials (EMA)

**Confidence: MEDIUM** -- Theoretically sound, less established for PBRS specifically.

Instead of computing PBRS from raw potentials, apply exponential moving average (EMA) smoothing to the potential before differencing:

```python
class SmoothedPBRS:
    def __init__(self, alpha=0.1, gamma=0.99):
        self.alpha = alpha  # EMA smoothing factor (lower = smoother)
        self.gamma = gamma
        self.smoothed_phi = None
        self.prev_smoothed_phi = None
    
    def compute(self, phi_current):
        if self.smoothed_phi is None:
            self.smoothed_phi = phi_current
        else:
            self.prev_smoothed_phi = self.smoothed_phi
            self.smoothed_phi = self.alpha * phi_current + (1 - self.alpha) * self.smoothed_phi
        
        if self.prev_smoothed_phi is None:
            return 0.0
        return self.gamma * self.smoothed_phi - self.prev_smoothed_phi
```

**WARNING:** This technically breaks the PBRS policy invariance guarantee because the smoothed potential is no longer a pure function of state -- it depends on history. However, in practice with function approximation, the guarantee is already approximate. The tradeoff is: slight theoretical impurity vs. much better training stability.

**DreamSmooth (2023)** validated a similar approach for model-based RL: temporally smoothing rewards before adding them to replay buffers improved reward prediction accuracy and task performance by up to 3x. While DreamSmooth targets MBRL, the principle (smoothing noisy per-step signals) transfers.

### Solution 6: Advantage Normalization (Already Implemented)

**Confidence: HIGH** -- Already in the codebase via `normalize_advantage=True`.

Per-minibatch advantage normalization (subtracting mean, dividing by std) is standard PPO practice and already enabled. This helps by ensuring roughly half the advantages are positive and half negative, regardless of reward scale.

**Note from Andrychowicz et al.:** This was found to "not affect performance much" -- it's a necessary hygiene factor but not a strong fix for the underlying noise problem.

### Solution 7: Larger Minibatches

**Confidence: MEDIUM** -- Standard variance reduction technique.

Larger minibatches provide more stable gradient estimates by averaging over more samples. If the per-step reward is noisy, a larger batch helps the critic fit the mean signal rather than individual noisy samples.

**Practical guidance:** Double or quadruple `frames_per_batch` if compute allows. The tradeoff is wall-clock time per update vs. gradient quality.

---

## Research Question 3: Does Combining PBRS with a Base Reward Help?

**Confidence: HIGH** -- This is the single most important finding.

**Yes, emphatically.** Combining PBRS with a base reward is not just helpful -- it's how PBRS was designed to be used. The evidence:

### Theoretical Argument

PBRS adds F = gamma*Phi(s') - Phi(s) to the base reward R(s,a,s'). At the episode level, F telescopes to gamma^T * Phi(s_T) - Phi(s_0), which is a constant that depends only on initial and terminal states. This means PBRS does not change the total episode reward (up to this constant) -- it REDISTRIBUTES credit across timesteps.

Without a base reward, there IS no credit to redistribute. The entire learning signal comes from noisy per-step potential differences. The critic has nothing smooth to anchor its value estimates to.

### Practical Evidence

- The project's own observation: PBRS-only training showed gradient escalation (1.1 -> 15.5). The same environment with `dist_weight=1.0` providing `exp(-5*dist)` as base reward is expected to be stable.

- Hussenot et al. (2025) showed that PBRS effectiveness depends on the relationship between potential, base reward, and Q-value initialization. Without a base reward, the Q-value initialization has no anchor point.

- Standard robotics RL practice (MuJoCo benchmarks, IsaacGym, etc.) always uses a base reward (typically distance/success/alive) with optional shaping on top.

### The "Anchor" Metaphor

Think of the critic as trying to learn a landscape. The base reward provides the coarse terrain (smooth, learnable). PBRS adds gradient information (which direction is downhill). Without the terrain, the gradient information is just noise floating in space -- the critic has no reference point to build its value estimates from.

### Specific Recommendation for Snake Robot

```python
# Minimal stable configuration:
dist_weight = 1.0      # Base signal: exp(-5*dist), smooth, [0,1]
pbrs_gamma = 0.99      # PBRS: credit redistribution
heading_weight = 0.0   # Optional: add heading base if needed

# The base reward alone gives episode-level signal
# PBRS redistributes that signal across timesteps
# The critic anchors to the base reward, uses PBRS for faster credit assignment
```

---

## Research Question 4: PBRS + PPO Specific Issues

**Confidence: MEDIUM-HIGH**

### Issue 1: GAE Amplifies Per-Step Noise

GAE computes advantages as exponentially-weighted sums of TD errors:
```
A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
```
where `delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`.

With PBRS, `r_t` already contains the noisy potential difference. If the critic also tries to learn the potential (since V*(s) shifts under PBRS), the TD error `delta_t` may have compounded noise from both the reward and value estimate.

**Fix:** Higher lambda (0.97-0.99) lets GAE average over more steps, where PBRS noise partially cancels due to telescoping.

### Issue 2: PPO Clip Range and Noisy Advantages

PPO clips the policy ratio at `1 +/- epsilon`. With noisy advantages, some actions get incorrectly classified as good/bad, leading to:
- Unnecessary clipping (ratio hits boundary when it shouldn't)
- Incorrect policy updates (the sign of the advantage is wrong)

**Fixes:**
- **Larger minibatches:** Reduce advantage estimate variance
- **Smaller clip epsilon:** More conservative updates when signal is noisy (try 0.1 instead of 0.2)
- **More PPO epochs with smaller learning rate:** Trade compute for stability

### Issue 3: Value Function Loss and Noisy Targets

The critic loss is MSE between predicted V(s) and GAE-computed targets. With noisy PBRS rewards, the targets themselves are noisy, and the critic may overfit to this noise.

**Fixes (from literature):**
- **DO NOT use value loss clipping:** Andrychowicz et al. (2021) found it hurts performance. The project has `value_clip_epsilon=0.2` but this should be evaluated.
- **Use L2 regularization on critic:** Small weight decay helps prevent overfitting to noisy targets.
- **Use a larger critic network:** Counter-intuitively, a larger critic with regularization can fit the smooth signal while averaging out noise, vs. a small critic that memorizes noise.

### Issue 4: Advantage Normalization Interaction

When advantages are normalized (zero mean, unit variance), noisy advantages still have unit variance -- the normalization masks the signal quality. A batch with 90% noise and 10% signal looks the same as a batch with 90% signal and 10% noise after normalization.

**This is actually good** in the PBRS case: it prevents the noisy PBRS signal from creating extremely large policy gradients (as observed in the 1.1->15.5 escalation). But it means the effective learning rate is reduced for the true signal.

### PPO-Specific Hyperparameter Recommendations for PBRS

| Parameter | Default | PBRS Recommendation | Rationale |
|-----------|---------|---------------------|-----------|
| `gae_lambda` | 0.95 | 0.97-0.99 | Let telescoping smooth noise |
| `clip_epsilon` | 0.2 | 0.1-0.15 | Conservative updates under noise |
| `max_grad_norm` | 0.5 | 0.5 (keep) | Already appropriate |
| `normalize_advantage` | True | True (keep) | Prevents gradient explosion |
| `frames_per_batch` | current | 2-4x increase | Better gradient estimates |
| `value_coef` | 0.5 | 0.25-0.5 | Lower if critic is overfitting |
| `num_epochs` | current | Reduce by 1-2 | Prevent critic overfitting noise |
| Weight decay | 0.0 | 1e-4 to 1e-3 | Regularize critic |

---

## Research Question 5: Alternative Approaches That Avoid the Problem

**Confidence: MEDIUM**

### Alternative 1: Dense Base Reward Instead of PBRS

The simplest approach: replace PBRS with a dense reward that provides the same directional signal but without the differencing noise.

```python
# PBRS approach (noisy per-step):
pbrs = gamma * (-dist'/R) - (-dist/R)  # Small, noisy difference

# Dense reward approach (smooth per-step):
reward = exp(-5 * dist) + 0.1 * improvement  # Direct distance-based signal
# where improvement = clip(prev_dist - dist, -0.1, 0.1)
```

**Tradeoff:** Dense rewards don't have PBRS's policy invariance guarantee. They can alter the optimal policy. But in practice, for robotics tasks with clear objectives, well-designed dense rewards work reliably.

**Recommendation:** If PBRS noise is problematic, fall back to dense reward shaping. The `exp(-5*dist)` base reward is already dense and informative. PBRS adds theoretical elegance but practical noise.

### Alternative 2: Bootstrapped Reward Shaping (BSRS)

**Confidence: LOW** -- Recent paper (Adamczyk et al., Jan 2025), not yet widely validated.

BSRS dynamically sets the potential to the agent's own value estimate: Phi(s) = V(s). This means the shaping reward is:

```
F = gamma * V(s') - V(s)
```

This is the TD error itself, which vanishes as the value function converges. The advantage: shaping is always calibrated to the agent's current understanding, avoiding the mismatch between a fixed potential and the agent's evolving value landscape.

**Concern:** If the value function is itself noisy (the problem we're trying to solve), bootstrapping from it may create circular instability. Needs empirical validation.

### Alternative 3: k-Step PBRS (Multi-Step Potential Differences)

Instead of computing gamma*Phi(s') - Phi(s) for adjacent states, compute the potential difference over k steps:

```python
# 1-step PBRS (noisy):
F_1 = gamma * Phi(s_{t+1}) - Phi(s_t)

# k-step PBRS (smoother):
F_k = gamma^k * Phi(s_{t+k}) - Phi(s_t)
# Applied as a single reward spread over k steps
```

**Property:** k-step PBRS still telescopes correctly at the episode level. The per-step signal is larger (states k apart have larger potential differences) and thus has better SNR.

**Implementation:** Compute in the GAE loop or as a post-processing step on collected trajectories.

**Confidence: LOW** -- This is a novel combination. Theoretically sound but not established in literature. Needs experimentation.

### Alternative 4: Reward Machines / Temporal Logic Shaping

For tasks with clear phase structure (approach -> contact -> coil), reward machines define an automaton over task progress and give rewards at phase transitions rather than every step:

```
State 0 (far): reward = 0 at each step; transition to State 1 when dist < threshold
State 1 (close): reward = +1 for entering; reward = 0 at each step; transition to State 2 on contact
State 2 (contact): reward = +2 for entering; per-step wrap bonus
```

**Advantage:** Large, clean reward signals at meaningful events. No per-step noise from potential differencing.
**Disadvantage:** Loses the fine-grained per-step guidance that PBRS provides. May slow learning in the "approach" phase where dense guidance is most helpful.

**Recommendation:** Not a direct replacement for PBRS, but consider for the coiling phase where discrete progress milestones are more meaningful than continuous potentials.

---

## Standard Stack

These are the established techniques for handling PBRS noise, ordered by priority of implementation:

| Technique | Type | Impact | Implementation Effort |
|-----------|------|--------|----------------------|
| Base reward + PBRS (not PBRS-only) | Architecture | CRITICAL | Already available in code |
| Reward normalization (running std) | Preprocessing | HIGH | Small (TorchRL transform or wrapper) |
| GAE lambda increase (0.97-0.99) | Hyperparameter | MEDIUM-HIGH | Config change only |
| Gradient clipping at 0.5 | Stability | MEDIUM | Already implemented |
| Advantage normalization | Preprocessing | MEDIUM | Already implemented |
| Smaller clip epsilon (0.1-0.15) | Hyperparameter | MEDIUM | Config change only |
| Larger minibatches | Variance reduction | MEDIUM | Config change (compute cost) |
| Critic weight decay (1e-4) | Regularization | MEDIUM | Config change only |
| EMA-smoothed potentials | Signal processing | MEDIUM | Moderate (custom smoothing code) |

---

## Architecture Patterns

### Pattern 1: Decomposed Reward with Stable Anchor

The primary architectural pattern for mitigating PBRS noise:

```python
def compute_reward(state, prev_state, gamma=0.99):
    # === ANCHOR: Smooth base reward (critic can fit this reliably) ===
    base_reward = exp(-5 * dist)  # [0, 1], smooth, low-noise
    
    # === SHAPING: PBRS for credit redistribution (noisy but policy-invariant) ===
    pbrs = gamma * phi(state) - phi(prev_state)  # Small, noisy
    
    # The critic learns: V(s) ~ base_value + potential_adjustment
    # Base value is smooth and learnable; PBRS noise averages out
    return base_reward + pbrs
```

**Why this works:** The critic first learns the smooth base reward landscape. PBRS then provides gradients that speed up credit assignment. Even if the per-step PBRS signal is noisy, the critic already has a stable foundation from the base reward.

### Pattern 2: Monitoring PBRS Health

Track these metrics during training to detect PBRS noise problems early:

```python
# Per logging interval, compute:
metrics = {
    "reward/base_mean": mean(base_rewards),
    "reward/pbrs_mean": mean(pbrs_rewards),
    "reward/pbrs_std": std(pbrs_rewards),
    "reward/pbrs_snr": abs(mean(pbrs_rewards)) / (std(pbrs_rewards) + 1e-8),
    "reward/total_std": std(total_rewards),
    "training/grad_norm": grad_norm,
    "training/value_loss": critic_loss,
    "training/explained_variance": explained_var,
}

# Alert conditions:
# - pbrs_snr < 0.1: PBRS signal is pure noise
# - grad_norm consistently > 5.0: gradient instability
# - explained_variance < 0.1: critic is not learning
# - value_loss increasing over 5+ updates: critic diverging
```

### Pattern 3: Critic Regularization Under Noisy Rewards

```python
# In PPOTrainer initialization:
self.optimizer = Adam(
    self.loss_module.parameters(),
    lr=self.config.learning_rate,
    weight_decay=1e-4,  # Regularize critic against noise overfitting
)

# Consider separate learning rates for actor/critic:
# Actor: standard lr (3e-4)
# Critic: potentially lower lr (1e-4) to prevent overfitting noisy targets
```

### Anti-Patterns to Avoid

- **PBRS-only rewards.** This is the root cause of the observed problem. Always include a base reward.
- **Aggressive critic training with noisy rewards.** Reducing PPO epochs or the value coefficient when rewards are noisy; don't let the critic overfit.
- **Value function clipping with PBRS.** Andrychowicz et al. showed this hurts; with PBRS noise it's worse because the value targets are themselves noisy.
- **Assuming PBRS is "free."** The policy invariance guarantee is for tabular RL with infinite samples. With function approximation and finite samples, PBRS adds noise that has real costs.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reward normalization | Custom running mean/std | `VecNormalize`-style wrapper or TorchRL `RewardScaling` | Edge cases with parallel envs, reset handling |
| Gradient monitoring | Custom gradient tracking | TorchRL's built-in loss logging + the existing `diagnostics.py` | Already handles NaN detection, clip fraction |
| Advantage estimation | Custom GAE with PBRS-aware modifications | TorchRL's `GAE` with tuned lambda | The standard GAE already handles PBRS correctly via lambda tuning |
| Reward decomposition tracking | Custom bookkeeping | Use `return_components=True` in reward function (already implemented) | Clean separation exists in `compute_follow_target_reward` |
| Value function diagnostics | Custom explained variance | Use existing `compute_explained_variance` from `diagnostics.py` | Already implemented correctly |

---

## Common Pitfalls

### Pitfall 1: Running PBRS Without Base Reward
**What goes wrong:** Gradient escalation, critic instability, training divergence.
**Why it happens:** PBRS potential differences are inherently high-variance step-to-step. Without a smooth base reward, the critic has no stable learning signal.
**How to avoid:** Always set `dist_weight >= 1.0` when `pbrs_gamma > 0`. Never run with `dist_weight=0, pbrs_gamma>0`.
**Warning signs:** Grad norm increasing over training (1.1 -> 15.5 was the observed pattern). Explained variance dropping below 0.1. Value loss increasing.

### Pitfall 2: Normalizing Rewards with Mean Subtraction
**What goes wrong:** Subtracting the running mean from PBRS rewards can shift the sign, creating misleading signals. An agent moving toward the target might receive negative normalized rewards.
**Why it happens:** Standard normalization subtracts mean and divides by std. For PBRS where the mean is near zero, this amplifies noise without adding signal.
**How to avoid:** Only divide by running std (no mean subtraction). This is what the PPO implementation detail recommends.
**Warning signs:** Reward distribution centered at zero after normalization when it shouldn't be.

### Pitfall 3: Low GAE Lambda with PBRS
**What goes wrong:** TD(0)-like advantage estimates pick up all the per-step PBRS noise.
**Why it happens:** Low lambda weights the 1-step TD error heavily. The 1-step TD error includes the full per-step PBRS noise.
**How to avoid:** Use lambda >= 0.97 when PBRS is active. This lets multi-step returns smooth out the noise via telescoping.
**Warning signs:** High variance in advantage estimates. Clip fraction oscillating erratically.

### Pitfall 4: Too Many PPO Epochs with Noisy Rewards  
**What goes wrong:** The critic overfits to noisy targets in the current batch.
**Why it happens:** Multiple passes over the same (noisy) batch let the critic memorize noise rather than learn signal.
**How to avoid:** Use fewer PPO epochs (3-5 instead of 10+) when reward is noisy. Add weight decay to critic.
**Warning signs:** Critic loss decreasing on training batch but explained variance not improving.

### Pitfall 5: Ignoring PBRS Contribution to Total Reward Variance
**What goes wrong:** Reward statistics look reasonable because base + PBRS average out, but the PBRS component is adding pure noise that destabilizes training.
**Why it happens:** Total reward metrics mask per-component problems. 
**How to avoid:** Always log PBRS components separately (the existing `return_components=True` supports this). Monitor PBRS SNR explicitly.
**Warning signs:** `reward_pbrs` std >> `reward_pbrs` abs mean.

---

## Code Examples

### Example 1: Reward Configuration That Avoids the SNR Problem

```python
# In training config / reward setup:
reward_config = {
    "dist_weight": 1.0,       # ALWAYS ON: smooth base signal, exp(-5*dist)
    "heading_weight": 0.5,    # Optional: smooth heading signal, (1+cos)/2
    "smooth_weight": 0.1,     # Optional: action smoothness
    "pbrs_gamma": 0.99,       # PBRS: redistributes credit (supplementary, not primary)
    "workspace_radius": 1.0,  # PBRS normalization
}
# Key: dist_weight > 0 provides the anchor. PBRS adds gradient info on top.
```

### Example 2: Running Reward Normalization (Std Only, No Mean)

```python
import torch

class RewardStdNormalizer:
    """Normalize rewards by running std without mean subtraction.
    
    This is the reward scaling from PPO implementation details:
    divide by std of discounted returns, no mean centering.
    """
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_ms = RunningMeanStd(shape=())  # Tracks var of discounted returns
        self.ret = 0.0  # Running discounted return
    
    def normalize(self, reward: float, done: bool) -> float:
        self.ret = self.ret * self.gamma * (1 - done) + reward
        self.running_ms.update(self.ret)
        return reward / (self.running_ms.std + self.epsilon)


class RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""
    def __init__(self, shape=()):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = float(x)
        batch_var = 0.0
        batch_count = 1
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    @property
    def std(self):
        return self.var ** 0.5
```

### Example 3: PBRS Health Monitoring

```python
def log_pbrs_health(components_list: list[dict], prefix="reward"):
    """Log PBRS signal quality metrics from a batch of reward components."""
    import numpy as np
    
    pbrs_dist = [c["reward_pbrs"] for c in components_list]
    pbrs_head = [c["reward_pbrs_head"] for c in components_list]
    base_dist = [c["reward_dist"] for c in components_list]
    
    metrics = {}
    
    for name, values in [("pbrs_dist", pbrs_dist), ("pbrs_head", pbrs_head), ("base_dist", base_dist)]:
        arr = np.array(values)
        metrics[f"{prefix}/{name}_mean"] = float(np.mean(arr))
        metrics[f"{prefix}/{name}_std"] = float(np.std(arr))
        metrics[f"{prefix}/{name}_snr"] = float(
            abs(np.mean(arr)) / (np.std(arr) + 1e-8)
        )
        metrics[f"{prefix}/{name}_abs_mean"] = float(np.mean(np.abs(arr)))
    
    # Alert: PBRS SNR below threshold
    for name in ["pbrs_dist", "pbrs_head"]:
        snr = metrics[f"{prefix}/{name}_snr"]
        if snr < 0.1:
            print(f"  [WARNING] {name} SNR={snr:.4f} -- PBRS signal is mostly noise")
    
    return metrics
```

### Example 4: EMA-Smoothed Potential (If Needed)

```python
class EMASmoothedPotential:
    """Exponential moving average smoothing of PBRS potential.
    
    Reduces per-step noise at the cost of slight policy invariance violation.
    Use only if base_reward + standard PBRS is still unstable.
    """
    def __init__(self, alpha: float = 0.2, gamma: float = 0.99):
        self.alpha = alpha  # Higher = less smoothing, more responsive
        self.gamma = gamma
        self._ema = None
        self._prev_ema = None
    
    def reset(self):
        self._ema = None
        self._prev_ema = None
    
    def step(self, phi: float) -> float:
        """Compute smoothed PBRS reward from current potential."""
        if self._ema is None:
            self._ema = phi
            return 0.0  # No shaping on first step
        
        self._prev_ema = self._ema
        self._ema = self.alpha * phi + (1.0 - self.alpha) * self._ema
        return self.gamma * self._ema - self._prev_ema
```

---

## Key Papers

| Paper | Year | Relevance | Key Insight |
|-------|------|-----------|-------------|
| Ng, Harada, Russell - "Policy Invariance Under Reward Transformations" | 1999 | Foundational | PBRS preserves optimal policy; designed as supplement to base reward |
| Andrychowicz et al. - "What Matters for On-Policy Deep AC" | 2021 | HIGH | Reward normalization critical; value loss clipping hurts; GAE lambda matters |
| Huang et al. - "37 Implementation Details of PPO" | 2022 | HIGH | Reward scaling (std only, no mean), gradient clipping at 0.5 |
| Romoff et al. - "Reward Estimation for Variance Reduction" | 2018 | MEDIUM | Learned reward estimators reduce variance in policy gradients |
| Hussenot et al. - "Improving Effectiveness of PBRS" | 2025 | MEDIUM | PBRS effectiveness depends on interaction with base reward and Q-init |
| Adamczyk et al. - "Bootstrapped Reward Shaping" | 2025 | LOW | Dynamic potential from value estimates; promising but unvalidated for this use case |
| VAR (Neurocomputing) - "Variance Aware Reward Smoothing" | 2021 | MEDIUM | Direct reward smoothing reduces variance without changing value function |
| DreamSmooth - "Improving MBRL via Reward Smoothing" | 2023 | LOW-MEDIUM | Temporal smoothing of rewards (EMA, Gaussian, uniform); MBRL-specific but principle transfers |

---

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Problem diagnosis (PBRS SNR) | **HIGH** | Mathematical properties are clear; observed symptoms match prediction; multiple indirect literature support |
| Solution: Base reward + PBRS | **HIGH** | This is how PBRS was designed; removing base reward violates the original formulation's intent |
| Solution: Reward normalization | **HIGH** | Standard PPO practice, extensively validated (Andrychowicz 2021, 37 details) |
| Solution: Higher GAE lambda | **MEDIUM-HIGH** | Theoretically sound (telescoping helps at longer horizons); specific lambda values need tuning |
| Solution: Hyperparameter adjustments | **MEDIUM** | Standard techniques, but optimal values are task-specific |
| Solution: EMA smoothing | **MEDIUM** | Theoretically sound, analogous to DreamSmooth; not established for PBRS specifically |
| Solution: k-step PBRS | **LOW** | Novel combination, theoretically sound, no established validation |
| Alternative: Reward machines | **MEDIUM** | Well-established technique but not a direct substitute for PBRS in continuous control |

### What We're Confident About

1. **The problem is real and the diagnosis is correct.** PBRS potential differences are inherently noisy step-to-step. This is structural, not a bug.
2. **Using a base reward alongside PBRS is critical.** The project's existing `dist_weight=1.0` default is correct.
3. **Standard PPO stabilization techniques help.** Reward normalization, gradient clipping, advantage normalization.
4. **Higher GAE lambda is beneficial.** 0.97-0.99 lets telescoping smooth out noise.

### What Needs Experimentation

1. **Exact hyperparameter values.** Lambda 0.97 vs 0.98 vs 0.99, clip epsilon 0.1 vs 0.15.
2. **Whether EMA smoothing is necessary.** May not be needed if base reward + normalization is sufficient.
3. **Optimal base reward weight relative to PBRS magnitude.** Should the base dominate or be roughly equal?
4. **Whether critic weight decay is needed.** Depends on severity of overfitting.

---

## Sources

### Primary (HIGH confidence)
- [Ng et al. 1999 - Policy Invariance Under Reward Transformations](https://www.andrewng.org/publications/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/) - Original PBRS theory
- [37 PPO Implementation Details (ICLR Blog Track 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Reward scaling, gradient clipping, advantage normalization
- [Andrychowicz et al. 2021 - What Matters for On-Policy Deep AC](http://rylanschaeffer.github.io/kernel_papers/andrychowicz_iclr_2021_what_matters_for_on_policy_deep_AC.html) - Reward normalization, value loss clipping hurts, GAE lambda
- [Stable Baselines3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) - Standard hyperparameter ranges

### Secondary (MEDIUM confidence)
- [Hussenot et al. 2025 - Improving PBRS Effectiveness](https://arxiv.org/html/2502.01307v1) - PBRS+base reward interaction, potential bias
- [DreamSmooth 2023](https://vint-1.github.io/dreamsmooth/) - Temporal reward smoothing (EMA, Gaussian, uniform)
- [Romoff et al. 2018 - Reward Estimation for Variance Reduction](https://arxiv.org/abs/1805.03359) - Variance reduction in policy gradients
- [VAR - Variance Aware Reward Smoothing (Neurocomputing 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0925231221009139) - Direct reward smoothing

### Tertiary (LOW confidence)
- [Adamczyk et al. 2025 - Bootstrapped Reward Shaping](https://arxiv.org/html/2501.00989v1) - Dynamic potentials from value estimates
- [SLOPE 2026 - Shaping Landscapes with Optimistic Potential Estimates](https://arxiv.org/html/2602.03201) - Optimistic potential landscapes for MBRL

## Metadata

**Confidence breakdown:**
- Problem diagnosis: HIGH - Mathematical analysis + empirical observation + indirect literature support
- Core solution (base+PBRS): HIGH - This is how PBRS was designed to work
- PPO-specific fixes: MEDIUM-HIGH - Standard techniques, task-specific tuning needed
- Novel approaches (EMA, k-step): LOW-MEDIUM - Theoretically sound, not validated for this case

**Research date:** 2026-04-02
**Valid until:** 2026-07-02 (stable field; core techniques well-established)
