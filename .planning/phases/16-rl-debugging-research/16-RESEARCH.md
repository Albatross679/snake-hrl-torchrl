# Phase 16: RL Debugging Research

**Researched:** 2026-03-25
**Domain:** Reinforcement learning diagnostics, training failure detection, systematic debugging
**Confidence:** HIGH

## Summary

RL debugging is fundamentally harder than supervised learning debugging because information flows in loops -- numerical errors in any component propagate through the entire system, causing all metrics to behave erratically simultaneously. The established practitioner methodology (Andy Jones, TorchRL docs, RLExplorer) converges on a consistent approach: (1) verify on progressively complex probe environments, (2) monitor a comprehensive set of diagnostic metrics beyond reward, (3) use specific metric signatures to identify failure modes, and (4) never hand-roll gradient monitoring or metric aggregation when W&B and PyTorch provide it natively.

Our codebase already logs basic metrics (reward, losses, gradient norms, Q-values, timing) to W&B. The gap is **systematic diagnostic interpretation** -- we log data but have no automated alerting, no probe environments, no health-check routines, and no standardized failure-mode lookup. The SAC gradient explosion issue (issues/sac-actor-gradient-explosion.md) was diagnosed ad-hoc; a systematic framework would have caught it earlier via actor_grad_norm growth rate monitoring.

**Primary recommendation:** Build a diagnostic middleware layer that wraps existing trainers, adds missing metrics (entropy, explained variance, action statistics, weight norms, ESS), configures W&B alerts for known failure signatures, and provides probe environment validation -- all using existing PyTorch/W&B APIs without new dependencies.

## Standard Stack

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| wandb | >=0.18.0 | Experiment tracking, metric logging, alerts | Already used; W&B alerts API enables automated failure detection |
| torch | >=2.6.0 | Gradient hooks, spectral norm, parameter monitoring | Native gradient monitoring via `register_hook`, `clip_grad_norm_` |
| torchrl | >=0.7.0 | RL training infrastructure | Already used; provides GAE, loss modules with built-in diagnostics |

### Supporting (No New Dependencies)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.nn.utils.parametrizations.spectral_norm` | PyTorch built-in | Lipschitz-constrain critic networks | SAC Q-sharpening prevention |
| `wandb.alert()` | W&B built-in | Automated training failure alerts | Gradient explosion, reward collapse, entropy collapse |
| `wandb.watch()` | W&B built-in | Gradient and weight histograms | Detailed per-layer gradient monitoring |
| `torch.nn.utils.clip_grad_norm_` | PyTorch built-in | Gradient clipping with norm reporting | Already used in trainers |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom diagnostic library | RLExplorer (arxiv:2410.04322) | Academic tool, not pip-installable, targets Gym/SB3 not TorchRL |
| Custom gradient monitoring | `wandb.watch(model)` | W&B watch logs per-layer gradients automatically but adds overhead |
| Hand-built probe envs | Gymnasium built-in envs (CartPole, Pendulum) | Less targeted but zero implementation cost |

## Architecture Patterns

### Recommended Project Structure
```
src/
├── trainers/
│   ├── diagnostics.py       # Diagnostic middleware (NEW)
│   ├── probe_envs.py        # Probe environments for validation (NEW)
│   ├── health_checks.py     # Automated health check routines (NEW)
│   ├── logging_utils.py     # Existing (extend with new metrics)
│   ├── sac.py               # Existing (add diagnostic hooks)
│   ├── ppo.py               # Existing (add diagnostic hooks)
│   └── otpg.py              # Existing (add diagnostic hooks)
```

### Pattern 1: Diagnostic Callback/Middleware
**What:** A `TrainingDiagnostics` class that wraps trainer update methods to compute and log additional diagnostic metrics without modifying core trainer logic.
**When to use:** Every training run.
**Example:**
```python
# Source: Adapted from Andy Jones RL debugging methodology + W&B API
class TrainingDiagnostics:
    """Non-invasive diagnostic layer for RL trainers."""

    def __init__(self, trainer, wandb_run, config=None):
        self.trainer = trainer
        self.wandb_run = wandb_run
        self.config = config or DiagnosticConfig()
        self._history = {
            "grad_norms": deque(maxlen=100),
            "rewards": deque(maxlen=100),
            "entropy": deque(maxlen=100),
        }

    def log_policy_health(self, actor, obs_batch):
        """Log policy distribution statistics."""
        with torch.no_grad():
            td = TensorDict({"observation": obs_batch}, batch_size=obs_batch.shape[0])
            td = actor(td)
            actions = td["action"]
            log_probs = td["action_log_prob"]

            # Action statistics
            action_mean = actions.mean(dim=0)
            action_std = actions.std(dim=0)

            # Entropy proxy (negative mean log prob)
            entropy = -log_probs.mean().item()

            metrics = {
                "diagnostics/action_mean": action_mean.mean().item(),
                "diagnostics/action_std_mean": action_std.mean().item(),
                "diagnostics/action_std_min": action_std.min().item(),
                "diagnostics/entropy_proxy": entropy,
            }

            # Per-dimension action stats (for detecting collapsed dimensions)
            for i in range(min(actions.shape[-1], 5)):
                metrics[f"diagnostics/action_dim{i}_std"] = action_std[i].item()

            wandb_utils.log_metrics(self.wandb_run, metrics, step=self.trainer.total_frames)

    def check_gradient_health(self, model, model_name):
        """Detect gradient explosion/vanishing."""
        grad_norm = compute_grad_norm(model)
        self._history["grad_norms"].append(grad_norm)

        # Exponential growth detection
        if len(self._history["grad_norms"]) >= 10:
            recent = list(self._history["grad_norms"])[-10:]
            growth_rate = recent[-1] / max(recent[0], 1e-10)
            if growth_rate > 100:
                wandb.alert(
                    title=f"{model_name} gradient explosion",
                    text=f"Grad norm grew {growth_rate:.0f}x in 10 steps: {recent[0]:.4f} -> {recent[-1]:.4f}",
                    level=wandb.AlertLevel.WARN,
                )

        return grad_norm
```

### Pattern 2: Probe Environment Validation
**What:** Progressively complex minimal environments that test specific trainer components in seconds, not hours.
**When to use:** Before any training run on the real environment, and after any trainer code change.
**Example:**
```python
# Source: Andy Jones probe environment methodology
class ProbeEnv1(EnvBase):
    """Single action, zero obs, one timestep, +1 reward.
    Tests: value network learns constant value.
    Expected: value -> 1.0 within ~100 updates.
    """
    pass

class ProbeEnv2(EnvBase):
    """Single action, random obs, obs-dependent reward.
    Tests: backpropagation through value network.
    Expected: value prediction correlates with reward.
    """
    pass

class ProbeEnv4(EnvBase):
    """Two actions, zero obs, action-dependent +/-1 reward.
    Tests: policy gradient and advantage computation.
    Expected: policy converges to always-select-positive-action.
    """
    pass
```

### Pattern 3: W&B Alert-Based Failure Detection
**What:** Programmatic W&B alerts triggered by metric thresholds.
**When to use:** Every production training run.
**Example:**
```python
# Source: W&B alerts API
import wandb

def setup_training_alerts(run):
    """Configure automated failure detection alerts."""
    # These fire when metrics cross thresholds
    # Checked via manual logic in training loop since W&B
    # alerts API is programmatic, not threshold-based
    pass

def check_and_alert(run, metrics, step):
    """Check metrics against known failure signatures."""
    # Gradient explosion
    if metrics.get("actor_grad_norm", 0) > 1e6:
        wandb.alert(
            title="Actor gradient explosion",
            text=f"Step {step}: actor_grad_norm = {metrics['actor_grad_norm']:.2e}",
            level=wandb.AlertLevel.ERROR,
        )

    # Entropy collapse
    if metrics.get("entropy", 1.0) < 0.01:
        wandb.alert(
            title="Entropy collapse",
            text=f"Step {step}: entropy = {metrics['entropy']:.4f}",
            level=wandb.AlertLevel.WARN,
        )

    # Reward collapse (>50% drop from best)
    if metrics.get("reward", 0) < 0.5 * metrics.get("best_reward", 0):
        wandb.alert(
            title="Reward collapse detected",
            text=f"Step {step}: reward={metrics['reward']:.2f}, best={metrics['best_reward']:.2f}",
            level=wandb.AlertLevel.WARN,
        )
```

### Anti-Patterns to Avoid
- **Debugging by hyperparameter sweep:** Tuning hyperparameters before confirming no implementation bugs exist. Always validate with probe environments first.
- **Trusting loss curves alone:** Loss provides only global information. Reward can improve while value function diverges, or vice versa.
- **Adaptive reward scaling as first resort:** Hand-tune reward scales before implementing adaptive schemes.
- **Pixel observations when unnecessary:** Our CPG state is already low-dimensional (124-dim) -- this is not a concern for us.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gradient monitoring per layer | Custom hook registry | `wandb.watch(model, log="gradients")` | Handles all parameter groups, histograms, and cleanup |
| Gradient clipping with reporting | Manual norm computation + clipping | `torch.nn.utils.clip_grad_norm_()` returns the norm | Already used in trainers; returns pre-clip norm for logging |
| Spectral normalization for Q-sharpening | Custom weight normalization | `torch.nn.utils.parametrizations.spectral_norm()` | Handles power iteration, training/eval mode, and gradient flow |
| Metric smoothing / rolling averages | Custom deque-based smoothing | W&B `define_metric()` with smoothing | W&B provides configurable smoothing in UI and API |
| Explained variance computation | Custom implementation | `1 - Var(y_true - y_pred) / Var(y_true)` (one-liner) | Standard formula; do not overthink this |
| Episode statistics aggregation | Custom episode tracker | TorchRL `SyncDataCollector` already tracks episode stats | Built into TorchRL data collection |

**Key insight:** The diagnostic gap in our codebase is not missing libraries but missing metric computation and interpretation logic. Everything we need is available through PyTorch + W&B APIs we already use.

## Common Pitfalls

### Pitfall 1: SAC Actor Gradient Explosion (OBSERVED IN OUR CODEBASE)
**What goes wrong:** Actor gradient norms grow exponentially (0.04 to 14.6B over 20M frames) while critic remains stable. Reward peaks early then collapses.
**Why it happens:** Without entropy regularization, Q-landscape becomes increasingly sharp/peaky. Actor loss `(-Q).mean()` reflects Q-landscape sharpness directly.
**How to avoid:** (a) Monitor actor_grad_norm growth rate, not just absolute value. (b) Alert when 10-step growth exceeds 10x. (c) Consider spectral normalization on critic. (d) Use entropy regularization.
**Warning signs:** Actor grad norm doubling every N steps while critic grad norm stays constant.
**Diagnostic signature:** `actor_grad_norm` exponential growth + `critic_grad_norm` stable + `reward` declining after peak.

### Pitfall 2: PPO Entropy Collapse
**What goes wrong:** Policy converges prematurely to near-deterministic actions. All action dimensions collapse to near-zero standard deviation.
**Why it happens:** Entropy coefficient too low, or reward signal so strong that exploitation overwhelms exploration.
**How to avoid:** Monitor per-dimension action standard deviation. Alert when any dimension std drops below 0.01.
**Warning signs:** Entropy drops to near-zero. Clip fraction drops to zero (no clipping needed because policy barely changes). KL stays very low.
**Diagnostic signature:** `entropy` -> 0 + `clip_fraction` -> 0 + `action_std_min` -> 0.

### Pitfall 3: Value Function Divergence / Overestimation
**What goes wrong:** Predicted values grow unbounded, causing incorrect advantage estimates and unstable policy updates.
**Why it happens:** Bootstrapping errors compound in off-policy learning. Target network drift in SAC.
**How to avoid:** Monitor Q-value magnitudes. Alert when Q-values exceed expected range (reward / (1 - gamma)).
**Warning signs:** Q-values growing steadily. Gap between Q1 and Q2 widening. Value targets outside [-100, 100] range.
**Diagnostic signature:** `q1_mean` or `q2_mean` growing unbounded + `critic_loss` increasing.

### Pitfall 4: Reward Plateau Due to Insufficient Exploration
**What goes wrong:** Agent converges to mediocre behavior and stops improving. Reward plateaus at a suboptimal level.
**Why it happens:** Agent finds a local optimum (e.g., moving toward prey but never coiling). Insufficient exploration after initial policy improvement.
**How to avoid:** Track reward component decomposition (distance reward vs alignment reward). Check entropy remains above minimum threshold.
**Warning signs:** Reward stable but below expected maximum. Entropy still moderate. Action distribution narrows to a subset of action space.
**Diagnostic signature:** `reward` plateau + `action_std` declining but not collapsed + reward components show one component dominant.

### Pitfall 5: Advantage Estimation Errors
**What goes wrong:** PPO/OTPG updates are too large or too small despite correct hyperparameters.
**Why it happens:** Unnormalized advantages, broken GAE computation, or incorrect reward-to-go calculation.
**How to avoid:** Log advantage mean (should be near zero when normalized), advantage std, advantage min/max. Check explained variance of value predictions.
**Warning signs:** Advantages with mean far from zero. Explained variance near zero or negative.
**Diagnostic signature:** `advantage_mean` != 0 + `explained_variance` < 0.

### Pitfall 6: Stale Experience in On-Policy Methods (PPO/OTPG)
**What goes wrong:** KL divergence grows steadily across training, indicating policy has diverged from data-generating policy.
**Why it happens:** Bug in data collection where old rollouts are reused, or too many epochs over the same batch.
**How to avoid:** Verify rollout buffer is cleared between collection phases. Monitor KL per epoch within each update.
**Warning signs:** KL growing across training (not just within epochs). Clip fraction very high.
**Diagnostic signature:** `kl_divergence` trending upward globally + `clip_fraction` > 0.3.

## Code Examples

### Diagnostic Metrics to Add to SAC Trainer
```python
# Source: Practitioner consensus (Andy Jones, TorchRL docs, W&B RL guide)
# Add to SACTrainer._update() return dict:

# Q-value health
metrics["q_value_spread"] = abs(q1.mean().item() - q2.mean().item())
metrics["q_max"] = max(q1.max().item(), q2.max().item())
metrics["q_min"] = min(q1.min().item(), q2.min().item())

# Target value statistics
metrics["target_value_mean"] = target_value.mean().item()
metrics["target_value_std"] = target_value.std().item()
metrics["target_value_max"] = target_value.max().item()

# Action distribution health (when actor updates)
if actor_loss is not None:
    metrics["action_mean"] = new_action.mean().item()
    metrics["action_std"] = new_action.std().item()
    metrics["log_prob_mean"] = log_prob.mean().item()
    metrics["log_prob_std"] = log_prob.std().item()
```

### Diagnostic Metrics to Add to PPO Trainer
```python
# Source: TorchRL PPOLoss + practitioner consensus
# Add to PPOTrainer._update() metrics:

# Explained variance of value predictions
with torch.no_grad():
    y_pred = value_pred.flatten()
    y_true = value_target.flatten()
    var_y = y_true.var()
    explained_var = 1 - (y_true - y_pred).var() / max(var_y, 1e-8)
    metrics["explained_variance"] = explained_var.item()

# Advantage statistics
metrics["advantage_mean"] = advantages.mean().item()
metrics["advantage_std"] = advantages.std().item()
metrics["advantage_max"] = advantages.abs().max().item()

# Ratio statistics (importance sampling)
ratio = torch.exp(new_log_prob - old_log_prob)
metrics["ratio_mean"] = ratio.mean().item()
metrics["ratio_max"] = ratio.max().item()
metrics["ratio_min"] = ratio.min().item()
```

### W&B Alert Setup
```python
# Source: W&B alerts API documentation
import wandb

def setup_diagnostic_alerts(wandb_run, algorithm="ppo"):
    """Register alert checks for known failure modes."""
    # Alerts are triggered programmatically, not via W&B threshold config
    # Call check_alerts() after each training update
    pass

def check_alerts(wandb_run, metrics, step, algorithm="ppo"):
    """Check for failure signatures and fire W&B alerts."""
    alerts = []

    # Universal checks
    grad_key = "actor_grad_norm" if algorithm == "sac" else "grad_norm"
    if metrics.get(grad_key, 0) > 1e4:
        alerts.append(("Gradient explosion", wandb.AlertLevel.ERROR,
                       f"Step {step}: {grad_key}={metrics[grad_key]:.2e}"))

    # NaN detection
    for key, val in metrics.items():
        if isinstance(val, float) and (val != val):  # NaN check
            alerts.append(("NaN in metrics", wandb.AlertLevel.ERROR,
                           f"Step {step}: {key} is NaN"))

    # SAC-specific
    if algorithm == "sac":
        if abs(metrics.get("q1_mean", 0)) > 1000:
            alerts.append(("Q-value divergence", wandb.AlertLevel.WARN,
                           f"Step {step}: q1_mean={metrics['q1_mean']:.1f}"))

    # PPO-specific
    if algorithm in ("ppo", "otpg"):
        if metrics.get("explained_variance", 1) < -0.5:
            alerts.append(("Value function anti-correlated", wandb.AlertLevel.WARN,
                           f"Step {step}: explained_var={metrics['explained_variance']:.3f}"))
        if metrics.get("clip_fraction", 0) > 0.5:
            alerts.append(("Excessive clipping", wandb.AlertLevel.WARN,
                           f"Step {step}: clip_fraction={metrics['clip_fraction']:.3f}"))

    for title, level, text in alerts:
        wandb.alert(title=title, text=text, level=level)
```

### Spectral Normalization for SAC Critic
```python
# Source: PyTorch docs + SNAC (Spectral Normalization Actor-Critic)
from torch.nn.utils.parametrizations import spectral_norm

def apply_spectral_norm_to_critic(critic):
    """Apply spectral normalization to all linear layers in critic.
    Prevents Q-landscape from becoming too sharp, reducing actor gradient explosion.
    """
    for name, module in critic.named_modules():
        if isinstance(module, nn.Linear):
            spectral_norm(module)
    return critic
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Reward curve only | Multi-metric dashboards (entropy, KL, grad norms, Q-values) | ~2020 onward | Catches failures 10x earlier |
| Ad-hoc debugging | Probe environments + systematic checklists | Andy Jones 2021, still current | Reduces debug time from days to hours |
| Manual log inspection | Automated alerts via W&B/custom monitoring | 2023-2025 | Catches issues during overnight runs |
| Fixed gradient clipping | Adaptive + spectral normalization | 2023-2025 (SNAC paper) | Prevents Q-sharpening at source |
| Single reward signal | Component-wise reward tracking | Standard practice | Identifies reward hacking and component imbalance |

**Deprecated/outdated:**
- TensorBoard alone for RL monitoring -- W&B provides better RL-specific features (alerts, custom dashboards, run comparison)
- RLExplorer is an academic paper (2024), not a mature library -- use its diagnostic taxonomy but implement checks natively

## Failure Mode Reference Table

| Failure Mode | Key Diagnostic Metrics | Healthy Range | Alarm Threshold | Algorithm |
|---|---|---|---|---|
| Gradient explosion | actor_grad_norm | 0.01 - 10 | > 1e4 or 10x growth in 10 steps | SAC, PPO, OTPG |
| Gradient vanishing | grad_norm | 0.01 - 10 | < 1e-6 for 100+ steps | PPO, OTPG |
| Entropy collapse | entropy_proxy, action_std_min | 0.1 - 5.0 | entropy < 0.01 or action_std < 0.01 | PPO, OTPG |
| Q-value divergence | q1_mean, q2_mean | [-100, 100] | abs(q) > 1000 | SAC |
| Q-sharpening | q_value_spread, actor_grad_norm | spread < 10 | spread > 100 + grad growth | SAC |
| Reward collapse | reward vs best_reward | monotonically improving | reward < 0.5 * best_reward | All |
| Value function failure | explained_variance | 0.5 - 1.0 | < 0 (anti-correlated) | PPO, OTPG |
| Stale experience | kl_divergence trend | stable or declining | monotonically increasing | PPO, OTPG |
| Policy collapse | action_std per dimension | > 0.05 | any dim < 0.01 | All |
| Advantage explosion | advantage_max | < 20 | > 100 | PPO, OTPG |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0.0 |
| Config file | pyproject.toml `[tool.pytest]` (implicit) |
| Quick run command | `pytest tests/ -x -q --timeout=30` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DIAG-01 | Diagnostic metrics computed correctly | unit | `pytest tests/test_diagnostics.py::test_metric_computation -x` | Wave 0 |
| DIAG-02 | Probe environments pass with PPO/SAC | integration | `pytest tests/test_probe_envs.py -x --timeout=60` | Wave 0 |
| DIAG-03 | W&B alerts fire on failure signatures | unit | `pytest tests/test_diagnostics.py::test_alert_thresholds -x` | Wave 0 |
| DIAG-04 | Health checks detect known failure modes | unit | `pytest tests/test_diagnostics.py::test_health_checks -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_diagnostics.py -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_diagnostics.py` -- covers DIAG-01, DIAG-03, DIAG-04
- [ ] `tests/test_probe_envs.py` -- covers DIAG-02

## Open Questions

1. **W&B alert rate limiting**
   - What we know: W&B alerts have rate limits to prevent spam. `wandb.alert()` has a `wait_duration` parameter (default 5 minutes).
   - What's unclear: Whether the 5-minute default is sufficient for long training runs, or if we need per-alert-type deduplication.
   - Recommendation: Use `wait_duration=300` (5 min) for most alerts, `wait_duration=3600` (1 hr) for persistent issues like gradient growth.

2. **Spectral normalization impact on SAC learning speed**
   - What we know: Spectral norm constrains Lipschitz constant, preventing Q-sharpening. SNAC paper shows improved stability.
   - What's unclear: Whether it significantly slows learning for our 5-action-dim CPG tasks.
   - Recommendation: Implement as optional flag in config. Benchmark on follow_target task with/without.

3. **Probe environment compatibility with TorchRL**
   - What we know: TorchRL uses `EnvBase` with `TensorDict` I/O, different from Gymnasium.
   - What's unclear: Exact boilerplate needed for minimal TorchRL probe environments.
   - Recommendation: Create TorchRL-native probe envs inheriting from `EnvBase` with proper `_step()` and `_reset()`.

## Sources

### Primary (HIGH confidence)
- [TorchRL Debugging RL Guide](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/DEBUGGING_RL.html) - Official TorchRL debugging checklist covering policy monitoring, reward function issues, exploration diagnostics, normalization
- [Andy Jones RL Debugging Guide](https://andyljones.com/posts/rl-debugging.html) - Comprehensive practitioner methodology: probe environments, diagnostic metrics (relative entropy, residual variance, terminal correlation), localization strategies
- [PyTorch clip_grad_norm_ docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) - Returns total norm before clipping
- [PyTorch spectral_norm docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html) - Parametrization-based spectral normalization

### Secondary (MEDIUM confidence)
- [RLExplorer paper (arxiv:2410.04322)](https://arxiv.org/html/2410.04322v1) - Fault taxonomy: action symptoms, agent symptoms, environment symptoms, exploration parameter symptoms, reward symptoms, state symptoms, Q-target symptoms, NN faults (vanishing/exploding gradient, dying ReLU, oscillating loss)
- [W&B RL Observability](https://wandb.ai/wandb_fc/genai-research/reports/Observability-tools-for-reinforcement-learning--VmlldzoxNDE3MzExMw) - W&B features for RL monitoring (alerts, `wandb.watch()`, custom dashboards)
- [9 Off-Policy RL Failures](https://medium.com/@hadiyolworld007/9-off-policy-rl-failures-that-happen-before-you-notice-instability-27b14c329674) - ESS collapse, Q-value divergence, replay buffer issues

### Tertiary (LOW confidence)
- [RL Convergence Debugging Guide (Medium)](https://medium.com/@tesfayzemuygebrekidan/reinforcement-learning-convergence-debugging-guide-73dcd9e56000) - General convergence debugging checklist
- [PPO Instability Troubleshooting](https://apxml.com/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/troubleshooting-ppo-instability) - PPO-specific clip fraction and entropy interpretation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies needed, all tools already in project
- Architecture: HIGH - Middleware pattern is well-established, probe environments are proven methodology
- Pitfalls: HIGH - SAC gradient explosion observed firsthand; other failure modes documented across multiple authoritative sources
- Failure mode signatures: MEDIUM - Threshold values are approximate and need tuning for our specific environment

**Research date:** 2026-03-25
**Valid until:** 2026-06-25 (stable domain, slow-moving)
