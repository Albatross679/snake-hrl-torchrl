---
name: rl-debug
description: >
  Systematic RL training debugger. Four-phase diagnostic process for identifying
  and fixing RL training failures. Use when (1) reward not growing or plateaued,
  (2) gradient explosion or vanishing, (3) entropy collapsed or policy died,
  (4) Q-values diverging in SAC, (5) user says "debug training", "reward stuck",
  "why isn't it learning", (6) validate trainer before real training,
  (7) run probe environments, (8) user mentions "explained variance",
  "advantage stats", "action collapse", or any diagnostic metric,
  (9) pre-flight check before training, (10) reward function debugging,
  reward component analysis, reward SNR, (11) environment verification,
  observation sufficiency, action space issues, (12) task design assessment,
  curriculum necessity, exploration problems.
  Works with PPO, SAC, MM-RKHS, and any TorchRL-based trainer.
---

# RL Training Debugger

Systematic 4-phase diagnostic process for reinforcement learning training failures. Do not tune hyperparameters until bugs are ruled out.

## Phase 1: Probe Environment Validation

Run BEFORE any real training. Catches implementation bugs in seconds.

**Probe envs location:** `src/trainers/probe_envs.py`

| Probe | Tests | Pass Criterion | Failure Means |
|-------|-------|----------------|---------------|
| ProbeEnv1 | Value loss + optimizer | V(s) -> 1.0 in ~100 updates | Broken loss or backprop |
| ProbeEnv2 | Backprop through value net | Value correlates with obs | Broken gradient flow |
| ProbeEnv3 | Reward discounting | V(s0) = gamma * 1.0 | Broken GAE or reward-to-go |
| ProbeEnv4 | Policy gradient + advantage | Always selects positive action | Broken advantage or policy update |
| ProbeEnv5 | Full policy-value interaction | Both networks learn | Stale experience / wrong batching |

If probe N fails but N-1 passes, the bug is in exactly the component N adds.

Run probes:
```python
from src.trainers.probe_envs import ALL_PROBES
# Instantiate each probe, run trainer on it for ~1000 frames, check convergence
```

Run tests:
```bash
python3 -m pytest tests/test_diagnostics.py -x -q
```

## Phase 2: Dashboard Diagnostic Metrics

Start real training. Check these W&B metrics in priority order.

**Diagnostics module:** `src/trainers/diagnostics.py`

### Priority 1: Value function health
- `diagnostics/explained_variance`: should rise from ~0 toward 0.5-1.0
  - Stays at 0 -> value network not learning, check reward scale
  - Negative -> value predictions anti-correlated (broken)

### Priority 2: Policy exploration
- `diagnostics/entropy_proxy`: should start high, decline gradually
- `diagnostics/action_std_min`: must stay above 0.01
  - Entropy -> 0 = policy died. Increase entropy coefficient or reduce reward scale

### Priority 3: Gradient health
- `gradients/grad_norm` (PPO/MM-RKHS) or `gradients/actor_grad_norm` (SAC): should stay 0.01-10
  - Exponential growth = gradient explosion
  - Stuck near 0 = vanishing gradients

### Priority 4: Advantage health (PPO/MM-RKHS)
- `diagnostics/advantage_mean`: should be ~0 if normalized
- `diagnostics/advantage_abs_max`: should be <20, alarm if >100

### Priority 5: Q-value health (SAC)
- `q_values/q1_mean`, `q2_mean`: should stay in [-100, 100]
- `diagnostics/q_value_spread`: alarm if >100

### Priority 6: Reward (check LAST)
- `episode/mean_reward`: lagging indicator, check everything above first

## Phase 3: Reward Not Growing — Decision Tree

Follow this tree top-to-bottom. Stop at the first match.

```
explained_variance < 0?
  -> Value function broken. Check reward scale (target [-3, +3]),
     try smaller network, verify GAE lambda.

entropy collapsed (< 0.01)?
  -> Policy died. Increase entropy coef, reduce LR.

grad_norm exploding (> 1e4)?
  -> Clip gradients tighter. For SAC: add spectral_norm on critic.
     torch.nn.utils.parametrizations.spectral_norm(critic_layer)

advantage_abs_max > 100?
  -> Reward scale too large. Divide rewards by 10.

clip_fraction > 0.3? (PPO)
  -> Policy changing too fast. Reduce LR or increase clip_epsilon.

clip_fraction = 0? (PPO)
  -> Policy not changing. Increase LR.

All diagnostics healthy?
  -> Trainer is working. Problem is reward, environment, or task design.
     Follow the Phase 4 sub-tree below (3 tiers, strict order).

Everything NaN?
  -> Division by zero in reward, log(0) in TanhNormal, or inf in observations.
```

## Phase 4: Reward Not Growing — Environment & Reward Sub-Tree

When all trainer diagnostics are healthy but reward isn't growing. Follow tiers in order — stop at the first match. **Diagnose with baselines before tuning.**

### Tier 1: Reward Function (check first — cheapest to fix)

```
1.1 Log each reward component separately.
    One component >90% of variance? Components cancel out?
      -> Reweight or remove the dominating term.

1.2 Compute reward SNR = abs(mean(reward)) / std(reward) over a batch.
    SNR < 0.1?
      -> Signal drowned in noise. Remove noisy bonus terms, normalize.
    SNR < 0.5?
      -> Marginal. Learning will be slow. Simplify reward.

1.3 Compute discounted returns. Check range.
    Returns outside [-10, +10]?
      -> Scale rewards to bring returns into [-3, +3].
    Return std < 0.01?
      -> Reward provides no differentiation between states. Increase shaping.

1.4 Fraction of steps with |reward| > 0.001?
    < 5%?
      -> Too sparse. Add potential-based reward shaping (PBRS).

1.5 Compare reward trend vs actual task metric (e.g., distance to target).
    Reward up but task metric flat?
      -> Reward hacking. Visualize policy. Randomize initial conditions.
```

**Diagnostic code for Tier 1:**
```python
# 1.1 Component decomposition — log in _compute_reward:
components = {"reward/dist": dist_r, "reward/heading": head_r, "reward/total": total}

# 1.2 Reward SNR:
rewards = collect_rollout_rewards(n_steps=1000)
snr = abs(rewards.mean()) / (rewards.std() + 1e-8)

# 1.3 Return scale:
returns = compute_discounted_returns(rewards, gamma=0.99)
print(f"Return range: [{returns.min():.1f}, {returns.max():.1f}]")

# 1.4 Sparsity:
nonzero_frac = (rewards.abs() > 0.001).float().mean()

# 1.5 Reward-observation correlation (bonus diagnostic):
obs_batch, rewards_batch = collect_obs_and_rewards(n=1000)
for dim in range(obs_dim):
    corr = np.corrcoef(obs_batch[:, dim], rewards_batch)[0, 1]
    if abs(corr) > 0.1:
        print(f"Obs dim {dim}: corr with reward = {corr:.3f}")
# If NO obs dim correlates > 0.1 with reward, the reward is not learnable from observations.
```

### Tier 2: Environment (check second)

```
2.1 Run random policy for 100 episodes. Measure mean return.
    Trained policy return within 1 std of random?
      -> Agent learned nothing. Continue to 2.2-2.6.

2.2 Write a simple heuristic/oracle policy (e.g., proportional controller).
    Oracle also fails?
      -> Environment or physics is broken. Check 2.6.
    Oracle succeeds, trained policy doesn't?
      -> Obs or action space problem. Check 2.3-2.4.

2.3 Observation sufficiency.
    Per-dim std analysis over 1000 steps:
      Dead dims (std < 1e-6)? -> Remove them.
      Scale mismatch (max/min std ratio > 100x)? -> Apply ObservationNorm.
      Missing info (target velocity, current state)? -> Augment obs space.

2.4 Action space verification.
    Sweep each action dim independently from -1 to +1:
      Some dims have no effect? -> Controller mapping broken.
      Extreme actions produce negligible effect? -> action_scale too small.
      Small actions produce huge effect? -> action_scale too large.

2.5 Reset bug detection.
    Reset env twice, compare initial obs:
      Identical when randomization is on? -> RNG not seeded properly.
      State leaking between episodes? -> Fix _reset().

2.6 Physics fidelity.
    Run 100 steps with zero actions:
      Tip drifts? -> Physics instability or wrong parameters.
      Silenced exceptions? -> Check convergence error handling.
```

**Diagnostic code for Tier 2:**
```python
# 2.1 Random baseline:
random_returns = []
for _ in range(100):
    td = env.reset()
    ep_reward = 0
    for _ in range(max_steps):
        action = env.action_spec.rand()
        td = env.step(TensorDict({"action": action}, batch_size=[]))
        ep_reward += td["reward"].item()
        if td["done"].item(): break
    random_returns.append(ep_reward)
print(f"Random: {np.mean(random_returns):.2f} +/- {np.std(random_returns):.2f}")

# 2.3 Observation analysis:
obs_buffer = collect_observations(n_steps=1000)  # (1000, obs_dim)
per_dim_std = obs_buffer.std(axis=0)
dead_dims = (per_dim_std < 1e-6).sum()
print(f"Dead dims: {dead_dims}/{obs_buffer.shape[1]}")
print(f"Scale ratio: {per_dim_std.max() / (per_dim_std[per_dim_std > 1e-6].min() + 1e-8):.1f}x")

# 2.4 Action sweep:
for dim in range(action_dim):
    action = np.zeros(action_dim)
    for val in [-1, 0, 1]:
        action[dim] = val
        td = env.step(TensorDict({"action": torch.tensor(action)}, batch_size=[]))
        print(f"dim={dim} val={val:.0f} -> tip_delta={get_tip_delta(td)}")
```

### Tier 3: Task Design (check last — hardest to fix)

```
3.1 Difficulty analysis.
    Compare: random return vs heuristic return vs trained return.
    Gap closed < 10% after significant training?
      -> Task too hard. Simplify or add curriculum.

3.2 Curriculum necessity.
    Reward flat from step 0? Agent never encounters positive reward?
      -> Start with easy goals (slow/close target), increase difficulty.
      -> Use PBRS from src/rewards/shaping.py.

3.3 Horizon analysis.
    Truncation rate (episodes hitting max_steps)?
      100%? -> Agent never reaches goal. Shorten episodes or add intermediate goals.
      Very short episodes (< 10 steps)? -> Agent dying immediately. Make task more forgiving.

3.4 Exploration sufficiency.
    State visitation near goal region?
      Agent never visits goal region? -> Add waypoint rewards, BC warmstart, or curiosity.
      Agent visits then leaves? -> Reward signal too weak at goal. Strengthen terminal reward.
```

### Quick Symptom Lookup

| Symptom | Start At |
|---------|----------|
| Reward flat from step 0 | 1.4 (sparsity) + 2.1 (random baseline) |
| Reward rises then plateaus | 1.1 (component decomposition) |
| Reward oscillates wildly | 1.2 (SNR) + 2.6 (physics drift) |
| High reward but bad behavior | 1.5 (reward hacking) |
| Trained return ~= random | Full Tier 2 sweep |
| Reward slowly declining | Re-check entropy + 1.5 (reward hacking) |

## W&B Alert System

Automated alerts fire via `wandb.alert()` in `src/trainers/diagnostics.py:check_alerts()`.
Already wired into all three trainers (PPO, SAC, MM-RKHS).

| Alert | Threshold | Level |
|-------|-----------|-------|
| Gradient explosion | grad_norm > 1e4 | ERROR |
| NaN in metrics | any value is NaN | ERROR |
| Entropy collapse | entropy < 0.01 | WARN |
| Action dim collapsed | action_std_min < 0.01 | WARN |
| Q-value divergence | abs(q_mean) > 1000 | WARN |
| Q-value twin divergence | q_value_spread > 100 | WARN |
| Value function anti-correlated | explained_variance < -0.5 | WARN |
| Excessive PPO clipping | clip_fraction > 0.5 | WARN |
| Advantage explosion | advantage_abs_max > 100 | WARN |

## Key Principle

Do NOT tune hyperparameters until implementation bugs are ruled out via probe envs.
Do NOT look at reward curves until all diagnostic metrics are checked.
Reward is a lagging indicator — by the time it drops, something upstream already broke.

## Failure Mode Reference

For detailed failure mode signatures with per-algorithm diagnostic patterns, see [failure-modes.md](references/failure-modes.md).
