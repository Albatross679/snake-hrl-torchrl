# Reinforcement Learning Data Validation

## Table of Contents
1. [Reward Signal Validation](#reward-signal-validation)
2. [Environment and State Validation](#environment-and-state-validation)
3. [Trajectory Data Quality (Offline RL)](#trajectory-data-quality-offline-rl)
4. [Replay Buffer Validation](#replay-buffer-validation)

---

## Reward Signal Validation

Flawed reward functions lead to reward hacking — the agent maximizes a proxy reward while failing the true objective.

**Reward problem categories (Pan et al. 2022):**

| Category | Description | Example |
|----------|-------------|---------|
| Misweighting | Goals valued differently than intended | Agent prioritizes easy sub-goals |
| Ontological | Reward captures different concept | Score doesn't reflect task quality |
| Scope | Measurement limited to narrow domain | Only measures local performance |
| Reward tampering | Agent modifies reward mechanism | Manipulates sensors or state |

**Validation approaches:**
1. **Bounded rewards:** Keep in [0,1] or [-1,1] to prevent exploitation
2. **Distribution monitoring:** Track mean, variance, skewness; spikes/collapse = hacking
3. **Multi-signal validation:** Use multiple independent reward signals; divergence = proxy gaming
4. **Verifiable rewards (RLVR):** Use binary ground-truth (correct/incorrect) over learned reward models
5. **Anomaly detection:** Isolation Forests or IQR on per-episode reward statistics
6. **Correlation decay:** Track proxy reward vs true objective; decay = hacking onset

**Practical checks:**
```python
import numpy as np
from scipy import stats

def validate_reward_signal(rewards, episode_returns):
    """Basic reward signal validation."""
    checks = {
        "bounded": all(-10 <= r <= 10 for r in rewards),
        "no_nans": not any(np.isnan(r) for r in rewards),
        "variance_nonzero": np.var(rewards) > 1e-8,
        "no_reward_explosion": max(abs(r) for r in rewards) < 1000,
        "return_distribution_normal": stats.normaltest(episode_returns).pvalue > 0.01,
    }
    return checks
```

---

## Environment and State Validation

Gymnasium `check_env` is the standard tool.

**Built-in validation:**
```python
from gymnasium.utils.env_checker import check_env

env = CustomEnv()
check_env(env)  # Validates obs_space, action_space, step(), reset()
# WARNING: Do not reuse env instance after check_env
```

**What `check_env` validates:**
- Observation/action spaces properly defined
- `reset()` returns valid observation within observation_space
- `step()` returns (observation, reward, terminated, truncated, info)
- Observations from step() within observation_space
- Rewards are scalar floats
- `render()` works with specified render_mode

**Additional manual checks:**

| Check | What to Verify | Why |
|-------|---------------|-----|
| Determinism | Same seed = same trajectory | Reproducibility |
| State bounds | Observations within declared space | Prevents NaN propagation |
| Reward consistency | Same state-action = consistent reward | Prevents instability |
| Episode termination | Episodes always terminate | Prevents training hangs |
| State transition validity | Next state reachable from current | Physical/logical consistency |
| Action masking | Invalid actions handled | Prevents undefined behavior |

**Full environment validation:**
```python
import gymnasium as gym
import numpy as np

env = gym.make("CustomEnv-v0")
obs, info = env.reset(seed=42)

assert env.observation_space.contains(obs), "Reset obs out of bounds"

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs), f"Step obs out of bounds: {obs}"
    assert isinstance(reward, (int, float)), f"Reward not scalar: {type(reward)}"
    assert not np.isnan(reward), "NaN reward"
    assert not np.isinf(reward), "Inf reward"
    if terminated or truncated:
        obs, info = env.reset()
```

---

## Trajectory Data Quality (Offline RL)

For offline RL, trajectory quality directly determines learning success.

**Quality dimensions:**

| Dimension | What It Measures | Impact |
|-----------|-----------------|--------|
| State-action coverage | State-action space representation | Low = poor generalization |
| Behavioral policy quality | How good was collecting policy | Garbage data = garbage policy |
| On-policyness | Closeness to current learning policy | Stale data degrades learning |
| Trajectory coherence | Temporal consistency of transitions | Corrupted transitions = noise |
| Return distribution | Episode returns in dataset | Skewed = biased learning |

**Validation checks:**
```python
import numpy as np

def validate_offline_dataset(dataset):
    """Validate offline RL dataset quality."""
    checks = {}

    # Coverage: fraction of state space represented
    state_coverage = estimate_state_coverage(dataset.observations)
    checks["state_coverage"] = state_coverage > 0.3  # At least 30%

    # Return distribution: should have variance
    returns = compute_episode_returns(dataset)
    checks["return_variance"] = np.var(returns) > 0
    checks["return_range"] = np.ptp(returns) > 0  # Non-degenerate

    # Transition consistency: s' from step t == s from step t+1
    for episode in dataset.episodes:
        for t in range(len(episode) - 1):
            assert np.allclose(episode[t].next_obs, episode[t + 1].obs)

    # Action diversity: should not be single-action
    action_entropy = compute_action_entropy(dataset.actions)
    checks["action_diversity"] = action_entropy > 0.5

    return checks
```

**Adaptive Replay Buffer (ARB):** Dynamically prioritize sampling based on on-policyness — how closely stored trajectories align with current policy. Assign proportional sampling weights.

---

## Replay Buffer Validation

| Issue | What Goes Wrong | Validation |
|-------|----------------|------------|
| Stale transitions | Old transitions dominate | Track insertion timestamps, monitor age distribution |
| Priority bias | Prioritized replay over-samples specific transitions | Monitor sampling vs uniform distribution |
| Capacity overflow | Important transitions evicted | Track value of evicted transitions |
| Corrupted transitions | NaN/Inf from numerical instability | Periodic buffer-wide NaN/Inf scan |
| Distribution shift | Buffer diverges from current policy | Track KL divergence between buffer and recent data |
