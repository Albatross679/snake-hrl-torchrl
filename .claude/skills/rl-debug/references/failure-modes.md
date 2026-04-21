# RL Failure Mode Reference

Detailed diagnostic signatures for each failure mode, organized by algorithm.

## Universal Failure Modes (All Algorithms)

### Gradient Explosion
- **Signature:** `grad_norm` grows exponentially over 10+ steps
- **Metric pattern:** grad_norm: 0.1 -> 1 -> 100 -> 10000 over consecutive batches
- **Root causes:** Learning rate too high, reward scale too large, unstable loss landscape
- **Fix:** Reduce LR, tighten `max_grad_norm`, reduce reward scale
- **SAC-specific:** Actor grad explosion with stable critic = Q-landscape sharpening. Apply `spectral_norm` to critic.

### Gradient Vanishing
- **Signature:** `grad_norm` < 1e-6 for 100+ consecutive steps
- **Metric pattern:** grad_norm flatlines near zero, loss unchanged
- **Root causes:** Dead ReLUs, LR too low, observation scale mismatch
- **Fix:** Check activation functions (use LeakyReLU), increase LR, normalize observations

### NaN Propagation
- **Signature:** Any metric becomes NaN, then all metrics become NaN
- **Metric pattern:** Sudden NaN in one metric, cascading to all within 1-2 steps
- **Root causes:** log(0) in TanhNormal, division by zero in reward, inf in observations
- **Fix:** Add epsilon to log computations, clamp rewards, validate observation bounds

### Entropy Collapse
- **Signature:** `entropy_proxy` -> 0 AND `action_std_min` -> 0
- **Metric pattern:** Entropy drops steadily, action stds collapse to near-zero
- **Root causes:** Entropy coef too low, reward signal too strong, LR too high
- **Fix:** Increase entropy coefficient, reduce reward scale, reduce LR

### Policy Collapse (Action Saturation)
- **Signature:** One or more `action_dim{i}_std` -> 0 while others remain normal
- **Metric pattern:** Specific action dimensions lose variance independently
- **Root causes:** Reward only depends on some action dims, TanhNormal saturation
- **Fix:** Check reward function uses all action dimensions, add per-dim entropy bonus

## PPO-Specific Failure Modes

### Excessive Clipping
- **Signature:** `clip_fraction` > 0.3 sustained
- **Metric pattern:** clip_fraction high + KL divergence high + unstable reward
- **Root causes:** Learning rate too high, too many epochs per batch, batch too small
- **Fix:** Reduce LR, reduce `num_epochs`, increase `frames_per_batch`

### Zero Clipping (Policy Stagnation)
- **Signature:** `clip_fraction` = 0 sustained
- **Metric pattern:** clip_fraction zero + KL near zero + reward flat
- **Root causes:** Learning rate too low, policy network too large (overparameterized)
- **Fix:** Increase LR, reduce network size, verify loss gradients flow

### Value Function Failure
- **Signature:** `explained_variance` < 0 sustained
- **Metric pattern:** explained_variance negative + advantage_std very high
- **Root causes:** Value network too small, reward scale too large, GAE lambda wrong
- **Fix:** Increase value network capacity, normalize rewards, try lambda=0.95

### Stale Experience (KL Drift)
- **Signature:** `kl_divergence` monotonically increasing across training
- **Metric pattern:** KL grows globally (not just within epochs) + clip_fraction very high
- **Root causes:** Bug in rollout buffer clearing, too many epochs over same batch
- **Fix:** Verify buffer clears between collections, reduce num_epochs

### Advantage Explosion
- **Signature:** `advantage_abs_max` > 100
- **Metric pattern:** Very large advantages + unstable policy updates + reward oscillation
- **Root causes:** Reward scale too large, broken normalization, GAE with high lambda + high gamma
- **Fix:** Reduce reward scale to [-3, +3] range, verify advantage normalization

## SAC-Specific Failure Modes

### Actor Gradient Explosion (Q-Sharpening)
- **Signature:** `actor_grad_norm` exponential growth + `critic_grad_norm` stable
- **Metric pattern:** actor_grad_norm: 0.04 -> 14.6B over 20M frames, critic stable
- **Root causes:** Q-landscape becomes increasingly sharp without entropy regularization
- **Fix:** Apply `spectral_norm` to critic, enable auto_alpha, increase alpha
- **Code:** `from torch.nn.utils.parametrizations import spectral_norm; spectral_norm(linear_layer)`

### Q-Value Divergence
- **Signature:** `q1_mean` or `q2_mean` growing unbounded (> 1000)
- **Metric pattern:** Q-values grow steadily, critic_loss increases, reward declines
- **Root causes:** Target network update too fast (tau too high), discount too high, reward unbounded
- **Fix:** Reduce tau (try 0.005 -> 0.001), reduce gamma, clamp rewards

### Twin Q Divergence
- **Signature:** `q_value_spread` > 100
- **Metric pattern:** Gap between q1_mean and q2_mean widens, training becomes unstable
- **Root causes:** Critic networks initialized differently and diverging, LR imbalance
- **Fix:** Re-initialize critics from same weights, verify same LR for both

### Alpha Collapse (SAC Entropy)
- **Signature:** `alpha` -> 0 when `auto_alpha=True`
- **Metric pattern:** alpha drops to near-zero, entropy follows, policy becomes deterministic
- **Root causes:** target_entropy set too low, alpha_lr too high
- **Fix:** Set target_entropy = -action_dim (default), reduce alpha_lr

### Replay Buffer Staleness
- **Signature:** Training loss decreases but reward doesn't improve
- **Metric pattern:** critic_loss declining, reward flat, Q-values don't reflect actual returns
- **Root causes:** Buffer too large relative to training speed, policy changed but buffer has old data
- **Fix:** Reduce buffer_size, increase update_frequency, increase warmup_steps

## MM-RKHS-Specific Failure Modes

### MMD Penalty Domination
- **Signature:** `mmd_penalty` >> `loss_policy` in magnitude
- **Metric pattern:** mmd_penalty large, policy barely updates, reward flat
- **Root causes:** beta too high, mmd_bandwidth too small
- **Fix:** Reduce beta, increase mmd_bandwidth, increase mmd_num_samples

### KL Regularizer Too Strong
- **Signature:** `kl_divergence` stays near zero, policy never changes
- **Metric pattern:** KL near zero + MMD near zero + reward flat (policy frozen)
- **Root causes:** eta too small (1/eta regularizer too strong)
- **Fix:** Increase eta (weakens KL regularizer)

## Diagnostic Metric Quick Reference

| Metric | Healthy Range | Alarm | Algorithm |
|--------|---------------|-------|-----------|
| explained_variance | 0.5 - 1.0 | < 0 | PPO, MM-RKHS |
| entropy_proxy | 0.1 - 5.0 | < 0.01 | All |
| action_std_min | > 0.05 | < 0.01 | All |
| grad_norm | 0.01 - 10 | > 1e4 | PPO, MM-RKHS |
| actor_grad_norm | 0.01 - 10 | > 1e4 or 10x growth in 10 steps | SAC |
| advantage_mean | ~0 (normalized) | abs > 1.0 | PPO, MM-RKHS |
| advantage_abs_max | < 20 | > 100 | PPO, MM-RKHS |
| clip_fraction | 0.05 - 0.2 | > 0.3 or = 0 | PPO |
| kl_divergence | 0.001 - 0.05 | > 0.1 or monotonic growth | PPO, MM-RKHS |
| q1_mean, q2_mean | [-100, 100] | abs > 1000 | SAC |
| q_value_spread | < 10 | > 100 | SAC |
| alpha | 0.01 - 1.0 | < 0.001 or > 10 | SAC |
| mmd_penalty | < loss_policy | >> loss_policy | MM-RKHS |
