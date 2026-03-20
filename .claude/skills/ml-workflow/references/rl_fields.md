# RL Fields Reference

Field specifications for Reinforcement Learning (Level 1c), classic control algorithms (PPO, SAC, DQN), and LLM alignment algorithms (DPO, GRPO, CISPO). All inherit Base fields and built-in infrastructure.

**Framework note:** Classic control RL (PPO, SAC, DQN) uses **TorchRL** for environments, collectors, and replay buffers. Training loops may use raw PyTorch or Lightning depending on complexity. LLM alignment (DPO, GRPO, CISPO) uses Lightning for training with `WandbLogger` and standard callbacks. VRAM management for all neural RL uses `BatchSizeFinder` where applicable.

## Table of Contents

- [RL Base](#level-1c-reinforcement-learning-rl)
- **Classic Control**
  - [PPO](#classic-control-ppo)
  - [SAC](#classic-control-sac)
  - [DQN](#classic-control-dqn)
  - [Classic RL Metrics](#classic-rl-metrics-contract)
- **LLM Alignment**
  - [DPO](#llm-alignment-dpo)
  - [GRPO](#llm-alignment-grpo)
  - [CISPO](#llm-alignment-cispo)
  - [Algorithm Comparison](#algorithm-comparison)
  - [Reward Design](#reward-design)
  - [RL Metrics](#rl-metrics-contract)
  - [Training Stability](#rl-training-stability)
  - [Auto Batch Size for DPO](#auto-batch-size-for-dpo)
  - [Encoder-Decoder Adaptations](#encoder-decoder-adaptations)

---

## Level 1c: Reinforcement Learning (RL)

For timestep-based RL training.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| total_timesteps | integer | `1,000,000` | Total environment timesteps |
| gamma | float | `0.99` | Discount factor |
| learning_rate | float | `3e-4` | Learning rate |
| num_envs | integer | `1` | Number of parallel environments |
| normalize_obs | boolean | `false` | Normalize observations |
| normalize_reward | boolean | `false` | Normalize rewards |

---

## Classic Control: PPO

Inherits all RL fields. Proximal Policy Optimization.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| frames_per_batch | integer | `2048` | Frames collected per rollout |
| num_epochs | integer | `10` | PPO update epochs per rollout batch |
| mini_batch_size | integer | `64` | Mini-batch size for PPO updates |
| clip_epsilon | float | `0.2` | PPO clipping range |
| gae_lambda | float | `0.95` | Generalized Advantage Estimation lambda |
| normalize_advantage | boolean | `true` | Normalize advantages before update |
| value_coef | float | `0.5` | Value loss coefficient |
| entropy_coef | float | `0.01` | Entropy bonus coefficient |
| max_grad_norm | float | `0.5` | Gradient clipping norm |
| target_kl | float or null | `null` | KL divergence threshold for early stopping; null disables |

---

## Classic Control: SAC (Soft Actor-Critic)

Inherits all RL fields. Off-policy, maximum-entropy RL for continuous action spaces. Learns a stochastic policy that maximizes both expected return and entropy, enabling robust exploration and multi-modal behavior.

**SAC fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| buffer_size | integer | `1,000,000` | Replay buffer capacity |
| batch_size | integer | `256` | Mini-batch size sampled from replay buffer |
| learning_starts | integer | `10,000` | Timesteps of random exploration before training begins |
| tau | float | `0.005` | Soft update coefficient for target networks: `target = tau * current + (1 - tau) * target` |
| actor_lr | float | `3e-4` | Actor (policy) network learning rate |
| critic_lr | float | `3e-4` | Critic (Q-function) network learning rate |
| alpha_lr | float | `3e-4` | Temperature (alpha) learning rate when `auto_alpha=true` |
| alpha | float | `0.2` | Entropy temperature. Higher = more exploration. Ignored when `auto_alpha=true` |
| auto_alpha | boolean | `true` | Automatically tune entropy temperature to match `target_entropy` |
| target_entropy | float or string | `"auto"` | Target entropy for auto-tuning. `"auto"` = `-dim(action_space)` |
| num_critics | integer | `2` | Number of Q-function networks (clipped double-Q reduces overestimation) |
| actor_hidden_dims | list of integers | `[256, 256]` | Actor MLP hidden layer sizes |
| critic_hidden_dims | list of integers | `[256, 256]` | Critic MLP hidden layer sizes |
| train_freq | integer | `1` | Gradient updates per environment step |
| gradient_steps | integer | `1` | Number of gradient steps per `train_freq` trigger |
| target_update_interval | integer | `1` | Steps between target network soft updates |

**SAC networks:**
- **Actor (policy):** Outputs mean and log_std of a squashed Gaussian (tanh-transformed Normal). Actions are sampled via reparameterization trick for differentiable sampling.
- **Critics (Q-functions):** N independent Q-networks (default 2). Target value uses minimum of all critics (clipped double-Q) to reduce overestimation bias.
- **Temperature (alpha):** Learned parameter balancing reward maximization vs entropy. Auto-tuned by default.

**SAC loss functions:**
```
# Critic loss (per critic)
L_Q = E[(Q(s,a) - (r + gamma * (min_j Q_target_j(s', a') - alpha * log_pi(a'|s'))))^2]

# Actor loss
L_pi = E[alpha * log_pi(a|s) - min_j Q_j(s, a)]

# Temperature loss (when auto_alpha=true)
L_alpha = E[-alpha * (log_pi(a|s) + target_entropy)]
```

**When to use SAC:**
- Continuous action spaces (robotics, locomotion, control)
- When sample efficiency matters (off-policy reuse of replay buffer)
- When robust exploration is needed (entropy maximization prevents premature convergence)
- Not suitable for discrete actions (use DQN) or LLM fine-tuning (use DPO/GRPO)

---

## Classic Control: DQN (Deep Q-Network)

Inherits all RL fields. Off-policy, value-based RL for discrete action spaces. Learns a Q-function and acts greedily (with epsilon-greedy exploration).

**DQN fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| buffer_size | integer | `1,000,000` | Replay buffer capacity |
| batch_size | integer | `32` | Mini-batch size sampled from replay buffer |
| learning_starts | integer | `50,000` | Timesteps of random exploration before training begins |
| tau | float | `1.0` | Target network update coefficient. `1.0` = hard copy, `<1.0` = soft (Polyak) |
| target_update_interval | integer | `10,000` | Steps between target network updates (hard copy when `tau=1.0`) |
| epsilon_start | float | `1.0` | Initial exploration rate |
| epsilon_end | float | `0.05` | Final exploration rate |
| epsilon_decay_steps | integer | `100,000` | Linear annealing steps from `epsilon_start` to `epsilon_end` |
| train_freq | integer | `4` | Environment steps between gradient updates |
| gradient_steps | integer | `1` | Gradient steps per training trigger |
| double_dqn | boolean | `true` | Use Double DQN (decouple action selection from evaluation to reduce overestimation) |
| dueling | boolean | `true` | Use Dueling DQN architecture (separate value and advantage streams) |
| prioritized_replay | boolean | `false` | Use Prioritized Experience Replay (sample transitions by TD error) |
| prioritized_replay_alpha | float | `0.6` | PER prioritization exponent (0 = uniform, 1 = full prioritization) |
| prioritized_replay_beta_start | float | `0.4` | PER importance-sampling correction start value |
| prioritized_replay_beta_end | float | `1.0` | PER IS correction annealed to this value |
| n_step_returns | integer | `1` | Multi-step returns (1 = standard TD, 3-5 = faster propagation but more bias) |
| hidden_dims | list of integers | `[64, 64]` | Q-network MLP hidden layer sizes |

**DQN loss:**
```
# Standard DQN
L = E[(Q(s,a) - (r + gamma * max_a' Q_target(s', a')))^2]

# Double DQN (reduces overestimation)
a* = argmax_a' Q(s', a')          # action selected by online network
L = E[(Q(s,a) - (r + gamma * Q_target(s', a*)))^2]  # evaluated by target network
```

**DQN variants hierarchy:**
- **Vanilla DQN:** Basic Q-learning with replay buffer + target network
- **Double DQN:** Decouple action selection (online net) from evaluation (target net)
- **Dueling DQN:** Split Q into V(s) + A(s,a) streams — better value estimation for states where action choice doesn't matter
- **Rainbow:** Combines Double + Dueling + PER + n-step + distributional + noisy nets

**When to use DQN:**
- Discrete action spaces (Atari, board games, routing, scheduling)
- When action space is small-to-medium (<1000 actions)
- Not suitable for continuous actions (use SAC) or high-dimensional discrete (consider PPO)

---

## Classic RL Metrics Contract

Metrics for SAC, DQN, and PPO (environment-based RL). Distinct from LLM alignment metrics.

### Episode Metrics (per episode)

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `episode/reward` | per episode | Total undiscounted episode return |
| `episode/length` | per episode | Episode length in timesteps |
| `episode/reward_mean_100` | per 100 episodes | Rolling mean reward (smoothed learning curve) |
| `episode/success_rate` | per episode | Task-specific binary success indicator |

### Training Metrics (per update)

| Metric Key | Algorithms | Purpose |
|------------|-----------|---------|
| `train/actor_loss` | SAC, PPO | Policy/actor loss |
| `train/critic_loss` | SAC, DQN | Q-function / value loss |
| `train/value_loss` | PPO | Value function loss |
| `train/entropy` | SAC, PPO | Policy entropy (exploration indicator) |
| `train/alpha` | SAC | Current entropy temperature (when auto-tuned) |
| `train/td_error` | DQN | Mean temporal-difference error |
| `train/q_value_mean` | SAC, DQN | Mean Q-value (rising too fast = overestimation) |
| `train/q_value_max` | SAC, DQN | Max Q-value (divergence detector) |

### Exploration Metrics

| Metric Key | Algorithms | Purpose |
|------------|-----------|---------|
| `exploration/epsilon` | DQN | Current epsilon-greedy exploration rate |
| `exploration/buffer_size` | SAC, DQN | Current replay buffer occupancy |
| `exploration/buffer_utilization` | SAC, DQN | `current_size / max_size` — when full, old transitions are overwritten |

---

## LLM Alignment Algorithms

Policy optimization methods for fine-tuning language models. These extend the RL hierarchy for seq2seq/decoder-only models where the "environment" is text generation and the "reward" comes from preference data or execution feedback.

### LLM Alignment: DPO (Direct Preference Optimization)

Offline alignment — no generation at training time. Learns from pre-collected preference pairs (chosen/rejected). Implemented in `part1/dpo_train.py` and `part1/dpo_loss.py`.

**DPO fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| dpo_beta | float | `0.1` | Temperature controlling divergence from reference policy. Lower = more deviation allowed |
| preference_data_path | string | — | Path to JSON file with preference pairs (`{query, chosen_sql, rejected_sql}`) |
| base_checkpoint_path | string | — | Path to base model checkpoint (serves as reference policy) |
| reference_free | boolean | `false` | Skip reference model logprobs (use implicit reference). Saves memory but less stable |
| dpo_loss_type | string | `"sigmoid"` | Loss variant: `"sigmoid"` (standard DPO), `"hinge"`, `"ipo"` (identity preference optimization) |

**DPO preference data generation fields** (for `part1/dpo_data.py`):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| n_candidates | integer | `8` | Number of SQL candidates to generate per query |
| sampling_temperature | float | `1.0` | Temperature for diverse candidate generation |
| top_k | integer | `50` | Top-k filtering during candidate sampling |
| num_beams_candidates | integer | `1` | Beam width for candidate generation (1 = sampling) |

**DPO training notes:**
- Uses single model copy with LoRA: `disable_adapter_layers()` computes reference logprobs, `enable_adapter_layers()` computes policy logprobs. Halves memory vs two separate models.
- Converges fast (5-20 epochs). Use aggressive early stopping (`patience_epochs=5`, `eval_every_n_epochs=1`).
- Low LR critical: `5e-6` for full FT, `1e-5` for LoRA. Higher LR causes reward hacking.

---

### LLM Alignment: GRPO (Group Relative Policy Optimization)

Online alignment — generates completions during training. Uses group-relative advantage estimation (no critic network). Planned for `part1/rl_train.py`.

**GRPO fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| group_size | integer | `8` | Number of completions (G) to generate per query for advantage estimation |
| clip_epsilon | float | `0.2` | PPO-style symmetric clipping range for surrogate objective |
| kl_coef | float | `0.0` | KL penalty coefficient against reference policy. 0 = no KL penalty |
| sampling_temperature | float | `1.0` | Temperature for generating completions |
| top_k | integer or null | `null` | Top-k filtering during completion sampling |
| top_p | float or null | `null` | Nucleus sampling threshold |
| max_completion_length | integer | `256` | Maximum tokens per generated completion |
| num_updates_per_rollout | integer | `1` | Gradient steps per batch of rollouts. >1 = off-policy reuse |
| normalize_advantage | boolean | `true` | Normalize advantages within each group (zero mean, unit variance) |
| reward_baseline | string | `"group_mean"` | Baseline for advantage: `"group_mean"` (GRPO standard), `"running_mean"`, `"none"` |
| reward_scale | float | `1.0` | Multiplicative scaling for raw rewards |
| reward_clip | float or null | `null` | Clamp rewards to [-reward_clip, reward_clip]. Prevents gradient explosion from outlier rewards |
| reference_model_update | string | `"none"` | How to update reference policy: `"none"` (frozen), `"periodic"` (copy every N steps), `"ema"` (exponential moving average) |
| reference_update_interval | integer | `100` | Steps between reference model syncs (when `reference_model_update="periodic"`) |
| ema_decay | float | `0.999` | EMA decay rate (when `reference_model_update="ema"`) |
| min_group_std | float | `0.0` | Minimum reward std within group. Groups below this threshold are skipped (DAPO-style dead group filtering) |

**GRPO loss:**
```
L_GRPO = -min(r_{i,t} * A_i, clip(r_{i,t}, 1-eps, 1+eps) * A_i)
```
Where `r_{i,t} = pi_theta / pi_old` is the per-token importance ratio and `A_i` is the group-relative advantage (same for all tokens in completion i).

**Key difference from PPO:** No critic/value network. Advantage is estimated by normalizing rewards within the group: `A_i = (R_i - mean(R)) / std(R)`.

---

### LLM Alignment: CISPO (Clipped Importance Sampling Policy Optimization)

Online alignment variant of GRPO. Clips the IS weight itself (detached) rather than the surrogate objective, preserving gradient flow for all tokens. From MiniMax-M1 (arXiv 2506.13585).

**CISPO-specific fields** (inherits all GRPO fields, overrides clipping behavior):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| epsilon_high | float | `0.3` | Upper IS weight clipping bound. Small-scale: 0.3, large-scale: 5.0 |
| normalize_by_total_tokens | boolean | `true` | Normalize loss by total completion tokens (not per-sample). More stable for variable-length outputs |

**CISPO loss:**
```
L_CISPO = -(1/N_tokens) * sum_i sum_t  detach(clamp(r_{i,t}, max=1+eps_high)) * A_i * log_pi_theta(o_{i,t})
```

**Key difference from GRPO:** The `detach()` on the clamped weight means gradient always flows through `log pi_theta`. No token is ever masked out. GRPO/PPO's `min()` zeros gradients for high-ratio tokens.

---

## Algorithm Comparison

| Property | PPO | DPO | GRPO | DAPO | CISPO |
|----------|-----|-----|------|------|-------|
| **Training mode** | Online | Offline | Online | Online | Online |
| **Requires generation** | Yes | No | Yes | Yes | Yes |
| **Clipping target** | Surrogate (min) | N/A | Surrogate (min) | Surrogate (min, asymmetric) | IS weight (detach) |
| **Gradient for clipped tokens** | Zero | N/A | Zero | Zero (higher ceiling) | Nonzero (scaled) |
| **Advantage estimation** | Learned critic | Implicit (preference pairs) | Group-relative | Group-relative | Group-relative |
| **Critic network** | Required | Not required | Not required | Not required | Not required |
| **KL constraint** | Optional penalty | Implicit via beta | Optional penalty | None | None |
| **Clipping bounds** | Symmetric | N/A | Symmetric | Asymmetric | Upper-only |
| **Dead group handling** | N/A | N/A | None | Dynamic sampling | None |
| **Memory overhead** | High (critic) | Low (single model w/ LoRA) | Medium (ref model) | Medium (ref model) | Medium (ref model) |
| **Typical epsilon** | 0.2 | N/A | 0.2 | low=0.2, high=0.28 | high=0.3 (small), 5.0 (large) |

---

## Reward Design

### Reward Types for Text-to-SQL

| Reward | Formula | Pros | Cons |
|--------|---------|------|------|
| **Binary execution** | `+1` if records match gold, `-1` otherwise | Simple, well-established (Seq2SQL, 2017) | Sparse signal, no partial credit |
| **F1-based** | `record_f1(pred, gold)` in [0, 1] | Continuous signal, rewards partial correctness | More expensive to compute |
| **Partial credit** | `+1` correct, `+0.5` valid SQL wrong result, `-1` SQL error | Distinguishes syntax from semantics | Requires error classification |
| **Shaped** | `F1 - 0.5 * error_penalty` | Combines correctness with penalty for crashes | Reward engineering fragility |

**For this project:** Start with binary execution reward (`+1`/`-1`). The flight database executes fast enough for online reward computation. F1-based is a good second option if binary proves too sparse.

### Reward Normalization

| Method | When to Use |
|--------|-------------|
| **Group-relative** (GRPO standard) | Default. `A_i = (R_i - mean(R)) / (std(R) + eps)` |
| **Running mean/std** | When groups are small (G < 4) and per-group normalization is noisy |
| **None** | When rewards are already well-scaled (e.g., F1 in [0,1]) |

---

## RL Metrics Contract

### Reward Metrics (per rollout / per epoch)

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `reward/mean` | per rollout | Average reward across completions — is the model learning? |
| `reward/std` | per rollout | Reward spread — too low means lack of diversity |
| `reward/max` | per rollout | Best completion reward |
| `reward/min` | per rollout | Worst completion reward |
| `reward/group_std` | per rollout | Per-group reward std averaged across groups. Near-zero = dead groups (no gradient signal) |
| `success_rate` | per rollout | Fraction of completions with positive reward |

### Policy Metrics (per update step)

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `kl_divergence` | per update | KL(policy ‖ reference). Drift monitor — alarm if > 10-15 |
| `policy_entropy` | per update | Entropy of action distribution. Dropping to near-zero = mode collapse |
| `clip_fraction` | per update | Fraction of tokens/samples that hit the clipping bound. Too high (>0.3) = step size too large |
| `importance_ratio/mean` | per update | Average IS ratio. Should stay near 1.0 |
| `importance_ratio/max` | per update | Max IS ratio. Spikes signal instability |

### Advantage Metrics (per rollout)

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `advantage/mean` | per rollout | Should be near 0 after normalization |
| `advantage/std` | per rollout | Should be near 1 after normalization. High variance = unstable |
| `advantage/max` | per rollout | Outlier detection |

### Generation Diversity (per rollout)

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `unique_completions_per_query` | per rollout | Average distinct completions per query out of G. If = 1, all completions identical (temperature too low or mode collapse) |
| `avg_completion_length` | per rollout | Mean generated sequence length. Sudden changes signal degeneration |

### DPO-Specific Metrics (per epoch)

| Metric Key | Frequency | Purpose |
|------------|-----------|---------|
| `dpo/chosen_reward` | per epoch | Implicit reward for chosen completions. Should increase |
| `dpo/rejected_reward` | per epoch | Implicit reward for rejected completions. Should decrease |
| `dpo/reward_margin` | per epoch | `chosen_reward - rejected_reward`. Should widen, then plateau |
| `dpo/accuracy` | per epoch | Fraction where model assigns higher prob to chosen vs rejected |

---

## RL Training Stability

Known failure modes and their detection/mitigation. Check these when monitoring RL training.

### Reward Hacking

**Symptom:** `reward/mean` increases but eval metric (Record F1) plateaus or drops.

**Cause:** Policy exploits reward function without actually improving. E.g., generating SQL that technically returns correct records for training queries via shortcut patterns that don't generalize.

**Detection:** Track both `reward/mean` and `eval/record_f1` — divergence signals hacking.

**Mitigation:** Add KL penalty (`kl_coef > 0`), use held-out eval set, regularize with reference model.

### Mode Collapse

**Symptom:** `policy_entropy` drops to near-zero, `unique_completions_per_query` = 1.

**Cause:** Policy converges to a single deterministic output for each query, losing ability to explore.

**Detection:** Monitor `policy_entropy` and `unique_completions_per_query` per rollout.

**Mitigation:** Increase `entropy_coef`, raise `sampling_temperature`, lower learning rate. For GRPO/CISPO: increase `group_size` to maintain diversity pressure.

### KL Explosion

**Symptom:** `kl_divergence` > 10-15, generated text becomes incoherent.

**Cause:** Policy drifts too far from reference model. Common when LR is too high or KL penalty is too low.

**Detection:** Monitor `kl_divergence` per update. Set alert threshold at 10.

**Mitigation:** Reduce LR, increase `kl_coef`, sync reference model more frequently (`reference_model_update="periodic"`), or use `target_kl` for early stopping within each rollout.

### Gradient Norm Explosion

**Symptom:** `gradient_norm` spikes 10-100x above baseline, loss becomes NaN.

**Cause:** Large importance ratios amplify gradients. Especially in CISPO where tokens are never masked out.

**Detection:** Monitor `gradient_norm` and `importance_ratio/max`.

**Mitigation:** Reduce `epsilon_high` (CISPO) or `clip_epsilon` (GRPO), reduce LR, tighten `max_grad_norm`.

### Dead Groups

**Symptom:** `reward/group_std` = 0 for many groups, training stalls.

**Cause:** All G completions for a query receive the same reward (all correct or all wrong). Group-relative advantage becomes 0/0 → no gradient.

**Detection:** Monitor `reward/group_std`. If near-zero for >50% of groups, training is getting no signal.

**Mitigation:** Increase `sampling_temperature`, increase `group_size`, use DAPO-style filtering (`min_group_std > 0` to skip dead groups), use mixed reward (F1 instead of binary for more variance).

### Reward Collapse

**Symptom:** `success_rate` stuck at 0% or 100%, `reward/std` = 0.

**Cause:** Task too hard (all fail) or too easy (all succeed) for current model.

**Detection:** Monitor `success_rate` over first few rollouts.

**Mitigation:** If 0%: warm-start from a stronger SFT checkpoint, use easier reward (partial credit). If 100%: increase task difficulty or use continuous reward (F1) for finer signal.

---

## Classic RL Stability (PPO/SAC — From Experience)

Failure modes observed in continuous control RL with TorchRL. These complement the LLM alignment stability section above.

### PPO: bf16 AMP Corrupts Importance Ratios

**Symptom:** Actor loss spikes to 1e11+ intermittently, critic loss stays stable.

**Cause:** bf16 autocast wraps the loss module including `TanhNormal.log_prob()`. bf16's ~3 decimal digits produce garbage log-probs → importance ratios exponentiate to inf.

**Detection:** Intermittent actor loss spikes 10+ orders of magnitude above baseline.

**Mitigation:** Keep loss computation (log-prob, importance ratio, advantage weighting) in f32. Only autocast network forward passes. Set `min_std >= 0.1`.

### PPO: NaN Cascade from Unguarded Loss

**Symptom:** All losses become NaN after step N, preceded by KL divergence spikes.

**Cause:** Extreme actor loss → gradient corruption → NaN in weights → all subsequent computations NaN.

**Detection:** Monitor `train/actor_loss` and `kl_divergence`. KL spikes precede NaN by a few updates.

**Mitigation:** Three-layer NaN guard:
1. `torch.isfinite(loss)` before `loss.backward()` — skip if false
2. `torch.isfinite(grad_norm)` after clipping — skip `optimizer.step()` if false
3. Per-batch KL early stopping at 1.5x `target_kl`

### PPO: Loss Metrics Appear as Zero

**Symptom:** All PPO metrics display 0.0000 despite rewards indicating learning.

**Cause:** Metrics divided by `num_epochs * num_batches` (theoretical max). KL early stopping means actual updates << theoretical max.

**Detection:** All loss metrics near-zero but `episode/reward` is improving.

**Mitigation:** Track `actual_updates` counter incremented only on `optimizer.step()`. Divide metrics by `actual_updates`, not theoretical max.

### PPO: Velocity-Based Reward Always Negative

**Symptom:** Mean reward always negative, PPO cannot learn a useful policy.

**Cause:** Velocity-based reward (v_g) is noisy, tiny magnitude, and negative whenever agent drifts. No positive signal to bootstrap.

**Mitigation:** Replace with potential-based reward: `reward = gamma * phi(s') - phi(s)` where `phi(s) = -distance_to_target`. Guarantees: positive when approaching, negative when retreating, zero-sum over cycles.

### SAC: Critic Divergence from Missing Gradient Clipping

**Symptom:** Critic loss rises from normal to 10K+ over ~100K steps. Q-values swing wildly. Alpha collapses to near-zero.

**Cause:** `max_grad_norm` exists in config but `_update()` never calls `clip_grad_norm_()`. With lr=0.001 and UTD>=4, one large gradient destabilizes the critic.

**Detection:** Monitor critic grad_norm. If it exceeds 1000 while `max_grad_norm` is set to 0.5, clipping is not applied.

**Mitigation:** Verify `nn.utils.clip_grad_norm_()` is actually called for both critic and actor in `_update()`.

### SAC: Vectorized Env Never Auto-Resets

**Symptom:** First batch of episodes OK (correct length), then all episodes length=1 with near-zero reward.

**Cause:** `ParallelEnv.step()` in TorchRL 0.11.x does NOT auto-reset done environments.

**Detection:** Episode length drops from expected (e.g., 200) to 1 after the first reset boundary.

**Mitigation:** Use `env.step_and_maybe_reset()` instead of `env.step()` + `step_mdp()` for vectorized paths. Also ensure `self._device` is `torch.device`, not a string.

---

## Auto Batch Size for DPO

DPO has unique VRAM characteristics that affect batch size tuning.

**Memory profile:**
- DPO has 4 forward passes per step (policy+ref × chosen+rejected), so activation memory is ~4x higher than standard training per sample.
- Full-FT DPO loads 2 complete model copies (policy + frozen reference), consuming significant fixed VRAM before any batch processing.
- LoRA DPO uses a single model with adapter toggling — roughly half the fixed VRAM cost, allowing larger batches.

**Lightning BatchSizeFinder for DPO:**
Use `BatchSizeFinder(mode="binsearch")` as a Trainer callback. The `LightningModule.training_step()` must perform the full DPO forward pass (all 4 passes) so that BatchSizeFinder measures realistic peak memory.

```python
from lightning.pytorch.callbacks import BatchSizeFinder

# Ensure LightningDataModule exposes batch_size attribute
class DPODataModule(L.LightningDataModule):
    def __init__(self, batch_size=8, ...):
        self.batch_size = batch_size  # BatchSizeFinder modifies this

trainer = Trainer(callbacks=[BatchSizeFinder(mode="binsearch")])
```

**Note:** Because DPO's memory profile is ~4x standard training, start with a conservative `batch_size` (e.g., 4-8) so the binary search has room to scale up.

---

## Encoder-Decoder Adaptations

GRPO/CISPO were designed for decoder-only models. Key adaptations for T5 encoder-decoder:

1. **Encoder output caching:** Compute encoder hidden states once per query, reuse across all G completions via `encoder_outputs` argument to `model.generate()`. Reduces compute by ~G× for the encoder pass.

2. **Per-token log probs:** Use `compute_restricted_log_probs()` from `dpo_loss.py` but return per-token (remove `.sum()`). Both GRPO and CISPO need token-level log probs for importance ratios.

3. **Constrained decoding:** T5ForFlightSQL's `prefix_allowed_tokens_fn` ensures all generated SQL uses valid vocabulary. This is compatible with `do_sample=True` for GRPO/CISPO rollout generation.

4. **Reward computation:** SQL execution reward uses the same `compute_metrics()` infrastructure as SL evaluation. Thread pool for parallel SQLite execution across G completions.
