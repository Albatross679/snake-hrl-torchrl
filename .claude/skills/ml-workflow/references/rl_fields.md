# RL Fields Reference

Field specifications for Reinforcement Learning (Level 1c) and PPO (Level 2c). Both inherit all Base fields and built-in infrastructure (output, console, checkpointing, metricslog, experiment tracking).

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

## Level 2c: PPO

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
