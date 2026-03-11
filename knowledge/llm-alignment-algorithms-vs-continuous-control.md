---
name: RL Algorithm Selection for Snake Locomotion
description: Comparison of RL algorithms (PPO, GRPO, DPO, DDPG, TD3, SAC, REDQ) for continuous control snake robot locomotion
type: knowledge
created: 2026-03-09
updated: 2026-03-09
tags: [rl, algorithms, ppo, grpo, dpo, ddpg, td3, sac, redq, continuous-control, llm-alignment]
aliases: [grpo-vs-ppo, dpo-vs-ppo, ddpg-vs-ppo, redq-vs-ppo, rl-algorithm-comparison]
---

# RL Algorithm Selection for Snake Locomotion

## Context

When selecting RL algorithms for the `locomotion_elastica` snake robot task, we evaluated algorithms from two families: LLM alignment (GRPO, DPO) and continuous control (DDPG, TD3, SAC, REDQ) against our current PPO baseline. This document captures the analysis.

## Algorithms Evaluated

### GRPO (Group Relative Policy Optimization)

- **Origin:** DeepSeek (DeepSeekMath, arXiv:2402.03300, Feb 2024; DeepSeek-R1, arXiv:2501.12948, Jan 2025)
- **Mechanism:** Eliminates the critic network. For each state, samples G completions (e.g., 64), scores all, normalizes advantages as `A_i = (r_i - mean(r)) / std(r)`. Retains PPO's clipped surrogate but replaces GAE with group-relative rewards. Adds explicit KL penalty against reference policy.
- **Designed for:** LLM reasoning tasks with verifiable outcome rewards (math, code)

### DPO (Direct Preference Optimization)

- **Origin:** Rafailov et al., Stanford, 2023 ("Direct Preference Optimization: Your Language Model is a Reward Model")
- **Mechanism:** Offline method that bypasses reward modeling entirely. Given pairs of trajectories labeled "preferred" vs "rejected," directly optimizes the policy via closed-form reparameterization of the Bradley-Terry preference model. No reward function, no critic, no online environment interaction.
- **Designed for:** LLM alignment from human preference data

### HRPO

- No established algorithm called "HRPO" exists in the continuous control literature.
- **HRPO (Hybrid Latent Reasoning via RL):** NeurIPS 2025, UIUC — LLM-specific technique, uses GRPO internally. Irrelevant to robotics.
- **Hybrid GRPO:** Single-author preprint (Jan 2025) — combines PPO's critic with GRPO's multi-sample evaluation. Untested on robotics, inherits GRPO's multi-sample cost.
- **HTRPO (Hindsight Trust Region Policy Optimization):** IJCAI 2021 — TRPO + hindsight experience replay. Relevant for sparse-reward goal-conditioned tasks, but our reward is already dense.

## Why They Don't Work for Snake Locomotion

### Fundamental assumption mismatch

| LLM alignment setting | Snake robot setting |
|---|---|
| Cheap forward pass (~ms) | Expensive physics sim (~17ms/step, ~57 FPS) |
| Discrete token output | 5-dim continuous actions (amp, freq, wave_num, phase, turn_bias) |
| Outcome-level reward (correct/incorrect) | Dense per-step shaped reward |
| No environment (offline or bandit) | Sequential MDP, 500 steps/episode |
| Preferences from humans | Reward from physics simulation |

### Specific issues

1. **No temporal credit assignment (GRPO, DPO).** Our reward is dense and per-step: `R = c_dist*(prev_dist - curr_dist) + c_align*cos(θ_g) + goal_bonus`. These algorithms have no value function for bootstrapping across 500 timesteps.

2. **Multi-sample overhead is prohibitive (GRPO).** Needs 16–64 rollouts per state. At ~57 FPS with PyElastica, this would reduce effective throughput to ~3.5 FPS — making training infeasible.

3. **Continuous actions don't group well (GRPO).** Designed for discrete completions. With 5-dim continuous actions, variance between random samples is enormous, making group normalization noisy.

4. **DPO is offline and requires preference data.** We already have a well-shaped reward function — DPO would throw it away and require a human oracle to label trajectory pairs instead.

5. **No proven robotics applications.** None of these algorithms have demonstrated results on continuous control or robotics benchmarks.

## Recommendation

**PPO remains the right choice** for `locomotion_elastica`. It natively handles dense rewards via GAE, works with continuous action spaces, and is extensively proven in robotics.

## Continuous Control Algorithms (Viable Alternatives)

### DDPG (Deep Deterministic Policy Gradient)

- **Origin:** Lillicrap et al., DeepMind, 2015
- **Mechanism:** Off-policy actor-critic with deterministic policy. Actor outputs a single action (no distribution), critic estimates Q(s, a). Uses replay buffer for sample reuse, target networks (soft-updated) for stability, and additive noise (OU or Gaussian) for exploration.
- **Status:** Largely **obsolete** — notoriously brittle, sensitive to hyperparameters, suffers from Q-value overestimation. Superseded by TD3.

### TD3 (Twin Delayed DDPG)

- **Origin:** Fujimoto et al., 2018 ("Addressing Function Approximation Error in Actor-Critic Methods")
- **Mechanism:** Fixes DDPG's problems with three techniques: (1) double critics (take minimum to prevent overestimation), (2) delayed policy updates (update actor less frequently than critic), (3) target policy smoothing (add noise to target actions). Off-policy with replay buffer.
- **For our task:** Strong baseline for continuous control. Deterministic policy may explore less effectively than SAC in our 5-dim action space.

### SAC (Soft Actor-Critic)

- **Origin:** Haarnoja et al., UC Berkeley, 2018
- **Mechanism:** Off-policy actor-critic with entropy-regularized objective: maximizes `E[Σ r_t + α H(π(·|s_t))]`. Stochastic policy (Gaussian), double critics, automatic temperature (α) tuning. Replay buffer for sample reuse.
- **For our task:** Best next step beyond PPO. Entropy bonus encourages exploration of the 5-dim serpenoid parameter space. Off-policy data reuse is valuable at 57 FPS. TorchRL has native SAC support.

### REDQ (Randomized Ensembled Double Q-learning)

- **Origin:** Chen et al., UC Berkeley, 2021
- **Mechanism:** Ensemble of N Q-functions (typically N=10). For each target computation, randomly samples M of N critics (typically M=2) and takes the minimum — controls overestimation without being overly pessimistic. Enables high UTD ratio (e.g., 20 gradient updates per environment step). Uses SAC-style entropy-regularized policy.
- **For our task:** Trades GPU compute for sample efficiency. With expensive PyElastica sim, this is a favorable trade — could reach same performance in 10–20x fewer env steps. But TorchRL doesn't have REDQ built-in (requires custom ensemble implementation).

### Off-policy advantage for expensive simulators

The DDPG/TD3/SAC/REDQ family shares a key advantage over PPO: **sample reuse via replay buffer**. PPO discards data after each update. Off-policy methods store transitions and learn from them many times, which matters when each simulation step costs ~17ms.

| Factor | PPO (on-policy) | Off-policy (SAC/TD3/REDQ) |
|--------|-----------------|---------------------------|
| Data reuse | Discarded after update | Replayed many times |
| Gradient updates per step | ~4 epochs | 1–20 (UTD ratio) |
| Env steps to converge | ~2M frames | ~100k–500k |
| Parallelism benefit | High (more data/batch) | Moderate (UTD matters more) |

## Recommendation

**PPO remains the right choice** for `locomotion_elastica`. It natively handles dense rewards via GAE, works with continuous action spaces, and is extensively proven in robotics.

### If moving beyond PPO

| Priority | Algorithm | Effort | Rationale |
|----------|-----------|--------|-----------|
| 1st | **SAC** | Moderate (TorchRL native) | Best sample efficiency / stability tradeoff, stochastic exploration |
| 2nd | **TD3** | Moderate (TorchRL native) | Good alternative if SAC's entropy tuning is problematic |
| 3rd | **REDQ** | High (custom ensemble) | Worth it if sample budget is very tight |
| Skip | **DDPG** | Low | Obsolete — use TD3 instead |
| Skip | **GRPO/DPO** | N/A | Wrong problem class (LLM alignment) |

## Full Comparison Table

| Factor | PPO | SAC | TD3 | REDQ | GRPO | DPO |
|--------|-----|-----|-----|------|------|-----|
| Dense per-step reward | ✅ GAE | ✅ TD | ✅ TD | ✅ TD | ❌ No credit assignment | ❌ No reward function |
| Continuous 5-dim actions | ✅ | ✅ Stochastic | ✅ Deterministic | ✅ Stochastic | ❌ Noisy group stats | ❌ Designed for discrete |
| Sim cost (57 FPS) | 1x | 1x (reuses data) | 1x (reuses data) | 1x (high UTD) | 16–64x (multi-sample) | N/A (offline) |
| Sample efficiency | Moderate | High | High | Very high | Low | N/A |
| Exploration | Entropy coef | Auto-tuned entropy | Additive noise | Auto-tuned entropy | N/A | N/A |
| TorchRL support | ✅ Native | ✅ Native | ✅ Native | ❌ Custom | ❌ No | ❌ No |
| Proven in robotics | ✅ Extensively | ✅ Extensively | ✅ Extensively | ✅ Yes | ❌ No | ❌ No |
| Stability | High (tuned) | High | High | Moderate | N/A | N/A |
