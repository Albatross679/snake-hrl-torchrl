---
name: GRPO and HRPO Algorithm Research
description: Research summary of Group Relative Policy Optimization (GRPO) and HRPO variants, comparing them to PPO for potential use in continuous control / robotics
type: knowledge
created: 2026-03-09
updated: 2026-03-09
tags: [reinforcement-learning, policy-optimization, GRPO, HRPO, PPO, algorithm-comparison]
aliases: [GRPO, HRPO, Group Relative Policy Optimization, Hybrid GRPO]
---

# GRPO and HRPO Algorithm Research

## 1. GRPO (Group Relative Policy Optimization)

### What it is

GRPO is a reinforcement learning algorithm that replaces the learned value function (critic) used in PPO with a **group-based advantage estimate**. Instead of training a separate critic network to estimate baselines, GRPO samples a *group* of completions for each prompt, computes rewards for all of them, and normalizes each completion's reward relative to the group mean and standard deviation.

### Who proposed it

Introduced by **DeepSeek** in the **DeepSeekMath** paper (Feb 2024), then used prominently in **DeepSeek-R1** (Jan 2025) for training reasoning capabilities via pure RL.

- **Key paper**: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300 (2024)
- **Follow-up**: DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv:2501.12948 (2025)

### Core mechanism (how it differs from PPO)

| Aspect | PPO | GRPO |
|--------|-----|------|
| Critic/value function | Yes (separate network, often same size as policy) | **No** -- eliminated entirely |
| Advantage estimation | GAE using learned V(s) | Group-normalized reward: `A_i = (r_i - mean(r)) / std(r)` |
| Samples per input | Typically 1 | **G samples** (e.g., G=64 in DeepSeekMath, G=16 in R1) |
| Memory cost | ~2x model size (policy + critic) | ~1x model size (policy only) |
| KL constraint | Clip ratio (epsilon) | Clip ratio + explicit KL penalty term `beta * KL(pi || pi_ref)` |
| Update style | Multiple epochs on collected batch | Single update per exploration stage (typical) |

The GRPO objective retains the clipped surrogate from PPO but replaces GAE advantages with group-normalized rewards and adds a per-token KL penalty against a reference policy.

### When GRPO excels

- **LLM alignment/reasoning** (its primary domain): eliminates the massive memory cost of a critic network the same size as the policy model
- **Tasks with verifiable rewards** (math, code): reward signal is clean and binary/discrete, group normalization works well
- **Large-scale training**: simpler pipeline, no critic training instability
- **Sample-rich regimes**: the group-based baseline becomes low-variance when G is large

### When GRPO struggles

- **Classic RL / continuous control**: designed for episodic, outcome-based rewards; does not handle dense, per-step rewards naturally (no value function for temporal credit assignment)
- **Sparse reward + no group variance**: if all G samples get the same reward, the normalized advantage is zero -- no learning signal
- **Mixture-of-Experts (MoE) architectures**: per-token importance sampling creates high variance
- **Theoretical guarantees**: less formally analyzed than PPO; convergence properties and failure modes not well-characterized
- **Long-horizon tasks with intermediate rewards**: no mechanism for bootstrapping or temporal difference learning

### Application to continuous control / robotics

GRPO has been **almost exclusively applied to LLM training** as of early 2026. One theoretical paper addresses extending it:

- Patel, "Extending Group Relative Policy Optimization to Continuous Control: A Theoretical Framework for Robotic Reinforcement Learning," arXiv:2507.19555 (2025) -- proposes trajectory-based policy clustering and state-aware advantage estimation, but notes significant computational overhead and complex hyperparameter tuning

**Bottom line for our project**: GRPO is not directly applicable to our snake locomotion task. It lacks temporal credit assignment (no value function, no GAE), requires multiple rollouts per state (expensive in physics simulation), and is designed for discrete token-level actions with outcome-based rewards.

---

## 2. HRPO -- Multiple Meanings

The acronym "HRPO" maps to **two distinct algorithms** in the literature. Neither is well-established in continuous control.

### 2a. HRPO: Hybrid Latent Reasoning via Reinforcement Learning

**What it is**: An RL framework for LLMs that fuses discrete token representations with continuous hidden-state representations through a learnable gating mechanism. Not a general-purpose RL algorithm -- it is specific to LLM latent reasoning.

**Authors**: Zhenrui Yue, Bowen Jin, Huimin Zeng, Honglei Zhuang, Zhen Qin, Jinsung Yoon, Lanyu Shang, Jiawei Han, Dong Wang (UIUC + collaborators)

**Key paper**: "Hybrid Latent Reasoning via Reinforcement Learning," arXiv:2505.18454, NeurIPS 2025

**Core mechanism**: Integrates prior hidden states into sampled tokens via a learnable gate. Initializes with predominantly token embeddings and progressively incorporates more hidden features. Uses GRPO-style optimization but for hybrid (discrete + continuous) reasoning representations. Does **not** apply to robotics or continuous control.

**Relevance to our project**: None. This is an LLM-specific technique for blending latent representations during autoregressive generation.

### 2b. Hybrid GRPO: Hybrid Group Relative Policy Optimization

**What it is**: An RL framework that combines PPO's value function with GRPO's group-based multi-sample evaluation. Retains the critic but augments advantage estimation with empirical multi-sample action evaluation.

**Author**: Soham Sane

**Key paper**: "Hybrid Group Relative Policy Optimization: A Multi-Sample Approach to Enhancing Policy Optimization," arXiv:2502.01652 (Jan 2025)

**Core mechanism**: Unlike pure GRPO (no critic) or pure PPO (single-sample + critic), Hybrid GRPO:
1. Keeps the value function for bootstrapped advantage estimation
2. Samples multiple actions per state and uses group statistics to refine advantages
3. Combines empirical reward signals with learned value baselines

Claims improved convergence speed, stability, and sample efficiency over both PPO and GRPO individually.

**Extensions explored**: entropy-regularized sampling, hierarchical multi-step sub-sampling, adaptive reward normalization, value-based action selection.

**Relevance to our project**: Theoretically more applicable than pure GRPO since it retains the value function. However, the paper is early-stage (single author, tested in simple environments), lists robotics as a "potential application" without demonstrating it, and multi-sample rollouts per state would multiply our already-expensive PyElastica simulation cost.

---

## 3. Summary Comparison

| Feature | PPO | GRPO | Hybrid GRPO | HRPO (Latent) |
|---------|-----|------|-------------|----------------|
| Critic model | Yes | No | Yes | N/A (LLM-specific) |
| Advantage estimation | GAE(lambda) | Group-normalized reward | Group + value hybrid | N/A |
| Samples per state/prompt | 1 | G (16-64) | Multiple | N/A |
| Temporal credit assignment | Yes (via V(s)) | No | Yes | N/A |
| Primary domain | General RL | LLM alignment | General RL (theoretical) | LLM reasoning |
| Continuous control proven | Yes (standard) | No (theoretical only) | No (claimed potential) | No |
| Maturity | Very high | High (in LLM domain) | Low (preprint) | Moderate (NeurIPS) |

## 4. Recommendation for This Project

**Stick with PPO.** For snake robot locomotion with dense per-step rewards, continuous action spaces, and expensive physics simulation:

- PPO's value function provides essential temporal credit assignment for our dense reward shaping
- GRPO's multi-sample requirement would multiply simulation cost by G (impractical with PyElastica at ~57 FPS)
- Neither HRPO variant addresses continuous control robotics
- Hybrid GRPO is interesting conceptually but unproven and would also increase sample requirements

If we want to explore beyond PPO, more relevant algorithms for our domain would be SAC (Soft Actor-Critic), TD3, or REDQ -- all designed for continuous control with strong sample efficiency.
