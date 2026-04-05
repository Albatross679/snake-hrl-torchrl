---
name: rl-reward-design
description: >
  Design and audit reward functions for reinforcement learning (continuous control, PPO, SAC).
  Covers: normalized multi-component reward architecture, potential-based reward shaping (PBRS),
  reward curriculum/scheduling, conditional gating, and common pitfalls.
  Use when: (1) designing a new reward function for an RL task, (2) debugging reward-related
  training issues (reward hacking, scale mismatch, value function instability), (3) adding
  reward shaping or auxiliary reward terms, (4) reviewing existing reward design for correctness,
  (5) implementing reward curriculum or scheduled weight transitions. Triggers on mentions of
  "reward design", "reward shaping", "PBRS", "reward curriculum", "reward normalization",
  "reward hacking", or "potential-based".
---

# RL Reward Design

## Core Architecture

Decompose total reward into normalized base components + raw PBRS shaping:

```
total = Sum( w_i * normalize(r_i) )        <- base: normalized then weighted
      + Sum( gamma*Phi_j(s') - Phi_j(s) )  <- PBRS: raw, unweighted
```

**Base components:** Normalize to [0,1] (objectives) or [-1,0] (penalties) before weighting.
Weights become pure importance coefficients. `w=0.3` means 30% of signal.

**PBRS components:** Never normalize or weight. They guide without changing the optimal policy.
Naturally centered near zero -- no value function baseline bloat.

## Normalization Rules

| Component type | Raw range | Normalize to | Method |
|---------------|-----------|-------------|--------|
| Exponential distance | [0, 1] | [0, 1] | Already normalized |
| Cosine similarity | [-1, 1] | [0, 1] | `(1 + cos_sim) / 2` |
| Action penalty | [-action_dim, 0] | [-1, 0] | `/ action_dim` or `/ (2*action_dim)` |
| Penetration/contact | [0, max_pen] | [-1, 0] | `-pen / max_pen` |
| Improvement (delta) | unbounded | [-1, 1] | `clip(delta, -c, c) / c` |

Static normalization preferred over running stats -- preserves PBRS guarantees, avoids
early-training noise.

## PBRS Design

Each objective gets its own potential function. Sum of valid PBRS terms is also valid PBRS.

```python
# Distance potential: Phi(s) = -dist / workspace_radius  (bounded [-1, 0])
f_dist = gamma * (-dist / R) - (-prev_dist / R)

# Heading potential: Phi(s) = cos_sim(tangent, to_target)  (bounded [-1, 1])
f_head = gamma * cos_sim - prev_cos_sim

# Smoothness potential: Phi(s) = -||a||^2 / action_dim  (bounded [-1, 0])
f_smooth = gamma * (-sum(a**2) / dim) - (-sum(prev_a**2) / dim)
```

**Critical rules:**
- Never normalize PBRS terms (breaks F = gamma*Phi(s') - Phi(s) form)
- Never weight PBRS terms (unnecessary -- they only guide, not control importance)
- Never apply conditional gates to PBRS (breaks policy-invariance guarantee)
- Apply gates to base reward components only
- Sum of valid PBRS terms is valid PBRS -- add per-objective potentials freely

## Reward Sign Convention

Sign does not affect training stability -- advantage normalization absorbs it (Liu 2023).

**Industry standard (MuJoCo, IsaacLab, DMControl):** Mixed positive + negative.
- Positive: task objectives (distance, contact, tracking)
- Negative: regularization (energy, jitter, penetration)

**What actually matters:** Reward centering (Naik et al. 2024). At gamma=0.99, value function
represents ~100x mean reward as constant baseline. Center rewards to keep critic on
state-dependent signal. PBRS is naturally centered.

## Conditional Gating

Gate base reward terms with smooth sigmoids, not hard if/else:

```python
gate = 1 / (1 + exp(-k * (threshold - state_value)))  # k=10 default
gated_reward = gate * reward_component
```

Use for: activating coil rewards only near prey, penalizing energy only after basic behavior
learned. Use hysteresis (different thresholds for activation vs deactivation) to prevent
reward flickering at boundaries.

## Reward Curriculum

Two-stage pattern (Fournier et al. 2024):

1. **Phase 1 (task mastery):** Base task reward only. Transition when `success_rate > 0.8`.
2. **Phase 2 (refinement):** Introduce auxiliary terms with cosine annealing:
   `w = w_target * (1 - cos(progress * pi)) / 2`

Off-policy (SAC): Store decomposed components in replay buffer, recompute weighted sum at
train time. On-policy (PPO): Weight change between rollouts is fine.

## Pitfalls & Audit Checklist

For detailed pitfall analysis, read [references/pitfalls.md](references/pitfalls.md).

Quick audit:
- [ ] All base components normalized to documented ranges before weighting?
- [ ] PBRS terms satisfy F = gamma*Phi(s') - Phi(s) exactly?
- [ ] No conditional gates on PBRS terms?
- [ ] Advantage normalization enabled (PPO)?
- [ ] Decomposed reward components logged separately?
- [ ] No duplicate shaping (e.g., improvement bonus AND distance PBRS)?
- [ ] Reward scales similar across sub-tasks in HRL?

## TorchRL Integration

For TorchRL-specific transform patterns and code, read
[references/torchrl-patterns.md](references/torchrl-patterns.md).

## Key References

| Paper | Year | Contribution |
|-------|------|-------------|
| Ng et al. | 1999 | PBRS preserves optimal policy |
| Devlin & Kudenko | 2012 | Dynamic PBRS (time-varying potentials) |
| Camacho et al. (Frontiers) | 2024 | Hierarchical PBRS with priority tiers |
| Fournier et al. | 2024 | Two-stage reward curriculum for robotics |
| Naik et al. (RLC) | 2024 | Reward centering for value function stability |
| Liu | 2023 | Advantage normalization absorbs reward scale |
| Huang et al. (ICLR) | 2022 | 37 PPO implementation details |
