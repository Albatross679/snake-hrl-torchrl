# Reward Design Pitfalls

## 1. Reward Hacking Through Multi-Term Exploitation

**What:** Agent maximizes easy-to-game terms while ignoring hard ones (e.g., oscillating near
target to farm velocity bonus instead of approaching).

**Warning signs:** High total reward but poor task completion. Individual components show
unexpected distributions.

**Fix:** (1) Validate each term correlates with desired behavior in isolation. (2) Cap
individual components. (3) Monitor decomposed rewards separately during training.

## 2. PBRS Guarantee Violation

**What:** Shaped reward changes the optimal policy -- agent learns wrong behavior that looks
good under shaping.

**Warning signs:** Performs well with shaping but poorly without. Shaping reward dominates base.

**Fix:** (1) Only apply conditional gates to base reward, never PBRS terms. (2) Verify shaping
satisfies `F = gamma*Phi(s') - Phi(s)` exactly. (3) Test with and without shaping to confirm
same optimal behavior.

## 3. Catastrophic Forgetting During Reward Transitions

**What:** Agent unlearns base task when auxiliary terms are introduced.

**Warning signs:** Sudden performance drop after phase transition. Base task metrics regressing.

**Fix:** (1) Cosine annealing (not step transitions). (2) Store decomposed rewards in replay
buffer for off-policy recomputation. (3) Keep training on base task alongside new terms.

## 4. Reward Scale Mismatch Between Tasks

**What:** Different sub-tasks have very different reward magnitudes, causing value function to
overfit to one.

**Warning signs:** Value estimates diverging. One sub-task learning much faster than the other.

**Fix:** Normalize all components to similar ranges [0,1] or [-1,1] before weighting. Use
`RewardScaling` transform in TorchRL.

## 5. Discontinuous Reward Landscapes from Hard Gates

**What:** Policy oscillates at gate boundaries. Gradient estimates become noisy.

**Warning signs:** Frequently switching behaviors near boundary. High reward variance in eval.

**Fix:** Smooth sigmoid gates: `gate = sigmoid(k * (threshold - x))`, k=10 default. Sweep
k in [5, 10, 20, 50] if needed.

## 6. Value Function Baseline Bloat

**What:** Critic wastes capacity representing large constant baseline instead of
state-dependent signal.

**Warning signs:** Large absolute value estimates. Slow critic convergence. Poor advantage
estimation despite correct rewards.

**Fix:** Reward centering (subtract running mean from total reward). Or use PBRS which is
naturally centered. At gamma=0.99, baseline is ~100x mean reward.

## 7. Duplicate Shaping Signals

**What:** Multiple terms encode the same information (e.g., improvement bonus `w*(prev_dist - dist)` AND distance PBRS `prev_dist - gamma*dist`), causing overweighting of one objective.

**Warning signs:** One objective dominates despite balanced weights. Removing one term has
little effect on total reward.

**Fix:** Each piece of information appears in exactly one term. Improvement bonus is strictly
dominated by distance PBRS (same signal + policy-invariance guarantee). Remove the duplicate.

## 8. Off-Policy Staleness After Weight Changes

**What:** Changing reward weights invalidates Q-value estimates from old experience in replay
buffer.

**Warning signs:** Q-value estimates suddenly inaccurate after weight transition. Training
instability in SAC/TD3 after curriculum stage change.

**Fix:** Store decomposed `{r_task, r_auxiliary, ...}` in replay buffer. Recompute weighted
sum `sum(w_i * r_i)` at training time with current weights. Only relevant for off-policy
methods (SAC, TD3). PPO discards rollouts each iteration so this is not an issue.
