# Reward Design Paradigms for HRL Snake Robot - Research

**Researched:** 2026-04-02
**Domain:** Reward function design for multi-objective hierarchical reinforcement learning
**Confidence:** HIGH (established RL techniques) / MEDIUM (application to this specific project)

## Summary

This research investigates three reward design paradigms relevant to the snake robot HRL project: (1) conditional/state-dependent reward activation, (2) evolving/scheduled reward weights, and (3) learned/discovered reward functions. All three are established concepts in the RL literature with well-defined terminology and known tradeoffs, though they sit on a spectrum from practical-and-proven to research-frontier.

**Conditional reward design** is the most mature and directly applicable. It maps naturally to the project's existing PBRS architecture and has formal theoretical backing (HPRS, DPBRS). **Reward weight scheduling** is well-studied with recent practical robotics papers providing concrete algorithms (two-stage reward curriculum). **Learned reward design** is the most ambitious and least practical for this project -- it requires significant infrastructure overhead and is better suited to cases where the reward function itself is unknown, which is not the case here.

**Primary recommendation:** Use conditional reward activation via hierarchical PBRS (already partially implemented in the codebase) and two-stage reward curriculum for weight scheduling. Do not invest in learned reward design at this stage -- the project's reward structure is well-understood enough to design by hand.

## Project Context

### Current Reward Architecture

The project already implements a solid reward shaping foundation:

- **PotentialBasedRewardShaping** (PBRS): `gamma * Phi(s') - Phi(s)` preserving optimal policies
- **ApproachPotential**: Distance + velocity-toward-prey potential
- **CoilPotential**: Contact fraction + wrap count + constriction potential
- **CompositeRewardShaping**: Multiple PBRS components with named weights
- **GaitPotential / CurriculumGaitPotential**: Gaussian potential with sigma annealing (curriculum)
- **AdaptiveGaitPotential**: Density-adaptive sigma based on demonstration coverage

The config system (`src/configs/env.py`) defines per-task reward weights:
- Approach: `energy_penalty_weight`, `success_bonus`, `distance_reward_weight`, `velocity_reward_weight`
- Coil: `stability_reward_weight`, `energy_penalty_weight`, `success_bonus`, `contact_reward_weight`, `wrap_reward_weight`, `constriction_reward_weight`
- HRL meta: `use_intrinsic_reward`, `intrinsic_reward_scale`, `use_curriculum`, `curriculum_stages`

### What the Project Needs

The snake robot predation task has a natural structure where:
1. During **approach**, coiling rewards are irrelevant (snake is far from prey)
2. During **coiling**, approach rewards can interfere (snake should stay close, not keep approaching)
3. **Energy efficiency** matters more once basic behavior is learned
4. **Gait quality** should be shaped loosely at first, then tightened

This maps directly to conditional activation (Topic 1) and reward scheduling (Topic 2).

---

## Topic 1: Conditional Reward Design (State-Dependent Reward Activation)

### Established Terminology

| Term | Definition | Key Reference |
|------|-----------|---------------|
| **Potential-Based Reward Shaping (PBRS)** | F(s,a,s') = gamma*Phi(s') - Phi(s); preserves optimal policies | Ng et al. 1999 |
| **Dynamic PBRS (DPBRS)** | Potential function changes over time/state; Phi(s,t) | Devlin & Kudenko 2012 |
| **Hierarchical PBRS (HPRS)** | Compositional potentials with strict priority ordering: safety > target > comfort. Higher-tier rewards only activate when lower-tier conditions are met | Camacho et al. 2024 (Frontiers) |
| **Conditional reward terms** | Reward components that activate/deactivate based on state predicates | Common practice, no single paper |
| **Gated rewards** | Rewards multiplied by indicator functions based on state conditions | Common practice |
| **Reward-Conditioned RL (RCRL)** | Policy conditioned on reward parameterization; single policy handles multiple reward specs | Kumar et al. 2019; recent 2025 work |

### Is "Conditional Reward Design" an Established Concept?

**Yes**, though it goes by several names. The most theoretically grounded framing is **Hierarchical Potential-Based Reward Shaping (HPRS)**, which formalizes conditional activation through priority tiers. The more informal practice of "gated rewards" (multiplying reward terms by state-dependent indicator functions) is ubiquitous in robotics RL but rarely formalized as a standalone concept. The key insight: **conditional activation of reward terms based on state predicates is standard engineering practice in RL, with HPRS providing the formal framework.**

### Canonical Pattern: Gated/Conditional Reward Terms

**Confidence: HIGH** -- This is standard practice in robotics RL.

The simplest and most common form of conditional reward is multiplying reward terms by indicator (gate) functions:

```python
def compute_reward(state, action):
    reward = 0.0
    
    # Always active: time penalty
    reward -= 0.001
    
    # Conditional: only penalize energy when close enough to matter
    if state["prey_distance"] < distance_threshold:
        reward -= energy_weight * np.sum(action**2)
    
    # Conditional: coiling reward only activates when in contact
    if state["contact_fraction"] > 0.1:
        reward += wrap_weight * state["wrap_count"]
    
    # Conditional: success bonus requires sustained contact
    if state["contact_fraction"] > 0.6 and state["wrap_count"] > 1.5:
        reward += success_bonus
    
    return reward
```

**Why this works:** Gate functions are piecewise constant, so they don't break PBRS guarantees when applied to the base reward (not the shaping reward). When applied to potential functions, they still preserve optimality as long as the gating is deterministic and state-dependent.

### Advanced Pattern: Hierarchical PBRS (HPRS)

**Confidence: HIGH** -- Published at Frontiers in Robotics and AI, 2024.

HPRS organizes reward terms into a strict priority hierarchy. For the snake robot:

```
Priority 1 (Safety):    Don't self-collide, stay within bounds
Priority 2 (Target):    Approach prey / achieve contact
Priority 3 (Comfort):   Energy efficiency, gait quality, smooth motion
```

The key mechanism: **target rewards only activate when safety is satisfied, and comfort rewards only activate when both safety and target are met.** This creates conditional activation through hierarchical dependencies rather than ad-hoc gates.

```python
class HierarchicalReward:
    """HPRS-style hierarchical reward with conditional activation."""
    
    def __call__(self, state, action):
        # Level 1: Safety (always active)
        safety_satisfied = self.check_safety(state)
        safety_reward = self.compute_safety_reward(state)
        
        # Level 2: Target (only if safety satisfied)
        target_reward = 0.0
        target_satisfied = False
        if safety_satisfied:
            target_satisfied = self.check_target(state)
            target_reward = self.compute_target_reward(state)
        
        # Level 3: Comfort (only if safety AND target satisfied)
        comfort_reward = 0.0
        if safety_satisfied and target_satisfied:
            comfort_reward = self.compute_comfort_reward(state, action)
        
        return safety_reward + target_reward + comfort_reward
```

### For HRL Specifically: Parent-Task-Aware Potentials

**Confidence: MEDIUM** -- IJCAI 2015 paper (Gao & Toni), niche but well-cited.

Standard PBRS defines Phi(s). In HRL, the same state can have different "goodness" depending on which sub-task/option is active. Gao & Toni (2015) showed that PBRS for HRL requires Phi(s, parent_task):

```python
# Standard PBRS: same potential regardless of active skill
phi = distance_potential(state)

# HRL-aware PBRS: potential depends on which skill is active
if active_skill == "approach":
    phi = approach_potential(state)
elif active_skill == "coil":
    phi = coil_potential(state)  # Different potential for same state!
```

**Implication for this project:** The existing `CompositeRewardShaping` already supports this by having separate `ApproachPotential` and `CoilPotential`. The meta-controller naturally provides the parent-task context. This is a validation that the current architecture is on the right track.

### Relevance to This Project

**HIGH relevance.** The approach/coil task decomposition is a textbook case for conditional rewards:
- During approach: Activate distance potential, velocity bonus. Deactivate coil terms.
- During coiling: Activate contact, wrap, constriction potentials. Reduce/deactivate approach terms.
- The meta-controller implicitly handles this by selecting which sub-policy (and thus which reward function) is active.

**What to implement:** The HRL structure already provides natural conditional activation through skill selection. Additional conditional gating within each sub-task (e.g., "only penalize energy when making contact" in the coil task) is straightforward and standard.

### Pitfalls for Conditional Rewards

1. **Discontinuous reward landscapes:** Abrupt activation/deactivation of reward terms creates discontinuities that can destabilize value function learning. **Mitigation:** Use smooth gates (sigmoid, tanh) rather than hard indicator functions.

2. **State-space boundary effects:** If the condition threshold (e.g., `prey_distance < 0.5`) falls in a frequently visited region, the agent experiences reward flickering. **Mitigation:** Use hysteresis (different thresholds for activation vs. deactivation) or smooth interpolation.

3. **Breaking PBRS guarantees:** If conditional gates are applied to the shaping reward F(s,a,s') rather than the base reward, the potential-based guarantee (preserving optimal policies) may no longer hold. **Mitigation:** Apply conditions to base rewards only, or verify that gated potentials still satisfy the PBRS conditions.

---

## Topic 2: Evolving/Scheduled Reward Weights

### Established Terminology

| Term | Definition | Key Reference |
|------|-----------|---------------|
| **Reward curriculum** | Staged introduction of reward complexity; train on simple reward first, then add terms | Fournier et al. 2024 (arXiv:2410.16790) |
| **Two-stage reward curriculum** | Phase 1: task-only reward. Phase 2: task + behavioral constraints | Decoupling Task and Behavior, 2025 (arXiv:2603.05113) |
| **Reward weight annealing** | Smooth interpolation of reward term coefficients over training | Common practice |
| **Curriculum learning** | General concept of training on progressively harder tasks/objectives | Bengio et al. 2009 |
| **Adaptive reward weighting** | Dynamically adjusting weights based on training progress/performance | PBT and meta-learning approaches |
| **Population-Based Training (PBT)** | Evolutionary search over hyperparameters (including reward weights) during training | Jaderberg et al. 2017 (DeepMind) |

### Canonical Pattern: Two-Stage Reward Curriculum

**Confidence: HIGH** -- Multiple 2024-2025 papers confirm this approach works well in robotics.

The clearest and best-validated approach splits training into two phases:

**Phase 1 (Task Mastery):** Train only on base task reward (e.g., approach prey, make contact).
**Phase 2 (Behavioral Refinement):** Introduce auxiliary terms (energy efficiency, gait quality, smoothness).

The combined reward:
```
r = (1 - w) * r_task + w * r_auxiliary
```

Where `w` transitions from 0 to `w_target` at the phase boundary.

```python
class TwoStageRewardCurriculum:
    """Two-stage reward curriculum from Fournier et al. 2024."""
    
    def __init__(self, w_target=0.5, transition_steps=100_000,
                 transition_schedule="cosine"):
        self.w_target = w_target
        self.transition_steps = transition_steps
        self.schedule = transition_schedule
        self.phase = 0  # 0 = task only, 1 = transitioning
        self.transition_step = 0
    
    def should_transition(self, metrics):
        """Check if agent has mastered base task."""
        # Option A: Actor-critic alignment (Fournier et al.)
        # Option B: Performance threshold
        # Option C: Reward convergence via trend analysis
        return metrics["success_rate"] > 0.8
    
    def get_weight(self):
        if self.phase == 0:
            return 0.0
        progress = min(self.transition_step / self.transition_steps, 1.0)
        if self.schedule == "cosine":
            return self.w_target * (1 - np.cos(progress * np.pi)) / 2
        return self.w_target * progress  # linear
    
    def compute_reward(self, r_task, r_auxiliary):
        w = self.get_weight()
        return (1 - w) * r_task + w * r_auxiliary
```

**Key implementation detail (from Fournier et al. 2024):** Store decomposed rewards `{r_task, r_auxiliary}` in the replay buffer. This allows reward recomputation when the weight changes, without re-collecting experience. Network weights persist across phase boundaries -- ablations showed that resetting the weights substantially decreases performance, while resetting the buffer has little influence.

### Pattern: Continuous Weight Annealing

**Confidence: HIGH** -- Standard practice, already partially implemented in this project.

The project's `CurriculumGaitPotential` already implements this pattern for the sigma parameter. The same approach generalizes to any reward weight:

```python
class RewardWeightScheduler:
    """Anneal multiple reward weights over training."""
    
    def __init__(self, schedules: Dict[str, Tuple[float, float, str]]):
        """
        schedules: {name: (initial_value, final_value, schedule_type)}
        """
        self.schedules = schedules
        self.progress = 0.0
    
    def set_progress(self, progress: float):
        self.progress = np.clip(progress, 0.0, 1.0)
    
    def get_weights(self) -> Dict[str, float]:
        weights = {}
        for name, (init, final, schedule) in self.schedules.items():
            if schedule == "linear":
                weights[name] = init + self.progress * (final - init)
            elif schedule == "cosine":
                factor = (1 - np.cos(self.progress * np.pi)) / 2
                weights[name] = init + factor * (final - init)
            elif schedule == "step":
                weights[name] = final if self.progress > 0.5 else init
        return weights

# Example for snake robot:
scheduler = RewardWeightScheduler({
    "energy_penalty": (0.0, 0.01, "linear"),      # Introduce energy penalty gradually
    "gait_quality": (0.0, 0.5, "cosine"),          # Fade in gait matching
    "distance_reward": (1.0, 0.5, "linear"),       # Reduce distance importance
    "contact_reward": (0.0, 1.0, "step"),           # Activate contact reward at midpoint
})
```

### Pattern: Automatic Phase Transition Detection

**Confidence: MEDIUM** -- Fournier et al. 2024 propose specific criteria, but tuning the transition threshold is itself a hyperparameter.

Three strategies for detecting when to transition between reward phases:

1. **Performance threshold:** Transition when success_rate > threshold (simple, requires tuning)
2. **Actor-critic alignment:** Transition when actor loss is consistently low for m steps (Fournier et al.: m=20, threshold Gamma_CR=-50 for DMC)
3. **Reward convergence:** Use Huber regression on reward trajectory to detect plateau

### Recommended Curriculum for Snake Robot

Given the approach/coil/HRL hierarchy:

**Stage 1: Approach mastery** -- Train approach policy with distance + velocity potential only. No energy penalty. Broad gait sigma.

**Stage 2: Approach refinement** -- Add energy penalty, tighten gait sigma. Wait for approach success rate > 80%.

**Stage 3: Coil mastery** -- Train coil policy with contact + wrap potential only. No energy/stability penalties.

**Stage 4: Coil refinement** -- Add stability reward, energy penalty. Wait for coil success rate > 80%.

**Stage 5: HRL integration** -- Meta-controller training with intrinsic reward. Curriculum stages already defined in `HRLConfig.curriculum_stages`.

This maps naturally to the existing `curriculum_stages: ["approach_only", "coil_only", "full"]` in the HRL config.

### Pitfalls for Reward Scheduling

1. **Catastrophic forgetting during transition:** When new reward terms are introduced, the value function estimates become incorrect. The agent may temporarily unlearn good behavior. **Mitigation:** Gradual weight annealing (cosine > linear > step). Keep replay buffer from Phase 1.

2. **Transition threshold sensitivity:** If the threshold is too low, the agent transitions before mastering the base task. If too high, training wastes time. **Mitigation:** Use multiple criteria (e.g., success rate AND low actor loss).

3. **Reward scale mismatch across phases:** If r_task and r_auxiliary have very different magnitudes, the transition creates a sudden change in reward scale even with smooth weight annealing. **Mitigation:** Normalize reward terms to similar scales before combining.

4. **Off-policy staleness:** In actor-critic methods, changing reward weights invalidates Q-value estimates from old experience. **Mitigation:** Store decomposed rewards in replay buffer and recompute combined rewards at training time (as in Fournier et al. 2024).

---

## Topic 3: Learned/Discovered Reward Functions

### Established Terminology

| Term | Definition | Key Reference |
|------|-----------|---------------|
| **Inverse Reinforcement Learning (IRL)** | Infer reward function from expert demonstrations | Ng & Russell 2000 |
| **Adversarial IRL (AIRL)** | Recover reward functions robust to domain variation | Fu et al. 2018 |
| **Reward learning from preferences** | Learn reward from human pairwise comparisons | Christiano et al. 2017 |
| **Evolutionary reward search** | Population-based search over reward function space | LaRes (2024), DERL (2025) |
| **Bilevel reward optimization** | Upper-level optimizes reward, lower-level trains policy | Nature Communications 2025 |
| **LLM-based reward design** | Use LLMs to generate reward function code, iterate based on training results | Eureka (2023), CARD (2024) |
| **Population-Based Training (PBT)** | Co-evolve hyperparameters (including reward weights) alongside training | Jaderberg et al. 2017 |

### Landscape (Ordered by Practicality for This Project)

#### Tier 1: Practical and Applicable Now

**Population-Based Training for Reward Weights**

**Confidence: HIGH** -- Well-established, used at DeepMind/OpenAI.

PBT treats reward weights as hyperparameters and evolves them alongside policy training. Multiple agents train in parallel with different reward weight configurations; poorly-performing agents periodically copy weights from better ones and mutate their reward weights.

**Tradeoff:** Requires 4-16x compute (parallel agents). For a project with limited GPU resources, this is expensive. Better suited to final tuning once the reward structure is fixed.

**Recommendation for this project:** LOW priority. The reward structure is understood; manual tuning is sufficient. Consider PBT only if multi-GPU access is available and final performance optimization is needed.

#### Tier 2: Relevant but Not Recommended

**Inverse RL from Demonstrations**

**Confidence: HIGH** (the technique) / **LOW** (relevance to this project)

The project already has a behavioral cloning pipeline with `DemonstrationBuffer` and `GaitPotential` that effectively does demonstration-guided reward shaping. This is simpler and more practical than full IRL.

Full IRL would: (1) assume optimal demonstrations, (2) require solving the RL problem as an inner loop, (3) recover a reward function that may not be interpretable. The project's current approach (Gaussian potential in feature space near demonstrations) achieves the same goal with less complexity.

**Recommendation:** Do NOT invest in IRL. The gait potential approach already captures demonstration guidance effectively.

**Bilevel Reward Optimization**

**Confidence: MEDIUM** -- Nature Communications 2025 paper demonstrates it for embodied agents.

A bilevel framework where the upper level optimizes the reward function (via regret minimization) and the lower level trains the policy. Requires a differentiable world model. Adds significant implementation complexity. Targets cases where reward design is genuinely unknown.

**Recommendation:** Do NOT invest at this stage. The snake robot's reward terms (distance, contact, wrapping) are physically meaningful and well-understood.

#### Tier 3: Research Frontier (Not Recommended)

**LLM-Based Reward Design (Eureka, CARD)**

Uses an LLM to generate reward function code, train a policy, evaluate, and iterate. Generated rewards may not satisfy PBRS guarantees. Adds nondeterminism. The project's reward structure is already well-defined.

**Evolutionary Reward Search (DERL, LaRes)**

Uses evolutionary algorithms to discover reward function structure (not just weights). Research-grade, not production-ready.

### What This Project Should Do Instead

The project already has the right primitives. What's missing is:
1. **Conditional activation** of reward terms based on task phase (Topic 1) -- straightforward to add
2. **Weight scheduling** for auxiliary terms over training (Topic 2) -- partially implemented via CurriculumGaitPotential, generalize to all weights
3. **Nothing from Topic 3** -- the reward structure is known, don't automate its discovery

---

## TorchRL Integration

### Built-in Reward Transforms

| Transform | Purpose | Useful Here? |
|-----------|---------|-------------|
| `RewardScaling` | Affine transform: `r * scale + loc` | Yes -- normalize reward magnitude |
| `RewardClipping` | Clip to `[min, max]` | Yes -- prevent extreme rewards |
| `RewardSum` | Track cumulative episode reward | Yes -- monitoring |
| `BinarizeReward` | Map to 0/1 | No |
| `Reward2GoTransform` | Compute reward-to-go | Maybe -- for replay buffer |
| `LineariseRewards` | Weighted sum of multi-objective rewards | Yes -- for MORL-style weighting |

### Custom Reward Transform Pattern

To implement conditional or scheduled reward shaping as a TorchRL transform:

```python
from torchrl.envs.transforms import Transform

class ConditionalRewardShaping(Transform):
    """Apply reward shaping conditionally based on state."""
    
    in_keys = ["reward", "prey_distance", "contact_fraction"]
    out_keys = ["reward"]
    
    def __init__(self, approach_weight=1.0, coil_weight=1.5,
                 activation_distance=0.3):
        super().__init__()
        self.approach_weight = approach_weight
        self.coil_weight = coil_weight
        self.activation_distance = activation_distance
    
    def _apply_transform(self, tensordict):
        reward = tensordict["reward"]
        distance = tensordict["prey_distance"]
        contact = tensordict["contact_fraction"]
        
        # Smooth gate: sigmoid activation for coil reward
        coil_gate = torch.sigmoid(10 * (self.activation_distance - distance))
        
        approach_shaping = self.approach_weight * (1 - coil_gate) * (-distance)
        coil_shaping = self.coil_weight * coil_gate * contact
        
        shaped_reward = reward + approach_shaping + coil_shaping
        return tensordict.set("reward", shaped_reward)
    
    def transform_reward_spec(self, reward_spec):
        return reward_spec  # Shape unchanged
```

### Integration with Existing Code

The project's current reward computation happens inside `base_env.py::_compute_reward()`. Two integration strategies:

**Strategy A (Recommended): Keep rewards in env, use transforms for shaping**
- Base reward computed in `_compute_reward()` (as now)
- PBRS, conditional gating, and scheduling applied as TorchRL transforms
- Cleaner separation of concerns

**Strategy B: All-in-env reward computation**
- Everything inside `_compute_reward()` including scheduling
- Simpler but harder to compose and test

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reward normalization | Custom running mean/std | `RewardScaling` transform | Numerical stability edge cases |
| Reward clipping | Manual `torch.clamp` | `RewardClipping` transform | Composable with other transforms |
| Episode reward tracking | Manual accumulator | `RewardSum` transform | Handles reset correctly |
| Multi-objective scalarization | Custom weighted sum | `LineariseRewards` transform | Handles spec propagation |
| Curriculum scheduling logic | Custom step counter | Extend `CurriculumGaitPotential.set_progress()` pattern | Already proven in codebase |

---

## Common Pitfalls (Cross-Cutting)

### Pitfall 1: Reward Hacking Through Multi-Term Exploitation
**What goes wrong:** Agent maximizes easy-to-game reward terms while ignoring hard ones. Example: snake maximizes velocity bonus by oscillating near prey instead of approaching.
**Why it happens:** Multi-term rewards create exploitable dimensional mismatches. Agents are adversarial optimizers.
**How to avoid:** (1) Validate that each reward term correlates with desired behavior in isolation, (2) Use reward capping to prevent extreme exploitation, (3) Monitor individual reward components separately during training.
**Warning signs:** High total reward but poor task completion. Individual reward terms showing unexpected distributions.

### Pitfall 2: PBRS Guarantee Violation
**What goes wrong:** Shaped reward changes the optimal policy (agent learns wrong behavior that looks good under shaping).
**Why it happens:** Shaping function doesn't satisfy F(s,a,s') = gamma*Phi(s') - Phi(s). Common when adding conditional gates to shaping terms or using non-potential-based shaping.
**How to avoid:** (1) Only apply conditions to base reward, not PBRS shaping terms, (2) Verify shaping satisfies potential-based form, (3) Test with and without shaping to confirm same optimal behavior.
**Warning signs:** Agent performs well with shaping but poorly without. Shaping reward dominates base reward.

### Pitfall 3: Catastrophic Forgetting During Reward Transitions
**What goes wrong:** Agent unlearns base task when auxiliary reward terms are introduced.
**Why it happens:** Value function estimates become incorrect when reward distribution shifts. Off-policy data becomes stale.
**How to avoid:** (1) Use gradual weight annealing (cosine > linear > step), (2) Store decomposed rewards in replay buffer, (3) Keep training on base task alongside new terms.
**Warning signs:** Sudden performance drop after phase transition. Base task metrics regressing.

### Pitfall 4: Reward Scale Mismatch Between Task Phases
**What goes wrong:** Approach rewards and coil rewards have very different magnitudes, causing the value function to overfit to one phase.
**Why it happens:** Different physical quantities (distance vs. contact fraction vs. wrap angle) have different natural scales.
**How to avoid:** Normalize reward terms to similar ranges [0, 1] or [-1, 1] before weighting. Use `RewardScaling` transform.
**Warning signs:** Value function estimates diverging. One sub-task learning much faster than the other.

### Pitfall 5: Discontinuous Reward Landscapes from Hard Gates
**What goes wrong:** Policy oscillates at gate boundaries. Gradient estimates become noisy.
**Why it happens:** Hard indicator functions (if/else) create reward discontinuities.
**How to avoid:** Use smooth sigmoid gates with tunable steepness: `gate = sigmoid(k * (threshold - state))` where k controls sharpness.
**Warning signs:** Policy frequently switching between behaviors near boundary. High variance in reward during evaluation.

---

## Architecture Patterns

### Recommended Reward Architecture for Snake HRL

```
Reward Architecture
|
+-- Meta-Controller Level
|   +-- Intrinsic reward (skill selection quality)
|   +-- Extrinsic reward (overall task progress)
|   +-- Curriculum: stages ["approach_only", "coil_only", "full"]
|
+-- Approach Sub-Policy Level
|   +-- Base reward: time penalty + success bonus
|   +-- PBRS: ApproachPotential (distance + velocity)
|   +-- Conditional: energy penalty (gated by distance)
|   +-- Scheduled: gait weight (anneal from 0 to target)
|
+-- Coil Sub-Policy Level
    +-- Base reward: time penalty + success bonus
    +-- PBRS: CoilPotential (contact + wrap + constriction)
    +-- Conditional: stability reward (gated by contact_fraction)
    +-- Scheduled: energy penalty (anneal from 0 to target)
```

### Anti-Patterns to Avoid

- **Monolithic reward function:** Don't put all reward logic in a single function with many nested conditionals. Use `CompositeRewardShaping` with named components.
- **Hard-coded weight transitions:** Don't use `if step > 50000: weight = 0.5`. Use a scheduler with smooth annealing.
- **Shaping without monitoring:** Don't add shaping terms without logging their individual contributions. Always track decomposed rewards.
- **Mixing base and shaping rewards:** Keep base (environment) rewards separate from PBRS shaping. The `PotentialBasedRewardShaping.__call__` already does this correctly.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|-------------|------------------|--------------|--------|
| Fixed reward weights | Two-stage reward curriculum | 2024-2025 | Better multi-objective balance in robotics |
| Manual reward engineering | LLM-assisted reward code generation | 2023-2024 | Faster prototyping (less theoretical rigor) |
| IRL for reward learning | Demonstration-guided PBRS (like GaitPotential) | Ongoing | Simpler, more interpretable, same benefits |
| Static PBRS | Dynamic/Curriculum PBRS | 2012-2024 | Better adaptation to training progress |
| Separate multi-objective rewards | Reward-conditioned single policy | 2024-2025 | One policy handles multiple objective tradeoffs |

---

## Open Questions

1. **Optimal transition criteria for reward curriculum stages**
   - What we know: Success rate thresholds work. Actor-critic alignment is more principled.
   - What's unclear: Best threshold values for snake robot tasks specifically.
   - Recommendation: Start with success_rate > 0.8, tune empirically. Log both metrics.

2. **Smooth gate steepness parameter (k in sigmoid gates)**
   - What we know: Smoother gates reduce gradient noise but provide less precise activation.
   - What's unclear: Optimal k for this domain's reward scales and state distributions.
   - Recommendation: Start with k=10, sweep over [5, 10, 20, 50] if needed.

3. **Interaction between gait shaping and task rewards during coiling**
   - What we know: Gait potential encourages specific locomotion patterns. Coiling requires different body configurations than locomotion.
   - What's unclear: Whether gait shaping should be deactivated during coiling phase.
   - Recommendation: Make gait weight a scheduled parameter that decreases during coil training. Monitor whether gait shaping interferes with wrap count.

---

## Sources

### Primary (HIGH confidence)
- [HPRS: Hierarchical PBRS from Task Specifications](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1444188/full) - Frontiers in Robotics and AI, 2024
- [Curriculum RL for Complex Reward Functions](https://arxiv.org/abs/2410.16790) - Fournier et al. 2024 (RC-SAC/RC-TD3)
- [Two-Stage Reward Curriculum for Robotics](https://arxiv.org/abs/2603.05113) - Decoupling Task and Behavior, 2025
- [Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) - Lilian Weng, 2024 (comprehensive blog post)
- [TorchRL Transforms Documentation](https://docs.pytorch.org/rl/main/reference/envs_transforms.html) - Official PyTorch docs
- [TorchRL Transform Base Class](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.Transform.html) - Official PyTorch docs

### Secondary (MEDIUM confidence)
- [Comprehensive Overview of Reward Engineering](https://arxiv.org/html/2408.10215v1) - Survey covering PBRS, DPBRS, and reward engineering taxonomy
- [PBRS for HRL (PBRS-MAXQ-0)](https://www.ijcai.org/Proceedings/15/Papers/493.pdf) - Gao & Toni, IJCAI 2015
- [Dynamic PBRS](https://www.ifaamas.org/Proceedings/aamas2012/papers/2C_3.pdf) - Devlin & Kudenko, AAMAS 2012
- [Discovery of Reward Functions for Embodied RL](https://www.nature.com/articles/s41467-025-66009-y) - Nature Communications 2025
- [Population-Based Training](https://deepmind.google/discover/blog/population-based-training-of-neural-networks/) - DeepMind 2017
- [Reward-Conditioned RL](https://arxiv.org/html/2603.05066) - 2025

### Tertiary (LOW confidence)
- [Reward Models in Deep RL: A Survey](https://arxiv.org/html/2506.15421v1) - IJCAI 2025 (recent)
- [Evolutionary RL Survey](https://arxiv.org/pdf/2303.04150) - 2023 survey covering reward search approaches

---

## Metadata

**Confidence breakdown:**
- Conditional reward design: HIGH - Well-established PBRS theory, directly applicable to project
- Reward weight scheduling: HIGH - Multiple 2024-2025 papers with concrete algorithms for robotics
- Learned reward design: MEDIUM - Well-studied field, but low applicability to this project
- TorchRL integration: MEDIUM - API documented, custom reward transforms less documented than observation transforms
- HRL-specific patterns: MEDIUM - PBRS-MAXQ-0 is niche; the general principle (parent-task-aware potentials) is sound

**Research date:** 2026-04-02
**Valid until:** 2026-07-02 (stable field, core techniques unlikely to change)
