# Phase 15: TorchRL-Native Environments and Benchmark Validation - Research

**Researched:** 2026-03-19
**Domain:** TorchRL native environments, BenchMARL, benchmark validation strategy
**Confidence:** HIGH

## Summary

This research addresses the user's requirement to use TorchRL-native environments and avoid gymnasium for MM-RKHS benchmark validation. The investigation covers five areas: (1) TorchRL's built-in environments, (2) BenchMARL's applicability, (3) official TorchRL tutorials, (4) TorchRL environment wrappers, and (5) a TorchRL-native benchmark validation strategy.

The key finding is that the project already has TorchRL-native environments (Choi2025 `SoftManipulatorEnv`) that work directly with the existing `create_actor()`/`create_critic()`/`SyncDataCollector`/`GAE` infrastructure. TorchRL's only built-in continuous-action environment is `PendulumEnv`, which uses non-standard observation keys (`th`, `thdot` instead of `observation`) and requires custom transforms or a wrapper to work with the project's training infrastructure. For a simple smoke-test environment, a lightweight `SimplePendulum` custom EnvBase implementation is the cleanest approach -- fully TorchRL-native, no external dependencies, passes `check_env_specs()`, and is compatible with all existing project infrastructure. BenchMARL is exclusively multi-agent and not suitable for single-agent MM-RKHS benchmarking.

**Primary recommendation:** Validate MM-RKHS exclusively on the Choi2025 4-task suite (already TorchRL-native). For a fast smoke-test during development, use a simple custom `SimplePendulum` EnvBase environment (3-dim obs, 1-dim action, ~50K frames to converge). Do NOT install gymnasium or dm_control.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Loss function: `L = -E[ratio * A] + beta * MMD^2(pi_new, pi_old) + (1/eta) * KL` (majorization bound from Eq 6.1)
- Architecture: Single V(s) critic with GAE, shared Adam optimizer, TanhNormal actor
- Continuous actions -- no discretization
- MMD: RBF kernel, linear-time unbiased estimator, 16 samples per state
- Hyperparameters: beta=1.0, eta=1.0, lr=3e-4, epochs=10, mini_batch=256, GAE lambda=0.95
- Benchmark on Choi2025 4-task suite only: follow_target, inverse_kinematics, tight_obstacles, random_obstacles
- Quick validation: 100K frames per task (learning signal check, not full training)
- 1 seed per configuration (4 runs total)
- Compare against Phase 14's existing PPO/SAC results (no fresh controls needed)
- W&B project: same `choi2025-replication` or new `otpg-validation` -- Claude's discretion

### Claude's Discretion
- W&B project naming
- Network size (3x1024 like Phase 14 or smaller)
- Whether to re-run PPO/SAC controls alongside OTPG
- RBF kernel bandwidth (sigma) default value
- Exact checkpoint save frequency
- Run naming convention

### Deferred Ideas (OUT OF SCOPE)
- Full 1M-frame training runs -- defer until 100K quick validation shows learning signal
- Multi-seed runs (3-5 seeds) for statistical significance
- Adaptive beta/eta scheduling
- W&B sweep over beta/eta
- Gymnasium baselines (HalfCheetah, Ant) for broader algorithm validation
- Elastica snake environment benchmark
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| OTPG-01 | MMRKHSConfig dataclass with beta, eta, mmd_bandwidth, mmd_num_samples, gae_lambda, value_coef | Config hierarchy pattern from `src/configs/training.py` -- extends RLConfig |
| OTPG-02 | MMRKHSTrainer class following PPOTrainer pattern | Full pipeline verified: `create_actor()` + `create_critic()` + `GAE` + `SyncDataCollector` all work with both Choi2025 and custom TorchRL envs |
| OTPG-03 | MMD penalty computation with RBF kernel | Linear-time estimator pattern documented; action samples via `actor.get_dist()` rsample |
| OTPG-04 | MM-RKHS loss function with clamped log-ratio, normalized advantage, NaN guards | **CRITICAL CORRECTION**: log prob key is `action_log_prob`, NOT `sample_log_prob` (verified empirically) |
| OTPG-05 | Choi2025 benchmark integration with train_mmrkhs.py | Choi2025 envs confirmed TorchRL-native; train_ppo.py pattern ready to mirror |
| OTPG-06 | Checkpoint save/load following PPOTrainer pattern | PPOTrainer checkpoint pattern documented with atomic saves |
</phase_requirements>

## Standard Stack

### Core (Already Installed -- No New Dependencies)
| Library | Version | Purpose | Verified |
|---------|---------|---------|----------|
| torchrl | 0.11.1 | RL environments, collectors, transforms, GAE | `python3 -c "import torchrl"` |
| torch | 2.x | Neural network training | Already in use |
| tensordict | (bundled) | TensorDict data containers | Already in use |

### NOT Needed (Explicitly Avoided Per User Constraint)
| Library | Reason |
|---------|--------|
| gymnasium | User explicitly wants TorchRL-native; not installed, not needed |
| gymnasium[mujoco] | Same as above |
| dm_control | Not installed; DMControlEnv raises import error |
| mujoco | Not installed |
| benchmarl | Multi-agent only; not suitable for single-agent MM-RKHS |
| gym (OpenAI) | Deprecated predecessor to gymnasium |

**Installation:** None required. The existing environment has everything needed.

## Architecture Patterns

### Pattern 1: TorchRL-Native Environment Structure (Verified)

The project's established pattern for TorchRL environments uses `EnvBase` subclasses that output a single `observation` key. This is the pattern used by `SoftManipulatorEnv` (Choi2025) and should be followed for any new environments.

**Critical observation keys:**
```python
# Reset returns:
{
    "observation": torch.Tensor,  # shape [obs_dim]
    "done": torch.Tensor,        # shape [1], dtype=bool
    "terminated": torch.Tensor,  # shape [1], dtype=bool
    "truncated": torch.Tensor,   # shape [1], dtype=bool
}

# Step returns (in "next" TensorDict):
{
    "observation": torch.Tensor,  # shape [obs_dim]
    "reward": torch.Tensor,       # shape [1], dtype=float32
    "done": torch.Tensor,         # shape [1], dtype=bool
    "terminated": torch.Tensor,   # shape [1], dtype=bool
    "truncated": torch.Tensor,    # shape [1], dtype=bool
}
```

**Spec construction:**
```python
# Source: papers/choi2025/env.py (verified working pattern)
self.observation_spec = Composite(
    observation=Unbounded(shape=(obs_dim,), dtype=torch.float32, device=self._device),
    shape=(),
)
self.action_spec = Bounded(
    low=-1.0, high=1.0,
    shape=(action_dim,), dtype=torch.float32, device=self._device,
)
self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=self._device)
self.done_spec = Composite(
    done=Unbounded(shape=(1,), dtype=torch.bool, device=self._device),
    terminated=Unbounded(shape=(1,), dtype=torch.bool, device=self._device),
    truncated=Unbounded(shape=(1,), dtype=torch.bool, device=self._device),
    shape=(),
)
```

### Pattern 2: SyncDataCollector Integration (Verified)

```python
# Source: Empirically verified with Choi2025 env + create_actor()
# Note: SyncDataCollector is deprecated in v0.11.1, will be removed in v0.13.
# Replacement: torchrl.collectors.Collector (same interface)
from torchrl.collectors import SyncDataCollector  # or Collector

collector = SyncDataCollector(
    create_env_fn=lambda: env,
    policy=actor,
    frames_per_batch=config.frames_per_batch,
    total_frames=config.total_frames,
    device=device,
)

for batch in collector:
    # batch keys: ['action', 'observation', 'done', 'terminated', 'truncated',
    #              'next', 'collector', 'action_log_prob', 'scale', 'loc']
    #
    # CRITICAL: log prob key is 'action_log_prob', NOT 'sample_log_prob'
    batch = batch.to(device)
    with torch.no_grad():
        gae(batch)
    # After GAE: batch also has 'advantage', 'value_target', 'state_value'
    metrics = trainer._update(batch)
```

### Pattern 3: Actor and Critic Compatibility (Verified)

```python
# Source: Empirically verified
from src.networks.actor import create_actor
from src.networks.critic import create_critic

# create_actor() outputs: ["loc", "scale", "action", "action_log_prob"]
# create_critic() outputs: ["state_value"]
# GAE reads: ["state_value", "reward", "done"]
# GAE writes: ["advantage", "value_target"]

# For MM-RKHS _update(), access log prob from collected batch:
log_prob_old = mb["action_log_prob"]  # NOT "sample_log_prob"

# To compute new log prob during update:
dist = actor.get_dist(mb)            # Get TanhNormal distribution
log_prob_new = dist.log_prob(mb["action"])
```

### Pattern 4: SimplePendulum for Smoke Testing (Verified)

A lightweight TorchRL-native pendulum environment that works with the full training pipeline without any external dependencies. Verified via `check_env_specs()`, `SyncDataCollector`, `create_actor()`, `create_critic()`, and `GAE`.

```python
# Source: Empirically verified in this research session
class SimplePendulum(EnvBase):
    """TorchRL-native pendulum. Obs=[cos(th), sin(th), thdot], act=torque[-2,2].

    200 steps/episode, reward = -(theta^2 + 0.1*vel^2 + 0.001*torque^2).
    Random policy: ~-1600 per episode. Good policy: ~-200.
    PPO converges in ~50K-100K frames.
    """
    def __init__(self, device='cpu'):
        super().__init__(device=torch.device(device), batch_size=torch.Size([]))
        # observation: [cos(th), sin(th), angular_velocity] -> 3-dim
        # action: torque in [-2, 2] -> 1-dim
        self.observation_spec = Composite(
            observation=Unbounded(shape=(3,), dtype=torch.float32, device=self.device),
            shape=(),
        )
        self.action_spec = Bounded(
            low=-2.0, high=2.0, shape=(1,),
            dtype=torch.float32, device=self.device,
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=self.device)
        self.done_spec = Composite(
            done=Unbounded(shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Unbounded(shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Unbounded(shape=(1,), dtype=torch.bool, device=self.device),
            shape=(),
        )
        # Physics parameters
        self._max_speed = 8.0
        self._dt = 0.05
        self._g = 10.0
        self._m = 1.0
        self._l = 1.0
        self._max_steps = 200
        self._step_count = 0
        self._th = 0.0
        self._thdot = 0.0

    def _set_seed(self, seed):
        self._rng = torch.manual_seed(seed)

    def _reset(self, tensordict=None, **kwargs):
        self._th = torch.empty(()).uniform_(-math.pi, math.pi).item()
        self._thdot = torch.empty(()).uniform_(-1.0, 1.0).item()
        self._step_count = 0
        obs = torch.tensor(
            [math.cos(self._th), math.sin(self._th), self._thdot],
            dtype=torch.float32, device=self.device,
        )
        return TensorDict({
            "observation": obs,
            "done": torch.tensor([False], dtype=torch.bool, device=self.device),
            "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
            "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict):
        u = tensordict["action"].squeeze(-1).clamp(-2.0, 2.0).item()
        th_norm = ((self._th + math.pi) % (2 * math.pi)) - math.pi
        cost = th_norm**2 + 0.1 * self._thdot**2 + 0.001 * u**2
        new_thdot = self._thdot + (
            3 * self._g / (2 * self._l) * math.sin(self._th)
            + 3.0 / (self._m * self._l**2) * u
        ) * self._dt
        new_thdot = max(-self._max_speed, min(self._max_speed, new_thdot))
        self._th = self._th + new_thdot * self._dt
        self._thdot = new_thdot
        self._step_count += 1
        truncated = self._step_count >= self._max_steps
        obs = torch.tensor(
            [math.cos(self._th), math.sin(self._th), self._thdot],
            dtype=torch.float32, device=self.device,
        )
        return TensorDict({
            "observation": obs,
            "reward": torch.tensor([-cost], dtype=torch.float32, device=self.device),
            "done": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
            "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
        }, batch_size=self.batch_size, device=self.device)
```

**Verified compatibility:**
- `check_env_specs(SimplePendulum())` -- PASSED
- `create_actor(obs_dim=3, action_spec=env.action_spec)` -- works, outputs `action_log_prob`
- `create_critic(obs_dim=3)` -- works, outputs `state_value`
- `GAE(gamma=0.99, lmbda=0.95, value_network=critic)` -- adds `advantage`, `value_target`
- `SyncDataCollector(create_env_fn=..., policy=actor)` -- collects batches correctly

### Anti-Patterns to Avoid

- **Installing gymnasium/mujoco for benchmarking:** The user explicitly wants TorchRL-native. The Choi2025 suite already provides 4 continuous-control tasks with known PPO/SAC baselines from Phase 14.
- **Using TorchRL's built-in PendulumEnv directly with create_actor():** PendulumEnv outputs `th` and `thdot` as scalar keys, not a single `observation` key. CatTensors fails on 0-dimensional tensors. Either write a custom EnvBase wrapper (SimplePendulum pattern) or use CosTransform/SinTransform + CatTensors.
- **Using `sample_log_prob` key name:** The actual key is `action_log_prob`. This is a correction to the existing 15-RESEARCH.md which incorrectly uses `sample_log_prob` throughout.
- **Using BenchMARL for single-agent benchmarking:** BenchMARL is exclusively multi-agent (VMAS, SMACv2, MPE, SISL, MeltingPot). It does not support single-agent environments.

## TorchRL Native Environments Analysis

### Built-in Environments (torchrl.envs.custom)
| Environment | Type | Obs Keys | Action | Suitable? |
|-------------|------|----------|--------|-----------|
| PendulumEnv | Continuous control | `th`, `thdot` (scalar) | 1-dim [-2,2] | Needs wrapper (scalar obs keys incompatible with `observation` key pattern) |
| ChessEnv | Board game | Board tensor | Discrete | No (discrete, non-RL-benchmark) |
| TicTacToeEnv | Board game | Board tensor | Discrete | No (trivial, discrete) |
| LLMHashingEnv | LLM | Text | Text | No (NLP-specific) |

**Conclusion:** Only PendulumEnv is relevant, and it requires a wrapper or custom transforms. Writing a SimplePendulum (as shown above) is cleaner.

### Environment Wrappers Available
| Wrapper | Backend Required | Status in Project |
|---------|-----------------|-------------------|
| GymEnv | gymnasium | NOT INSTALLED |
| DMControlEnv | dm_control | NOT INSTALLED |
| BraxEnv | brax (JAX) | NOT INSTALLED |
| JumanjiEnv | jumanji (JAX) | NOT INSTALLED |
| VmasEnv | vmas | NOT INSTALLED, multi-agent only |
| IsaacGymEnv | Isaac Gym | NOT INSTALLED |
| PettingZooEnv | PettingZoo | NOT INSTALLED, multi-agent only |

**Conclusion:** None of these wrapper backends are installed. The project operates entirely with custom EnvBase environments (Choi2025 suite).

### BenchMARL Assessment
BenchMARL (Facebook Research) is built on TorchRL but is **exclusively multi-agent**:
- Supports: VMAS, SMACv2, MPE, SISL, MeltingPot, MAgent2
- Algorithms: MAPPO, IPPO, MADDPG, IDDPG, MASAC, ISAC, QMIX, VDN, IQL
- NOT suitable for single-agent MM-RKHS benchmarking
- Cannot be used with Choi2025 environments (single-agent)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Benchmark environments | gymnasium wrappers | Choi2025 4-task suite + SimplePendulum | Already TorchRL-native, no dependencies |
| Data collection | Custom rollout loop | `SyncDataCollector` (or `Collector` in v0.13) | Handles stepping, resets, device transfer |
| Actor network | New policy network | `create_actor()` | ProbabilisticActor with TanhNormal, tested |
| Critic network | New value network | `create_critic()` | ValueOperator, tested |
| Advantage estimation | Custom GAE | `torchrl.objectives.value.GAE` | Already used by PPOTrainer |
| Episode reward tracking | Custom counter | `RewardSum` transform | Standard TorchRL transform |
| Environment verification | Manual testing | `check_env_specs()` | Catches spec mismatches automatically |

## Common Pitfalls

### Pitfall 1: Wrong Log Prob Key Name
**What goes wrong:** MM-RKHS trainer accesses `batch["sample_log_prob"]` for the old policy log probability, causing a KeyError.
**Why it happens:** The existing 15-RESEARCH.md and 15-01-PLAN.md incorrectly use `sample_log_prob`. TorchRL's `ClipPPOLoss` internally maps `sample_log_prob -> action_log_prob`, hiding this from PPO users. The MM-RKHS trainer accesses batch keys directly.
**How to avoid:** Use `batch["action_log_prob"]` or `mb["action_log_prob"]`. Verified empirically: `create_actor()` with `return_log_prob=True` outputs `action_log_prob` key.
**Warning signs:** `KeyError: 'sample_log_prob'` during first training batch.

### Pitfall 2: TorchRL PendulumEnv Scalar Observation Keys
**What goes wrong:** TorchRL's built-in `PendulumEnv` outputs `th` (scalar, shape []) and `thdot` (scalar, shape []) instead of a single `observation` vector. `CatTensors` fails on 0-dimensional tensors with: `RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated`.
**Why it happens:** PendulumEnv is designed as a stateless env teaching tool, not as a standard RL benchmark. Its observation spec uses named scalar keys.
**How to avoid:** Do NOT use PendulumEnv directly. Either (a) use `SimplePendulum` custom EnvBase that outputs `observation` directly, or (b) use the Choi2025 environments which already output `observation`.
**Warning signs:** CatTensors RuntimeError or shape mismatch when stacking rollout batches.

### Pitfall 3: SyncDataCollector Deprecation
**What goes wrong:** `SyncDataCollector` prints deprecation warnings in TorchRL 0.11.1 and will be removed in v0.13.
**Why it happens:** TorchRL renamed `SyncDataCollector` to `Collector` with identical API.
**How to avoid:** For now, use `SyncDataCollector` (matches PPOTrainer pattern). When upgrading to v0.13+, change imports to `from torchrl.collectors import Collector`. The constructor signature is identical.
**Warning signs:** `DeprecationWarning: SyncDataCollector has been deprecated and will be removed in v0.13.`

### Pitfall 4: Accessing actor.get_dist() for MMD Computation
**What goes wrong:** Calling `actor.get_dist(mb)` on a ProbabilisticActor to get the distribution for sampling actions for MMD computation may fail or return unexpected types.
**Why it happens:** TorchRL's `ProbabilisticActor` wraps distributions inside `SafeProbabilisticModule`. The `get_dist()` method may require specific TensorDict structure.
**How to avoid:** For MMD computation, compute new log probs via forward pass and sample via `dist.rsample()`. Specifically:
```python
# Forward the actor to get dist params
td_with_dist = actor(mb.clone())
# Build distribution manually from loc/scale
from torchrl.modules import TanhNormal
dist_new = TanhNormal(td_with_dist["loc"], td_with_dist["scale"],
                       low=action_spec.space.low, high=action_spec.space.high)
action_samples_new = dist_new.rsample((num_samples,))
```
**Warning signs:** `AttributeError` on `get_dist` or wrong distribution type.

### Pitfall 5: RewardSum Transform Missing
**What goes wrong:** Episode reward metrics not logged. `mean_episode_reward` never appears in training logs.
**Why it happens:** Standard `EnvBase` environments do not automatically accumulate rewards into `episode_reward`. The `RewardSum` transform is needed.
**How to avoid:** Always append `RewardSum()` transform before passing env to trainer, exactly as `train_ppo.py` does:
```python
env = env.append_transform(RewardSum())
```
**Warning signs:** `episode_reward` key missing from `next` TensorDict when `done=True`.

## Code Examples

### Example 1: Choi2025 Environment Usage (Verified Working)
```python
# Source: Empirically verified in this research session
import sys
sys.path.insert(0, '/home/user/snake-hrl-torchrl/papers')
from choi2025.env import SoftManipulatorEnv
from choi2025.config import Choi2025EnvConfig, TaskType

config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)
env = SoftManipulatorEnv(config=config, device='cpu')

# Verified specs:
# observation_spec: Composite(observation=Unbounded(shape=(148,)))
# action_spec: Bounded(shape=(10,), low=-1.0, high=1.0)
# Note: 3D sim uses 10 action dims; 2D sim uses 5 action dims
```

### Example 2: Full OTPG-Compatible Pipeline (Verified)
```python
# Source: Empirically verified end-to-end
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value import GAE
from src.networks.actor import create_actor
from src.networks.critic import create_critic

env = SoftManipulatorEnv(config=config, device='cpu')
obs_dim = env.observation_spec["observation"].shape[-1]

actor = create_actor(obs_dim=obs_dim, action_spec=env.action_spec, device='cpu')
critic = create_critic(obs_dim=obs_dim, device='cpu')
gae = GAE(gamma=0.99, lmbda=0.95, value_network=critic)

collector = SyncDataCollector(
    create_env_fn=lambda: SoftManipulatorEnv(config=config, device='cpu'),
    policy=actor,
    frames_per_batch=64,
    total_frames=128,
    device='cpu',
)

for batch in collector:
    batch = batch.to('cpu')
    with torch.no_grad():
        gae(batch)

    # MM-RKHS-specific: access log probs for importance ratio
    log_prob_old = batch["action_log_prob"]  # CORRECT key name

    # For each mini-batch in _update():
    # Re-compute log prob under new policy
    td_fwd = actor(batch.clone())
    log_prob_new = td_fwd["action_log_prob"]
    ratio = (log_prob_new - log_prob_old).exp()
    advantage = batch["advantage"]

    # Surrogate advantage (no clipping -- trust region via MMD)
    surr = (ratio * advantage).mean()
    break
```

### Example 3: SimplePendulum for Unit Testing
```python
# Source: Verified with check_env_specs(), SyncDataCollector, create_actor(), GAE
# Use this for unit tests that need a fast, simple, TorchRL-native env

from torchrl.envs import check_env_specs

env = SimplePendulum()
check_env_specs(env)  # PASSES

# Compatible with existing infrastructure:
actor = create_actor(obs_dim=3, action_spec=env.action_spec,
                     config=ActorConfig(hidden_dims=[64, 64]))
critic = create_critic(obs_dim=3, config=CriticConfig(hidden_dims=[64, 64]))
gae = GAE(gamma=0.99, lmbda=0.95, value_network=critic)

# Quick smoke test: 200 frames
collector = SyncDataCollector(
    create_env_fn=lambda: SimplePendulum(),
    policy=actor,
    frames_per_batch=200,
    total_frames=200,
    device='cpu',
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `SyncDataCollector` | `Collector` (same API) | TorchRL 0.11.1 (deprecation) | Must migrate before v0.13 |
| `gym` (OpenAI) | `gymnasium` (Farama) | 2023 | Project uses neither; custom EnvBase instead |
| Gymnasium MuJoCo for benchmarks | Custom TorchRL EnvBase | This project | No external dependencies for benchmarking |
| `sample_log_prob` in code examples | `action_log_prob` | TorchRL naming convention | Existing RESEARCH.md has wrong key name |

## Open Questions

1. **Computing new log_prob during MM-RKHS update**
   - What we know: The batch stores `action_log_prob` from the policy that collected the data. For importance ratio, we need log_prob under the *current* (updated) policy.
   - What's unclear: The cleanest way to compute new log_prob during _update() mini-batch processing. Options: (a) re-forward the actor module on mini-batch and read `action_log_prob`, (b) manually build the TanhNormal distribution from `loc`/`scale` and call `log_prob(action)`.
   - Recommendation: Use approach (a) -- `td_fwd = actor(mb.clone()); log_prob_new = td_fwd["action_log_prob"]` -- because `create_actor()` handles all the TanhNormal parameterization correctly. Do NOT manually construct TanhNormal; let the existing infrastructure handle it.

2. **MMD computation: sampling from old vs new policy**
   - What we know: MMD requires samples from both old and new policy. The new policy is the current actor. The old policy is the actor that collected the batch.
   - What's unclear: How to sample from the "old" policy distribution during _update(). The batch stores `loc`/`scale` from collection time, which can be used to reconstruct the old distribution.
   - Recommendation: Reconstruct old distribution from stored `loc`/`scale`: `old_dist = TanhNormal(mb["loc"].detach(), mb["scale"].detach(), ...)`. Sample new distribution from current actor forward pass. This avoids storing a copy of the old policy parameters.

3. **SimplePendulum as development smoke test**
   - What we know: SimplePendulum passes check_env_specs and works with the full pipeline.
   - What's unclear: Whether it's worth including in the test suite or only using during development.
   - Recommendation: Include SimplePendulum in `tests/test_mmrkhs.py` for fast unit tests (3-dim obs, 1-dim action, ~30 second training). Use Choi2025 for actual benchmark validation (OTPG-05).

## Benchmark Validation Strategy (TorchRL-Native Only)

### Primary: Choi2025 4-Task Suite
The user's locked decision is to benchmark on Choi2025 only. This is already TorchRL-native.

| Task | Obs Dim | Act Dim | Act Range | Phase 14 PPO | Phase 14 SAC | MM-RKHS Target |
|------|---------|---------|-----------|-------------|-------------|-------------|
| follow_target | 148 (3D) / ~80 (2D) | 10 (3D) / 5 (2D) | [-1, 1] | Inconclusive (short run) | Learning signal confirmed | Any improvement over random |
| inverse_kinematics | 151 (3D) | 10 (3D) / 5 (2D) | [-1, 1] | Inconclusive | Learning signal confirmed | Any improvement over random |
| tight_obstacles | 148 (3D) | 10 (3D) / 5 (2D) | [-1, 1] | Inconclusive | Learning signal confirmed | Any improvement over random |
| random_obstacles | 148 (3D) | 10 (3D) / 5 (2D) | [-1, 1] | Inconclusive | Learning signal confirmed | Any improvement over random |

### Secondary: SimplePendulum (Development Smoke Test)
| Property | Value |
|----------|-------|
| Obs dim | 3 (cos, sin, angular_vel) |
| Act dim | 1 (torque) |
| Act range | [-2, 2] |
| Episode length | 200 steps |
| Random policy reward | ~-1600 |
| Good policy reward | ~-200 |
| Training to learn | ~50K-100K frames |
| Training time | ~30 seconds (CPU) |
| Dependencies | None (pure TorchRL EnvBase) |

### Validation Protocol
1. **Unit tests (SimplePendulum):** MMRKHSTrainer initializes, _update() produces finite losses, MMD computes without NaN, checkpoint save/load round-trips. All in `tests/test_mmrkhs.py`.
2. **Smoke test (SimplePendulum):** 10K-frame training run, verify reward improves from random baseline. ~30 seconds.
3. **Benchmark validation (Choi2025):** 100K-frame training on follow_target, verify learning signal. ~15 minutes.
4. **Full benchmark (Choi2025):** 100K frames on all 4 tasks, compare with Phase 14 PPO/SAC. 4 runs total.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none (pytest defaults) |
| Quick run command | `python3 -m pytest tests/test_mmrkhs.py -x` |
| Full suite command | `python3 -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OTPG-01 | MMRKHSConfig dataclass validates and inherits RLConfig | unit | `python3 -m pytest tests/test_mmrkhs.py::test_config -x` | Wave 0 |
| OTPG-02 | MMRKHSTrainer initializes with SimplePendulum env | unit | `python3 -m pytest tests/test_mmrkhs.py::test_trainer_init -x` | Wave 0 |
| OTPG-03 | MMD penalty computes without NaN, returns finite scalar | unit | `python3 -m pytest tests/test_mmrkhs.py::test_mmd_penalty -x` | Wave 0 |
| OTPG-04 | _update() produces finite loss values, uses action_log_prob | unit | `python3 -m pytest tests/test_mmrkhs.py::test_update_step -x` | Wave 0 |
| OTPG-05 | Short training run on SimplePendulum completes without crash | smoke | `python3 -m pytest tests/test_mmrkhs.py::test_short_training -x` | Wave 0 |
| OTPG-06 | Checkpoint save/load round-trip preserves state | unit | `python3 -m pytest tests/test_mmrkhs.py::test_checkpoint -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/test_mmrkhs.py -x`
- **Per wave merge:** `python3 -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_mmrkhs.py` -- covers OTPG-01 through OTPG-06, uses SimplePendulum for fast tests
- [ ] Framework install: pytest already available

## Sources

### Primary (HIGH confidence)
- TorchRL 0.11.1 installed -- `python3 -c "import torchrl; print(torchrl.__version__)"` outputs `0.11.1`
- `papers/choi2025/env.py` -- SoftManipulatorEnv confirmed TorchRL-native, outputs `observation` key
- `papers/choi2025/train_ppo.py` -- PPO training pattern for Choi2025 (verified working in Phase 14)
- `src/networks/actor.py` -- `create_actor()` outputs `action_log_prob` (verified empirically)
- `src/networks/critic.py` -- `create_critic()` outputs `state_value` (verified empirically)
- `torchrl.collectors.SyncDataCollector` -- batch keys verified: `action_log_prob` not `sample_log_prob`
- `torchrl.objectives.ClipPPOLoss.tensor_keys` -- confirms `sample_log_prob` maps to `action_log_prob`
- `check_env_specs(SimplePendulum())` -- PASSED
- [Official PyTorch PPO tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html) -- GymEnv pattern with transforms
- [Official TorchRL Pendulum tutorial](https://docs.pytorch.org/tutorials/advanced/pendulum.html) -- Custom EnvBase patterns, CatTensors, transforms

### Secondary (MEDIUM confidence)
- [BenchMARL README](https://github.com/facebookresearch/BenchMARL) -- Confirmed multi-agent only
- [TorchRL Multi-Agent PPO tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html) -- VMAS environment patterns
- [TorchRL EnvBase docs](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html) -- EnvBase API reference

### Tertiary (LOW confidence)
- SimplePendulum reward baselines -- estimated from gymnasium Pendulum-v1 equivalence; actual values may differ slightly due to implementation differences
- Training time estimates -- based on CPU testing, GPU times will differ

## Metadata

**Confidence breakdown:**
- TorchRL native environment compatibility: HIGH -- empirically verified end-to-end
- Log prob key correction: HIGH -- empirically verified, critical for MM-RKHS implementation
- BenchMARL assessment: HIGH -- confirmed multi-agent only from official docs
- SimplePendulum as smoke test: HIGH -- passes check_env_specs, full pipeline verified
- Choi2025 benchmark strategy: HIGH -- already working from Phase 14
- MMD computation approach: MEDIUM -- pattern documented but not yet tested in training loop

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (stable; no API changes expected before TorchRL 0.13)
