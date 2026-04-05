# TorchRL Reward Integration Patterns

## Built-in Reward Transforms

| Transform | Purpose | When to use |
|-----------|---------|-------------|
| `RewardScaling` | Affine: `r * scale + loc` | Normalize reward magnitude |
| `RewardClipping` | Clip to `[min, max]` | Prevent extreme rewards |
| `RewardSum` | Cumulative episode reward | Monitoring |
| `LineariseRewards` | Weighted sum of multi-objective | MORL-style scalarization |
| `Reward2GoTransform` | Compute reward-to-go | Replay buffer |

## Custom Conditional Reward Transform

```python
from torchrl.envs.transforms import Transform

class ConditionalRewardShaping(Transform):
    """Sigmoid-gated reward activation based on state."""

    in_keys = ["reward", "prey_distance", "contact_fraction"]
    out_keys = ["reward"]

    def __init__(self, approach_weight=1.0, coil_weight=1.5,
                 activation_distance=0.3, gate_steepness=10.0):
        super().__init__()
        self.approach_weight = approach_weight
        self.coil_weight = coil_weight
        self.activation_distance = activation_distance
        self.k = gate_steepness

    def _apply_transform(self, tensordict):
        reward = tensordict["reward"]
        distance = tensordict["prey_distance"]
        contact = tensordict["contact_fraction"]

        # Smooth sigmoid gate
        coil_gate = torch.sigmoid(self.k * (self.activation_distance - distance))

        approach_shaping = self.approach_weight * (1 - coil_gate) * (-distance)
        coil_shaping = self.coil_weight * coil_gate * contact

        return tensordict.set("reward", reward + approach_shaping + coil_shaping)

    def transform_reward_spec(self, reward_spec):
        return reward_spec
```

## Custom Reward Weight Scheduler Transform

```python
class ScheduledRewardWeights(Transform):
    """Anneal reward component weights over training."""

    in_keys = ["reward_task", "reward_auxiliary"]
    out_keys = ["reward"]

    def __init__(self, w_target=0.5, transition_steps=100_000,
                 schedule="cosine"):
        super().__init__()
        self.w_target = w_target
        self.transition_steps = transition_steps
        self.schedule = schedule
        self._step = 0
        self._active = False

    def activate(self):
        """Call when base task is mastered (success_rate > threshold)."""
        self._active = True
        self._step = 0

    def _get_weight(self):
        if not self._active:
            return 0.0
        progress = min(self._step / self.transition_steps, 1.0)
        if self.schedule == "cosine":
            return self.w_target * (1 - math.cos(progress * math.pi)) / 2
        return self.w_target * progress

    def _apply_transform(self, tensordict):
        w = self._get_weight()
        r_task = tensordict["reward_task"]
        r_aux = tensordict["reward_auxiliary"]
        combined = (1 - w) * r_task + w * r_aux
        self._step += 1
        return tensordict.set("reward", combined)

    def transform_reward_spec(self, reward_spec):
        return reward_spec
```

## Integration Strategy

**Recommended: Keep base rewards in env, use transforms for shaping/scheduling.**

```python
# In env._compute_reward(): compute base + PBRS, expose components
# In training script: add transforms for scheduling/gating

env = SoftManipulatorEnv(config)
env = env.append_transform(ObservationNorm(in_keys=["observation"]))
env = env.append_transform(RewardScaling(loc=0.0, scale=1.0))  # if needed
env = env.append_transform(RewardSum())  # episode tracking
```

This separates concerns: env owns physics-derived rewards, transforms own
training-schedule-dependent modifications.

## Logging Decomposed Rewards

Register per-component keys in the observation spec so ParallelEnv allocates shared memory:

```python
# In _make_spec():
self.observation_spec = Composite(
    observation=Unbounded(shape=(obs_dim,), ...),
    reward_dist=Unbounded(shape=(1,), ...),
    reward_heading=Unbounded(shape=(1,), ...),
    reward_pbrs_dist=Unbounded(shape=(1,), ...),
    reward_pbrs_head=Unbounded(shape=(1,), ...),
    reward_smooth=Unbounded(shape=(1,), ...),
)
```

Then log each component to W&B/TensorBoard for per-component diagnostics.