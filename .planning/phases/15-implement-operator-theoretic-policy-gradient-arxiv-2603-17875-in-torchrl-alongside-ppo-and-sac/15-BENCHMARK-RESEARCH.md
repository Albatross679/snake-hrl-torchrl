# Phase 15: OTPG Benchmark Environment Research

**Researched:** 2026-03-19
**Domain:** Standard RL benchmark environments for algorithm validation
**Confidence:** HIGH

## Summary

This research identifies the best standard RL benchmark environments for validating the OTPG algorithm before deploying on the complex Choi2025 snake robot tasks. The goal is fast iteration on simple, well-understood environments with known PPO/SAC baselines, continuous action spaces, and training times under 500K frames.

The recommended benchmark suite is a three-tier ladder: (1) **Pendulum-v1** (no MuJoCo needed, 3-dim obs, 1-dim action, ~100K frames to converge), (2) **InvertedPendulum-v4** (trivial MuJoCo task, 4-dim obs, 1-dim action, ~100K frames), and (3) **HalfCheetah-v4** (standard MuJoCo benchmark, 17-dim obs, 6-dim action, ~500K-1M frames). These cover increasing complexity and action-space dimensionality while all being environments where PPO and SAC have well-documented baselines.

**Primary recommendation:** Start with Pendulum-v1 (zero MuJoCo dependency, fastest iteration). Graduate to HalfCheetah-v4 once OTPG shows learning signal on Pendulum. Use `gymnasium[mujoco]` for MuJoCo environments -- no separate mujoco-py installation needed.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Loss function: `L = -E[ratio * A] + beta * MMD^2(pi_new, pi_old) + (1/eta) * KL` (majorization bound from Eq 6.1)
- Architecture: Single V(s) critic with GAE, shared Adam optimizer, TanhNormal actor
- Continuous actions -- no discretization
- MMD: RBF kernel, linear-time unbiased estimator, 16 samples per state
- Hyperparameters: beta=1.0, eta=1.0, lr=3e-4, epochs=10, mini_batch=256, GAE lambda=0.95

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
- **Gymnasium baselines (HalfCheetah, Ant) for broader algorithm validation** -- originally deferred but NOW being researched as the user explicitly requested it
- Elastica snake environment benchmark
</user_constraints>

## Standard Stack

### Core (to install)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gymnasium | latest (1.1.x) | Standard RL environment API | Required by TorchRL's GymEnv wrapper |
| gymnasium[mujoco] | (same) | MuJoCo continuous control envs | HalfCheetah, InvertedPendulum, Hopper, etc. |
| mujoco | >= 2.3.3 | Physics engine for MuJoCo envs | Bundled with `gymnasium[mujoco]`, no separate install |

### Already Installed
| Library | Version | Purpose |
|---------|---------|---------|
| torchrl | 0.11.1 | GymEnv wrapper, SyncDataCollector, GAE |
| torch | 2.10.0 | Neural networks, training |

### Not Needed
| Library | Reason |
|---------|--------|
| mujoco-py | Legacy -- gymnasium uses `mujoco` package directly since v4 envs |
| dm-control | Unnecessary -- gymnasium MuJoCo envs are sufficient |
| gym (OpenAI) | Deprecated -- use gymnasium |

**Installation:**
```bash
pip install "gymnasium[mujoco]"
```

This single command installs gymnasium and the mujoco package together.

## Benchmark Recommendations

### Recommended Benchmark Suite (Ranked)

| Rank | Environment | Obs Dim | Act Dim | Act Range | MuJoCo? | Frames to Learn | PPO Baseline | SAC Baseline | Why |
|------|-------------|---------|---------|-----------|---------|-----------------|--------------|--------------|-----|
| 1 | **Pendulum-v1** | 3 | 1 | [-2, 2] | No | ~50K-100K | ~-200 | ~-150 | Zero dependencies, fastest iteration |
| 2 | **InvertedPendulum-v4** | 4 | 1 | [-3, 3] | Yes | ~50K-100K | 1000.0 | 1000.0 | Trivially solvable, sanity check |
| 3 | **HalfCheetah-v4** | 17 | 6 | [-1, 1] | Yes | ~500K-1M | ~5000-8000 | ~10000-12000 | Standard benchmark, multi-dim action |
| 4 | **Hopper-v4** | 11 | 3 | [-1, 1] | Yes | ~500K-1M | ~2500-3500 | ~3000-3500 | Medium complexity, good middle ground |

### Tier 1: Minimal Validation (Pendulum-v1)
- **No MuJoCo required** -- pure Python, ships with gymnasium base install
- Observation: [cos(theta), sin(theta), angular_velocity] (3-dim)
- Action: torque [-2, 2] (1-dim continuous)
- Max episode length: 200 steps
- Reward: negative cost, best possible ~0, typical good policy ~-150 to -200
- **Training time:** ~50K-100K frames with PPO/SAC, ~30 seconds on GPU
- **Use case:** First smoke test. If OTPG cannot learn Pendulum, it will not learn anything else.

### Tier 2: MuJoCo Sanity Check (InvertedPendulum-v4)
- Observation: [cart_pos, pole_angle, cart_vel, pole_vel] (4-dim)
- Action: cart force [-3, 3] (1-dim continuous)
- Max episode length: 1000 steps
- Reward threshold: 950.0 (both PPO and SAC reach 1000.0)
- **Training time:** ~50K-100K frames, ~1 minute on GPU
- **Use case:** Confirms MuJoCo integration works. Nearly all algorithms solve this trivially.

### Tier 3: Real Benchmark (HalfCheetah-v4)
- Observation: body positions, velocities, joint angles (17-dim)
- Action: 6 joint torques [-1, 1] (6-dim continuous)
- Max episode length: 1000 steps
- PPO baseline: ~5000-8000 at 1M frames (SB3/SpinningUp)
- SAC baseline: ~10000-12000 at 1M frames
- **Training time:** ~500K-1M frames, ~10-30 minutes on GPU
- **Use case:** True algorithm comparison. Multi-dimensional action space. If OTPG matches or exceeds PPO here, it validates the continuous adaptation.

### Tier 4: Optional Additional (Hopper-v4)
- Observation: body positions, velocities (11-dim)
- Action: 3 joint torques [-1, 1] (3-dim continuous)
- More challenging than HalfCheetah due to balance requirements
- PPO baseline: ~2500-3500 at 1M frames
- **Use case:** Only if HalfCheetah results are promising and more evidence is needed.

### Environments NOT Recommended
| Environment | Why Not |
|-------------|---------|
| Ant-v4 | High-dimensional (111-dim obs, 8-dim act), slow to train, overkill for validation |
| Humanoid-v4 | Very high-dimensional (376-dim obs), extremely slow, hard to get working |
| Walker2d-v4 | Similar to Hopper but harder; HalfCheetah is a better first benchmark |
| LunarLander-v2 | Discrete actions (default), continuous variant is Box2D not MuJoCo |
| Swimmer-v4 | Known to have flat reward landscape, algorithms struggle to differentiate |

## Architecture Patterns

### Pattern 1: TorchRL GymEnv Creation
**What:** Create standard benchmark environments using TorchRL's GymEnv wrapper with proper transforms.
**When to use:** For all benchmark environments.
```python
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)

def make_benchmark_env(env_name: str, device: str = "cpu") -> TransformedEnv:
    """Create a TorchRL-wrapped gymnasium benchmark environment.

    Args:
        env_name: Gymnasium env ID (e.g., "Pendulum-v1", "HalfCheetah-v4")
        device: Target device
    """
    base_env = GymEnv(env_name, device=device)

    env = TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),        # MuJoCo returns float64, networks need float32
            StepCounter(),          # Track episode step counts
        ),
    )
    return env
```

### Pattern 2: Benchmark Training Script
**What:** A standalone script that trains OTPG (and optionally PPO/SAC) on a benchmark environment.
**When to use:** For quick algorithm validation before Choi2025 deployment.
```python
# scripts/benchmark_otpg.py
"""Quick OTPG validation on standard benchmark environments."""

import argparse
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter

from src.configs.training import OTPGConfig
from src.configs.network import NetworkConfig, ActorConfig, CriticConfig
from src.trainers.otpg import OTPGTrainer

BENCHMARKS = {
    "pendulum": {
        "env_name": "Pendulum-v1",
        "total_frames": 100_000,
        "frames_per_batch": 1024,
        "good_reward": -200,   # PPO baseline
    },
    "inverted_pendulum": {
        "env_name": "InvertedPendulum-v4",
        "total_frames": 100_000,
        "frames_per_batch": 2048,
        "good_reward": 950,
    },
    "halfcheetah": {
        "env_name": "HalfCheetah-v4",
        "total_frames": 500_000,
        "frames_per_batch": 4096,
        "good_reward": 5000,
    },
}

def make_env(env_name: str, device: str = "cpu"):
    base_env = GymEnv(env_name, device=device)
    return TransformedEnv(
        base_env,
        Compose(DoubleToFloat(), StepCounter()),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=list(BENCHMARKS.keys()),
                        default="pendulum")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-frames", type=int, default=None)
    args = parser.parse_args()

    bench = BENCHMARKS[args.benchmark]
    env = make_env(bench["env_name"], args.device)

    config = OTPGConfig(
        name=f"otpg-{args.benchmark}",
        total_frames=args.total_frames or bench["total_frames"],
        frames_per_batch=bench["frames_per_batch"],
        device=args.device,
    )

    # Use smaller network for benchmarks (faster iteration)
    net_config = NetworkConfig(
        actor=ActorConfig(hidden_dims=[256, 256]),
        critic=CriticConfig(hidden_dims=[256, 256]),
    )

    trainer = OTPGTrainer(
        env=env,
        config=config,
        network_config=net_config,
        device=args.device,
    )

    result = trainer.train()
    print(f"Final best reward: {result['best_reward']:.1f}")
    print(f"Baseline ({args.benchmark}): {bench['good_reward']}")

if __name__ == "__main__":
    main()
```

### Pattern 3: Comparison Script (PPO vs OTPG)
**What:** Run PPO and OTPG on the same environment for direct comparison.
**When to use:** After OTPG shows basic learning signal.
```python
# Run both algorithms on the same benchmark
for algo in ["ppo", "otpg"]:
    for benchmark in ["pendulum", "halfcheetah"]:
        env = make_env(BENCHMARKS[benchmark]["env_name"])
        if algo == "ppo":
            config = PPOConfig(name=f"ppo-{benchmark}", ...)
            trainer = PPOTrainer(env, config, ...)
        else:
            config = OTPGConfig(name=f"otpg-{benchmark}", ...)
            trainer = OTPGTrainer(env, config, ...)
        trainer.train()
```

### Anti-Patterns to Avoid
- **Skipping Pendulum and going straight to HalfCheetah:** Pendulum trains in 30 seconds. If OTPG fails on it, you save hours of debugging on harder envs.
- **Using ObservationNorm with MuJoCo envs:** MuJoCo observations are already well-scaled. Adding ObservationNorm requires a separate normalization pass and can hurt performance if the statistics are computed from too few samples.
- **Using 3x1024 networks on benchmarks:** Overkill. Use 2x256 for Pendulum, 2x256 for HalfCheetah. The Choi2025 3x1024 networks are sized for ~100-dim observations, not 3-17 dim.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Benchmark environments | Custom envs | `GymEnv("Pendulum-v1")` | Standard, reproducible, known baselines |
| Env transforms | Custom normalization | `DoubleToFloat()`, `StepCounter()` | TorchRL transforms handle device/dtype correctly |
| PPO comparison | Reimplement PPO | Existing `PPOTrainer` from `src/trainers/ppo.py` | Already tested on Choi2025, same infrastructure |
| Gym/Gymnasium compat | Manual version checks | `set_gym_backend("gymnasium")` | TorchRL handles the abstraction |

## Common Pitfalls

### Pitfall 1: gymnasium Not Installed
**What goes wrong:** `GymEnv("Pendulum-v1")` raises `ModuleNotFoundError: No module named 'gymnasium'` because gymnasium is not currently installed in the project virtualenv.
**How to avoid:** Run `pip install "gymnasium[mujoco]"` before any benchmark code. The `[mujoco]` extra installs both gymnasium and the mujoco physics engine.
**Warning signs:** Import errors when creating GymEnv.

### Pitfall 2: Float64 vs Float32
**What goes wrong:** MuJoCo environments return float64 observations. Neural networks expect float32. Without `DoubleToFloat()` transform, you get dtype mismatches or silent precision issues.
**How to avoid:** Always include `DoubleToFloat()` in the transform pipeline for MuJoCo environments.
**Warning signs:** `RuntimeError: expected Float but got Double`.

### Pitfall 3: Action Space Range Mismatch
**What goes wrong:** Pendulum-v1 has action range [-2, 2] while TanhNormal squashes to [-1, 1]. InvertedPendulum-v4 has [-3, 3]. HalfCheetah-v4 has [-1, 1]. If the actor does not properly scale actions, the agent operates in a restricted subspace.
**How to avoid:** TorchRL's `create_actor()` with `TanhNormal` already handles this by mapping the tanh output to the environment's action bounds via the `action_spec`. Verify by checking `env.action_spec.space.low` and `env.action_spec.space.high`.
**Warning signs:** Reward plateau far below baseline despite correct loss computation.

### Pitfall 4: Episode Reward Tracking Differences
**What goes wrong:** The Choi2025 environments use custom `episode_reward` tracking in TorchRL's `next` TensorDict. Standard GymEnv may not populate `episode_reward` the same way, causing the PPOTrainer's reward tracking to produce no metrics.
**How to avoid:** Ensure the benchmark env includes `StepCounter()` transform. For episode rewards, check that `next["episode_reward"]` is populated. If not, add a `RewardSum` transform.
**Warning signs:** `mean_episode_reward` never appears in logs.

### Pitfall 5: Comparing Against Wrong Baselines
**What goes wrong:** PyBullet baselines (SB3) are different from MuJoCo baselines. SB3 PPO gets ~2000 on PyBullet HalfCheetah vs ~8000 on MuJoCo HalfCheetah. Different env versions (v2 vs v4) also have different reward scales.
**How to avoid:** Only compare within the same gymnasium version (use v4 consistently). Reference MuJoCo-specific baselines, not PyBullet. The scores in the Benchmark Recommendations table above are for MuJoCo gymnasium envs.
**Warning signs:** Algorithm appears to "fail" when actually performing normally for the env version.

### Pitfall 6: SyncDataCollector with GymEnv Lambda
**What goes wrong:** Passing `create_env_fn=lambda: env` to SyncDataCollector reuses the same env object. This works for single collection but can cause issues if the env state is shared.
**How to avoid:** For benchmark envs, pass a factory function: `create_env_fn=lambda: make_env("Pendulum-v1")`. This creates a fresh env for the collector. The existing PPOTrainer pattern (`lambda: env`) works because each training run uses one env instance.
**Warning signs:** Subtle bugs in episode boundary handling.

## Code Examples

### Example 1: Minimal Pendulum Validation
```python
"""Smallest possible OTPG validation: train on Pendulum-v1."""
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter

# Create environment
env = TransformedEnv(
    GymEnv("Pendulum-v1", device="cpu"),
    Compose(DoubleToFloat(), StepCounter()),
)

# Verify dimensions
print(f"Obs dim: {env.observation_spec['observation'].shape}")  # [3]
print(f"Act dim: {env.action_spec.shape}")                       # [1]
print(f"Act range: [{env.action_spec.space.low}, {env.action_spec.space.high}]")  # [-2, 2]

# Quick rollout test
td = env.reset()
print(f"Reset keys: {list(td.keys())}")
td = env.rand_step()
print(f"Step keys: {list(td.keys())}")
```

### Example 2: HalfCheetah with SyncDataCollector
```python
"""HalfCheetah data collection using TorchRL infrastructure."""
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter
from torchrl.collectors import SyncDataCollector
from src.networks.actor import create_actor

# Create environment
env = TransformedEnv(
    GymEnv("HalfCheetah-v4", device="cpu"),
    Compose(DoubleToFloat(), StepCounter()),
)

# Create actor compatible with this env
obs_dim = env.observation_spec["observation"].shape[-1]  # 17
action_spec = env.action_spec  # Bounded(-1, 1, shape=[6])

actor = create_actor(
    obs_dim=obs_dim,
    action_spec=action_spec,
    config=ActorConfig(hidden_dims=[256, 256]),
    device="cpu",
)

# Create collector
collector = SyncDataCollector(
    create_env_fn=lambda: TransformedEnv(
        GymEnv("HalfCheetah-v4", device="cpu"),
        Compose(DoubleToFloat(), StepCounter()),
    ),
    policy=actor,
    frames_per_batch=4096,
    total_frames=500_000,
    device="cpu",
)

for batch in collector:
    print(f"Batch shape: {batch.shape}, keys: {list(batch.keys())}")
    break
```

### Example 3: Gymnasium Backend Selection
```python
"""Ensure gymnasium (not legacy gym) is used."""
from torchrl.envs import set_gym_backend, GymEnv

# Explicitly set gymnasium as the backend
with set_gym_backend("gymnasium"):
    env = GymEnv("Pendulum-v1")

# Alternative: set globally
import torchrl.envs
torchrl.envs.set_gym_backend("gymnasium")
```

## Benchmark Environment Details

### Pendulum-v1 (Classic Control, No MuJoCo)
- **State:** [cos(theta), sin(theta), angular_velocity]
- **Action:** torque in [-2, 2]
- **Reward:** `-(theta^2 + 0.1*vel^2 + 0.001*torque^2)`, max per step ~0
- **Episode length:** 200 steps (fixed)
- **Goal:** Swing pendulum up and balance (start hanging down)
- **PPO ~100K frames:** reaches -200 to -300 (good)
- **SAC ~100K frames:** reaches -150 to -200 (good)
- **Install:** `pip install gymnasium` (no MuJoCo needed)

### InvertedPendulum-v4 (MuJoCo, trivial)
- **State:** [cart_x, pole_angle, cart_vel, pole_vel]
- **Action:** cart force in [-3, 3]
- **Reward:** +1 per step alive
- **Episode length:** 1000 steps (terminated if pole falls)
- **Goal:** Balance pole upright
- **PPO/SAC ~100K frames:** reaches 1000 (maximum, trivially solved)
- **Install:** `pip install "gymnasium[mujoco]"`

### HalfCheetah-v4 (MuJoCo, standard benchmark)
- **State:** 8 joint angles + 9 velocities = 17-dim
- **Action:** 6 joint torques in [-1, 1]
- **Reward:** forward velocity - control cost
- **Episode length:** 1000 steps (no early termination)
- **Goal:** Run forward as fast as possible
- **PPO ~1M frames:** ~5000-8000 (well-known benchmark)
- **SAC ~1M frames:** ~10000-12000 (SAC outperforms PPO here)
- **Why ideal for OTPG:** No exploration challenge, pure optimization. If OTPG's trust region works, it should match or exceed PPO's convergence speed.

### Hopper-v4 (MuJoCo, medium difficulty)
- **State:** 5 joint angles + 6 velocities = 11-dim
- **Action:** 3 joint torques in [-1, 1]
- **Reward:** forward velocity + alive bonus - control cost
- **Episode length:** 1000 steps (terminated on fall)
- **Goal:** Hop forward without falling
- **PPO ~1M frames:** ~2500-3500
- **SAC ~1M frames:** ~3000-3500

## Recommended Hyperparameters for Benchmarks

### Pendulum-v1
```python
config = OTPGConfig(
    name="otpg-pendulum",
    total_frames=100_000,
    frames_per_batch=1024,
    num_epochs=10,
    mini_batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    beta=1.0,
    eta=1.0,
    mmd_num_samples=16,
    mmd_bandwidth=1.0,
    patience_batches=50,  # Short patience for fast iteration
)
net_config = NetworkConfig(
    actor=ActorConfig(hidden_dims=[64, 64]),
    critic=CriticConfig(hidden_dims=[64, 64]),
)
```

### HalfCheetah-v4
```python
config = OTPGConfig(
    name="otpg-halfcheetah",
    total_frames=500_000,
    frames_per_batch=4096,
    num_epochs=10,
    mini_batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    beta=1.0,
    eta=1.0,
    mmd_num_samples=16,
    mmd_bandwidth=1.0,
    patience_batches=100,
)
net_config = NetworkConfig(
    actor=ActorConfig(hidden_dims=[256, 256]),
    critic=CriticConfig(hidden_dims=[256, 256]),
)
```

## Validation Strategy

### Phase 1: Smoke Test (Pendulum-v1)
1. Train OTPG on Pendulum-v1 for 100K frames
2. **Pass criteria:** reward improves from ~-1600 (random) to ~-400 or better
3. If fails: debug loss function, check MMD computation, verify gradients flow
4. Estimated time: 1-2 minutes

### Phase 2: MuJoCo Integration (InvertedPendulum-v4)
1. Train OTPG on InvertedPendulum-v4 for 100K frames
2. **Pass criteria:** reward reaches 900+ (near maximum of 1000)
3. If fails: likely action space scaling issue or GymEnv integration bug
4. Estimated time: 2-3 minutes

### Phase 3: Real Benchmark (HalfCheetah-v4)
1. Train OTPG on HalfCheetah-v4 for 500K frames
2. Train PPO on same env for comparison
3. **Pass criteria:** OTPG reaches at least 50% of PPO's reward
4. **Stretch goal:** OTPG matches or exceeds PPO's reward
5. Estimated time: 15-30 minutes per algorithm

### Comparison to Choi2025
| Property | Pendulum-v1 | HalfCheetah-v4 | Choi2025 Tasks |
|----------|-------------|----------------|----------------|
| Obs dim | 3 | 17 | ~100-200 |
| Act dim | 1 | 6 | 5 |
| Act range | [-2, 2] | [-1, 1] | [-1, 1] |
| Physics | None | MuJoCo | Elastica/DisMech |
| Episode length | 200 | 1000 | varies |
| Known baselines | Yes | Yes | Phase 14 only |
| Train time (100K) | ~30s | ~5 min | ~15 min |

## Open Questions

1. **ObservationNorm for benchmarks?**
   - The TorchRL PPO tutorial uses `ObservationNorm` for InvertedDoublePendulum. However, most MuJoCo environments have well-scaled observations, and PPO/SAC benchmarks in SB3 do not normalize.
   - Recommendation: Skip observation normalization for benchmark validation. Add it only if OTPG fails to learn and observation scale is suspected.

2. **RewardSum transform for episode tracking?**
   - TorchRL's `GymEnv` may or may not populate `episode_reward` in the TensorDict depending on the gymnasium version and wrapper setup.
   - Recommendation: Test with a quick rollout. If `episode_reward` is missing, add `RewardSum(in_keys=["reward"])` to the transform pipeline.

3. **Vectorized envs for benchmarks?**
   - The Choi2025 setup uses single-env (`num_envs=1`). For faster benchmark training, `GymEnv(env_name, num_envs=4)` could parallelize.
   - Recommendation: Start with `num_envs=1` for simplicity. Only parallelize if training is too slow.

## Sources

### Primary (HIGH confidence)
- [TorchRL GymEnv docs](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.GymEnv.html) -- API reference
- [TorchRL PPO Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html) -- InvertedDoublePendulum example with transforms
- [Gymnasium MuJoCo docs](https://gymnasium.farama.org/environments/mujoco/) -- Environment specifications (obs/act dims, reward thresholds)
- [Gymnasium InvertedPendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) -- v4/v5 specs
- [Gymnasium HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) -- v4/v5 specs

### Secondary (MEDIUM confidence)
- [SB3 PPO Benchmarks](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) -- PyBullet scores (not MuJoCo; used for relative comparison)
- [SB3 SAC Benchmarks](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) -- PyBullet scores
- [SpinningUp Benchmarks](https://spinningup.openai.com/en/latest/spinningup/bench.html) -- MuJoCo scores (graph-only, no exact numbers extracted)
- [Open RL Benchmark](https://arxiv.org/html/2402.03046v1) -- Comprehensive tracked experiments

### Tertiary (LOW confidence)
- PPO/SAC reward numbers for MuJoCo v4 -- compiled from multiple sources, ranges rather than exact numbers
- Training time estimates -- based on known GPU throughput for similar workloads, not measured on this specific system

## Metadata

**Confidence breakdown:**
- Environment selection: HIGH -- these are universally used benchmarks
- TorchRL integration: HIGH -- GymEnv is well-documented and tested
- Installation: HIGH -- `gymnasium[mujoco]` is standard
- PPO/SAC baselines: MEDIUM -- scores vary by implementation, hyperparameters, and env version
- Training time estimates: MEDIUM -- depends on GPU, batch size, network size
- OTPG expected performance: LOW -- completely novel adaptation, no precedent

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (stable; environments and TorchRL API are stable)
