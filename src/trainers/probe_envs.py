"""Probe environments for RL trainer validation (Andy Jones methodology).

Five progressively complex minimal environments that isolate specific trainer
components. Each runs in seconds and produces a decisive pass/fail result.

If probe N fails but N-1 passes, the bug is in exactly the component that N adds.

Usage:
    from src.trainers.probe_envs import ProbeEnv1, run_probe_validation

    # Run single probe
    env = ProbeEnv1(device="cpu")
    td = env.reset()

    # Run all probes against a trainer class
    results = run_probe_validation(PPOTrainer, device="cpu")
    assert all(r["passed"] for r in results.values())
"""

from typing import Dict, Any, Optional, Type

import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite,
    Bounded,
    Unbounded,
    Categorical,
)


class ProbeEnv1(EnvBase):
    """Probe 1: Value network learns a constant.

    Setup: 1 action, zero observation (dim=1, always 0), 1 timestep, +1 reward.
    Tests: Value loss calculation, optimizer works at all.
    Expected: Value network learns V(s) ≈ 1.0 within ~100 updates.
    Failure indicates: Broken value loss or backpropagation.
    """

    def __init__(self, device="cpu", batch_size=None):
        super().__init__(device=device, batch_size=batch_size or torch.Size([]))
        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.action_spec = Bounded(
            low=-1.0, high=1.0, shape=(1,),
            dtype=torch.float32, device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=(1,), dtype=torch.float32, device=self.device,
        )
        self.done_spec = Composite(
            done=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            shape=self.batch_size, device=self.device,
        )

    def _reset(self, tensordict=None):
        return TensorDict({
            "observation": torch.zeros(1, device=self.device),
            "done": torch.tensor([False], device=self.device),
            "terminated": torch.tensor([False], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict):
        return TensorDict({
            "observation": torch.zeros(1, device=self.device),
            "reward": torch.tensor([1.0], device=self.device),
            "done": torch.tensor([True], device=self.device),
            "terminated": torch.tensor([True], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed):
        torch.manual_seed(seed if seed is not None else 0)


class ProbeEnv2(EnvBase):
    """Probe 2: Value network learns from observations.

    Setup: 1 action, random ±1 observation (dim=1), 1 timestep,
           reward = observation value.
    Tests: Backpropagation through the value network.
    Expected: Value prediction correlates with observation.
    Failure indicates: Broken gradient flow through value network.
    """

    def __init__(self, device="cpu", batch_size=None):
        super().__init__(device=device, batch_size=batch_size or torch.Size([]))
        self._obs = None
        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            shape=self.batch_size, device=self.device,
        )
        self.action_spec = Bounded(
            low=-1.0, high=1.0, shape=(1,),
            dtype=torch.float32, device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=(1,), dtype=torch.float32, device=self.device,
        )
        self.done_spec = Composite(
            done=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            shape=self.batch_size, device=self.device,
        )

    def _reset(self, tensordict=None):
        self._obs = torch.sign(torch.randn(1, device=self.device))  # ±1
        return TensorDict({
            "observation": self._obs.clone(),
            "done": torch.tensor([False], device=self.device),
            "terminated": torch.tensor([False], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict):
        reward = self._obs.clone()  # reward = observation
        self._obs = torch.sign(torch.randn(1, device=self.device))
        return TensorDict({
            "observation": self._obs.clone(),
            "reward": reward,
            "done": torch.tensor([True], device=self.device),
            "terminated": torch.tensor([True], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed):
        torch.manual_seed(seed if seed is not None else 0)


class ProbeEnv3(EnvBase):
    """Probe 3: Reward discounting works.

    Setup: 1 action, zero observation (dim=1), 2 timesteps,
           reward = 0 at step 0, reward = +1 at step 1 (terminal).
    Tests: Correct reward discounting across timesteps.
    Expected: V(s0) ≈ gamma * 1.0.
    Failure indicates: Broken reward accumulation or discount factor.
    """

    def __init__(self, device="cpu", batch_size=None):
        super().__init__(device=device, batch_size=batch_size or torch.Size([]))
        self._step_count = 0
        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            shape=self.batch_size, device=self.device,
        )
        self.action_spec = Bounded(
            low=-1.0, high=1.0, shape=(1,),
            dtype=torch.float32, device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=(1,), dtype=torch.float32, device=self.device,
        )
        self.done_spec = Composite(
            done=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            shape=self.batch_size, device=self.device,
        )

    def _reset(self, tensordict=None):
        self._step_count = 0
        return TensorDict({
            "observation": torch.zeros(1, device=self.device),
            "done": torch.tensor([False], device=self.device),
            "terminated": torch.tensor([False], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict):
        self._step_count += 1
        done = self._step_count >= 2
        reward = 1.0 if done else 0.0
        return TensorDict({
            "observation": torch.zeros(1, device=self.device),
            "reward": torch.tensor([reward], device=self.device),
            "done": torch.tensor([done], device=self.device),
            "terminated": torch.tensor([done], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed):
        torch.manual_seed(seed if seed is not None else 0)


class ProbeEnv4(EnvBase):
    """Probe 4: Policy gradient learns to select the right action.

    Setup: 2 actions (continuous dim=1), zero observation (dim=1), 1 timestep,
           reward = +1 if action > 0, reward = -1 if action <= 0.
    Tests: Advantage computation and policy gradient update.
    Expected: Policy converges to always output positive action.
    Failure indicates: Broken advantage computation or policy update.
    """

    def __init__(self, device="cpu", batch_size=None):
        super().__init__(device=device, batch_size=batch_size or torch.Size([]))
        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            shape=self.batch_size, device=self.device,
        )
        self.action_spec = Bounded(
            low=-1.0, high=1.0, shape=(1,),
            dtype=torch.float32, device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=(1,), dtype=torch.float32, device=self.device,
        )
        self.done_spec = Composite(
            done=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            shape=self.batch_size, device=self.device,
        )

    def _reset(self, tensordict=None):
        return TensorDict({
            "observation": torch.zeros(1, device=self.device),
            "done": torch.tensor([False], device=self.device),
            "terminated": torch.tensor([False], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict):
        action = tensordict["action"]
        reward = torch.where(action > 0, 1.0, -1.0).sum().unsqueeze(0)
        return TensorDict({
            "observation": torch.zeros(1, device=self.device),
            "reward": reward.to(self.device),
            "done": torch.tensor([True], device=self.device),
            "terminated": torch.tensor([True], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed):
        torch.manual_seed(seed if seed is not None else 0)


class ProbeEnv5(EnvBase):
    """Probe 5: Joint policy-value learning from observations.

    Setup: 2 actions (continuous dim=1), random ±1 observation (dim=1),
           1 timestep, reward = obs * sign(action).
    Tests: Full policy-value interaction, batching correctness.
    Expected: Policy learns action = sign(obs); value learns V(s) ≈ 1.
    Failure indicates: Stale experience or incorrect sample pairing.
    """

    def __init__(self, device="cpu", batch_size=None):
        super().__init__(device=device, batch_size=batch_size or torch.Size([]))
        self._obs = None
        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            shape=self.batch_size, device=self.device,
        )
        self.action_spec = Bounded(
            low=-1.0, high=1.0, shape=(1,),
            dtype=torch.float32, device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=(1,), dtype=torch.float32, device=self.device,
        )
        self.done_spec = Composite(
            done=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            shape=self.batch_size, device=self.device,
        )

    def _reset(self, tensordict=None):
        self._obs = torch.sign(torch.randn(1, device=self.device))
        return TensorDict({
            "observation": self._obs.clone(),
            "done": torch.tensor([False], device=self.device),
            "terminated": torch.tensor([False], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict):
        action = tensordict["action"]
        # reward = obs * sign(action): correct action matches obs sign
        reward = (self._obs * torch.sign(action)).sum().unsqueeze(0)
        self._obs = torch.sign(torch.randn(1, device=self.device))
        return TensorDict({
            "observation": self._obs.clone(),
            "reward": reward.to(self.device),
            "done": torch.tensor([True], device=self.device),
            "terminated": torch.tensor([True], device=self.device),
            "truncated": torch.tensor([False], device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed):
        torch.manual_seed(seed if seed is not None else 0)


# Ordered list of all probes for validation
ALL_PROBES = [
    ("probe1_constant_value", ProbeEnv1),
    ("probe2_obs_dependent_value", ProbeEnv2),
    ("probe3_reward_discounting", ProbeEnv3),
    ("probe4_policy_gradient", ProbeEnv4),
    ("probe5_joint_policy_value", ProbeEnv5),
]
