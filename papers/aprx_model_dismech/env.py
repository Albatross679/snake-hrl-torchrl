"""GPU-batched TorchRL environment using the trained neural surrogate (DisMech).

Replaces DisMech's implicit Euler integration with a single MLP forward
pass. Runs N environments simultaneously on GPU in a single EnvBase instance.

Observation (14 dims) and reward are computed identically to
the Elastica surrogate env, but using DisMech physics config.
"""

import math
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

try:
    from torchrl.data import (
        BoundedTensorSpec,
        CompositeSpec,
        UnboundedContinuousTensorSpec,
    )
except ImportError:
    from torchrl.data import (
        Bounded as BoundedTensorSpec,
        Composite as CompositeSpec,
        Unbounded as UnboundedContinuousTensorSpec,
    )
from torchrl.envs import EnvBase

from aprx_model_dismech.train_config import SurrogateEnvConfig, SurrogateModelConfig
from aprx_model_dismech.model import SurrogateModel
from aprx_model_dismech.state import (
    STATE_DIM,
    NUM_NODES,
    NUM_ELEMENTS,
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
    StateNormalizer,
    action_to_omega_batch,
    encode_phase_batch,
)


class SurrogateLocomotionEnv(EnvBase):
    """GPU-batched snake locomotion environment using a neural surrogate (DisMech).

    Observation (14 dims):
        [0-2]  Curvature modes (amplitude, wave_number, phase) via FFT
        [3-4]  Body heading (cos, sin)
        [5]    Yaw angular velocity
        [6-7]  CoM velocity in body frame (forward, lateral)
        [8]    Distance to goal
        [9]    Heading error to goal (radians)
        [10]   Velocity toward goal
        [11]   Forward speed (scalar)
        [12]   Energy rate
        [13]   Current turn bias

    Action (5 dims, [-1, 1]):
        [0] amplitude, [1] frequency, [2] wave_number, [3] phase, [4] turn_bias
    """

    OBS_DIM = 14

    def __init__(
        self,
        config: SurrogateEnvConfig,
        device: str = "cpu",
    ):
        self._batch = config.batch_size
        super().__init__(device=device, batch_size=torch.Size([self._batch]))
        self.config = config
        self._device = device

        # Physics constants from DisMech config
        physics = config.physics
        self._num_nodes = physics.geometry.num_nodes        # 21
        self._num_segments = physics.geometry.num_segments  # 20
        self._snake_length = physics.geometry.snake_length  # 0.5
        self._snake_radius = physics.geometry.snake_radius  # 0.02
        self._density = physics.density                     # 1200
        self._dt_rl = physics.dt  # 0.05s (single implicit step)

        # Segment mass (for energy calculation)
        seg_len = self._snake_length / self._num_segments
        self._seg_mass = self._density * math.pi * self._snake_radius ** 2 * seg_len

        # Load surrogate model
        self._model: Optional[SurrogateModel] = None
        self._normalizer: Optional[StateNormalizer] = None
        if config.surrogate_checkpoint:
            self._load_surrogate(config.surrogate_checkpoint)

        # Internal state tensors (initialized in _reset)
        self._rod_state = torch.zeros(self._batch, STATE_DIM, device=device)
        self._serpenoid_time = torch.zeros(self._batch, device=device)
        self._prev_com = torch.zeros(self._batch, 2, device=device)
        self._prev_heading_angle = torch.zeros(self._batch, device=device)
        self._prev_dist_to_goal = torch.full((self._batch,), 1.0, device=device)
        self._goal_xy = torch.zeros(self._batch, 2, device=device)
        self._step_count = torch.zeros(self._batch, dtype=torch.long, device=device)
        self._no_progress_count = torch.zeros(self._batch, dtype=torch.long, device=device)
        self._last_turn_bias = torch.zeros(self._batch, device=device)

        # RNG
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(42)

        self._make_spec()

    def _load_surrogate(self, checkpoint_dir: str):
        """Load trained surrogate model and normalizer."""
        from pathlib import Path
        import json

        ckpt_dir = Path(checkpoint_dir)
        config_path = ckpt_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = SurrogateModelConfig(**json.load(f))
        else:
            model_config = SurrogateModelConfig()

        self._model = SurrogateModel(model_config).to(self._device)
        model_path = ckpt_dir / "model.pt"
        self._model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        self._model.eval()

        norm_path = ckpt_dir / "normalizer.pt"
        self._normalizer = StateNormalizer.load(str(norm_path), device=self._device)

    def _make_spec(self):
        """Define observation, action, reward, and done specs."""
        B = self._batch
        _scalar = lambda dtype=torch.float32: UnboundedContinuousTensorSpec(
            shape=(B, 1), dtype=dtype, device=self._device
        )

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(B, self.OBS_DIM), dtype=torch.float32, device=self._device
            ),
            shape=(B,),
        )

        self.action_spec = BoundedTensorSpec(
            low=-1.0, high=1.0,
            shape=(B, 5),
            dtype=torch.float32,
            device=self._device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(B, 1), dtype=torch.float32, device=self._device
        )

        self.done_spec = CompositeSpec(
            done=UnboundedContinuousTensorSpec(
                shape=(B, 1), dtype=torch.bool, device=self._device
            ),
            terminated=UnboundedContinuousTensorSpec(
                shape=(B, 1), dtype=torch.bool, device=self._device
            ),
            truncated=UnboundedContinuousTensorSpec(
                shape=(B, 1), dtype=torch.bool, device=self._device
            ),
            shape=(B,),
        )

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        B = self._batch
        _false = torch.zeros(B, 1, dtype=torch.bool, device=self._device)
        obs = torch.zeros(B, self.OBS_DIM, device=self._device)

        return TensorDict(
            {
                "observation": obs,
                "done": _false.clone(),
                "terminated": _false.clone(),
                "truncated": _false.clone(),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict["action"]
        B = self._batch

        # Surrogate forward pass
        omega = action_to_omega_batch(action)
        time_enc = encode_phase_batch(omega * self._serpenoid_time)
        with torch.no_grad():
            self._rod_state = self._model.predict_next_state(
                self._rod_state, action, time_enc, self._normalizer
            )
        self._serpenoid_time = self._serpenoid_time + self._dt_rl
        self._step_count = self._step_count + 1

        obs = torch.zeros(B, self.OBS_DIM, device=self._device)
        reward = torch.zeros(B, 1, device=self._device)
        truncated = self._step_count >= self.config.max_episode_steps
        _false = torch.zeros(B, 1, dtype=torch.bool, device=self._device)

        return TensorDict(
            {
                "observation": obs,
                "reward": reward,
                "done": truncated.unsqueeze(-1),
                "terminated": _false,
                "truncated": truncated.unsqueeze(-1),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def close(self, **kwargs):
        self._model = None
        self._normalizer = None
        super().close(**kwargs)
