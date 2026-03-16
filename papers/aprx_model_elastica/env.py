"""GPU-batched TorchRL environment using the trained neural surrogate.

Replaces PyElastica's 500 integration substeps with a single MLP forward
pass. Runs N environments simultaneously on GPU in a single EnvBase instance.

Observation (14 dims) and reward are computed identically to
LocomotionElasticaEnv, but reimplemented in vectorized PyTorch.
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

from aprx_model_elastica.train_config import SurrogateEnvConfig, SurrogateModelConfig
from aprx_model_elastica.model import SurrogateModel
from aprx_model_elastica.state import (
    STATE_DIM,
    NUM_NODES,
    NUM_ELEMENTS,
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
    StateNormalizer,
    action_to_omega_batch,
    encode_phase_batch,
)


class SurrogateLocomotionEnv(EnvBase):
    """GPU-batched snake locomotion environment using a neural surrogate.

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

        # Physics constants
        physics = config.physics
        self._num_nodes = physics.geometry.num_nodes        # 21
        self._num_segments = physics.geometry.num_segments  # 20
        self._snake_length = physics.geometry.snake_length  # 0.5
        self._snake_radius = physics.geometry.snake_radius  # 0.02
        self._density = physics.density                     # 1200
        self._dt_rl = physics.dt_substep * config.control.substeps_per_action  # 0.5s

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
        self._prev_dist_to_goal = torch.full((self._batch,), config.goal.goal_distance, device=device)
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

        # Load model config
        config_path = ckpt_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = SurrogateModelConfig(**json.load(f))
        else:
            model_config = SurrogateModelConfig()

        # Load model
        self._model = SurrogateModel(model_config).to(self._device)
        model_path = ckpt_dir / "model.pt"
        self._model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        self._model.eval()

        # Load normalizer
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
            v_g=_scalar(),
            dist_to_goal=_scalar(),
            theta_g=_scalar(),
            reward_dist=_scalar(),
            reward_align=_scalar(),
            episode_wall_time_s=_scalar(),
            final_dist_to_goal=_scalar(),
            goal_reached=_scalar(torch.bool),
            starvation=_scalar(torch.bool),
            shape=(B,),
        )

        self.action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(B, self.config.control.action_dim),
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

    # === Initial state generation ===

    def _generate_initial_state(self, mask: Optional[torch.Tensor] = None) -> None:
        """Generate straight-rod initial state for environments indicated by mask.

        Args:
            mask: (B,) bool tensor. If None, reset all environments.
        """
        if mask is None:
            mask = torch.ones(self._batch, dtype=torch.bool, device=self._device)

        n_reset = mask.sum().item()
        if n_reset == 0:
            return

        # Random heading angle
        if self.config.randomize_initial_heading:
            theta = torch.rand(n_reset, device=self._device, generator=self._rng) * 2 * math.pi
        else:
            theta = torch.zeros(n_reset, device=self._device)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Node positions along direction: node_i at (i/20 - 0.5) * L * direction
        t_vals = torch.linspace(0, 1, self._num_nodes, device=self._device)  # (21,)
        t_centered = t_vals - 0.5  # centered at 0

        # positions: (n_reset, 21) for x and y
        pos_x = t_centered.unsqueeze(0) * self._snake_length * cos_t.unsqueeze(1)  # (n, 21)
        pos_y = t_centered.unsqueeze(0) * self._snake_length * sin_t.unsqueeze(1)  # (n, 21)

        # Build state vector
        state = torch.zeros(n_reset, STATE_DIM, device=self._device)
        state[:, POS_X] = pos_x
        state[:, POS_Y] = pos_y
        # velocities = 0 (already zeros)
        # yaw = theta for all elements
        state[:, YAW] = theta.unsqueeze(1).expand(-1, NUM_ELEMENTS)
        # omega_z = 0 (already zeros)

        self._rod_state[mask] = state
        self._serpenoid_time[mask] = 0.0

        # CoM = mean of positions
        com_x = pos_x.mean(dim=1)
        com_y = pos_y.mean(dim=1)
        com = torch.stack([com_x, com_y], dim=-1)
        self._prev_com[mask] = com
        self._prev_heading_angle[mask] = theta

        # Goal placement
        goal_dist = self.config.goal.goal_distance
        goal = com + goal_dist * torch.stack([cos_t, sin_t], dim=-1)
        self._goal_xy[mask] = goal
        self._prev_dist_to_goal[mask] = goal_dist

        self._step_count[mask] = 0
        self._no_progress_count[mask] = 0
        self._last_turn_bias[mask] = 0.0

    # === Vectorized observation extraction ===

    def _compute_obs_batch(self) -> torch.Tensor:
        """Compute 14-dim observation for all environments. Returns (B, 14)."""
        state = self._rod_state

        # Positions and velocities
        pos_x = state[:, POS_X]  # (B, 21)
        pos_y = state[:, POS_Y]  # (B, 21)
        vel_x = state[:, VEL_X]  # (B, 21)
        vel_y = state[:, VEL_Y]  # (B, 21)

        # 1. Curvature modes via FFT (3)
        curv_modes = self._curvature_modes_batch(pos_x, pos_y)  # (B, 3)

        # 2. Body heading (cos, sin) (2)
        head_x = pos_x[:, -1] - pos_x[:, 0]
        head_y = pos_y[:, -1] - pos_y[:, 0]
        head_norm = torch.sqrt(head_x ** 2 + head_y ** 2).clamp(min=1e-8)
        heading_cos = head_x / head_norm
        heading_sin = head_y / head_norm

        # 3. Yaw angular velocity (1)
        heading_angle = torch.atan2(heading_sin, heading_cos)
        delta_angle = heading_angle - self._prev_heading_angle
        # Wrap to [-pi, pi]
        delta_angle = (delta_angle + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = delta_angle / self._dt_rl

        # 4. CoM velocity in body frame (forward, lateral) (2)
        com_x = pos_x.mean(dim=1)
        com_y = pos_y.mean(dim=1)
        com = torch.stack([com_x, com_y], dim=-1)  # (B, 2)
        v_com = (com - self._prev_com) / self._dt_rl  # (B, 2)
        heading_vec = torch.stack([heading_cos, heading_sin], dim=-1)  # (B, 2)
        lateral_vec = torch.stack([-heading_sin, heading_cos], dim=-1)  # (B, 2)
        v_forward = (v_com * heading_vec).sum(dim=-1)
        v_lateral = (v_com * lateral_vec).sum(dim=-1)

        # 5. Distance to goal (1)
        to_goal = self._goal_xy - com  # (B, 2)
        dist_to_goal = torch.norm(to_goal, dim=-1)  # (B,)

        # 6. Heading error to goal (1)
        goal_angle = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        theta_g = goal_angle - heading_angle
        theta_g = (theta_g + math.pi) % (2 * math.pi) - math.pi

        # 7. Velocity toward goal (1)
        goal_unit = to_goal / dist_to_goal.clamp(min=1e-6).unsqueeze(-1)
        v_g = (v_com * goal_unit).sum(dim=-1)

        # 8. Forward speed (1)
        speed = torch.norm(v_com, dim=-1)

        # 9. Energy rate (1)
        ke = 0.5 * self._seg_mass * (vel_x ** 2 + vel_y ** 2).sum(dim=1)
        ref = self._seg_mass * self._num_nodes * 0.1 ** 2
        energy_rate = ke / ref

        # 10. Turn bias (1)
        turn_bias = self._last_turn_bias

        obs = torch.stack([
            curv_modes[:, 0], curv_modes[:, 1], curv_modes[:, 2],
            heading_cos, heading_sin,
            yaw_rate,
            v_forward, v_lateral,
            dist_to_goal,
            theta_g,
            v_g,
            speed,
            energy_rate,
            turn_bias,
        ], dim=-1)  # (B, 14)

        return obs, com, heading_angle, dist_to_goal, theta_g, v_g

    def _curvature_modes_batch(
        self, pos_x: torch.Tensor, pos_y: torch.Tensor
    ) -> torch.Tensor:
        """Extract curvature modes (amplitude, wave_number, phase) via FFT.

        Args:
            pos_x: (B, 21) x positions.
            pos_y: (B, 21) y positions.

        Returns:
            (B, 3) tensor of [amplitude, wave_number, phase].
        """
        B = pos_x.shape[0]

        # Compute discrete curvatures at internal nodes (19 values)
        # v1[i] = pos[i] - pos[i-1], v2[i] = pos[i+1] - pos[i]
        dx1 = pos_x[:, 1:-1] - pos_x[:, :-2]  # (B, 19)
        dy1 = pos_y[:, 1:-1] - pos_y[:, :-2]
        dx2 = pos_x[:, 2:] - pos_x[:, 1:-1]
        dy2 = pos_y[:, 2:] - pos_y[:, 1:-1]

        # Cross product z-component (signed angle)
        cross_z = dx1 * dy2 - dy1 * dx2  # (B, 19)

        # Norms
        norm1 = torch.sqrt(dx1 ** 2 + dy1 ** 2).clamp(min=1e-8)
        norm2 = torch.sqrt(dx2 ** 2 + dy2 ** 2).clamp(min=1e-8)

        # cos(angle) and angle
        dot = dx1 * dx2 + dy1 * dy2
        cos_angle = (dot / (norm1 * norm2)).clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)

        # Signed curvature
        avg_len = (norm1 + norm2) / 2
        kappa = torch.sign(cross_z) * angle / avg_len  # (B, 19)

        # FFT
        fft = torch.fft.rfft(kappa, dim=-1)  # (B, 10)
        magnitudes = torch.abs(fft)
        n = kappa.shape[-1]  # 19

        # Dominant frequency (skip DC component)
        dominant_idx = 1 + torch.argmax(magnitudes[:, 1:], dim=-1)  # (B,)

        # Extract modes for each batch element
        amplitude = 2.0 * magnitudes[torch.arange(B), dominant_idx] / n
        wave_number = dominant_idx.float() / n * (2 * math.pi)
        phase = torch.angle(fft[torch.arange(B), dominant_idx])

        return torch.stack([amplitude, wave_number, phase], dim=-1)  # (B, 3)

    # === EnvBase interface ===

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        # Determine which envs to reset
        if tensordict is not None and "_reset" in tensordict.keys():
            mask = tensordict["_reset"].squeeze(-1)
        else:
            mask = None  # reset all

        self._generate_initial_state(mask)

        obs, com, heading_angle, dist_to_goal, theta_g, v_g = self._compute_obs_batch()

        # Update cached state for next step
        self._prev_com = com
        self._prev_heading_angle = heading_angle

        B = self._batch
        _zero = torch.zeros(B, 1, dtype=torch.float32, device=self._device)
        _false = torch.zeros(B, 1, dtype=torch.bool, device=self._device)

        return TensorDict(
            {
                "observation": obs,
                "done": _false.clone(),
                "terminated": _false.clone(),
                "truncated": _false.clone(),
                "v_g": _zero.clone(),
                "dist_to_goal": dist_to_goal.unsqueeze(-1),
                "theta_g": _zero.clone(),
                "reward_dist": _zero.clone(),
                "reward_align": _zero.clone(),
                "episode_wall_time_s": _zero.clone(),
                "final_dist_to_goal": _zero.clone(),
                "goal_reached": _false.clone(),
                "starvation": _false.clone(),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict["action"]  # (B, 5)

        # Store turn bias for observation
        # Denormalize turn_bias: [-1, 1] → turn_bias_range
        tb_range = self.config.control.turn_bias_range
        tb_norm = (action[:, 4] + 1) / 2
        self._last_turn_bias = tb_range[0] + tb_norm * (tb_range[1] - tb_range[0])

        # Record pre-step state
        obs_pre, com_pre, heading_angle_pre, _, _, _ = self._compute_obs_batch()
        self._prev_com = com_pre
        self._prev_heading_angle = heading_angle_pre

        # Surrogate forward pass — encode oscillation phase omega*t
        freq_range = self.config.control.frequency_range
        omega = action_to_omega_batch(action, freq_range)
        time_enc = encode_phase_batch(omega * self._serpenoid_time)
        with torch.no_grad():
            self._rod_state = self._model.predict_next_state(
                self._rod_state, action, time_enc, self._normalizer
            )

        # Update serpenoid time
        self._serpenoid_time = self._serpenoid_time + self._dt_rl

        # Post-step observation
        obs, com, heading_angle, dist_to_goal, theta_g, v_g = self._compute_obs_batch()

        # Reward
        c = self.config.rewards
        reward_dist = c.c_dist * (self._prev_dist_to_goal - dist_to_goal)
        reward_align = c.c_align * torch.cos(theta_g)
        goal_reached = dist_to_goal <= self.config.goal.goal_radius
        goal_bonus = torch.where(goal_reached, c.goal_bonus, 0.0)
        reward = reward_dist + reward_align + goal_bonus
        reward = torch.nan_to_num(reward, nan=0.0)

        # Starvation tracking
        no_progress = dist_to_goal >= self._prev_dist_to_goal
        self._no_progress_count = torch.where(
            no_progress,
            self._no_progress_count + 1,
            torch.zeros_like(self._no_progress_count),
        )

        # Update cached state
        self._prev_dist_to_goal = dist_to_goal
        self._prev_com = com
        self._prev_heading_angle = heading_angle
        self._step_count = self._step_count + 1

        # Termination
        starvation = self._no_progress_count >= self.config.goal.starvation_timeout
        terminated = goal_reached | starvation
        truncated = self._step_count >= self.config.max_episode_steps
        done = terminated | truncated

        return TensorDict(
            {
                "observation": obs,
                "reward": reward.unsqueeze(-1),
                "done": done.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),
                "v_g": v_g.unsqueeze(-1),
                "dist_to_goal": dist_to_goal.unsqueeze(-1),
                "theta_g": theta_g.unsqueeze(-1),
                "reward_dist": reward_dist.unsqueeze(-1),
                "reward_align": reward_align.unsqueeze(-1),
                "episode_wall_time_s": torch.zeros(self._batch, 1, device=self._device),
                "final_dist_to_goal": torch.where(
                    done, dist_to_goal, torch.zeros_like(dist_to_goal)
                ).unsqueeze(-1),
                "goal_reached": goal_reached.unsqueeze(-1),
                "starvation": starvation.unsqueeze(-1),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def close(self, **kwargs):
        self._model = None
        self._normalizer = None
        super().close(**kwargs)
