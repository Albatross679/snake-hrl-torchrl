"""DDPG (Deep Deterministic Policy Gradient) trainer using TorchRL."""

import time
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from trainers.logging_utils import compute_grad_norm, log_system_metrics

from torchrl.envs import EnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm

from configs.training import DDPGConfig
from configs.network import NetworkConfig
from configs.base import resolve_device
from configs.run_dir import setup_run_dir
from networks.actor import ActorNetwork, get_activation
from networks.critic import QNetwork


class DeterministicActor(nn.Module):
    """Deterministic policy network for DDPG.

    Outputs tanh-squashed actions directly (no stochastic sampling).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list,
        activation: str = "relu",
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        super().__init__()
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev_dim, action_dim)

        # Initialize
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        nn.init.uniform_(self.output_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_head.bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.mlp(obs)
        raw = self.output_head(features)
        return torch.tanh(raw) * self.action_scale + self.action_bias


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.copy()


class DDPGTrainer:
    """DDPG trainer for continuous control tasks."""

    def __init__(
        self,
        env: EnvBase,
        config: Optional[DDPGConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
    ):
        self.env = env
        self.config = config or DDPGConfig()
        self.network_config = network_config or NetworkConfig()
        self.device = resolve_device(device)

        # Get dimensions from environment
        obs_dim = env.observation_spec["observation"].shape[-1]
        action_spec = env.action_spec
        action_dim = action_spec.shape[-1]
        action_low = float(action_spec.space.low.min())
        action_high = float(action_spec.space.high.max())

        # Create deterministic actor
        actor_hidden = self.network_config.actor.hidden_dims
        actor_activation = self.network_config.actor.activation
        self.actor = DeterministicActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden,
            activation=actor_activation,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)

        # Create target actor
        self.actor_target = DeterministicActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden,
            activation=actor_activation,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Create critic (single Q-network)
        critic_hidden = self.network_config.critic.hidden_dims
        critic_activation = self.network_config.critic.activation
        self.critic = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden,
            activation=critic_activation,
        ).to(self.device)

        # Create target critic
        self.critic_target = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden,
            activation=critic_activation,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.config.buffer_size, device=self.device),
            batch_size=self.config.batch_size,
        )

        # Exploration noise
        if self.config.noise_type == "ou":
            self.noise = OUNoise(
                action_dim=action_dim,
                theta=self.config.noise_theta,
                sigma=self.config.noise_sigma,
            )
        else:
            self.noise = None  # Gaussian noise applied inline

        self._action_dim = action_dim
        self._action_low = action_low
        self._action_high = action_high

        # Training state
        self.total_frames = 0
        self.total_episodes = 0
        self.best_reward = float("-inf")

        # Metric tracking
        self._train_start_time = 0.0
        self._episode_start_time = 0.0
        self._system_log_counter = [0]

        # Logging / output directories (auto-create consolidated run dir if not provided)
        if run_dir is None:
            run_dir = setup_run_dir(self.config)
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir
        self.save_dir = self.run_dir / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = self.run_dir / "tensorboard"

        # TensorBoard
        self.writer = None
        if self.config.tensorboard.enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                log_dir=str(tb_dir),
                flush_secs=self.config.tensorboard.flush_secs,
            )

    def _select_action(self, obs: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Select action with optional exploration noise."""
        with torch.no_grad():
            action = self.actor(obs)

        if add_noise:
            if self.config.noise_type == "ou":
                noise = torch.tensor(
                    self.noise.sample(), dtype=torch.float32, device=self.device
                )
            else:
                noise = torch.randn_like(action) * self.config.noise_sigma
            action = action + noise
            action = action.clamp(self._action_low, self._action_high)

        return action

    def train(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run DDPG training loop."""
        pbar = tqdm(total=self.config.total_frames, desc="Training")
        all_metrics = []
        metrics_cfg = self.config.tensorboard.metrics

        self._train_start_time = time.monotonic()
        self._episode_start_time = time.monotonic()
        max_wall_time = self.config.max_wall_time

        episode_reward = 0.0
        episode_length = 0
        td = self.env.reset()

        if self.noise is not None:
            self.noise.reset()

        while self.total_frames < self.config.total_frames:
            # Check wall-clock limit
            if max_wall_time is not None:
                elapsed = time.monotonic() - self._train_start_time
                if elapsed >= max_wall_time:
                    tqdm.write(
                        f"Wall-clock limit reached ({elapsed:.0f}s / {max_wall_time:.0f}s). "
                        f"Stopping at {self.total_frames} frames."
                    )
                    break
            obs = td["observation"]

            # Select action
            if self.total_frames < self.config.warmup_steps:
                action = torch.rand(self._action_dim, device=self.device)
                action = action * (self._action_high - self._action_low) + self._action_low
            else:
                action = self._select_action(obs)

            td["action"] = action

            # Environment step
            next_td = self.env.step(td)

            # Store transition
            self.replay_buffer.add(next_td)

            # Update counters
            episode_reward += next_td["next", "reward"].item()
            episode_length += 1
            self.total_frames += 1
            pbar.update(1)

            # Check episode end
            done = next_td["next", "done"].item()
            if done:
                self.total_episodes += 1
                metrics = {
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "total_frames": self.total_frames,
                }
                all_metrics.append(metrics)

                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.save_checkpoint("best")

                if self.total_episodes % self.config.log_interval == 0:
                    tqdm.write(
                        f"Episode {self.total_episodes}: "
                        f"reward={episode_reward:.2f}, "
                        f"length={episode_length}"
                    )

                if self.writer is not None:
                    if metrics_cfg.episode:
                        self.writer.add_scalar("episode/reward", episode_reward, self.total_frames)
                        self.writer.add_scalar("episode/length", episode_length, self.total_frames)
                        self.writer.add_scalar("episode/count", self.total_episodes, self.total_frames)
                    if metrics_cfg.timing:
                        wall = time.monotonic() - self._train_start_time
                        ep_time = time.monotonic() - self._episode_start_time
                        self.writer.add_scalar("timing/wall_clock_secs", wall, self.total_frames)
                        self.writer.add_scalar("timing/episode_time_secs", ep_time, self.total_frames)
                    if metrics_cfg.system:
                        log_system_metrics(
                            self.writer,
                            self.total_frames,
                            self.device,
                            self._system_log_counter,
                            metrics_cfg.system_interval,
                        )

                self._episode_start_time = time.monotonic()
                td = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                if self.noise is not None:
                    self.noise.reset()

                if callback:
                    callback(metrics)
            else:
                td = next_td["next"]

            # Training updates
            if (
                self.total_frames >= self.config.warmup_steps
                and self.total_frames % self.config.update_frequency == 0
            ):
                update_metrics = {}
                for _ in range(self.config.num_updates):
                    update_metrics = self._update()

                if self.writer is not None:
                    if metrics_cfg.train:
                        for k in ("critic_loss", "actor_loss"):
                            if k in update_metrics:
                                self.writer.add_scalar(f"train/{k}", update_metrics[k], self.total_frames)
                    if metrics_cfg.gradients:
                        for k in ("actor_grad_norm", "critic_grad_norm"):
                            if k in update_metrics:
                                self.writer.add_scalar(f"gradients/{k}", update_metrics[k], self.total_frames)
                    if metrics_cfg.q_values:
                        for k in ("q_mean", "q_max", "q_min"):
                            if k in update_metrics:
                                self.writer.add_scalar(f"q_values/{k}", update_metrics[k], self.total_frames)

            # Save checkpoint
            if self.total_frames % (self.config.save_interval * 1000) == 0:
                self.save_checkpoint(f"step_{self.total_frames}")

        pbar.close()
        self.save_checkpoint("final")

        if self.writer is not None:
            self.writer.close()

        return {
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "metrics": all_metrics,
        }

    def _update(self) -> Dict[str, float]:
        """Perform single DDPG update."""
        batch = self.replay_buffer.sample()
        obs = batch["observation"]
        action = batch["action"]
        reward = batch["next", "reward"]
        next_obs = batch["next", "observation"]
        done = batch["next", "done"].float()

        # Update critic
        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_action)
            target_value = reward + (1 - done) * self.config.gamma * target_q

        current_q = self.critic(obs, action)
        critic_loss = nn.functional.mse_loss(current_q, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        metrics_cfg = self.config.tensorboard.metrics
        critic_grad_norm = (
            compute_grad_norm(self.critic) if metrics_cfg.gradients else None
        )

        self.critic_optimizer.step()

        # Update actor
        predicted_action = self.actor(obs)
        actor_loss = -self.critic(obs, predicted_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        actor_grad_norm = (
            compute_grad_norm(self.actor) if metrics_cfg.gradients else None
        )

        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        result = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

        if metrics_cfg.gradients:
            result["critic_grad_norm"] = critic_grad_norm
            result["actor_grad_norm"] = actor_grad_norm

        if metrics_cfg.q_values:
            result["q_mean"] = current_q.mean().item()
            result["q_max"] = current_q.max().item()
            result["q_min"] = current_q.min().item()

        return result

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update of target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "config": self.config,
        }
        path = self.save_dir / f"{name}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_frames = checkpoint["total_frames"]
        self.total_episodes = checkpoint["total_episodes"]
        self.best_reward = checkpoint["best_reward"]

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy (deterministic, no noise)."""
        rewards = []
        lengths = []

        for _ in range(num_episodes):
            td = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            done = False
            while not done:
                obs = td["observation"]
                action = self._select_action(obs, add_noise=False)
                td["action"] = action
                td = self.env.step(td)
                episode_reward += td["next", "reward"].item()
                episode_length += 1
                done = td["next", "done"].item()
                td = td["next"]

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }
