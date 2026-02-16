"""SAC (Soft Actor-Critic) trainer using TorchRL."""

from typing import Optional, Dict, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from torchrl.envs import EnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm

from configs.training import SACConfig
from configs.network import NetworkConfig
from networks.actor import create_actor
from networks.critic import TwinQNetwork


class SACTrainer:
    """SAC trainer for continuous control tasks."""

    def __init__(
        self,
        env: EnvBase,
        config: Optional[SACConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cpu",
    ):
        """Initialize SAC trainer.

        Args:
            env: TorchRL environment
            config: Training configuration
            network_config: Network architecture configuration
            device: Device for training
        """
        self.env = env
        self.config = config or SACConfig()
        self.network_config = network_config or NetworkConfig()
        self.device = device

        # Get dimensions from environment
        obs_dim = env.observation_spec["observation"].shape[-1]
        action_spec = env.action_spec
        action_dim = action_spec.shape[-1]

        # Create actor
        self.actor = create_actor(
            obs_dim=obs_dim,
            action_spec=action_spec,
            config=self.network_config.actor,
            device=device,
        )

        # Create twin Q-networks
        self.critic = TwinQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.network_config.critic.hidden_dims,
            activation=self.network_config.critic.activation,
        ).to(device)

        # Create target Q-networks
        self.critic_target = TwinQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.network_config.critic.hidden_dims,
            activation=self.network_config.critic.activation,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy coefficient
        if self.config.auto_alpha:
            # Learnable log_alpha
            self.log_alpha = torch.tensor(
                np.log(self.config.alpha), requires_grad=True, device=device
            )
            self.target_entropy = (
                self.config.target_entropy
                if self.config.target_entropy is not None
                else -action_dim
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.config.alpha), device=device)

        # Create optimizers
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.config.buffer_size, device=device),
            batch_size=self.config.batch_size,
        )

        # Training state
        self.total_frames = 0
        self.total_episodes = 0
        self.best_reward = float("-inf")
        self._update_count = 0

        # Logging
        self.log_dir = Path(self.config.log_dir) / self.config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(self.config.save_dir) / self.config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def alpha(self) -> torch.Tensor:
        """Get current entropy coefficient."""
        return self.log_alpha.exp()

    def train(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run SAC training loop.

        Args:
            callback: Optional callback function called after each episode

        Returns:
            Dictionary with training statistics
        """
        pbar = tqdm(total=self.config.total_frames, desc="Training")
        all_metrics = []

        episode_reward = 0.0
        episode_length = 0
        td = self.env.reset()

        while self.total_frames < self.config.total_frames:
            # Select action
            if self.total_frames < self.config.warmup_steps:
                # Random action during warmup
                action = torch.rand(self.env.action_spec.shape) * 2 - 1
                td["action"] = action.to(self.device)
            else:
                with torch.no_grad():
                    td = self.actor(td)

            # Environment step
            next_td = self.env.step(td)

            # Store transition
            self.replay_buffer.add(next_td)

            # Update counters
            episode_reward += next_td["reward"].item()
            episode_length += 1
            self.total_frames += 1
            pbar.update(1)

            # Check episode end
            if next_td["done"].item():
                self.total_episodes += 1
                metrics = {
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "total_frames": self.total_frames,
                }
                all_metrics.append(metrics)

                # Track best reward
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.save_checkpoint("best")

                # Log
                if self.total_episodes % self.config.log_interval == 0:
                    tqdm.write(
                        f"Episode {self.total_episodes}: "
                        f"reward={episode_reward:.2f}, "
                        f"length={episode_length}"
                    )

                # Reset
                td = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

                if callback:
                    callback(metrics)
            else:
                td = next_td

            # Training updates
            if (
                self.total_frames >= self.config.warmup_steps
                and self.total_frames % self.config.update_frequency == 0
            ):
                for _ in range(self.config.num_updates):
                    self._update()

            # Save checkpoint
            if self.total_frames % (self.config.save_interval * 1000) == 0:
                self.save_checkpoint(f"step_{self.total_frames}")

        pbar.close()
        self.save_checkpoint("final")

        return {
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "metrics": all_metrics,
        }

    def _update(self) -> Dict[str, float]:
        """Perform single SAC update.

        Returns:
            Dictionary with loss metrics
        """
        # Sample batch
        batch = self.replay_buffer.sample()
        obs = batch["observation"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next"]["observation"]
        done = batch["done"].float()

        # Update critics
        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_td = TensorDict({"observation": next_obs}, batch_size=obs.shape[0])
            next_td = self.actor(next_td)
            next_action = next_td["action"]
            next_log_prob = next_td["sample_log_prob"]

            # Target Q-values
            q1_target, q2_target = self.critic_target(next_obs, next_action)
            q_target = torch.min(q1_target, q2_target)
            target_value = reward + (1 - done) * self.config.gamma * (
                q_target - self.alpha * next_log_prob.unsqueeze(-1)
            )

        # Current Q-values
        q1, q2 = self.critic(obs, action)
        critic_loss = nn.functional.mse_loss(q1, target_value) + nn.functional.mse_loss(
            q2, target_value
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor (less frequently)
        self._update_count += 1
        if self._update_count % self.config.actor_update_frequency == 0:
            # Get current actions and log probs
            td = TensorDict({"observation": obs}, batch_size=obs.shape[0])
            td = self.actor(td)
            new_action = td["action"]
            log_prob = td["sample_log_prob"]

            # Q-values for new actions
            q1_new, q2_new = self.critic(obs, new_action)
            q_new = torch.min(q1_new, q2_new)

            # Actor loss: maximize Q - alpha * log_prob
            actor_loss = (self.alpha * log_prob.unsqueeze(-1) - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update alpha
            if self.config.auto_alpha:
                alpha_loss = -(
                    self.log_alpha * (log_prob + self.target_entropy).detach()
                ).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        # Update target networks
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self) -> None:
        """Soft update of target networks."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "config": self.config,
        }

        if self.config.auto_alpha:
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        path = self.save_dir / f"{name}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.log_alpha = checkpoint["log_alpha"]
        self.total_frames = checkpoint["total_frames"]
        self.total_episodes = checkpoint["total_episodes"]
        self.best_reward = checkpoint["best_reward"]

        if self.config.auto_alpha and "alpha_optimizer_state_dict" in checkpoint:
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )

    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate current policy."""
        rewards = []
        lengths = []

        for _ in range(num_episodes):
            td = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            done = False
            while not done:
                with torch.no_grad():
                    if deterministic:
                        self.actor(td)
                        td["action"] = td["loc"]
                    else:
                        td = self.actor(td)

                td = self.env.step(td)
                episode_reward += td["reward"].item()
                episode_length += 1
                done = td["done"].item()

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }
