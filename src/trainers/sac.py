"""SAC (Soft Actor-Critic) trainer using TorchRL."""

import time
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

from src.configs.training import SACConfig
from src.configs.network import NetworkConfig
from src.configs.base import resolve_device
from src.configs.run_dir import setup_run_dir
from src.networks.actor import create_actor
from src.networks.critic import TwinQNetwork
from .logging_utils import collect_system_metrics, compute_grad_norm


class SACTrainer:
    """SAC trainer for continuous control tasks."""

    def __init__(
        self,
        env: EnvBase,
        config: Optional[SACConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
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
        self.device = resolve_device(device)

        # Get dimensions from environment
        obs_dim = env.observation_spec["observation"].shape[-1]
        action_spec = env.action_spec
        action_dim = action_spec.shape[-1]

        # Create actor
        self.actor = create_actor(
            obs_dim=obs_dim,
            action_spec=action_spec,
            config=self.network_config.actor,
            device=self.device,
        )

        # Create twin Q-networks
        self.critic = TwinQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.network_config.critic.hidden_dims,
            activation=self.network_config.critic.activation,
        ).to(self.device)

        # Create target Q-networks
        self.critic_target = TwinQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.network_config.critic.hidden_dims,
            activation=self.network_config.critic.activation,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy coefficient
        if self.config.auto_alpha:
            # Learnable log_alpha
            self.log_alpha = torch.tensor(
                np.log(self.config.alpha), requires_grad=True, device=self.device
            )
            self.target_entropy = (
                self.config.target_entropy
                if self.config.target_entropy is not None
                else -action_dim
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.config.alpha), device=self.device)

        # Create optimizers
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.config.buffer_size, device=self.device),
            batch_size=self.config.batch_size,
        )

        # Training state
        self.total_frames = 0
        self.total_episodes = 0
        self.best_reward = float("-inf")
        self._update_count = 0

        # Metric tracking
        self._train_start_time = 0.0
        self._system_log_counter = [0]

        # Logging / output directories (auto-create consolidated run dir if not provided)
        if run_dir is None:
            run_dir = setup_run_dir(self.config)
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir
        self.save_dir = self.run_dir / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Weights & Biases
        self.wandb_run = None
        if self.config.wandb.enabled:
            import wandb
            from dataclasses import asdict

            wandb_cfg = self.config.wandb
            self.wandb_run = wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity or None,
                group=wandb_cfg.group or None,
                tags=wandb_cfg.tags or None,
                name=self.config.name,
                config=asdict(self.config),
                dir=str(self.run_dir),
            )

    @property
    def alpha(self) -> torch.Tensor:
        """Get current entropy coefficient."""
        return self.log_alpha.exp()

    def train(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run SAC training loop.

        Supports both single-env and vectorized (batched) environments.
        When the environment has a non-empty batch_size, transitions are
        stored with ``extend`` and per-env episode stats are tracked.

        Args:
            callback: Optional callback function called after each episode

        Returns:
            Dictionary with training statistics
        """
        # Detect vectorized env
        is_vec = len(self.env.batch_size) > 0
        num_envs = self.env.batch_size[0] if is_vec else 1

        pbar = tqdm(total=self.config.total_frames, desc="Training")
        all_metrics = []

        # Wall-clock limit
        start_time = time.monotonic()
        self._train_start_time = start_time
        max_wall_time = self.config.max_wall_time

        # Per-env episode accumulators
        episode_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs, dtype=int)

        td = self.env.reset()

        while self.total_frames < self.config.total_frames:
            # Check wall-clock limit
            if max_wall_time is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= max_wall_time:
                    tqdm.write(
                        f"Wall-clock limit reached ({elapsed:.0f}s / {max_wall_time:.0f}s). "
                        f"Stopping at {self.total_frames} frames."
                    )
                    break
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

            # Store transition(s)
            if is_vec:
                self.replay_buffer.extend(next_td)
            else:
                self.replay_buffer.add(next_td)

            # Update counters
            frames_this_step = num_envs
            self.total_frames += frames_this_step
            pbar.update(frames_this_step)

            # TorchRL nests step results under "next"
            step_result = next_td["next"] if "next" in next_td.keys() else next_td

            if is_vec:
                # Vectorized: per-env episode tracking
                rewards = step_result["reward"].cpu().numpy().reshape(num_envs)
                dones = step_result["done"].cpu().numpy().reshape(num_envs)
                episode_rewards += rewards
                episode_lengths += 1

                # Handle completed episodes
                done_mask = dones.astype(bool)
                if done_mask.any():
                    for i in np.where(done_mask)[0]:
                        self.total_episodes += 1
                        metrics = {
                            "episode_reward": float(episode_rewards[i]),
                            "episode_length": int(episode_lengths[i]),
                            "total_frames": self.total_frames,
                        }
                        all_metrics.append(metrics)

                        if episode_rewards[i] > self.best_reward:
                            self.best_reward = float(episode_rewards[i])
                            self.save_checkpoint("best")

                        if self.total_episodes % self.config.log_interval == 0:
                            tqdm.write(
                                f"Episode {self.total_episodes}: "
                                f"reward={episode_rewards[i]:.2f}, "
                                f"length={episode_lengths[i]}"
                            )

                        self._log_episode_metrics(episode_rewards[i], episode_lengths[i])

                        if callback:
                            callback(metrics)

                    # Reset accumulators for done envs
                    episode_rewards[done_mask] = 0.0
                    episode_lengths[done_mask] = 0

                # TorchRL SerialEnv auto-resets; use next_td for next step
                td = next_td
            else:
                # Single env path (original logic)
                episode_rewards[0] += step_result["reward"].item()
                episode_lengths[0] += 1

                if step_result["done"].item():
                    self.total_episodes += 1
                    metrics = {
                        "episode_reward": float(episode_rewards[0]),
                        "episode_length": int(episode_lengths[0]),
                        "total_frames": self.total_frames,
                    }
                    all_metrics.append(metrics)

                    if episode_rewards[0] > self.best_reward:
                        self.best_reward = float(episode_rewards[0])
                        self.save_checkpoint("best")

                    if self.total_episodes % self.config.log_interval == 0:
                        tqdm.write(
                            f"Episode {self.total_episodes}: "
                            f"reward={episode_rewards[0]:.2f}, "
                            f"length={episode_lengths[0]}"
                        )

                    self._log_episode_metrics(episode_rewards[0], episode_lengths[0])

                    td = self.env.reset()
                    episode_rewards[0] = 0.0
                    episode_lengths[0] = 0

                    if callback:
                        callback(metrics)
                else:
                    td = step_result

            # Training updates
            if (
                self.total_frames >= self.config.warmup_steps
                and self.total_frames % self.config.update_frequency == 0
            ):
                update_metrics = {}
                for _ in range(self.config.num_updates):
                    update_metrics = self._update()

                self._log_train_metrics(update_metrics)

            # Save checkpoint
            if self.total_frames % (self.config.save_interval * 1000) == 0:
                self.save_checkpoint(f"step_{self.total_frames}")

        pbar.close()
        self.save_checkpoint("final")

        # Close W&B run
        if self.wandb_run is not None:
            self.wandb_run.finish()

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
            next_log_prob = next_td["action_log_prob"]

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
        critic_grad_norm = compute_grad_norm(self.critic)
        self.critic_optimizer.step()

        # Update actor (less frequently)
        actor_loss = None
        alpha_loss = None
        actor_grad_norm = None
        self._update_count += 1
        if self._update_count % self.config.actor_update_frequency == 0:
            # Get current actions and log probs
            td = TensorDict({"observation": obs}, batch_size=obs.shape[0])
            td = self.actor(td)
            new_action = td["action"]
            log_prob = td["action_log_prob"]

            # Q-values for new actions
            q1_new, q2_new = self.critic(obs, new_action)
            q_new = torch.min(q1_new, q2_new)

            # Actor loss: maximize Q - alpha * log_prob
            actor_loss = (self.alpha * log_prob.unsqueeze(-1) - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = compute_grad_norm(self.actor)
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

        metrics = {
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha.item(),
            "critic_grad_norm": critic_grad_norm,
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }
        if actor_loss is not None:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_grad_norm"] = actor_grad_norm
        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss.item()
        return metrics

    def _soft_update(self) -> None:
        """Soft update of target networks."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def _log_episode_metrics(self, episode_reward: float, episode_length: int) -> None:
        """Log episode metrics to W&B."""
        if self.wandb_run is None:
            return
        self.wandb_run.log({
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/count": self.total_episodes,
        }, step=self.total_frames)

    def _log_train_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training update metrics to W&B."""
        if self.wandb_run is None:
            return
        wandb_log = {}
        for key in ("critic_loss", "actor_loss", "alpha", "alpha_loss"):
            if key in metrics:
                wandb_log[f"train/{key}"] = metrics[key]
        for key in ("q1_mean", "q2_mean"):
            if key in metrics:
                wandb_log[f"q_values/{key}"] = metrics[key]
        for key in ("critic_grad_norm", "actor_grad_norm"):
            if key in metrics:
                wandb_log[f"gradients/{key}"] = metrics[key]
        wall = time.monotonic() - self._train_start_time
        wandb_log["timing/wall_clock_mins"] = wall / 60.0

        sys_metrics = collect_system_metrics(
            self.device, self._system_log_counter, 10,
        )
        wandb_log.update(sys_metrics)

        self.wandb_run.log(wandb_log, step=self.total_frames)

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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

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
                step_result = td["next"] if "next" in td.keys() else td
                episode_reward += step_result["reward"].item()
                episode_length += 1
                done = step_result["done"].item()
                td = step_result

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }
