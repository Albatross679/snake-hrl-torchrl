"""PPO (Proximal Policy Optimization) trainer using TorchRL."""

import time
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import os
import signal
import tempfile
import shutil
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from trainers.logging_utils import log_system_metrics

from torchrl.envs import EnvBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm

from configs.training import PPOConfig
from configs.network import NetworkConfig
from configs.base import resolve_device
from configs.run_dir import setup_run_dir
from networks.actor import create_actor
from networks.critic import create_critic


class PPOTrainer:
    """PPO trainer for continuous control tasks."""

    def __init__(
        self,
        env: EnvBase,
        config: Optional[PPOConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
    ):
        """Initialize PPO trainer.

        Args:
            env: TorchRL environment
            config: Training configuration
            network_config: Network architecture configuration
            device: Device for training
        """
        self.env = env
        self.config = config or PPOConfig()
        self.network_config = network_config or NetworkConfig()
        self.device = resolve_device(device)

        # Get dimensions from environment
        obs_dim = env.observation_spec["observation"].shape[-1]
        action_spec = env.action_spec

        # Create actor and critic
        self.actor = create_actor(
            obs_dim=obs_dim,
            action_spec=action_spec,
            config=self.network_config.actor,
            device=self.device,
        )

        self.critic = create_critic(
            obs_dim=obs_dim,
            config=self.network_config.critic,
            device=self.device,
        )

        # Create PPO loss module
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.config.clip_epsilon,
            entropy_coef=self.config.entropy_coef,
            critic_coef=self.config.value_coef,
            normalize_advantage=self.config.normalize_advantage,
        )

        # Create GAE module for advantage estimation
        self.advantage_module = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.gae_lambda,
            value_network=self.critic,
        )

        # Create optimizer
        self.optimizer = Adam(
            self.loss_module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        if self.config.lr_schedule == "linear":
            total_updates = self.config.total_frames // self.config.frames_per_batch
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(
                    self.config.lr_end / self.config.learning_rate,
                    1.0 - step / total_updates,
                ),
            )
        else:
            self.scheduler = None

        # Create data collector
        self.collector = SyncDataCollector(
            create_env_fn=lambda: env,
            policy=self.actor,
            frames_per_batch=self.config.frames_per_batch,
            total_frames=self.config.total_frames,
            device=self.device,
        )

        # Training state
        self.total_frames = 0
        self.total_episodes = 0
        self.best_reward = float("-inf")

        # Metric tracking
        self._train_start_time = 0.0
        self._batch_start_time = 0.0
        self._system_log_counter = [0]

        # Graceful shutdown handling
        self._shutdown_requested = False
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)

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

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        signal_name = signal.Signals(signum).name
        print(f"\n{signal_name} received, requesting graceful shutdown...")
        self._shutdown_requested = True

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    def train(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run PPO training loop.

        Args:
            callback: Optional callback function called after each batch

        Returns:
            Dictionary with training statistics
        """
        pbar = tqdm(total=self.config.total_frames, desc="Training")
        all_metrics = []
        metrics_cfg = self.config.tensorboard.metrics

        self._train_start_time = time.monotonic()
        self._batch_start_time = time.monotonic()
        max_wall_time = self.config.max_wall_time

        for batch_idx, batch in enumerate(self.collector):
            # Check for graceful shutdown
            if self._shutdown_requested:
                print("Shutdown requested, saving checkpoint...")
                self.save_checkpoint("interrupted")
                print("Checkpoint saved to 'interrupted.pt'. Exiting.")
                self._restore_signal_handlers()
                break

            # Check wall-clock limit
            if max_wall_time is not None:
                elapsed = time.monotonic() - self._train_start_time
                if elapsed >= max_wall_time:
                    tqdm.write(
                        f"Wall-clock limit reached ({elapsed:.0f}s / {max_wall_time:.0f}s). "
                        f"Stopping at {self.total_frames} frames."
                    )
                    break

            # Compute advantages
            with torch.no_grad():
                self.advantage_module(batch)

            # PPO update
            metrics = self._update(batch)
            metrics["batch_idx"] = batch_idx
            metrics["total_frames"] = self.total_frames

            # Update frame count
            self.total_frames += batch.numel()
            pbar.update(batch.numel())

            # Episode statistics from batch
            if "episode_reward" in batch.keys():
                episode_rewards = batch["episode_reward"][batch["done"].squeeze(-1)]
                if len(episode_rewards) > 0:
                    metrics["mean_episode_reward"] = episode_rewards.mean().item()
                    metrics["max_episode_reward"] = episode_rewards.max().item()
                    self.total_episodes += len(episode_rewards)

                    # Track best reward
                    if metrics["mean_episode_reward"] > self.best_reward:
                        self.best_reward = metrics["mean_episode_reward"]
                        self.save_checkpoint("best")

            if metrics_cfg.episode:
                metrics["total_episodes"] = self.total_episodes

            if metrics_cfg.timing:
                metrics["batch_time_secs"] = time.monotonic() - self._batch_start_time
            self._batch_start_time = time.monotonic()

            all_metrics.append(metrics)

            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._log_metrics(metrics)

            # Save checkpoint
            if batch_idx % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.total_frames}")

            # Callback
            if callback:
                callback(metrics)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

        pbar.close()

        # Final save
        self.save_checkpoint("final")

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        return {
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "metrics": all_metrics,
        }

    def _update(self, batch: TensorDict) -> Dict[str, float]:
        """Perform PPO update on batch.

        Args:
            batch: Batch of experience

        Returns:
            Dictionary with loss metrics
        """
        metrics = {
            "loss_actor": 0.0,
            "loss_critic": 0.0,
            "loss_entropy": 0.0,
            "kl_divergence": 0.0,
            "grad_norm": 0.0,
        }

        # Multiple epochs over the batch
        for epoch in range(self.config.num_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(batch.numel())
            num_batches = max(1, batch.numel() // self.config.mini_batch_size)

            for i in range(num_batches):
                start = i * self.config.mini_batch_size
                end = min((i + 1) * self.config.mini_batch_size, batch.numel())
                mb_indices = indices[start:end]

                mini_batch = batch[mb_indices]

                # Compute loss
                loss_dict = self.loss_module(mini_batch)

                # Total loss
                loss = (
                    loss_dict["loss_objective"]
                    + self.config.value_coef * loss_dict["loss_critic"]
                    + self.config.entropy_coef * loss_dict.get("loss_entropy", 0.0)
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(),
                        self.config.max_grad_norm,
                    )
                else:
                    grad_norm = None

                self.optimizer.step()

                # Accumulate metrics
                metrics["loss_actor"] += loss_dict["loss_objective"].item()
                metrics["loss_critic"] += loss_dict["loss_critic"].item()
                if "loss_entropy" in loss_dict:
                    metrics["loss_entropy"] += loss_dict["loss_entropy"].item()
                if "kl_approx" in loss_dict:
                    metrics["kl_divergence"] += loss_dict["kl_approx"].item()
                if grad_norm is not None:
                    metrics["grad_norm"] += float(grad_norm)

            # Early stopping on KL divergence
            if self.config.target_kl and metrics["kl_divergence"] > self.config.target_kl:
                break

        # Average metrics
        num_updates = self.config.num_epochs * num_batches
        for key in metrics:
            metrics[key] /= max(1, num_updates)

        return metrics

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics.

        Args:
            metrics: Dictionary of metrics to log
        """
        log_str = f"Step {self.total_frames}: "
        log_str += f"actor_loss={metrics['loss_actor']:.4f}, "
        log_str += f"critic_loss={metrics['loss_critic']:.4f}"

        if "mean_episode_reward" in metrics:
            log_str += f", reward={metrics['mean_episode_reward']:.2f}"

        tqdm.write(log_str)

        # TensorBoard
        if self.writer is not None:
            metrics_cfg = self.config.tensorboard.metrics
            if metrics_cfg.train:
                self.writer.add_scalar("train/loss_actor", metrics["loss_actor"], self.total_frames)
                self.writer.add_scalar("train/loss_critic", metrics["loss_critic"], self.total_frames)
                self.writer.add_scalar("train/loss_entropy", metrics["loss_entropy"], self.total_frames)
                self.writer.add_scalar("train/kl_divergence", metrics["kl_divergence"], self.total_frames)
                if self.scheduler:
                    self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.total_frames)
            if metrics_cfg.episode:
                if "mean_episode_reward" in metrics:
                    self.writer.add_scalar("episode/mean_reward", metrics["mean_episode_reward"], self.total_frames)
                    self.writer.add_scalar("episode/max_reward", metrics["max_episode_reward"], self.total_frames)
                if "total_episodes" in metrics:
                    self.writer.add_scalar("episode/count", metrics["total_episodes"], self.total_frames)
            if metrics_cfg.gradients and "grad_norm" in metrics:
                self.writer.add_scalar("gradients/grad_norm", metrics["grad_norm"], self.total_frames)
            if metrics_cfg.timing:
                wall = time.monotonic() - self._train_start_time
                self.writer.add_scalar("timing/wall_clock_secs", wall, self.total_frames)
                if "batch_time_secs" in metrics:
                    self.writer.add_scalar("timing/batch_time_secs", metrics["batch_time_secs"], self.total_frames)
            if metrics_cfg.system:
                log_system_metrics(
                    self.writer,
                    self.total_frames,
                    self.device,
                    self._system_log_counter,
                    metrics_cfg.system_interval,
                )

    def save_checkpoint(self, name: str) -> None:
        """Save training checkpoint atomically.

        Uses atomic save pattern: write to temp file, then rename.
        Creates backup of existing checkpoint before overwriting.

        Args:
            name: Checkpoint name
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "config": self.config,
        }

        path = self.save_dir / f"{name}.pt"

        # Backup existing checkpoint if it exists
        if path.exists():
            backup_path = self.save_dir / f"{name}.pt.backup"
            shutil.copy2(path, backup_path)

        # Atomic save: write to temp file, then rename
        fd, temp_path = tempfile.mkstemp(dir=self.save_dir, suffix='.pt.tmp')
        try:
            torch.save(checkpoint, temp_path)
            os.rename(temp_path, path)  # Atomic on same filesystem
        except Exception:
            os.unlink(temp_path)
            raise
        finally:
            os.close(fd)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_frames = checkpoint["total_frames"]
        self.total_episodes = checkpoint["total_episodes"]
        self.best_reward = checkpoint["best_reward"]

    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate current policy.

        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions

        Returns:
            Dictionary with evaluation metrics
        """
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
                        # Use mean action
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
