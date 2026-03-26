"""MM-RKHS (Gupta & Mahajan) trainer using TorchRL.

Implements the MM-RKHS algorithm from Gupta & Mahajan (2026, arXiv:2603.17875),
adapted for continuous action spaces with neural network function approximation.

Key differences from PPOTrainer:
- No ClipPPOLoss module -- custom loss computed directly in _update()
- Trust region enforced by MMD penalty + KL regularizer (no clipping)
- No entropy bonus -- KL regularizer handles exploration
- MMD computed via linear-time RBF kernel estimator
- Old distribution reconstructed from stored loc/scale in batch
"""

import json
import math
import time
from collections import deque
from contextlib import nullcontext
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import os
import signal
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from .logging_utils import collect_system_metrics
from .diagnostics import (
    compute_explained_variance,
    compute_action_stats,
    compute_advantage_stats,
    compute_log_prob_stats,
    check_alerts,
)

from torchrl.envs import EnvBase
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.modules import TanhNormal
from tensordict import TensorDict
from tqdm import tqdm

from src.configs.training import MMRKHSConfig
from src.configs.network import NetworkConfig
from src.configs.base import resolve_device
from src.configs.run_dir import setup_run_dir
from src.networks.actor import create_actor
from src.networks.critic import create_critic
from src import wandb_utils


# Default STOP file path (relative to working directory)
STOP_FILE = "STOP"


def _amp_context(use_amp: bool, device: str):
    """Create AMP autocast context for bf16 mixed precision."""
    if use_amp and 'cuda' in str(device):
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


class MMRKHSTrainer:
    """MM-RKHS trainer for continuous control tasks (Gupta & Mahajan, 2026).

    Implements Majorization-Minimization in RKHS with MM-RKHS loss:
    L = -E[ratio * A] + beta * MMD^2(pi_new, pi_old) + (1/eta) * KL + value_coef * critic_loss

    Trust region enforced entirely by MMD penalty + KL regularizer (no PPO clipping).
    """

    def __init__(
        self,
        env: EnvBase,
        config: Optional[MMRKHSConfig] = None,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
    ):
        self.env = env
        self.config = config or MMRKHSConfig()
        self.network_config = network_config or NetworkConfig()
        self.device = resolve_device(device)

        # Get dimensions from environment
        obs_dim = env.observation_spec["observation"].shape[-1]
        action_spec = env.action_spec
        self.action_spec = action_spec  # Store for TanhNormal reconstruction in MMD

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

        # Collect all trainable parameters (no ClipPPOLoss wrapper)
        self.loss_params = list(self.actor.parameters()) + list(self.critic.parameters())

        # Create optimizer over combined actor + critic parameters
        self.optimizer = Adam(
            self.loss_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Create GAE module for advantage estimation
        self.advantage_module = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.gae_lambda,
            value_network=self.critic,
        )

        # Learning rate scheduler
        if self.config.lr_schedule == "linear":
            total_updates = max(1, self.config.total_frames // self.config.frames_per_batch)
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

        # Rolling 100-episode mean reward
        self._episode_reward_buffer = deque(maxlen=100)

        # Early stopping state
        self._batches_since_improvement = 0
        self._patience_batches = self.config.patience_batches

        # Metric tracking
        self._train_start_time = 0.0
        self._batch_start_time = 0.0
        self._system_log_counter = [0]

        # Graceful shutdown handling
        self._shutdown_requested = False
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)

        # Logging / output directories
        if run_dir is None:
            run_dir = setup_run_dir(self.config)
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir
        self.save_dir = self.run_dir / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics log file (JSON-lines)
        self._metrics_log_path = self.run_dir / "metrics.jsonl"
        self._metrics_log_file = open(self._metrics_log_path, "a", encoding="utf-8")

        # Weights & Biases (via shared utility)
        self.wandb_run = wandb_utils.setup_run(self.config, self.run_dir)

        # Log one-time params (count params manually since no loss_module)
        total_params = sum(p.numel() for p in self.loss_params)
        trainable_params = sum(p.numel() for p in self.loss_params if p.requires_grad)
        extra_params = {
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
        extra_params["num_envs"] = self.config.num_envs
        extra_params["use_amp"] = self.config.use_amp
        extra_params["amp_dtype"] = "bfloat16" if self.config.use_amp else "float32"
        extra_params["algorithm"] = "MM-RKHS"
        extra_params["beta"] = self.config.beta
        extra_params["eta"] = self.config.eta
        extra_params["mmd_bandwidth"] = self.config.mmd_bandwidth
        extra_params["mmd_num_samples"] = self.config.mmd_num_samples
        extra_params.update(wandb_utils.collect_hardware_info(self.device))
        wandb_utils.log_extra_params(self.wandb_run, extra_params)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        signal_name = signal.Signals(signum).name
        print(f"\n{signal_name} received, requesting graceful shutdown...")
        self._shutdown_requested = True

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    def _check_stop_file(self) -> bool:
        """Check if a STOP file exists (user requested graceful stop)."""
        return Path(STOP_FILE).exists()

    def _write_metrics_jsonl(self, metrics: Dict[str, Any]) -> None:
        """Append a metrics dict as one JSON line to metrics.jsonl."""
        line = json.dumps(metrics, default=str)
        self._metrics_log_file.write(line + "\n")
        self._metrics_log_file.flush()

    def train(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run MM-RKHS training loop.

        Returns:
            Dictionary with training statistics
        """
        pbar = tqdm(total=self.config.total_frames, desc="Training")
        all_metrics = []

        self._train_start_time = time.monotonic()
        self._batch_start_time = time.monotonic()
        max_wall_time = self.config.max_wall_time
        stop_reason = "completed"

        try:
            # Track env step time across collector iterations
            _collector_start = time.monotonic()

            for batch_idx, batch in enumerate(self.collector):
                # --- Timing: env step (rollout collection via SyncDataCollector) ---
                env_dt = time.monotonic() - _collector_start

                # Check for graceful shutdown (SIGINT/SIGTERM)
                if self._shutdown_requested:
                    print("Shutdown requested, saving checkpoint...")
                    self.save_checkpoint("interrupted")
                    print("Checkpoint saved to 'interrupted.pt'. Exiting.")
                    self._restore_signal_handlers()
                    stop_reason = "signal"
                    break

                # Check for STOP file
                if self._check_stop_file():
                    tqdm.write("STOP file detected. Finishing current batch and stopping.")
                    stop_reason = "stop_file"
                    break

                # Check wall-clock limit
                if max_wall_time is not None:
                    elapsed = time.monotonic() - self._train_start_time
                    if elapsed >= max_wall_time:
                        tqdm.write(
                            f"Wall-clock limit reached ({elapsed:.0f}s / {max_wall_time:.0f}s). "
                            f"Stopping at {self.total_frames} frames."
                        )
                        stop_reason = "wall_time"
                        break

                # Check early stopping
                if self._patience_batches > 0 and self._batches_since_improvement >= self._patience_batches:
                    tqdm.write(
                        f"Early stopping: no reward improvement for {self._batches_since_improvement} batches "
                        f"(patience={self._patience_batches})."
                    )
                    stop_reason = "early_stopping"
                    break

                # --- Timing: data (batch transfer + GAE computation) ---
                t0 = time.monotonic()

                # Move batch to training device (ParallelEnv produces CPU tensors)
                batch = batch.to(self.device)

                # Flatten batch if from ParallelEnv (shape [num_envs, T] -> [num_envs*T])
                if batch.ndim > 1:
                    batch = batch.reshape(-1)

                # Compute advantages
                with torch.no_grad():
                    self.advantage_module(batch)
                data_dt = time.monotonic() - t0

                # --- Timing: backward (MM-RKHS update epochs) ---
                t0 = time.monotonic()

                # MM-RKHS update
                metrics = self._update(batch)
                backward_dt = time.monotonic() - t0

                metrics["batch_idx"] = batch_idx
                metrics["total_frames"] = self.total_frames

                # Update frame count
                self.total_frames += batch.numel()
                pbar.update(batch.numel())

                # Episode statistics from batch
                next_td = batch.get("next", batch)
                done_mask = next_td["done"].squeeze(-1)

                if "episode_reward" in next_td.keys():
                    episode_rewards = next_td["episode_reward"][done_mask]
                    if len(episode_rewards) > 0:
                        metrics["mean_episode_reward"] = episode_rewards.mean().item()
                        metrics["max_episode_reward"] = episode_rewards.max().item()
                        metrics["min_episode_reward"] = episode_rewards.min().item()
                        self.total_episodes += len(episode_rewards)

                        # Feed rolling buffer
                        for r in episode_rewards.tolist():
                            self._episode_reward_buffer.append(r)

                        # Rolling 100-episode mean
                        if len(self._episode_reward_buffer) > 0:
                            metrics["rolling_mean_reward_100"] = float(
                                np.mean(list(self._episode_reward_buffer))
                            )

                        # Track best reward + early stopping
                        if metrics["mean_episode_reward"] > self.best_reward:
                            self.best_reward = metrics["mean_episode_reward"]
                            self._batches_since_improvement = 0
                            self.save_checkpoint("best")
                        else:
                            self._batches_since_improvement += 1

                # Episode lengths
                if "step_count" in next_td.keys():
                    episode_lengths = next_td["step_count"][done_mask]
                    if len(episode_lengths) > 0:
                        metrics["mean_episode_length"] = episode_lengths.float().mean().item()

                # Episode wall-clock time
                if "episode_wall_time_s" in next_td.keys():
                    ep_wall_times = next_td["episode_wall_time_s"][done_mask]
                    if len(ep_wall_times) > 0:
                        metrics["mean_episode_wall_time_s"] = ep_wall_times.mean().item()

                # Goal metrics (termination reason + final distance)
                n_done = done_mask.sum().item()
                if n_done > 0:
                    if "goal_reached" in next_td.keys():
                        n_goal = next_td["goal_reached"][done_mask].sum().item()
                        metrics["goal_reach_rate"] = n_goal / n_done
                    if "starvation" in next_td.keys():
                        n_starve = next_td["starvation"][done_mask].sum().item()
                        metrics["starvation_rate"] = n_starve / n_done
                        metrics["truncation_rate"] = (n_done - n_goal - n_starve) / n_done
                    if "final_dist_to_goal" in next_td.keys():
                        finals = next_td["final_dist_to_goal"][done_mask]
                        metrics["mean_final_dist_to_goal"] = finals.mean().item()

                # Reward diagnostics (batch means)
                for key in ("v_g", "dist_to_goal", "theta_g", "reward_dist", "reward_align"):
                    if key in next_td.keys():
                        metrics[f"mean_{key}"] = next_td[key].mean().item()

                metrics["total_episodes"] = self.total_episodes
                metrics["best_reward"] = self.best_reward
                metrics["batches_since_improvement"] = self._batches_since_improvement
                batch_time_s = time.monotonic() - self._batch_start_time
                metrics["batch_time_mins"] = batch_time_s / 60.0
                metrics["fps"] = batch.numel() / batch_time_s if batch_time_s > 0 else 0.0
                metrics["step_time_ms"] = (batch_time_s / batch.numel()) * 1000 if batch.numel() > 0 else 0.0
                self._batch_start_time = time.monotonic()

                # --- Timing: overhead (logging, checkpointing) ---
                t0 = time.monotonic()

                all_metrics.append(metrics)

                # Write to metrics.jsonl (every batch)
                self._write_metrics_jsonl(metrics)

                # Console + W&B logging
                if batch_idx % self.config.log_interval == 0:
                    self._log_metrics(metrics)

                # Save checkpoint periodically
                if batch_idx % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.total_frames}")

                # Callback
                if callback:
                    callback(metrics)

                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()

                overhead_dt = time.monotonic() - t0

                # Compute timing fractions and add to metrics
                total_dt = env_dt + data_dt + backward_dt + overhead_dt
                metrics["timing/env_step_seconds"] = env_dt
                metrics["timing/data_seconds"] = data_dt
                metrics["timing/backward_seconds"] = backward_dt
                metrics["timing/overhead_seconds"] = overhead_dt
                metrics["timing/env_step_pct"] = env_dt / total_dt * 100 if total_dt > 0 else 0
                metrics["timing/backward_pct"] = backward_dt / total_dt * 100 if total_dt > 0 else 0

                # Start timing for next iteration's env step (collector.__next__)
                _collector_start = time.monotonic()

        finally:
            pbar.close()
            # Close metrics log file
            self._metrics_log_file.close()

        # Final save
        self.save_checkpoint("final")

        # Upload best model as W&B artifact
        best_path = self.save_dir / "best.pt"
        if best_path.exists():
            wandb_utils.log_model_artifact(
                self.wandb_run,
                best_path,
                artifact_name=self.config.name,
                metadata={
                    "best_reward": self.best_reward,
                    "total_frames": self.total_frames,
                    "stop_reason": stop_reason,
                },
            )

        # Close W&B run
        wandb_utils.end_run(self.wandb_run)

        return {
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "stop_reason": stop_reason,
            "metrics": all_metrics,
        }

    def _compute_mmd_penalty(
        self,
        obs: torch.Tensor,
        old_loc: torch.Tensor,
        old_scale: torch.Tensor,
        new_loc: torch.Tensor,
        new_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MMD^2 penalty between old and new policy distributions.

        Uses linear-time unbiased RBF kernel estimator (Gretton et al., 2012).
        Samples actions from both distributions and computes MMD^2 in action space.

        Args:
            obs: Observation tensor [batch_size, obs_dim] (unused but kept for API consistency)
            old_loc: Old policy mean [batch_size, action_dim]
            old_scale: Old policy std [batch_size, action_dim]
            new_loc: New policy mean [batch_size, action_dim]
            new_scale: New policy std [batch_size, action_dim]

        Returns:
            Scalar MMD^2 penalty (non-negative)
        """
        num_samples = self.config.mmd_num_samples
        bandwidth = self.config.mmd_bandwidth

        # Reconstruct distributions from loc/scale
        action_low = self.action_spec.space.low.to(old_loc.device)
        action_high = self.action_spec.space.high.to(old_loc.device)

        old_dist = TanhNormal(
            old_loc.detach(), old_scale.detach(),
            low=action_low, high=action_high,
        )
        new_dist = TanhNormal(
            new_loc, new_scale,
            low=action_low, high=action_high,
        )

        # Sample actions: [num_samples, batch_size, action_dim]
        with torch.no_grad():
            x = old_dist.rsample((num_samples,))
        y = new_dist.rsample((num_samples,))

        # Linear-time MMD^2 estimator using pairs
        # Split samples into pairs for unbiased estimation
        n_pairs = num_samples // 2
        if n_pairs < 1:
            return torch.tensor(0.0, device=old_loc.device, requires_grad=True)

        x1, x2 = x[:n_pairs], x[n_pairs:2 * n_pairs]
        y1, y2 = y[:n_pairs], y[n_pairs:2 * n_pairs]

        # RBF kernel: k(a, b) = exp(-||a - b||^2 / (2 * bandwidth^2))
        def rbf_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            diff = (a - b).pow(2).sum(dim=-1)  # [n_pairs, batch_size]
            return torch.exp(-diff / (2.0 * bandwidth ** 2))

        # Linear-time unbiased estimator:
        # MMD^2 = E[k(x1,x2)] + E[k(y1,y2)] - E[k(x1,y2)] - E[k(x2,y1)]
        k_xx = rbf_kernel(x1, x2)  # [n_pairs, batch_size]
        k_yy = rbf_kernel(y1, y2)
        k_xy = rbf_kernel(x1, y2)
        k_yx = rbf_kernel(x2, y1)

        # Average over pairs and batch
        mmd_sq = (k_xx + k_yy - k_xy - k_yx).mean()

        # Clamp to non-negative (estimator can be slightly negative due to variance)
        return mmd_sq.clamp(min=0.0)

    def _update(self, batch: TensorDict) -> Dict[str, float]:
        """Perform MM-RKHS update on batch.

        Loss = -E[ratio * A] + beta * MMD^2 + (1/eta) * KL + value_coef * critic_loss

        Returns:
            Dictionary with loss metrics.
        """
        metrics = {
            "loss_policy": 0.0,
            "loss_critic": 0.0,
            "mmd_penalty": 0.0,
            "kl_divergence": 0.0,
            "grad_norm": 0.0,
            "policy_entropy": 0.0,
        }

        # --- Diagnostics: compute stats from batch before update ---
        with torch.no_grad():
            advantages = batch["advantage"]
            metrics.update(compute_advantage_stats(advantages))

            # Explained variance
            if "value_target" in batch.keys() and "state_value" in batch.keys():
                metrics["explained_variance"] = compute_explained_variance(
                    batch["state_value"], batch["value_target"],
                )
            elif "value_target" in batch.keys():
                v_pred = self.critic(batch)
                if "state_value" in v_pred.keys():
                    metrics["explained_variance"] = compute_explained_variance(
                        v_pred["state_value"], batch["value_target"],
                    )

            # Action distribution stats
            if "action" in batch.keys():
                metrics.update({
                    f"diagnostics/{k}": v
                    for k, v in compute_action_stats(batch["action"]).items()
                })

            # Log probability stats
            if "action_log_prob" in batch.keys():
                metrics.update({
                    f"diagnostics/{k}": v
                    for k, v in compute_log_prob_stats(batch["action_log_prob"]).items()
                })

            # Reward stats (step-level)
            if "next" in batch.keys() and "reward" in batch["next"].keys():
                rewards = batch["next"]["reward"]
                metrics["diagnostics/reward_mean"] = rewards.mean().item()
                metrics["diagnostics/reward_std"] = rewards.std().item()

        # Multiple epochs over the batch
        actual_updates = 0

        for epoch in range(self.config.num_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(batch.numel())
            num_batches = max(1, batch.numel() // self.config.mini_batch_size)

            for i in range(num_batches):
                start = i * self.config.mini_batch_size
                end = min((i + 1) * self.config.mini_batch_size, batch.numel())
                mb_indices = indices[start:end]

                mb = batch[mb_indices]

                # --- Get old log_prob from batch (collected by previous policy) ---
                log_prob_old = mb["action_log_prob"].detach()

                # --- Re-forward actor to get new log_prob and distribution params ---
                td_fwd = self.actor(mb.clone())
                log_prob_new = td_fwd["action_log_prob"]
                new_loc = td_fwd["loc"]
                new_scale = td_fwd["scale"]

                # Old distribution params (stored in batch from collection time)
                old_loc = mb["loc"].detach()
                old_scale = mb["scale"].detach()

                # --- Compute importance ratio with clamping for stability ---
                log_ratio = (log_prob_new - log_prob_old).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()

                # --- Normalize advantages ---
                advantage = mb["advantage"]
                if self.config.normalize_advantage:
                    adv_mean = advantage.mean()
                    adv_std = advantage.std() + 1e-8
                    advantage = (advantage - adv_mean) / adv_std

                # --- Surrogate advantage (no clipping) ---
                surr_advantage = ratio * advantage

                # --- Compute MMD penalty ---
                mmd = self._compute_mmd_penalty(
                    obs=mb["observation"],
                    old_loc=old_loc,
                    old_scale=old_scale,
                    new_loc=new_loc,
                    new_scale=new_scale,
                )

                # --- Compute KL divergence (approximate via log ratio) ---
                # KL(pi_old || pi_new) approx = E[(ratio - 1) - log(ratio)]
                with torch.no_grad():
                    kl = (ratio - 1.0 - log_ratio).mean()

                # --- Compute critic loss ---
                value_pred = self.critic(mb)["state_value"]
                critic_loss = F.mse_loss(value_pred, mb["value_target"])

                # --- Compute policy entropy (for diagnostics, not in loss) ---
                with torch.no_grad():
                    # Entropy from current distribution params
                    action_low = self.action_spec.space.low.to(new_loc.device)
                    action_high = self.action_spec.space.high.to(new_loc.device)
                    current_dist = TanhNormal(
                        new_loc, new_scale,
                        low=action_low, high=action_high,
                    )
                    # Use approximate entropy from Gaussian base (TanhNormal entropy
                    # is intractable, but the base Normal entropy is a good proxy)
                    entropy = 0.5 * torch.log(2 * math.pi * math.e * new_scale.pow(2)).mean()

                # --- Combined loss ---
                # L = -surr_advantage + beta * MMD^2 + (1/eta) * KL + value_coef * critic_loss
                total_loss = (
                    -surr_advantage.mean()
                    + self.config.beta * mmd
                    + (1.0 / self.config.eta) * kl
                    + self.config.value_coef * critic_loss
                )

                # --- Backward pass ---
                self.optimizer.zero_grad()

                # NaN guard: skip backward if loss is not finite
                if not torch.isfinite(total_loss):
                    print(f"  [WARNING] NaN/inf loss detected at step {self.total_frames}, skipping update")
                    continue

                total_loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.loss_params,
                        self.config.max_grad_norm,
                    )
                else:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.loss_params,
                        float('inf'),
                    )

                # NaN guard: skip step if gradients are not finite
                if not torch.isfinite(grad_norm):
                    print(f"  [WARNING] NaN/inf gradients (norm={grad_norm:.4f}) at step {self.total_frames}, skipping update")
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.step()
                actual_updates += 1

                # Accumulate metrics
                metrics["loss_policy"] += (-surr_advantage.mean()).item()
                metrics["loss_critic"] += critic_loss.item()
                metrics["mmd_penalty"] += mmd.item()
                metrics["kl_divergence"] += kl.item()
                metrics["grad_norm"] += float(grad_norm)
                metrics["policy_entropy"] += entropy.item()

        # Average metrics over actual updates
        for key in metrics:
            metrics[key] /= max(1, actual_updates)

        return metrics

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics to console and W&B."""
        # Console output
        log_str = f"Step {self.total_frames}: "
        log_str += f"actor_loss={metrics['loss_policy']:.4f}, "
        log_str += f"critic_loss={metrics['loss_critic']:.4f}, "
        log_str += f"mmd={metrics['mmd_penalty']:.4f}, "
        log_str += f"kl={metrics['kl_divergence']:.4f}"

        if "mean_episode_reward" in metrics:
            log_str += f", reward={metrics['mean_episode_reward']:.2f}"
        if "rolling_mean_reward_100" in metrics:
            log_str += f", rolling100={metrics['rolling_mean_reward_100']:.2f}"

        tqdm.write(log_str)

        # Build W&B log dict
        wandb_log = {
            "train/actor_loss": metrics["loss_policy"],
            "train/critic_loss": metrics["loss_critic"],
            "train/mmd_penalty": metrics["mmd_penalty"],
            "train/kl_divergence": metrics["kl_divergence"],
            "train/policy_entropy": metrics.get("policy_entropy", 0.0),
            "gradients/grad_norm": metrics.get("grad_norm", 0.0),
        }

        # Episode metrics
        if "mean_episode_reward" in metrics:
            wandb_log["episode/mean_reward"] = metrics["mean_episode_reward"]
            wandb_log["episode/max_reward"] = metrics["max_episode_reward"]
            wandb_log["episode/min_reward"] = metrics["min_episode_reward"]
        if "rolling_mean_reward_100" in metrics:
            wandb_log["episode/rolling_mean_reward_100"] = metrics["rolling_mean_reward_100"]
        if "mean_episode_length" in metrics:
            wandb_log["episode/mean_length"] = metrics["mean_episode_length"]
        if "mean_episode_wall_time_s" in metrics:
            wandb_log["episode/mean_wall_time_s"] = metrics["mean_episode_wall_time_s"]

        # Goal metrics
        for key, wandb_key in (
            ("goal_reach_rate", "episode/goal_reach_rate"),
            ("starvation_rate", "episode/starvation_rate"),
            ("truncation_rate", "episode/truncation_rate"),
            ("mean_final_dist_to_goal", "episode/final_dist_to_goal"),
        ):
            if key in metrics:
                wandb_log[wandb_key] = metrics[key]

        # Reward diagnostics
        for key, wandb_key in (
            ("mean_v_g", "reward/v_g"),
            ("mean_dist_to_goal", "reward/dist_to_goal"),
            ("mean_theta_g", "reward/theta_g"),
            ("mean_reward_dist", "reward/component_dist"),
            ("mean_reward_align", "reward/component_align"),
        ):
            if key in metrics:
                wandb_log[wandb_key] = metrics[key]

        if "total_episodes" in metrics:
            wandb_log["episode/count"] = metrics["total_episodes"]

        # Tracking: best reward + patience counter
        wandb_log["tracking/best_reward"] = self.best_reward
        wandb_log["tracking/batches_since_improvement"] = self._batches_since_improvement

        if self.scheduler:
            wandb_log["train/learning_rate"] = self.scheduler.get_last_lr()[0]

        # Timing
        if "batch_time_mins" in metrics:
            wandb_log["timing/batch_time_mins"] = metrics["batch_time_mins"]
        if "fps" in metrics:
            wandb_log["timing/fps"] = metrics["fps"]
        if "step_time_ms" in metrics:
            wandb_log["timing/step_time_ms"] = metrics["step_time_ms"]
        wandb_log["timing/wall_clock_mins"] = (time.monotonic() - self._train_start_time) / 60.0

        # Per-section timing metrics
        for key in (
            "timing/env_step_seconds", "timing/data_seconds",
            "timing/backward_seconds", "timing/overhead_seconds",
            "timing/env_step_pct", "timing/backward_pct",
        ):
            if key in metrics:
                wandb_log[key] = metrics[key]

        # System metrics (throttled)
        sys_metrics = collect_system_metrics(
            self.device,
            self._system_log_counter,
            10,
        )
        wandb_log.update(sys_metrics)

        # Diagnostics: explained variance, action stats, advantage stats, log prob stats
        if "explained_variance" in metrics:
            wandb_log["diagnostics/explained_variance"] = metrics["explained_variance"]
        for key in metrics:
            if key.startswith("diagnostics/"):
                wandb_log[key] = metrics[key]

        # Log via shared utility
        wandb_utils.log_metrics(self.wandb_run, wandb_log, step=self.total_frames)

        # Check for failure signatures and fire W&B alerts
        check_alerts(self.wandb_run, {**metrics, **wandb_log}, self.total_frames, algorithm="mmrkhs")

    def save_checkpoint(self, name: str) -> None:
        """Save training checkpoint atomically.

        Uses atomic save pattern: write to temp file, then rename.
        Creates backup of existing checkpoint before overwriting.
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
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_frames = checkpoint["total_frames"]
        self.total_episodes = checkpoint["total_episodes"]
        self.best_reward = checkpoint["best_reward"]

    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate current policy."""
        rewards = []
        lengths = []
        amp_ctx = _amp_context(self.config.use_amp, self.device)

        for _ in range(num_episodes):
            td = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            done = False
            while not done:
                with torch.inference_mode():
                    with amp_ctx:
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
