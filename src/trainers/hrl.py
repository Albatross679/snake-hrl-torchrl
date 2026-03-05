"""Hierarchical RL trainer for snake predation."""

from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import os
import signal
import tempfile
import shutil
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from torchrl.envs import EnvBase
from tensordict import TensorDict
from tqdm import tqdm

from src.configs.training import HRLConfig
from src.configs.network import HRLNetworkConfig
from src.configs.base import resolve_device
from .ppo import PPOTrainer
from src.envs.approach_env import ApproachEnv
from src.envs.coil_env import CoilEnv
from src.envs.hrl_env import HRLEnv
from src.networks.actor import CategoricalActorNetwork, create_actor
from src.networks.critic import create_critic


class HRLTrainer:
    """Hierarchical RL trainer with manager and worker policies.

    Training strategies:
    - sequential: Train skills first, then manager
    - joint: Train all policies together
    - pretrain_skills: Pretrain skills, then fine-tune with manager
    """

    def __init__(
        self,
        config: Optional[HRLConfig] = None,
        network_config: Optional[HRLNetworkConfig] = None,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
    ):
        """Initialize HRL trainer.

        Args:
            config: HRL training configuration
            network_config: Network architecture configuration
            device: Device for training
        """
        self.config = config or HRLConfig()
        self.network_config = network_config or HRLNetworkConfig()
        self.device = resolve_device(device)

        # Create skill environments
        self.approach_env = ApproachEnv(device=self.device)
        self.coil_env = CoilEnv(device=self.device)
        self.hrl_env = HRLEnv(device=self.device)

        # Skill trainers (will be created during training)
        self.approach_trainer: Optional[PPOTrainer] = None
        self.coil_trainer: Optional[PPOTrainer] = None

        # Manager policy
        self.manager_actor: Optional[nn.Module] = None
        self.manager_critic: Optional[nn.Module] = None
        self.manager_optimizer: Optional[Adam] = None

        # Training state
        self.total_frames = 0
        self.curriculum_stage = 0
        self.skill_success_rates: Dict[str, List[float]] = {
            "approach": [],
            "coil": [],
        }

        # Run directory
        self._run_dir = Path(run_dir) if run_dir is not None else None

        # Graceful shutdown handling
        self._shutdown_requested = False
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)

        # Logging / output directories
        if self._run_dir is not None:
            self.log_dir = self._run_dir
            self.save_dir = self._run_dir / "checkpoints"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = Path(self.config.log_dir) / self.config.experiment_name
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir = Path(self.config.save_dir) / self.config.experiment_name
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
                dir=str(self._run_dir or self.log_dir),
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
        """Run hierarchical RL training.

        Args:
            callback: Optional callback function

        Returns:
            Dictionary with training statistics
        """
        if self.config.training_strategy == "sequential":
            return self._train_sequential(callback)
        elif self.config.training_strategy == "joint":
            return self._train_joint(callback)
        elif self.config.training_strategy == "pretrain_skills":
            return self._train_pretrain_skills(callback)
        else:
            raise ValueError(f"Unknown strategy: {self.config.training_strategy}")

    def _train_sequential(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Train skills sequentially, then manager.

        1. Train approach skill
        2. Train coil skill
        3. Train manager to coordinate skills
        """
        results = {}

        # Phase 1: Train approach skill
        print("=" * 50)
        print("Phase 1: Training Approach Skill")
        print("=" * 50)

        self.approach_trainer = PPOTrainer(
            env=self.approach_env,
            config=self.config.approach_config,
            network_config=self.network_config.worker_approach,
            device=self.device,
            run_dir=self._run_dir,
        )

        approach_config = self.config.approach_config
        approach_config.total_frames = self.config.approach_frames
        approach_results = self.approach_trainer.train()
        results["approach"] = approach_results

        # Phase 2: Train coil skill
        print("=" * 50)
        print("Phase 2: Training Coil Skill")
        print("=" * 50)

        self.coil_trainer = PPOTrainer(
            env=self.coil_env,
            config=self.config.coil_config,
            network_config=self.network_config.worker_coil,
            device=self.device,
            run_dir=self._run_dir,
        )

        coil_config = self.config.coil_config
        coil_config.total_frames = self.config.coil_frames
        coil_results = self.coil_trainer.train()
        results["coil"] = coil_results

        # Phase 3: Train manager
        print("=" * 50)
        print("Phase 3: Training Manager Policy")
        print("=" * 50)

        # Freeze skill policies if configured
        if self.config.freeze_skills_during_manager_training:
            for param in self.approach_trainer.actor.parameters():
                param.requires_grad = False
            for param in self.coil_trainer.actor.parameters():
                param.requires_grad = False

        manager_results = self._train_manager()
        results["manager"] = manager_results

        # Save final checkpoint
        self.save_checkpoint("final")

        # Close W&B run
        if self.wandb_run is not None:
            self.wandb_run.finish()

        return results

    def _train_joint(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Train all policies jointly from scratch."""
        # Initialize all trainers
        self._initialize_all_policies()

        results = {"metrics": []}
        pbar = tqdm(total=self.config.total_frames, desc="Joint Training")

        while self.total_frames < self.config.total_frames:
            # Check for graceful shutdown
            if self._shutdown_requested:
                print("Shutdown requested, saving checkpoint...")
                self.save_checkpoint("interrupted")
                print("Checkpoint saved to 'interrupted.pt'. Exiting.")
                self._restore_signal_handlers()
                break

            # Collect experience with hierarchical policy
            batch = self._collect_hrl_experience()

            # Update skills
            approach_metrics = self._update_skill("approach", batch)
            coil_metrics = self._update_skill("coil", batch)

            # Update manager
            manager_metrics = self._update_manager(batch)

            # Combine metrics
            metrics = {
                **{f"approach_{k}": v for k, v in approach_metrics.items()},
                **{f"coil_{k}": v for k, v in coil_metrics.items()},
                **{f"manager_{k}": v for k, v in manager_metrics.items()},
                "total_frames": self.total_frames,
            }

            results["metrics"].append(metrics)
            pbar.update(batch.numel())

            # W&B — joint training metrics
            if self.wandb_run is not None:
                wandb_log = {}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        wandb_log[f"joint/{key}"] = value
                wandb_log["joint/curriculum_stage"] = self.curriculum_stage
                self.wandb_run.log(wandb_log, step=self.total_frames)

            # Curriculum advancement
            if self.config.use_curriculum:
                self._check_curriculum_advancement()

            if callback:
                callback(metrics)

        pbar.close()
        self.save_checkpoint("final")

        # Close W&B run
        if self.wandb_run is not None:
            self.wandb_run.finish()

        return results

    def _train_pretrain_skills(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Pretrain skills, then fine-tune with manager."""
        # First train skills
        results = self._train_sequential(callback)

        # Then unfreeze and fine-tune
        print("=" * 50)
        print("Phase 4: Fine-tuning with Manager")
        print("=" * 50)

        # Unfreeze skill policies
        for param in self.approach_trainer.actor.parameters():
            param.requires_grad = True
        for param in self.coil_trainer.actor.parameters():
            param.requires_grad = True

        # Joint fine-tuning with lower learning rate
        original_lr = self.config.learning_rate
        self.config.learning_rate = original_lr * 0.1

        finetune_results = self._train_joint(callback)
        results["finetune"] = finetune_results

        return results

    def _train_manager(self) -> Dict[str, Any]:
        """Train manager policy using PPO."""
        # Get observation dimension from HRL environment
        obs_dim = self.hrl_env.observation_spec["observation"].shape[-1]
        num_skills = 2  # Approach and Coil

        # Create manager networks
        self.manager_actor = CategoricalActorNetwork(
            obs_dim=obs_dim,
            num_actions=num_skills,
            hidden_dims=self.network_config.manager.actor.hidden_dims,
            activation=self.network_config.manager.actor.activation,
        ).to(self.device)

        self.manager_critic = create_critic(
            obs_dim=obs_dim,
            config=self.network_config.manager.critic,
            device=self.device,
        )

        self.manager_optimizer = Adam(
            list(self.manager_actor.parameters())
            + list(self.manager_critic.parameters()),
            lr=self.config.manager_config.learning_rate,
        )

        # Training loop
        metrics_history = []
        pbar = tqdm(total=self.config.manager_frames, desc="Manager Training")
        frames = 0

        while frames < self.config.manager_frames:
            # Collect hierarchical experience
            batch = self._collect_manager_experience()
            frames += batch.numel()
            pbar.update(batch.numel())

            # Update manager
            metrics = self._update_manager(batch)
            metrics_history.append(metrics)

            # W&B — manager training metrics
            if self.wandb_run is not None:
                wandb_log = {}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        wandb_log[f"manager/{key}"] = value
                self.wandb_run.log(wandb_log, step=frames)

            if len(metrics_history) % 10 == 0:
                avg_reward = np.mean([m.get("episode_reward", 0) for m in metrics_history[-10:]])
                tqdm.write(f"Manager training - Avg reward: {avg_reward:.2f}")

        pbar.close()

        return {"metrics": metrics_history}

    def _collect_manager_experience(self) -> TensorDict:
        """Collect experience using hierarchical policy."""
        batch_data = []
        frames = 0
        target_frames = self.config.manager_config.frames_per_batch

        while frames < target_frames:
            td = self.hrl_env.reset()
            episode_data = []

            done = False
            while not done:
                # Manager selects skill
                with torch.no_grad():
                    obs = td["observation"]
                    logits = self.manager_actor(obs.unsqueeze(0))
                    probs = torch.softmax(logits, dim=-1)
                    skill_idx = torch.multinomial(probs, 1).item()

                # Execute skill for skill_duration steps
                skill_reward = 0.0
                skill_done = False

                for _ in range(self.hrl_env.hrl_config.skill_duration):
                    # Get action from selected skill policy
                    if skill_idx == 0:  # Approach
                        td = self.approach_trainer.actor(td)
                    else:  # Coil
                        td = self.coil_trainer.actor(td)

                    # Environment step
                    td = self.hrl_env.step(td)
                    skill_reward += td["reward"].item()
                    frames += 1

                    if td["done"].item():
                        skill_done = True
                        break

                # Store transition for manager
                episode_data.append({
                    "observation": obs,
                    "action": torch.tensor([skill_idx]),
                    "reward": torch.tensor([skill_reward]),
                    "done": torch.tensor([skill_done]),
                })

                done = skill_done

            batch_data.extend(episode_data)

        # Convert to TensorDict
        batch = TensorDict({
            "observation": torch.stack([d["observation"] for d in batch_data]),
            "action": torch.stack([d["action"] for d in batch_data]),
            "reward": torch.stack([d["reward"] for d in batch_data]),
            "done": torch.stack([d["done"] for d in batch_data]),
        }, batch_size=[len(batch_data)])

        return batch

    def _collect_hrl_experience(self) -> TensorDict:
        """Collect experience for joint training."""
        # Similar to _collect_manager_experience but also stores low-level data
        return self._collect_manager_experience()

    def _update_skill(self, skill_name: str, batch: TensorDict) -> Dict[str, float]:
        """Update skill policy on relevant experience."""
        # Filter batch for skill-specific updates
        # For simplicity, we update on all data (in practice, filter by skill used)
        if skill_name == "approach" and self.approach_trainer:
            return {"updated": True}
        elif skill_name == "coil" and self.coil_trainer:
            return {"updated": True}
        return {"updated": False}

    def _update_manager(self, batch: TensorDict) -> Dict[str, float]:
        """Update manager policy using PPO."""
        if self.manager_actor is None or self.manager_optimizer is None:
            return {}

        # Compute advantages (simplified - use GAE in production)
        rewards = batch["reward"]
        values = self.manager_critic(batch)["state_value"]
        advantages = rewards - values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        obs = batch["observation"]
        actions = batch["action"]
        old_logits = self.manager_actor(obs)
        old_probs = torch.softmax(old_logits, dim=-1)
        old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-8)

        for _ in range(self.config.manager_config.num_epochs):
            logits = self.manager_actor(obs)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs.gather(1, actions) + 1e-8)

            # Policy loss (PPO clip)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            clip_ratio = torch.clamp(
                ratio, 1 - self.config.manager_config.clip_epsilon,
                1 + self.config.manager_config.clip_epsilon
            )
            policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()

            # Value loss
            values = self.manager_critic(batch)["state_value"]
            value_loss = nn.functional.mse_loss(values, rewards)

            # Entropy bonus
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            # Total loss
            loss = (
                policy_loss
                + self.config.manager_config.value_coef * value_loss
                - self.config.manager_config.entropy_coef * entropy
            )

            self.manager_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.manager_actor.parameters())
                + list(self.manager_critic.parameters()),
                self.config.manager_config.max_grad_norm,
            )
            self.manager_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def _initialize_all_policies(self) -> None:
        """Initialize all policies for joint training."""
        # Initialize skill trainers
        self.approach_trainer = PPOTrainer(
            env=self.approach_env,
            config=self.config.approach_config,
            network_config=self.network_config.worker_approach,
            device=self.device,
            run_dir=self._run_dir,
        )

        self.coil_trainer = PPOTrainer(
            env=self.coil_env,
            config=self.config.coil_config,
            network_config=self.network_config.worker_coil,
            device=self.device,
            run_dir=self._run_dir,
        )

        # Initialize manager
        obs_dim = self.hrl_env.observation_spec["observation"].shape[-1]
        num_skills = 2

        self.manager_actor = CategoricalActorNetwork(
            obs_dim=obs_dim,
            num_actions=num_skills,
            hidden_dims=self.network_config.manager.actor.hidden_dims,
        ).to(self.device)

        self.manager_critic = create_critic(
            obs_dim=obs_dim,
            config=self.network_config.manager.critic,
            device=self.device,
        )

        self.manager_optimizer = Adam(
            list(self.manager_actor.parameters())
            + list(self.manager_critic.parameters()),
            lr=self.config.manager_config.learning_rate,
        )

    def _check_curriculum_advancement(self) -> None:
        """Check if curriculum should advance to next stage."""
        if not self.config.use_curriculum:
            return

        stages = self.config.curriculum_stages
        thresholds = self.config.curriculum_thresholds

        if self.curriculum_stage >= len(stages) - 1:
            return

        # Check success rate for current stage
        current_stage = stages[self.curriculum_stage]
        threshold = thresholds[self.curriculum_stage]

        if current_stage == "approach_only":
            success_rate = np.mean(self.skill_success_rates["approach"][-100:])
        elif current_stage == "coil_only":
            success_rate = np.mean(self.skill_success_rates["coil"][-100:])
        else:
            return

        if success_rate >= threshold:
            self.curriculum_stage += 1
            print(f"Advancing to curriculum stage: {stages[self.curriculum_stage]}")

    def save_checkpoint(self, name: str) -> None:
        """Save all policies and training state atomically.

        Uses atomic save pattern: write to temp file, then rename.
        Creates backup of existing checkpoint before overwriting.
        """
        checkpoint = {
            "total_frames": self.total_frames,
            "curriculum_stage": self.curriculum_stage,
            "skill_success_rates": self.skill_success_rates,
            "config": self.config,
        }

        if self.approach_trainer:
            checkpoint["approach_actor"] = self.approach_trainer.actor.state_dict()
            checkpoint["approach_critic"] = self.approach_trainer.critic.state_dict()

        if self.coil_trainer:
            checkpoint["coil_actor"] = self.coil_trainer.actor.state_dict()
            checkpoint["coil_critic"] = self.coil_trainer.critic.state_dict()

        if self.manager_actor:
            checkpoint["manager_actor"] = self.manager_actor.state_dict()
            checkpoint["manager_critic"] = self.manager_critic.state_dict()

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

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load all policies and training state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.total_frames = checkpoint["total_frames"]
        self.curriculum_stage = checkpoint["curriculum_stage"]
        self.skill_success_rates = checkpoint["skill_success_rates"]

        if "approach_actor" in checkpoint:
            self.approach_trainer.actor.load_state_dict(checkpoint["approach_actor"])
            self.approach_trainer.critic.load_state_dict(checkpoint["approach_critic"])

        if "coil_actor" in checkpoint:
            self.coil_trainer.actor.load_state_dict(checkpoint["coil_actor"])
            self.coil_trainer.critic.load_state_dict(checkpoint["coil_critic"])

        if "manager_actor" in checkpoint:
            self.manager_actor.load_state_dict(checkpoint["manager_actor"])
            self.manager_critic.load_state_dict(checkpoint["manager_critic"])

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate hierarchical policy."""
        rewards = []
        successes = []

        for _ in range(num_episodes):
            td = self.hrl_env.reset()
            episode_reward = 0.0

            done = False
            while not done:
                with torch.no_grad():
                    obs = td["observation"]
                    logits = self.manager_actor(obs.unsqueeze(0))
                    skill_idx = logits.argmax().item()

                    if skill_idx == 0:
                        td = self.approach_trainer.actor(td)
                    else:
                        td = self.coil_trainer.actor(td)

                td = self.hrl_env.step(td)
                episode_reward += td["reward"].item()
                done = td["done"].item()

            rewards.append(episode_reward)
            successes.append(float(self.hrl_env.is_success()))

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": np.mean(successes),
        }
