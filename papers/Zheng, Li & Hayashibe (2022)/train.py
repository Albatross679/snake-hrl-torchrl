"""PPO training with curriculum learning for Zheng, Li & Hayashibe (2022).

Two-phase curriculum:
    Phase 1 (epochs 0-2000): Maximize forward velocity with power penalty.
    Phase 2 (epochs 2000+): Match decreasing target velocities with low power.

Uses separate learning rates for actor (0.003) and critic (0.001) as per the paper.
"""

import sys
import os
import signal
import argparse
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

# Add parent directory so we can import snake_hrl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict

from snake_hrl.networks.actor import create_actor
from snake_hrl.networks.critic import create_critic
from snake_hrl.configs.network import ActorConfig, CriticConfig

from configs import ZhengConfig
from env import UnderwaterSnakeEnv
from reward import CurriculumReward


def make_env(config: ZhengConfig) -> UnderwaterSnakeEnv:
    """Create and return an environment instance."""
    return UnderwaterSnakeEnv(config=config, device=config.device)


def rewrite_rewards(batch: TensorDict, curriculum: CurriculumReward) -> None:
    """Replace environment rewards with curriculum rewards in-place.

    The env stores raw head_velocity_x and power. This function applies the
    curriculum reward function to compute the actual training reward.
    """
    head_vx = batch["head_velocity_x"].squeeze(-1).cpu().numpy()
    power = batch["power"].squeeze(-1).cpu().numpy()

    rewards = np.zeros_like(head_vx)
    for i in range(len(head_vx)):
        rewards[i] = curriculum.compute_reward(float(head_vx[i]), float(power[i]))

    batch["reward"] = torch.tensor(
        rewards, dtype=torch.float32, device=batch.device,
    ).unsqueeze(-1)


def save_checkpoint(
    save_dir: Path,
    name: str,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    epoch: int,
    best_reward: float,
    config: ZhengConfig,
):
    """Save training checkpoint atomically."""
    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "epoch": epoch,
        "best_reward": best_reward,
        "config": config,
    }

    path = save_dir / f"{name}.pt"

    if path.exists():
        backup_path = save_dir / f"{name}.pt.backup"
        shutil.copy2(path, backup_path)

    fd, temp_path = tempfile.mkstemp(dir=save_dir, suffix=".pt.tmp")
    try:
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, path)
    except Exception:
        os.unlink(temp_path)
        raise
    finally:
        os.close(fd)


def train(config: ZhengConfig):
    """Run PPO training with curriculum learning."""
    # Setup directories
    save_dir = Path(config.save_dir) / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.log_dir) / config.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = make_env(config)

    # Create actor and critic using snake_hrl networks
    actor_config = ActorConfig(
        hidden_dims=config.hidden_dims,
        activation=config.activation,
    )
    critic_config = CriticConfig(
        hidden_dims=config.hidden_dims,
        activation=config.activation,
    )

    obs_dim = config.obs_dim
    action_spec = env.action_spec

    actor = create_actor(
        obs_dim=obs_dim,
        action_spec=action_spec,
        config=actor_config,
        device=config.device,
    )
    critic = create_critic(
        obs_dim=obs_dim,
        config=critic_config,
        device=config.device,
    )

    # PPO loss
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_coef=config.entropy_coef,
        critic_coef=config.value_coef,
        normalize_advantage=True,
    )

    # GAE for advantage estimation
    advantage_module = GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=critic,
    )

    # Separate optimizers (paper specifies different LRs)
    actor_params = list(actor.parameters())
    critic_params = list(critic.parameters())
    actor_optimizer = Adam(actor_params, lr=config.policy_lr)
    critic_optimizer = Adam(critic_params, lr=config.value_lr)

    # Data collector
    collector = SyncDataCollector(
        create_env_fn=lambda: make_env(config),
        policy=actor,
        frames_per_batch=config.steps_per_epoch,
        total_frames=config.total_epochs * config.steps_per_epoch,
        device=config.device,
    )

    # Curriculum reward
    curriculum = CurriculumReward(config)

    # Training state
    best_reward = float("-inf")
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        print(f"\n{signal.Signals(signum).name} received, requesting graceful shutdown...")
        shutdown_requested = True

    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

    # Training loop
    pbar = tqdm(total=config.total_epochs, desc="Training")
    metrics_history = []

    for epoch, batch in enumerate(collector):
        if shutdown_requested:
            print("Shutdown requested, saving checkpoint...")
            save_checkpoint(
                save_dir, "interrupted", actor, critic,
                actor_optimizer, critic_optimizer, epoch, best_reward, config,
            )
            break

        # Update curriculum
        curriculum.set_epoch(epoch)

        # Rewrite rewards using curriculum function
        rewrite_rewards(batch, curriculum)

        # Compute advantages
        with torch.no_grad():
            advantage_module(batch)

        # PPO update with multiple epochs and mini-batches
        epoch_metrics = {
            "loss_actor": 0.0,
            "loss_critic": 0.0,
            "loss_entropy": 0.0,
        }
        num_updates = 0

        for _ in range(config.num_ppo_epochs):
            indices = torch.randperm(batch.numel())
            num_batches = max(1, batch.numel() // config.mini_batch_size)

            for i in range(num_batches):
                start = i * config.mini_batch_size
                end = min((i + 1) * config.mini_batch_size, batch.numel())
                mb_indices = indices[start:end]
                mini_batch = batch[mb_indices]

                loss_dict = loss_module(mini_batch)

                # Actor update (policy loss + entropy)
                actor_loss = loss_dict["loss_objective"]
                if "loss_entropy" in loss_dict:
                    actor_loss = actor_loss + config.entropy_coef * loss_dict["loss_entropy"]

                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                if config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(actor_params, config.max_grad_norm)
                actor_optimizer.step()

                # Critic update
                critic_loss = loss_dict["loss_critic"]
                critic_optimizer.zero_grad()
                critic_loss.backward()
                if config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(critic_params, config.max_grad_norm)
                critic_optimizer.step()

                epoch_metrics["loss_actor"] += loss_dict["loss_objective"].item()
                epoch_metrics["loss_critic"] += loss_dict["loss_critic"].item()
                if "loss_entropy" in loss_dict:
                    epoch_metrics["loss_entropy"] += loss_dict["loss_entropy"].item()
                num_updates += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, num_updates)

        # Episode statistics
        head_vx = batch["head_velocity_x"].squeeze(-1)
        power = batch["power"].squeeze(-1)
        rewards = batch["reward"].squeeze(-1)

        epoch_metrics.update({
            "epoch": epoch,
            "phase": curriculum.phase,
            "mean_head_vx": head_vx.mean().item(),
            "mean_power": power.mean().item(),
            "mean_reward": rewards.mean().item(),
            "max_head_vx": head_vx.max().item(),
        })

        if curriculum.phase == 2:
            epoch_metrics["target_velocity"] = curriculum.target_velocity

        metrics_history.append(epoch_metrics)

        # Logging
        if epoch % config.log_interval == 0:
            phase_str = f"Phase {curriculum.phase}"
            if curriculum.phase == 2:
                phase_str += f" (v_target={curriculum.target_velocity:.3f})"
            tqdm.write(
                f"Epoch {epoch} [{phase_str}]: "
                f"reward={epoch_metrics['mean_reward']:.4f}, "
                f"vx={epoch_metrics['mean_head_vx']:.4f} m/s, "
                f"power={epoch_metrics['mean_power']:.4f} W, "
                f"actor_loss={epoch_metrics['loss_actor']:.4f}, "
                f"critic_loss={epoch_metrics['loss_critic']:.4f}"
            )

        # Track best reward
        mean_reward = epoch_metrics["mean_reward"]
        if mean_reward > best_reward:
            best_reward = mean_reward
            save_checkpoint(
                save_dir, "best", actor, critic,
                actor_optimizer, critic_optimizer, epoch, best_reward, config,
            )

        # Periodic checkpoint
        if epoch % config.save_interval == 0 and epoch > 0:
            save_checkpoint(
                save_dir, f"epoch_{epoch}", actor, critic,
                actor_optimizer, critic_optimizer, epoch, best_reward, config,
            )

        pbar.update(1)

    pbar.close()

    # Final checkpoint
    save_checkpoint(
        save_dir, "final", actor, critic,
        actor_optimizer, critic_optimizer, epoch, best_reward, config,
    )

    # Restore signal handlers
    signal.signal(signal.SIGINT, original_sigint)
    signal.signal(signal.SIGTERM, original_sigterm)

    # Save metrics
    torch.save(metrics_history, log_dir / "metrics.pt")
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"Metrics saved to: {log_dir / 'metrics.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train underwater snake (Zheng et al. 2022)")
    parser.add_argument("--epochs", type=int, default=None, help="Override total_epochs")
    parser.add_argument("--stiffness", type=float, default=0.0, help="Joint stiffness (Nm/rad)")
    parser.add_argument("--fluid-density", type=float, default=1000.0, help="Fluid density (kg/m^3)")
    parser.add_argument("--fluid-viscosity", type=float, default=0.0009, help="Fluid viscosity (Pa-s)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--name", type=str, default="zheng2022", help="Experiment name")
    args = parser.parse_args()

    config = ZhengConfig(
        joint_stiffness=args.stiffness,
        fluid_density=args.fluid_density,
        fluid_viscosity=args.fluid_viscosity,
        device=args.device,
        experiment_name=args.name,
    )
    if args.epochs is not None:
        config.total_epochs = args.epochs

    train(config)


if __name__ == "__main__":
    main()
