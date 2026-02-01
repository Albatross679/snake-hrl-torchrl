#!/usr/bin/env python
"""Train the approach skill with direct curvature control using BC pretrained weights.

This script trains the approaching worker RL policy initialized with behavioral
cloning pretrained weights, using:
- State: REDUCED_APPROACH (13-dim)
- Action: DIRECT control (19-dim curvatures)

The BC policy provides a warm start with reasonable locomotion, and RL fine-tuning
improves goal-directed behavior.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt

from snake_hrl.envs import ApproachEnv
from snake_hrl.trainers import PPOTrainer
from snake_hrl.configs.env import (
    ApproachEnvConfig,
    CPGConfig,
    ControlMethod,
    StateRepresentation,
)
from snake_hrl.configs.training import PPOConfig
from snake_hrl.configs.network import NetworkConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train approach skill with direct curvature control"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="checkpoints/approach_curvature_policy.pt",
        help="Path to BC pretrained weights",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=500_000,
        help="Total training frames",
    )
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=4096,
        help="Frames per training batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Log directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="approach_curvature_rl",
        help="Experiment name",
    )
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
        help="Skip loading pretrained weights (train from scratch)",
    )
    return parser.parse_args()


def load_bc_weights_to_actor(actor, bc_weights_path: str, device: str = "cpu") -> bool:
    """Transfer BC pretrained weights to ActorNetwork mean pathway.

    The BC policy is a simple Sequential MLP:
        Linear(13→128) → ReLU → Linear(128→128) → ReLU →
        Linear(128→64) → ReLU → Linear(64→19) → Tanh

    The ActorNetwork has:
        mlp: Sequential with Linear layers interleaved with activations
        mean_head: Linear(64→19)
        log_std_head: Linear(64→19)

    We map:
        BC layer 0 → ActorNetwork.mlp[0]
        BC layer 2 → ActorNetwork.mlp[2]
        BC layer 4 → ActorNetwork.mlp[4]
        BC layer 6 → ActorNetwork.mean_head

    Args:
        actor: TorchRL ProbabilisticActor wrapping ActorNetwork
        bc_weights_path: Path to BC pretrained weights (.pt file)
        device: Device to load weights on

    Returns:
        True if weights loaded successfully, False otherwise
    """
    if not os.path.exists(bc_weights_path):
        print(f"Warning: BC weights not found at {bc_weights_path}")
        return False

    try:
        bc_state = torch.load(bc_weights_path, map_location=device)

        # Access the ActorNetwork inside TorchRL wrappers
        # ProbabilisticActor -> ModuleList[0] (TensorDictModule) -> ActorNetwork
        actor_net = actor.module[0].module

        # Verify architecture compatibility
        expected_keys = ['0.weight', '0.bias', '2.weight', '2.bias',
                         '4.weight', '4.bias', '6.weight', '6.bias']
        if not all(k in bc_state for k in expected_keys):
            print(f"Warning: BC weights missing expected keys. Found: {list(bc_state.keys())}")
            return False

        # Check dimensions
        bc_input_dim = bc_state['0.weight'].shape[1]
        bc_output_dim = bc_state['6.weight'].shape[0]
        actor_input_dim = actor_net.obs_dim
        actor_output_dim = actor_net.action_dim

        if bc_input_dim != actor_input_dim:
            print(f"Warning: Input dim mismatch - BC: {bc_input_dim}, Actor: {actor_input_dim}")
            return False
        if bc_output_dim != actor_output_dim:
            print(f"Warning: Output dim mismatch - BC: {bc_output_dim}, Actor: {actor_output_dim}")
            return False

        # Transfer weights to MLP layers
        # BC uses indices 0, 2, 4 for Linear layers (1, 3, 5 are ReLU/activations)
        # ActorNetwork.mlp has same structure
        actor_net.mlp[0].weight.data = bc_state['0.weight'].clone()
        actor_net.mlp[0].bias.data = bc_state['0.bias'].clone()
        actor_net.mlp[2].weight.data = bc_state['2.weight'].clone()
        actor_net.mlp[2].bias.data = bc_state['2.bias'].clone()
        actor_net.mlp[4].weight.data = bc_state['4.weight'].clone()
        actor_net.mlp[4].bias.data = bc_state['4.bias'].clone()

        # Transfer BC output layer to mean_head
        actor_net.mean_head.weight.data = bc_state['6.weight'].clone()
        actor_net.mean_head.bias.data = bc_state['6.bias'].clone()

        print(f"Successfully loaded BC weights from {bc_weights_path}")
        print(f"  Input dim: {bc_input_dim}, Output dim: {bc_output_dim}")
        print(f"  Hidden dims: {bc_state['0.weight'].shape[0]} -> "
              f"{bc_state['2.weight'].shape[0]} -> {bc_state['4.weight'].shape[0]}")

        return True

    except Exception as e:
        print(f"Error loading BC weights: {e}")
        return False


def save_training_plots(
    metrics: List[Dict[str, Any]],
    save_path: str,
    experiment_name: str,
) -> None:
    """Generate and save training plots.

    Args:
        metrics: List of metrics dictionaries from training
        save_path: Directory to save plots
        experiment_name: Name for the plot file
    """
    if not metrics:
        print("No metrics to plot")
        return

    # Extract data from metrics
    frames = [m.get('total_frames', i * 4096) for i, m in enumerate(metrics)]
    actor_losses = [m.get('loss_actor', 0) for m in metrics]
    critic_losses = [m.get('loss_critic', 0) for m in metrics]
    rewards = [m.get('mean_episode_reward', None) for m in metrics]

    # Filter out None rewards and their corresponding frames
    reward_frames = [f for f, r in zip(frames, rewards) if r is not None]
    rewards = [r for r in rewards if r is not None]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Progress: {experiment_name}', fontsize=14)

    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    if rewards:
        ax1.plot(reward_frames, rewards, 'b-', alpha=0.6, label='Episode Reward')
        # Add smoothed curve
        if len(rewards) > 10:
            window = min(50, len(rewards) // 5)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            smoothed_frames = reward_frames[window-1:]
            ax1.plot(smoothed_frames, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
        ax1.legend()
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Training Reward')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Actor Loss
    ax2 = axes[0, 1]
    ax2.plot(frames, actor_losses, 'g-', alpha=0.6)
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Loss')
    ax2.set_title('Actor (Policy) Loss')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Critic Loss
    ax3 = axes[1, 0]
    ax3.plot(frames, critic_losses, 'm-', alpha=0.6)
    ax3.set_xlabel('Frames')
    ax3.set_ylabel('Loss')
    ax3.set_title('Critic (Value) Loss')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Combined losses
    ax4 = axes[1, 1]
    ax4.plot(frames, actor_losses, 'g-', alpha=0.6, label='Actor Loss')
    ax4.plot(frames, critic_losses, 'm-', alpha=0.6, label='Critic Loss')
    ax4.set_xlabel('Frames')
    ax4.set_ylabel('Loss')
    ax4.set_title('Combined Losses')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f'{experiment_name}_training.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training plots saved to {plot_path}")


def save_metrics_npz(
    metrics: List[Dict[str, Any]],
    save_path: str,
    experiment_name: str,
) -> None:
    """Save training metrics to npz file.

    Args:
        metrics: List of metrics dictionaries from training
        save_path: Directory to save metrics
        experiment_name: Name for the metrics file
    """
    if not metrics:
        return

    # Extract arrays
    frames = np.array([m.get('total_frames', 0) for m in metrics])
    actor_losses = np.array([m.get('loss_actor', 0) for m in metrics])
    critic_losses = np.array([m.get('loss_critic', 0) for m in metrics])
    entropy_losses = np.array([m.get('loss_entropy', 0) for m in metrics])
    kl_divergences = np.array([m.get('kl_divergence', 0) for m in metrics])

    # Rewards (with NaN for missing values)
    rewards = np.array([
        m.get('mean_episode_reward', np.nan) for m in metrics
    ])

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / f'{experiment_name}_metrics.npz'

    np.savez(
        metrics_path,
        frames=frames,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        entropy_losses=entropy_losses,
        kl_divergences=kl_divergences,
        rewards=rewards,
    )

    print(f"Training metrics saved to {metrics_path}")


def main():
    args = parse_args()

    print("=" * 70)
    print("Training Approach Skill with Direct Curvature Control")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Total frames: {args.total_frames:,}")
    print(f"Frames per batch: {args.frames_per_batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Pretrained weights: {args.pretrained}")
    print("=" * 70)

    # Set random seeds
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment with REDUCED_APPROACH state and DIRECT control
    cpg_config = CPGConfig(control_method=ControlMethod.DIRECT)

    env_config = ApproachEnvConfig(
        state_representation=StateRepresentation.REDUCED_APPROACH,
        cpg=cpg_config,
        max_episode_steps=500,
        use_reward_shaping=True,
        # Reward shaping parameters
        energy_penalty_weight=0.01,
        success_bonus=1.0,
        distance_reward_weight=1.0,
        velocity_reward_weight=0.1,
    )

    print(f"\nEnvironment Configuration:")
    print(f"  State representation: {env_config.state_representation.value}")
    print(f"  Control method: {cpg_config.control_method.value}")
    print(f"  Observation dim: {env_config.obs_dim}")
    print(f"  Action dim: {env_config.action_dim}")

    env = ApproachEnv(config=env_config, device=args.device)

    # Create training config
    train_config = PPOConfig(
        total_frames=args.total_frames,
        frames_per_batch=args.frames_per_batch,
        learning_rate=args.lr,
        num_epochs=10,
        mini_batch_size=256,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )

    # Create network config matching BC architecture
    # BC network: [128, 128, 64] hidden dims with ReLU activations
    network_config = NetworkConfig()
    network_config.actor.hidden_dims = [128, 128, 64]
    network_config.actor.activation = "relu"  # Match BC network
    network_config.critic.hidden_dims = [128, 128, 64]
    network_config.critic.activation = "relu"

    print(f"\nNetwork Configuration:")
    print(f"  Actor hidden dims: {network_config.actor.hidden_dims}")
    print(f"  Actor activation: {network_config.actor.activation}")
    print(f"  Critic hidden dims: {network_config.critic.hidden_dims}")

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        config=train_config,
        network_config=network_config,
        device=args.device,
    )

    # Load pretrained BC weights
    if not args.skip_pretrained:
        print(f"\nLoading pretrained BC weights...")
        success = load_bc_weights_to_actor(
            trainer.actor,
            args.pretrained,
            args.device
        )
        if success:
            print("BC weights loaded successfully - policy initialized with learned locomotion")
        else:
            print("Warning: Could not load BC weights - training from scratch")
    else:
        print("\nSkipping pretrained weights - training from scratch")

    # Verify dimensions
    print(f"\nDimension Verification:")
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]
    print(f"  Environment obs_dim: {obs_dim}")
    print(f"  Environment action_dim: {action_dim}")
    print(f"  Expected obs_dim (REDUCED_APPROACH): 13")
    print(f"  Expected action_dim (DIRECT): 19")

    assert obs_dim == 13, f"Expected obs_dim=13, got {obs_dim}"
    assert action_dim == 19, f"Expected action_dim=19, got {action_dim}"
    print("  Dimensions verified!")

    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")

    # Train
    results = trainer.train()

    # Print final results
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total frames: {results['total_frames']:,}")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Best reward: {results['best_reward']:.2f}")

    # Save training plots
    print("\nGenerating training plots...")
    save_training_plots(
        results['metrics'],
        'figures',
        args.experiment_name,
    )

    # Save metrics to npz
    save_metrics_npz(
        results['metrics'],
        f'checkpoints/{args.experiment_name}',
        args.experiment_name,
    )

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate(num_episodes=20)
    print(f"Evaluation Results:")
    print(f"  Mean reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    print(f"  Mean episode length: {eval_results['mean_length']:.0f}")
    print(f"  Min reward: {eval_results['min_reward']:.2f}")
    print(f"  Max reward: {eval_results['max_reward']:.2f}")

    return results


if __name__ == "__main__":
    main()
