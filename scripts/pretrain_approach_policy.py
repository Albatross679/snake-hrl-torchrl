#!/usr/bin/env python3
"""Pretrain approach policy from experiences using behavioral cloning.

This script loads approach experiences and trains a policy network using
supervised learning (behavioral cloning). The pretrained weights can then
be used to initialize RL training for faster convergence.

Usage:
    python scripts/pretrain_approach_policy.py \\
        --experiences data/approach_experiences.npz \\
        --epochs 100 \\
        --batch-size 64 \\
        --lr 1e-3 \\
        --output checkpoints/approach_policy_pretrained.pt

Examples:
    # Basic pretraining
    python scripts/pretrain_approach_policy.py \\
        --experiences data/approach_experiences.npz \\
        --output checkpoints/approach_policy_pretrained.pt

    # Custom network and training
    python scripts/pretrain_approach_policy.py \\
        --experiences data/approach_experiences.npz \\
        --hidden-dims 128 128 64 \\
        --activation elu \\
        --epochs 200 \\
        --batch-size 128 \\
        --lr 5e-4 \\
        --weight-decay 1e-4 \\
        --validation-split 0.1 \\
        --early-stopping 20 \\
        --verbose \\
        --output checkpoints/approach_policy_pretrained.pt
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake_hrl.demonstrations.approach_experiences import ApproachExperienceBuffer
from snake_hrl.trainers.behavioral_cloning import (
    BehavioralCloningPretrainer,
    create_mlp_policy,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain approach policy using behavioral cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/Output
    parser.add_argument(
        "--experiences", "-e",
        type=str,
        required=True,
        help="Path to experience buffer (.npz file)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="checkpoints/approach_policy_pretrained.pt",
        help="Output path for pretrained weights (default: checkpoints/approach_policy_pretrained.pt)",
    )

    # Network architecture
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Hidden layer dimensions (default: 64 64)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "tanh", "elu", "leaky_relu"],
        default="relu",
        help="Activation function (default: relu)",
    )
    parser.add_argument(
        "--output-activation",
        type=str,
        choices=["none", "tanh", "sigmoid"],
        default="none",
        help="Output activation function (default: none)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", "-n",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization weight (default: 1e-4)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Stop if val loss doesn't improve for N epochs (default: disabled)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training (default: cpu)",
    )

    # Visualization
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training curves and save to file",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Path to save plot (default: same as output with .png extension)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )

    return parser.parse_args()


def plot_training_curves(
    metrics: dict,
    output_path: str,
    title: str = "Behavioral Cloning Training",
):
    """Plot training and validation loss curves.

    Args:
        metrics: Dictionary with train_loss and val_loss lists
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(metrics["train_loss"]) + 1)

    ax.plot(epochs, metrics["train_loss"], label="Training Loss", color="blue")

    if metrics.get("val_loss"):
        ax.plot(epochs, metrics["val_loss"], label="Validation Loss", color="orange")

        # Mark best epoch
        best_epoch = metrics.get("best_epoch", 0) + 1
        best_val_loss = metrics["val_loss"][metrics.get("best_epoch", 0)]
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5)
        ax.scatter([best_epoch], [best_val_loss], color="green", s=100, zorder=5)
        ax.annotate(
            f"Best: {best_val_loss:.4f} (epoch {best_epoch})",
            xy=(best_epoch, best_val_loss),
            xytext=(best_epoch + 5, best_val_loss),
            fontsize=10,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis to log scale if range is large
    if max(metrics["train_loss"]) / min(metrics["train_loss"]) > 10:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Main entry point."""
    args = parse_args()

    # Load experiences
    if args.verbose:
        print("=" * 60)
        print("Approach Policy Pretraining")
        print("=" * 60)
        print(f"\nLoading experiences from: {args.experiences}")

    buffer = ApproachExperienceBuffer()
    buffer.load(args.experiences)

    states, actions = buffer.to_dataset()
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    if args.verbose:
        print(f"  Experiences: {len(buffer)}")
        print(f"  State dim:   {state_dim}")
        print(f"  Action dim:  {action_dim}")

    # Create policy network
    output_act = None if args.output_activation == "none" else args.output_activation

    policy = create_mlp_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        output_activation=output_act,
    )

    if args.verbose:
        print(f"\nNetwork architecture:")
        print(f"  Hidden dims: {args.hidden_dims}")
        print(f"  Activation:  {args.activation}")
        print(f"  Output act:  {args.output_activation}")
        num_params = sum(p.numel() for p in policy.parameters())
        print(f"  Parameters:  {num_params:,}")

    # Create pretrainer
    pretrainer = BehavioralCloningPretrainer(
        policy=policy,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    if args.verbose:
        print(f"\nTraining configuration:")
        print(f"  Epochs:      {args.epochs}")
        print(f"  Batch size:  {args.batch_size}")
        print(f"  LR:          {args.lr}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Val split:   {args.validation_split}")
        print(f"  Early stop:  {args.early_stopping}")
        print(f"  Device:      {args.device}")
        print()

    # Train
    metrics = pretrainer.train(
        experience_buffer=buffer,
        num_epochs=args.epochs,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping,
        verbose=args.verbose,
    )

    # Evaluate
    eval_metrics = pretrainer.evaluate(buffer)

    # Save weights
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pretrainer.save_policy(str(output_path))

    # Save training metrics
    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "train_loss": metrics["train_loss"],
                "val_loss": metrics["val_loss"],
                "best_epoch": metrics["best_epoch"],
                "final_mse": eval_metrics["mse"],
                "final_mae": eval_metrics["mae"],
                "action_error_per_dim": eval_metrics["action_error_per_dim"],
                "config": {
                    "hidden_dims": args.hidden_dims,
                    "activation": args.activation,
                    "output_activation": args.output_activation,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
            },
            f,
            indent=2,
        )

    # Plot training curves
    if args.plot:
        plot_path = args.plot_output or str(output_path.with_suffix(".png"))
        plot_training_curves(metrics, plot_path)
        if args.verbose:
            print(f"  Plot saved to: {plot_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Pretraining Complete")
    print("=" * 60)
    print(f"\nFinal metrics:")
    print(f"  MSE:  {eval_metrics['mse']:.6f}")
    print(f"  MAE:  {eval_metrics['mae']:.6f}")
    print(f"  Per-dim error: {eval_metrics['action_error_per_dim']}")
    print(f"\nBest validation loss at epoch {metrics['best_epoch'] + 1}")
    print(f"\nOutput files:")
    print(f"  Weights: {output_path}")
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
