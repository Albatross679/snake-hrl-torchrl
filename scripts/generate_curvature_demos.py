#!/usr/bin/env python3
"""Generate curvature action experiences for direct control RL.

This script generates locomotion experiences where:
- State: REDUCED_APPROACH (13-dim)
- Action: 19 joint curvatures (computed from serpenoid steering)

The script uses DirectSerpenoidSteeringTransform (5 params) to generate diverse
locomotion behaviors, but records the 19-dim curvatures as the action.

Features:
- Randomized initial snake direction for each trajectory
- Grid search over serpenoid steering parameters
- Retrospective goal setting for state-action pairs
- Distribution analysis with direction and L2 norm histograms

Usage:
    python scripts/generate_curvature_demos.py \\
        --output data/demos/curvature_experiences.npz \\
        --amplitudes 0.5 1.0 1.5 \\
        --frequencies 0.5 1.0 1.5 \\
        --wave-numbers 1.0 2.0 \\
        --phases 0.0 3.14159 \\
        --turn-biases -1.5 -0.75 0.0 0.75 1.5 \\
        --duration 3.0 \\
        --sample-interval 0.1 \\
        --plot-distributions \\
        --verbose

Examples:
    # Default grid (180 trajectories)
    python scripts/generate_curvature_demos.py \\
        --output data/demos/curvature_experiences.npz \\
        --verbose

    # Small test run
    python scripts/generate_curvature_demos.py \\
        --output data/demos/test_curvature.npz \\
        --amplitudes 1.0 \\
        --frequencies 1.0 \\
        --wave-numbers 1.5 \\
        --phases 0.0 \\
        --turn-biases -1.0 0.0 1.0 \\
        --duration 2.0 \\
        --plot-distributions \\
        --verbose
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake_hrl.configs.env import PhysicsConfig
from snake_hrl.demonstrations.curvature_action_experiences import (
    CurvatureActionExperienceGenerator,
    analyze_trajectory_distribution,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate curvature action experiences for direct control RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/demos/curvature_experiences.npz",
        help="Output file path (default: data/demos/curvature_experiences.npz)",
    )

    # Grid parameters
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Amplitude values for grid search (default: 0.5 1.0 1.5)",
    )
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Frequency values for grid search (default: 0.5 1.0 1.5)",
    )
    parser.add_argument(
        "--wave-numbers",
        type=float,
        nargs="+",
        default=[1.0, 2.0],
        help="Wave number values for grid search (default: 1.0 2.0)",
    )
    parser.add_argument(
        "--phases",
        type=float,
        nargs="+",
        default=[0.0, np.pi],
        help="Phase values for grid search in radians (default: 0.0 pi)",
    )
    parser.add_argument(
        "--turn-biases",
        type=float,
        nargs="+",
        default=[-1.5, -0.75, 0.0, 0.75, 1.5],
        help="Turn bias values for grid search (default: -1.5 -0.75 0.0 0.75 1.5)",
    )

    # Simulation parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of each trajectory in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Time between recorded samples in seconds (default: 0.1)",
    )

    # Filtering
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=0.01,
        help="Minimum displacement threshold in meters (default: 0.01)",
    )

    # Physics configuration
    parser.add_argument(
        "--snake-length",
        type=float,
        default=1.0,
        help="Snake length in meters (default: 1.0)",
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=20,
        help="Number of snake segments (default: 20)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=5e-2,
        help="Simulation timestep (default: 0.05)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    # Output options
    parser.add_argument(
        "--plot-distributions",
        action="store_true",
        help="Show distribution plots after generation",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save distribution plot to file (e.g., 'distributions.png')",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )

    return parser.parse_args()


def print_direction_histogram(directions: np.ndarray, num_bins: int = 8) -> None:
    """Print ASCII histogram of trajectory directions.

    Args:
        directions: Array of direction angles in radians
        num_bins: Number of direction bins
    """
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    hist, _ = np.histogram(directions, bins=bin_edges)

    # Direction labels
    labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    if num_bins == 8:
        bin_labels = labels
    else:
        bin_labels = [f"Bin {i}" for i in range(num_bins)]

    # Find max for scaling
    max_count = max(hist) if len(hist) > 0 else 1
    bar_width = 40

    print("\nDirection Distribution:")
    print("-" * 60)
    for i, (label, count) in enumerate(zip(bin_labels, hist)):
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        angle = np.degrees((bin_edges[i] + bin_edges[i + 1]) / 2)
        pct = count / len(directions) * 100 if len(directions) > 0 else 0
        bar = "#" * bar_len
        print(f"  {label:3s} ({angle:6.1f}deg): {bar:<{bar_width}} {count:4d} ({pct:5.1f}%)")


def print_l2_histogram(l2_norms: np.ndarray, num_bins: int = 10) -> None:
    """Print ASCII histogram of L2 norms.

    Args:
        l2_norms: Array of displacement magnitudes
        num_bins: Number of bins
    """
    hist, bin_edges = np.histogram(l2_norms, bins=num_bins)

    # Find max for scaling
    max_count = max(hist) if len(hist) > 0 else 1
    bar_width = 40

    print("\nL2 Norm (Displacement) Distribution:")
    print("-" * 60)
    for i, count in enumerate(hist):
        low, high = bin_edges[i], bin_edges[i + 1]
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        pct = count / len(l2_norms) * 100 if len(l2_norms) > 0 else 0
        bar = "#" * bar_len
        print(f"  [{low:.3f}, {high:.3f}]: {bar:<{bar_width}} {count:4d} ({pct:5.1f}%)")


def plot_distributions(buffer, save_path: str = None) -> None:
    """Plot distribution histograms using matplotlib.

    Args:
        buffer: Experience buffer with trajectory info
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    directions = np.array(buffer.trajectory_info["directions"])
    l2_norms = np.array(buffer.trajectory_info["l2_norms"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Direction histogram (polar)
    ax1 = plt.subplot(121, projection='polar')
    num_bins = 16
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    hist, _ = np.histogram(directions, bins=bin_edges)

    # Center bars on bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 2 * np.pi / num_bins

    bars = ax1.bar(bin_centers, hist, width=width, alpha=0.7, edgecolor='black')
    ax1.set_title("Trajectory Direction Distribution", pad=20)
    ax1.set_theta_zero_location("E")  # 0 degrees = East

    # L2 norm histogram
    ax2 = axes[1]
    ax2.hist(l2_norms, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel("L2 Norm (Displacement in meters)")
    ax2.set_ylabel("Count")
    ax2.set_title("Displacement Magnitude Distribution")
    ax2.axvline(np.mean(l2_norms), color='r', linestyle='--', label=f'Mean: {np.mean(l2_norms):.3f}m')
    ax2.axvline(np.median(l2_norms), color='g', linestyle='--', label=f'Median: {np.median(l2_norms):.3f}m')
    ax2.legend()

    # Add statistics text
    stats_text = (
        f"N = {len(l2_norms)}\n"
        f"Min: {np.min(l2_norms):.3f}m\n"
        f"Max: {np.max(l2_norms):.3f}m\n"
        f"Std: {np.std(l2_norms):.3f}m"
    )
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def main():
    """Main entry point."""
    args = parse_args()

    start_time = time.time()

    # Calculate total combinations
    total = (
        len(args.amplitudes)
        * len(args.frequencies)
        * len(args.wave_numbers)
        * len(args.phases)
        * len(args.turn_biases)
    )

    # Print configuration
    if args.verbose:
        print("=" * 60)
        print("Curvature Action Experience Generation")
        print("=" * 60)
        print(f"\nGrid parameters:")
        print(f"  Amplitudes:    {args.amplitudes}")
        print(f"  Frequencies:   {args.frequencies}")
        print(f"  Wave numbers:  {args.wave_numbers}")
        print(f"  Phases:        {[f'{p:.4f}' for p in args.phases]}")
        print(f"  Turn biases:   {args.turn_biases}")
        print(f"  Total combinations: {total}")
        print(f"\nSimulation:")
        print(f"  Duration:        {args.duration}s")
        print(f"  Sample interval: {args.sample_interval}s")
        print(f"  Timestep:        {args.dt}s")
        print(f"\nFiltering:")
        print(f"  Min displacement: {args.min_displacement}m")
        print(f"\nOutput:")
        print(f"  File: {args.output}")
        if args.seed is not None:
            print(f"  Seed: {args.seed}")
        print()

    # Create physics config
    physics_config = PhysicsConfig(
        snake_length=args.snake_length,
        num_segments=args.num_segments,
        dt=args.dt,
    )

    # Create generator
    generator = CurvatureActionExperienceGenerator(physics_config)

    # Generate experiences
    buffer, stats = generator.generate_from_grid(
        amplitude_values=args.amplitudes,
        frequency_values=args.frequencies,
        wave_number_values=args.wave_numbers,
        phase_values=args.phases,
        turn_bias_values=args.turn_biases,
        duration=args.duration,
        sample_interval=args.sample_interval,
        min_displacement=args.min_displacement,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Save experiences
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer.save(str(output_path))

    elapsed = time.time() - start_time

    # Print summary
    states, actions = buffer.to_dataset()
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Experiences:   {len(buffer)}")
    print(f"  Trajectories:  {buffer.metadata.get('num_trajectories', 'N/A')}")
    print(f"  State shape:   {states.shape}")
    print(f"  Action shape:  {actions.shape}")

    print(f"\nState statistics (13-dim REDUCED_APPROACH):")
    state_labels = [
        "curv_amp", "curv_wn", "curv_ph",
        "orient_x", "orient_y", "orient_z",
        "angvel_x", "angvel_y", "angvel_z",
        "goal_x", "goal_y", "goal_z",
        "goal_dist",
    ]
    print(f"  {'Component':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for i, label in enumerate(state_labels):
        print(f"  {label:<12} {states[:, i].mean():8.4f} {states[:, i].std():8.4f} "
              f"{states[:, i].min():8.4f} {states[:, i].max():8.4f}")

    print(f"\nAction statistics (19-dim curvatures):")
    print(f"  Mean: {actions.mean():.4f}")
    print(f"  Std:  {actions.std():.4f}")
    print(f"  Min:  {actions.min():.4f}")
    print(f"  Max:  {actions.max():.4f}")

    # Print trajectory distribution analysis
    if buffer.trajectory_info["directions"]:
        directions = np.array(buffer.trajectory_info["directions"])
        l2_norms = np.array(buffer.trajectory_info["l2_norms"])

        print("\n" + "=" * 60)
        print("Trajectory Distribution Analysis")
        print("=" * 60)

        print_direction_histogram(directions)

        print(f"\nL2 Norm (Displacement) Statistics:")
        print(f"  Min:    {np.min(l2_norms):.4f} m")
        print(f"  Max:    {np.max(l2_norms):.4f} m")
        print(f"  Mean:   {np.mean(l2_norms):.4f} m")
        print(f"  Median: {np.median(l2_norms):.4f} m")
        print(f"  Std:    {np.std(l2_norms):.4f} m")

        print_l2_histogram(l2_norms)

    print(f"\nOutput:")
    print(f"  File:     {output_path}")
    print(f"  Size:     {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Time:     {elapsed:.1f}s")

    # Plot distributions if requested
    if args.plot_distributions or args.save_plot:
        plot_distributions(buffer, args.save_plot)


if __name__ == "__main__":
    main()
