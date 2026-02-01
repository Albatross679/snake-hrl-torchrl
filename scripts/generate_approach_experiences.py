#!/usr/bin/env python3
"""Generate approach experiences for behavioral cloning pretraining.

This script generates successful locomotion experiences via grid search over
serpenoid parameters. Experiences are saved in .npz format for use in
pretraining the approach worker policy via behavioral cloning.

The script uses "retrospective goal labeling": since grid search has no predefined
target, we run the simulation, compute the actual displacement, and set the goal
direction as the normalized displacement direction.

Usage:
    python scripts/generate_approach_experiences.py \\
        --amplitude 0.5 1.0 1.5 2.0 \\
        --frequency 0.5 1.0 1.5 2.0 \\
        --wave-number 1.0 1.5 2.0 2.5 \\
        --phase 0.0 1.57 3.14 4.71 \\
        --duration 5.0 \\
        --min-displacement 0.05 \\
        --output data/approach_experiences.npz \\
        --verbose

Examples:
    # Basic grid search
    python scripts/generate_approach_experiences.py \\
        --amplitude 0.5 1.0 1.5 \\
        --frequency 0.5 1.0 1.5 \\
        --wave-number 1.0 2.0 \\
        --output data/approach_experiences.npz

    # Dense grid with direction diversity
    python scripts/generate_approach_experiences.py \\
        --amplitude 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \\
        --frequency 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \\
        --wave-number 1.0 1.5 2.0 2.5 3.0 \\
        --phase 0.0 0.79 1.57 2.36 3.14 3.93 4.71 5.50 \\
        --duration 5.0 \\
        --min-displacement 0.1 \\
        --ensure-direction-diversity \\
        --top-k-per-bin 5 \\
        --verbose \\
        --output data/approach_experiences_dense.npz
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake_hrl.configs.env import PhysicsConfig
from snake_hrl.demonstrations.approach_experiences import (
    ApproachExperienceGenerator,
)
from snake_hrl.demonstrations.fitness import compute_direction_coverage


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate approach experiences for behavioral cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Grid parameters
    parser.add_argument(
        "--amplitude",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Amplitude values for grid search (default: 0.5 1.0 1.5 2.0)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Frequency values for grid search (default: 0.5 1.0 1.5 2.0)",
    )
    parser.add_argument(
        "--wave-number",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5],
        help="Wave number values for grid search (default: 1.0 1.5 2.0 2.5)",
    )
    parser.add_argument(
        "--phase",
        type=float,
        nargs="+",
        default=[0.0, 1.57, 3.14, 4.71],
        help="Phase values for grid search (default: 0.0 1.57 3.14 4.71)",
    )

    # Simulation parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of each trajectory in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Sample every N states from each trajectory (default: 5)",
    )

    # Filtering parameters
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=0.05,
        help="Minimum displacement threshold in meters (default: 0.05)",
    )
    parser.add_argument(
        "--ensure-direction-diversity",
        action="store_true",
        help="Keep best trajectories per direction bin",
    )
    parser.add_argument(
        "--num-direction-bins",
        type=int,
        default=8,
        help="Number of direction bins for diversity (default: 8)",
    )
    parser.add_argument(
        "--top-k-per-bin",
        type=int,
        default=3,
        help="Keep top K trajectories per direction bin (default: 3)",
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

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/approach_experiences.npz",
        help="Output file path (default: data/approach_experiences.npz)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    start_time = time.time()

    # Print configuration
    if args.verbose:
        total = (
            len(args.amplitude)
            * len(args.frequency)
            * len(args.wave_number)
            * len(args.phase)
        )
        print("=" * 60)
        print("Approach Experience Generation")
        print("=" * 60)
        print(f"\nGrid parameters:")
        print(f"  Amplitudes:    {args.amplitude}")
        print(f"  Frequencies:   {args.frequency}")
        print(f"  Wave numbers:  {args.wave_number}")
        print(f"  Phases:        {args.phase}")
        print(f"  Total combinations: {total}")
        print(f"\nSimulation:")
        print(f"  Duration:      {args.duration}s")
        print(f"  Sample interval: {args.sample_interval}")
        print(f"\nFiltering:")
        print(f"  Min displacement: {args.min_displacement}m")
        print(f"  Direction diversity: {args.ensure_direction_diversity}")
        if args.ensure_direction_diversity:
            print(f"  Direction bins: {args.num_direction_bins}")
            print(f"  Top K per bin: {args.top_k_per_bin}")
        print()

    # Create physics config
    physics_config = PhysicsConfig(
        snake_length=args.snake_length,
        num_segments=args.num_segments,
        dt=args.dt,
    )

    # Create generator
    generator = ApproachExperienceGenerator(physics_config)

    # Generate experiences
    buffer = generator.generate_from_grid(
        amplitude_values=args.amplitude,
        wave_number_values=args.wave_number,
        frequency_values=args.frequency,
        phase_values=args.phase,
        duration=args.duration,
        min_displacement=args.min_displacement,
        ensure_direction_diversity=args.ensure_direction_diversity,
        num_direction_bins=args.num_direction_bins,
        top_k_per_bin=args.top_k_per_bin,
        sample_interval=args.sample_interval,
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
    print(f"\nState statistics:")
    print(f"  Mean: {states.mean(axis=0)[:5]}...")
    print(f"  Std:  {states.std(axis=0)[:5]}...")
    print(f"\nAction statistics:")
    print(f"  Mean: {actions.mean(axis=0)}")
    print(f"  Std:  {actions.std(axis=0)}")
    print(f"\nOutput:")
    print(f"  File:     {output_path}")
    print(f"  Size:     {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Time:     {elapsed:.1f}s")


if __name__ == "__main__":
    main()
