#!/usr/bin/env python3
"""Generate demonstration trajectories for gait-based reward shaping.

This script generates serpenoid locomotion demonstrations using the physics
simulation and saves them for later use in reward shaping.

Usage:
    python scripts/generate_demos.py --num-demos 100 --output data/demos/serpenoid.pkl
    python scripts/generate_demos.py --num-demos 50 --duration 10.0 --output data/demos/long.pkl
    python scripts/generate_demos.py --grid --output data/demos/grid.pkl  # Parameter grid

Examples:
    # Generate 100 random demonstrations
    python scripts/generate_demos.py \\
        --num-demos 100 \\
        --duration 5.0 \\
        --output data/demos/serpenoid_demos.pkl

    # Generate demonstrations on a parameter grid
    python scripts/generate_demos.py \\
        --grid \\
        --amplitudes 0.5 1.0 1.5 2.0 \\
        --wave-numbers 1.0 2.0 3.0 \\
        --frequencies 0.5 1.0 1.5 2.0 \\
        --output data/demos/grid_demos.pkl

    # Grid search with fitness evaluation and direction diversity
    python scripts/generate_demos.py \\
        --grid \\
        --amplitudes 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \\
        --wave-numbers 1.0 1.5 2.0 2.5 3.0 \\
        --frequencies 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \\
        --phases 0.0 1.57 3.14 4.71 \\
        --duration 5.0 \\
        --evaluate \\
        --min-displacement 0.1 \\
        --ensure-direction-diversity \\
        --verbose \\
        --output data/demos/successful_gaits.pkl
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake_hrl.configs.env import PhysicsConfig
from snake_hrl.demonstrations.generators import SerpenoidGenerator
from snake_hrl.demonstrations.io import save_demonstrations
from snake_hrl.demonstrations.fitness import (
    evaluate_trajectory,
    filter_successful_trajectories,
    compute_direction_coverage,
    get_best_parameters_per_direction,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate serpenoid locomotion demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/demos/serpenoid_demos.pkl",
        help="Output file path (default: data/demos/serpenoid_demos.pkl)",
    )

    # Generation mode
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Generate on parameter grid instead of random sampling",
    )

    # Random sampling parameters
    parser.add_argument(
        "--num-demos", "-n",
        type=int,
        default=100,
        help="Number of demonstrations to generate (default: 100)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Duration of each trajectory in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Parameter ranges for random sampling
    parser.add_argument(
        "--amplitude-min",
        type=float,
        default=0.5,
        help="Minimum amplitude (default: 0.5)",
    )
    parser.add_argument(
        "--amplitude-max",
        type=float,
        default=2.0,
        help="Maximum amplitude (default: 2.0)",
    )
    parser.add_argument(
        "--wave-number-min",
        type=float,
        default=1.0,
        help="Minimum wave number (default: 1.0)",
    )
    parser.add_argument(
        "--wave-number-max",
        type=float,
        default=3.0,
        help="Maximum wave number (default: 3.0)",
    )
    parser.add_argument(
        "--frequency-min",
        type=float,
        default=0.5,
        help="Minimum frequency (default: 0.5)",
    )
    parser.add_argument(
        "--frequency-max",
        type=float,
        default=2.0,
        help="Maximum frequency (default: 2.0)",
    )

    # Grid parameters
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Amplitude values for grid (default: 0.5 1.0 1.5 2.0)",
    )
    parser.add_argument(
        "--wave-numbers",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Wave number values for grid (default: 1.0 2.0 3.0)",
    )
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Frequency values for grid (default: 0.5 1.0 1.5 2.0)",
    )
    parser.add_argument(
        "--phases",
        type=float,
        nargs="+",
        default=[0.0],
        help="Phase values for grid (default: 0.0). Use multiple phases for direction diversity.",
    )

    # Fitness evaluation options
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable fitness evaluation (compute displacement metrics)",
    )
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=0.1,
        help="Minimum displacement threshold in meters (default: 0.1)",
    )
    parser.add_argument(
        "--ensure-direction-diversity",
        action="store_true",
        help="Keep best trajectory per direction bin to ensure directional coverage",
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
        default=1,
        help="Keep top K trajectories per direction bin (default: 1)",
    )
    parser.add_argument(
        "--sort-by-fitness",
        action="store_true",
        help="Sort output trajectories by displacement magnitude (descending)",
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
        default=1e-3,
        help="Simulation timestep (default: 1e-3)",
    )

    # Sampling
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="State recording rate in Hz (default: every step)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )

    return parser.parse_args()


def generate_random_demos(
    generator: SerpenoidGenerator,
    num_demos: int,
    duration: float,
    amplitude_range: tuple,
    wave_number_range: tuple,
    frequency_range: tuple,
    seed: Optional[int],
    sample_rate: Optional[float],
    verbose: bool,
) -> tuple:
    """Generate random demonstrations.

    Returns:
        Tuple of (trajectories, parameters, metadata)
    """
    if verbose:
        print(f"Generating {num_demos} random demonstrations...")
        print(f"  Amplitude range: {amplitude_range}")
        print(f"  Wave number range: {wave_number_range}")
        print(f"  Frequency range: {frequency_range}")
        print(f"  Duration: {duration}s")

    if seed is not None:
        np.random.seed(seed)

    trajectories = []
    parameters = []

    start_time = time.time()

    for i in range(num_demos):
        # Sample random parameters
        amplitude = np.random.uniform(*amplitude_range)
        wave_number = np.random.uniform(*wave_number_range)
        frequency = np.random.uniform(*frequency_range)
        phase = np.random.uniform(0, 2 * np.pi)

        # Generate trajectory
        traj = generator.generate_physics_trajectory(
            amplitude=amplitude,
            wave_number=wave_number,
            frequency=frequency,
            duration=duration,
            phase=phase,
            sample_rate=sample_rate,
        )

        trajectories.append(traj)
        parameters.append({
            "amplitude": amplitude,
            "wave_number": wave_number,
            "frequency": frequency,
            "phase": phase,
        })

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (num_demos - i - 1)
            print(f"  Generated {i + 1}/{num_demos} demos (ETA: {eta:.1f}s)")

    metadata = {
        "generation_mode": "random",
        "num_demos": num_demos,
        "duration": duration,
        "amplitude_range": amplitude_range,
        "wave_number_range": wave_number_range,
        "frequency_range": frequency_range,
        "seed": seed,
    }

    if verbose:
        elapsed = time.time() - start_time
        total_states = sum(len(t) for t in trajectories)
        print(f"Generated {num_demos} demos ({total_states} states) in {elapsed:.1f}s")

    return trajectories, parameters, metadata


def generate_grid_demos(
    generator: SerpenoidGenerator,
    amplitudes: List[float],
    wave_numbers: List[float],
    frequencies: List[float],
    phases: List[float],
    duration: float,
    sample_rate: Optional[float],
    verbose: bool,
) -> tuple:
    """Generate demonstrations on a parameter grid.

    Returns:
        Tuple of (trajectories, parameters, metadata)
    """
    num_demos = len(amplitudes) * len(wave_numbers) * len(frequencies) * len(phases)

    if verbose:
        print(f"Generating demonstrations on parameter grid...")
        print(f"  Amplitudes: {amplitudes}")
        print(f"  Wave numbers: {wave_numbers}")
        print(f"  Frequencies: {frequencies}")
        print(f"  Phases: {phases}")
        print(f"  Total: {num_demos} demos")
        print(f"  Duration: {duration}s")

    trajectories, parameters = generator.get_parameter_grid(
        amplitude_values=amplitudes,
        wave_number_values=wave_numbers,
        frequency_values=frequencies,
        phase_values=phases,
        duration=duration,
        sample_rate=sample_rate,
        verbose=verbose,
    )

    metadata = {
        "generation_mode": "grid",
        "num_demos": num_demos,
        "duration": duration,
        "amplitudes": amplitudes,
        "wave_numbers": wave_numbers,
        "frequencies": frequencies,
        "phases": phases,
    }

    if verbose:
        total_states = sum(len(t) for t in trajectories)
        print(f"Generated {num_demos} demos ({total_states} states)")

    return trajectories, parameters, metadata


def main():
    """Main entry point."""
    args = parse_args()

    # Create physics config
    physics_config = PhysicsConfig(
        snake_length=args.snake_length,
        num_segments=args.num_segments,
        dt=args.dt,
    )

    # Create generator
    generator = SerpenoidGenerator(physics_config)

    # Generate demonstrations
    if args.grid:
        trajectories, parameters, metadata = generate_grid_demos(
            generator=generator,
            amplitudes=args.amplitudes,
            wave_numbers=args.wave_numbers,
            frequencies=args.frequencies,
            phases=args.phases,
            duration=args.duration,
            sample_rate=args.sample_rate,
            verbose=args.verbose,
        )
    else:
        trajectories, parameters, metadata = generate_random_demos(
            generator=generator,
            num_demos=args.num_demos,
            duration=args.duration,
            amplitude_range=(args.amplitude_min, args.amplitude_max),
            wave_number_range=(args.wave_number_min, args.wave_number_max),
            frequency_range=(args.frequency_min, args.frequency_max),
            seed=args.seed,
            sample_rate=args.sample_rate,
            verbose=args.verbose,
        )

    # Add physics config to metadata
    metadata["physics_config"] = {
        "snake_length": physics_config.snake_length,
        "num_segments": physics_config.num_segments,
        "dt": physics_config.dt,
    }

    # Fitness evaluation and filtering
    fitness_info = None
    if args.evaluate:
        if args.verbose:
            print(f"\nEvaluating trajectory fitness...")

        # Evaluate all trajectories
        fitness_info = [evaluate_trajectory(t) for t in trajectories]

        # Print pre-filter statistics
        magnitudes = [info["displacement_magnitude"] for info in fitness_info]
        if args.verbose:
            print(f"  Pre-filter statistics:")
            print(f"    Displacement range: {min(magnitudes):.3f}m - {max(magnitudes):.3f}m")
            print(f"    Mean displacement: {np.mean(magnitudes):.3f}m")
            print(f"    Passing threshold ({args.min_displacement}m): {sum(m >= args.min_displacement for m in magnitudes)}/{len(magnitudes)}")

        # Filter by displacement threshold and optionally direction diversity
        original_count = len(trajectories)
        trajectories, parameters, fitness_info = filter_successful_trajectories(
            trajectories,
            parameters,
            min_displacement=args.min_displacement,
            ensure_direction_diversity=args.ensure_direction_diversity,
            num_direction_bins=args.num_direction_bins,
            top_k_per_bin=args.top_k_per_bin,
        )

        if args.verbose:
            print(f"  After filtering: {len(trajectories)}/{original_count} trajectories")

        # Compute and print direction coverage
        if fitness_info:
            coverage = compute_direction_coverage(fitness_info, args.num_direction_bins)
            if args.verbose:
                print(f"\n  Direction coverage: {coverage['bins_covered']}/{coverage['total_bins']} bins")
                print(f"    Coverage ratio: {coverage['coverage_ratio']*100:.1f}%")
                if coverage["bin_counts"]:
                    print(f"    Bin counts: {coverage['bin_counts']}")
                    print(f"    Best displacement per bin:")
                    for bin_name, disp in sorted(coverage["best_per_bin"].items()):
                        print(f"      {bin_name}: {disp:.3f}m")

            # Get best parameters per direction
            best_params = get_best_parameters_per_direction(
                parameters, fitness_info, args.num_direction_bins
            )
            if args.verbose and best_params:
                print(f"\n  Best CPG parameters per direction:")
                for bin_name, info in sorted(best_params.items()):
                    p = info["parameters"]
                    print(f"    {bin_name}: A={p['amplitude']:.2f}, k={p['wave_number']:.2f}, "
                          f"f={p['frequency']:.2f}, phi={p.get('phase', 0.0):.2f} "
                          f"-> {info['displacement']:.3f}m")

            # Sort by fitness if requested
            if args.sort_by_fitness and fitness_info:
                sorted_indices = sorted(
                    range(len(fitness_info)),
                    key=lambda i: fitness_info[i]["displacement_magnitude"],
                    reverse=True,
                )
                trajectories = [trajectories[i] for i in sorted_indices]
                parameters = [parameters[i] for i in sorted_indices]
                fitness_info = [fitness_info[i] for i in sorted_indices]

        # Update metadata with evaluation info
        metadata["evaluation"] = {
            "min_displacement": args.min_displacement,
            "ensure_direction_diversity": args.ensure_direction_diversity,
            "num_direction_bins": args.num_direction_bins,
            "top_k_per_bin": args.top_k_per_bin,
            "original_count": original_count,
            "filtered_count": len(trajectories),
        }
        if fitness_info:
            metadata["evaluation"]["direction_coverage"] = compute_direction_coverage(
                fitness_info, args.num_direction_bins
            )

    # Save demonstrations
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_demonstrations(
        trajectories=trajectories,
        path=output_path,
        metadata=metadata,
        parameters=parameters,
        fitness_info=fitness_info,
    )

    if args.verbose:
        print(f"\nSaved demonstrations to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Print summary
    total_states = sum(len(t) for t in trajectories)
    print(f"\nGeneration complete:")
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Total states: {total_states}")
    if fitness_info:
        mags = [info["displacement_magnitude"] for info in fitness_info]
        print(f"  Displacement range: {min(mags):.3f}m - {max(mags):.3f}m")
        coverage = compute_direction_coverage(fitness_info, args.num_direction_bins)
        print(f"  Direction coverage: {coverage['bins_covered']}/{coverage['total_bins']} bins")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
