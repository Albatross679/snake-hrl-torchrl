#!/usr/bin/env python3
"""High curvature stability test.

This script tests whether the snake can remain stable when applying
high curvature values (κ = 10) over extended simulation time.

Key questions:
1. Does the snake maintain stable curvature at κ = 10?
2. Does the simulation converge or diverge over time?
3. What is the relationship between target and achieved curvature?
4. Are there numerical stability issues at high curvature?
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snake_hrl.configs.env import PhysicsConfig
from snake_hrl.physics.snake_robot import SnakeRobot


@dataclass
class StabilityTestResult:
    """Results from a stability test."""
    target_curvature: float
    num_steps: int

    # Curvature metrics over time
    mean_curvatures: List[float]
    std_curvatures: List[float]
    max_curvatures: List[float]
    min_curvatures: List[float]

    # Position metrics
    head_positions: np.ndarray  # (num_steps, 3)
    cog_positions: np.ndarray   # (num_steps, 3)

    # Energy metrics
    total_displacements: List[float]

    # Convergence info
    convergence_failures: int

    # Final state
    final_positions: np.ndarray
    final_curvatures: np.ndarray

    @property
    def is_stable(self) -> bool:
        """Check if simulation remained stable."""
        # Stable if:
        # 1. Curvature std didn't explode
        # 2. No NaN values
        # 3. Snake didn't fly off to infinity
        if any(np.isnan(self.mean_curvatures)):
            return False
        if any(np.isnan(self.head_positions.flatten())):
            return False
        if max(self.std_curvatures) > 100:  # Curvature became chaotic
            return False
        if np.max(np.abs(self.head_positions)) > 10:  # Snake flew away
            return False
        return True

    @property
    def curvature_tracking_error(self) -> float:
        """Mean absolute error between target and achieved curvature."""
        return np.mean(np.abs(np.array(self.mean_curvatures) - self.target_curvature))


def run_stability_test(
    target_curvature: float,
    num_steps: int = 500,
    config: PhysicsConfig = None,
    verbose: bool = True,
) -> StabilityTestResult:
    """Run a stability test at given curvature.

    Args:
        target_curvature: Target curvature to apply
        num_steps: Number of simulation steps
        config: Physics configuration
        verbose: Print progress

    Returns:
        StabilityTestResult with metrics
    """
    if config is None:
        config = PhysicsConfig()

    num_joints = config.num_segments - 1
    num_nodes = config.num_segments + 1

    # Create robot starting from origin
    robot = SnakeRobot(
        config=config,
        initial_snake_position=np.array([0.0, 0.0, 0.0]),
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"STABILITY TEST: κ = {target_curvature}")
        print(f"{'='*60}")
        print(f"Steps: {num_steps}, dt: {config.dt}s, total time: {num_steps * config.dt:.1f}s")

    # Set target curvatures
    target_curvatures = np.full(num_joints, target_curvature)
    robot.set_curvature_control(target_curvatures)

    # Storage for metrics
    mean_curvatures = []
    std_curvatures = []
    max_curvatures = []
    min_curvatures = []
    head_positions = []
    cog_positions = []
    total_displacements = []
    convergence_failures = 0

    initial_head_pos = robot.snake.positions[0].copy()

    for step in range(num_steps):
        try:
            state = robot.step()
        except Exception as e:
            if verbose:
                print(f"Step {step}: Exception - {e}")
            convergence_failures += 1
            # Try to continue
            state = robot.get_state()

        # Track curvature metrics
        curvs = state['curvatures']
        mean_curvatures.append(np.mean(np.abs(curvs)))
        std_curvatures.append(np.std(curvs))
        max_curvatures.append(np.max(np.abs(curvs)))
        min_curvatures.append(np.min(np.abs(curvs)))

        # Track position metrics
        positions = state['positions']
        head_positions.append(positions[0].copy())
        cog = np.mean(positions, axis=0)
        cog_positions.append(cog)

        # Track displacement
        displacement = np.linalg.norm(positions[0] - initial_head_pos)
        total_displacements.append(displacement)

        # Check for convergence warnings (they're printed by DisMech)
        # We count them via the iteration limit message

        if verbose and step % 100 == 0:
            print(f"Step {step:4d}: κ_mean={mean_curvatures[-1]:6.2f}, "
                  f"κ_std={std_curvatures[-1]:6.2f}, "
                  f"head_pos=({positions[0][0]:6.3f}, {positions[0][1]:6.3f})")

    final_state = robot.get_state()

    result = StabilityTestResult(
        target_curvature=target_curvature,
        num_steps=num_steps,
        mean_curvatures=mean_curvatures,
        std_curvatures=std_curvatures,
        max_curvatures=max_curvatures,
        min_curvatures=min_curvatures,
        head_positions=np.array(head_positions),
        cog_positions=np.array(cog_positions),
        total_displacements=total_displacements,
        convergence_failures=convergence_failures,
        final_positions=final_state['positions'],
        final_curvatures=final_state['curvatures'],
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Stable: {result.is_stable}")
        print(f"  Final mean κ: {mean_curvatures[-1]:.2f} (target: {target_curvature})")
        print(f"  Tracking error: {result.curvature_tracking_error:.2f}")
        print(f"  Max κ observed: {max(max_curvatures):.2f}")
        print(f"  Total head displacement: {total_displacements[-1]:.3f}m")

    return result


def run_curvature_sweep_stability(
    curvatures: List[float] = None,
    num_steps: int = 300,
    config: PhysicsConfig = None,
    verbose: bool = True,
) -> Dict[float, StabilityTestResult]:
    """Run stability tests across multiple curvature values.

    Args:
        curvatures: List of curvature values to test
        num_steps: Steps per test
        config: Physics configuration
        verbose: Print progress

    Returns:
        Dict mapping curvature to StabilityTestResult
    """
    if curvatures is None:
        curvatures = [2.0, 4.0, 6.0, 8.0, 9.0, 9.5, 10.0]

    if config is None:
        config = PhysicsConfig()

    results = {}

    print("\n" + "="*70)
    print("HIGH CURVATURE STABILITY SWEEP")
    print("="*70)

    for kappa in curvatures:
        result = run_stability_test(
            target_curvature=kappa,
            num_steps=num_steps,
            config=config,
            verbose=verbose,
        )
        results[kappa] = result

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'κ_target':>10} {'Stable':>8} {'κ_final':>10} {'κ_error':>10} {'Displacement':>12}")
    print("-"*70)

    for kappa in curvatures:
        r = results[kappa]
        stable_str = "YES" if r.is_stable else "NO"
        print(f"{kappa:>10.1f} {stable_str:>8} {r.mean_curvatures[-1]:>10.2f} "
              f"{r.curvature_tracking_error:>10.2f} {r.total_displacements[-1]:>12.3f}m")

    return results


def run_long_term_stability_test(
    target_curvature: float = 10.0,
    duration_seconds: float = 30.0,
    config: PhysicsConfig = None,
    verbose: bool = True,
) -> StabilityTestResult:
    """Run extended stability test to check long-term behavior.

    Args:
        target_curvature: Target curvature
        duration_seconds: Total simulation time in seconds
        config: Physics configuration
        verbose: Print progress

    Returns:
        StabilityTestResult
    """
    if config is None:
        config = PhysicsConfig()

    num_steps = int(duration_seconds / config.dt)

    print("\n" + "="*70)
    print(f"LONG-TERM STABILITY TEST: κ = {target_curvature}")
    print(f"Duration: {duration_seconds}s ({num_steps} steps)")
    print("="*70)

    return run_stability_test(
        target_curvature=target_curvature,
        num_steps=num_steps,
        config=config,
        verbose=verbose,
    )


def visualize_stability_results(
    results: Dict[float, StabilityTestResult],
    save_path: str = None,
):
    """Visualize stability test results.

    Args:
        results: Dict from run_curvature_sweep_stability
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))

    curvatures = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(curvatures)))

    # 1. Curvature evolution over time
    ax1 = fig.add_subplot(2, 3, 1)
    for kappa, color in zip(curvatures, colors):
        r = results[kappa]
        steps = np.arange(len(r.mean_curvatures))
        ax1.plot(steps, r.mean_curvatures, color=color, label=f'κ={kappa}', alpha=0.8)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Curvature')
    ax1.set_title('Curvature Evolution Over Time')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Curvature std over time (stability indicator)
    ax2 = fig.add_subplot(2, 3, 2)
    for kappa, color in zip(curvatures, colors):
        r = results[kappa]
        steps = np.arange(len(r.std_curvatures))
        ax2.plot(steps, r.std_curvatures, color=color, label=f'κ={kappa}', alpha=0.8)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Curvature Std')
    ax2.set_title('Curvature Uniformity Over Time')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Target vs Achieved curvature
    ax3 = fig.add_subplot(2, 3, 3)
    final_curvs = [results[k].mean_curvatures[-1] for k in curvatures]
    ax3.scatter(curvatures, final_curvs, s=100, c='blue', label='Achieved')
    ax3.plot([0, max(curvatures)], [0, max(curvatures)], 'r--', label='Ideal')
    ax3.set_xlabel('Target Curvature')
    ax3.set_ylabel('Achieved Curvature')
    ax3.set_title('Target vs Achieved Curvature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Head trajectory (top-down view)
    ax4 = fig.add_subplot(2, 3, 4)
    for kappa, color in zip(curvatures, colors):
        r = results[kappa]
        ax4.plot(r.head_positions[:, 0], r.head_positions[:, 1],
                color=color, label=f'κ={kappa}', alpha=0.7)
        ax4.scatter(r.head_positions[-1, 0], r.head_positions[-1, 1],
                   color=color, s=50, marker='o')

    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Head Trajectory (Top-Down)')
    ax4.set_aspect('equal')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Displacement over time
    ax5 = fig.add_subplot(2, 3, 5)
    for kappa, color in zip(curvatures, colors):
        r = results[kappa]
        steps = np.arange(len(r.total_displacements))
        ax5.plot(steps, r.total_displacements, color=color, label=f'κ={kappa}', alpha=0.8)

    ax5.set_xlabel('Step')
    ax5.set_ylabel('Head Displacement (m)')
    ax5.set_title('Total Displacement Over Time')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Stability summary bar chart
    ax6 = fig.add_subplot(2, 3, 6)
    tracking_errors = [results[k].curvature_tracking_error for k in curvatures]
    bars = ax6.bar(range(len(curvatures)), tracking_errors, color=colors)
    ax6.set_xticks(range(len(curvatures)))
    ax6.set_xticklabels([f'{k}' for k in curvatures])
    ax6.set_xlabel('Target Curvature')
    ax6.set_ylabel('Mean Tracking Error')
    ax6.set_title('Curvature Tracking Error')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add stability indicators
    for i, kappa in enumerate(curvatures):
        if results[kappa].is_stable:
            ax6.text(i, tracking_errors[i] + 0.1, '✓', ha='center', fontsize=14, color='green')
        else:
            ax6.text(i, tracking_errors[i] + 0.1, '✗', ha='center', fontsize=14, color='red')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def visualize_single_test(
    result: StabilityTestResult,
    save_path: str = None,
):
    """Visualize a single stability test result.

    Args:
        result: StabilityTestResult
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(14, 10))

    steps = np.arange(result.num_steps)

    # 1. Curvature over time
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(steps, result.mean_curvatures, 'b-', label='Mean κ', linewidth=2)
    ax1.fill_between(steps,
                     np.array(result.mean_curvatures) - np.array(result.std_curvatures),
                     np.array(result.mean_curvatures) + np.array(result.std_curvatures),
                     alpha=0.3, color='blue', label='±1 std')
    ax1.axhline(y=result.target_curvature, color='r', linestyle='--',
                label=f'Target κ={result.target_curvature}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Curvature')
    ax1.set_title(f'Curvature Stability Test (κ_target = {result.target_curvature})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Snake configuration snapshots
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_aspect('equal')

    # Plot trajectory
    ax2.plot(result.head_positions[:, 0], result.head_positions[:, 1],
            'b-', alpha=0.5, linewidth=1, label='Head trajectory')

    # Plot final snake configuration
    ax2.plot(result.final_positions[:, 0], result.final_positions[:, 1],
            'g-', linewidth=3, label='Final snake')
    ax2.plot(result.final_positions[0, 0], result.final_positions[0, 1],
            'ro', markersize=10, label='Head')
    ax2.plot(result.final_positions[-1, 0], result.final_positions[-1, 1],
            'ko', markersize=8, label='Tail')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Snake Configuration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Curvature profile
    ax3 = fig.add_subplot(2, 2, 3)
    joint_indices = np.arange(1, len(result.final_curvatures) + 1)
    ax3.bar(joint_indices, result.final_curvatures, alpha=0.7, color='blue')
    ax3.axhline(y=result.target_curvature, color='r', linestyle='--',
                label=f'Target κ={result.target_curvature}')
    ax3.set_xlabel('Joint Index')
    ax3.set_ylabel('Curvature')
    ax3.set_title('Final Curvature Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Statistics summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    stats_text = f"""
    STABILITY TEST RESULTS
    ══════════════════════════════════════

    Target Curvature:     κ = {result.target_curvature}
    Simulation Steps:     {result.num_steps}

    CURVATURE METRICS
    ────────────────────────────────────
    Final Mean κ:         {result.mean_curvatures[-1]:.3f}
    Final Std κ:          {result.std_curvatures[-1]:.3f}
    Max κ Observed:       {max(result.max_curvatures):.3f}
    Min κ Observed:       {min(result.min_curvatures):.3f}
    Tracking Error:       {result.curvature_tracking_error:.3f}

    STABILITY METRICS
    ────────────────────────────────────
    Simulation Stable:    {'YES ✓' if result.is_stable else 'NO ✗'}
    Total Displacement:   {result.total_displacements[-1]:.3f} m
    Convergence Issues:   {result.convergence_failures}

    ASSESSMENT
    ────────────────────────────────────
    {'PASSED: Snake remains stable at high curvature' if result.is_stable else 'FAILED: Simulation unstable'}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='High curvature stability test')
    parser.add_argument('--test', type=str, default='sweep',
                       choices=['single', 'sweep', 'long'],
                       help='Test type: single (one κ), sweep (multiple κ), long (extended time)')
    parser.add_argument('--curvature', type=float, default=10.0,
                       help='Target curvature for single/long tests')
    parser.add_argument('--steps', type=int, default=300,
                       help='Number of simulation steps')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Duration in seconds for long test')
    parser.add_argument('--save-fig', type=str, default=None,
                       help='Path to save figure')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    config = PhysicsConfig()

    if args.test == 'single':
        result = run_stability_test(
            target_curvature=args.curvature,
            num_steps=args.steps,
            config=config,
            verbose=args.verbose,
        )

        save_path = args.save_fig or os.path.join(
            os.path.dirname(__file__), '..', 'figures',
            f'stability_test_k{args.curvature}.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_single_test(result, save_path)

    elif args.test == 'sweep':
        results = run_curvature_sweep_stability(
            curvatures=[2.0, 4.0, 6.0, 8.0, 9.0, 9.5, 10.0],
            num_steps=args.steps,
            config=config,
            verbose=args.verbose,
        )

        save_path = args.save_fig or os.path.join(
            os.path.dirname(__file__), '..', 'figures', 'stability_sweep.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_stability_results(results, save_path)

    elif args.test == 'long':
        result = run_long_term_stability_test(
            target_curvature=args.curvature,
            duration_seconds=args.duration,
            config=config,
            verbose=args.verbose,
        )

        save_path = args.save_fig or os.path.join(
            os.path.dirname(__file__), '..', 'figures',
            f'stability_long_k{args.curvature}.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_single_test(result, save_path)


if __name__ == '__main__':
    main()
