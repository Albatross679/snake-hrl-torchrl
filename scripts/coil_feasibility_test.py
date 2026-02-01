#!/usr/bin/env python3
"""Coil feasibility test script.

This script tests whether the snake can actually coil around the prey using
curvature control, with different strategies:
1. Gradual curvature ramping
2. Different starting positions (closer to prey)
3. Analysis of what curvature range is actually achievable

The key insight is that we need to:
- Start the snake close to or touching the prey
- Gradually increase curvature to avoid numerical instability
- Verify contact and wrap metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snake_hrl.configs.env import PhysicsConfig
from snake_hrl.physics.snake_robot import SnakeRobot


def test_gradual_coiling(
    target_curvature: float = 9.0,
    ramp_steps: int = 50,
    hold_steps: int = 150,
    config: PhysicsConfig = None,
    verbose: bool = True,
):
    """Test coiling with gradual curvature ramping.

    Args:
        target_curvature: Final target curvature
        ramp_steps: Steps to ramp up curvature
        hold_steps: Steps to hold target curvature
        config: Physics configuration
        verbose: Print progress

    Returns:
        history: List of states
        success: Whether coil criteria were met
    """
    if config is None:
        config = PhysicsConfig()

    num_joints = config.num_segments - 1

    # Position snake so its head is touching the prey
    # Prey is at default position: [snake_length + 2*prey_radius, 0, prey_length/2]
    prey_x = config.snake_length + config.prey_radius * 2
    prey_center = np.array([prey_x, 0.0, config.prey_length / 2])

    # Snake starts from origin pointing toward prey
    snake_start = np.array([0.0, 0.0, 0.0])

    robot = SnakeRobot(
        config=config,
        initial_snake_position=snake_start,
        initial_prey_position=prey_center,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"GRADUAL COILING TEST (target κ = {target_curvature})")
        print(f"{'=' * 60}")
        print(f"Prey center: {prey_center}")
        print(f"Snake start: {snake_start}")

    history = []
    total_steps = ramp_steps + hold_steps

    for step in range(total_steps):
        # Ramp curvature
        if step < ramp_steps:
            t = step / ramp_steps
            current_curvature = target_curvature * t
        else:
            current_curvature = target_curvature

        # Apply uniform curvature
        curvatures = np.full(num_joints, current_curvature)
        robot.set_curvature_control(curvatures)

        state = robot.step()
        history.append(state)

        if verbose and step % 50 == 0:
            print(f"Step {step}: κ_target={current_curvature:.2f}, "
                  f"κ_actual={np.mean(np.abs(state['curvatures'])):.2f}, "
                  f"wrap={state['wrap_count']:.2f}, "
                  f"contact={state['contact_fraction']:.1%}")

    final = history[-1]
    success = final['wrap_count'] >= 1.0 and final['contact_fraction'] >= 0.3

    if verbose:
        print(f"\nFinal: wrap_count={final['wrap_count']:.2f}, "
              f"contact={final['contact_fraction']:.1%}")
        print(f"Success: {success}")

    return history, success


def test_approach_then_coil(
    approach_steps: int = 100,
    coil_steps: int = 200,
    approach_curvature: float = 2.0,
    coil_curvature: float = 8.0,
    config: PhysicsConfig = None,
    verbose: bool = True,
):
    """Two-phase strategy: approach with low curvature, then increase for coiling.

    Args:
        approach_steps: Steps for approach phase
        coil_steps: Steps for coiling phase
        approach_curvature: Curvature during approach (low, for forward motion)
        coil_curvature: Curvature during coiling (high)
        config: Physics configuration
        verbose: Print progress

    Returns:
        history: List of states
        success: Whether coil criteria were met
    """
    if config is None:
        config = PhysicsConfig()

    num_joints = config.num_segments - 1

    # Position prey closer to snake for faster approach
    prey_center = np.array([0.5, 0.0, config.prey_length / 2])
    snake_start = np.array([0.0, 0.0, 0.0])

    robot = SnakeRobot(
        config=config,
        initial_snake_position=snake_start,
        initial_prey_position=prey_center,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"APPROACH-THEN-COIL TEST")
        print(f"{'=' * 60}")
        print(f"Phase 1: Approach with κ={approach_curvature}")
        print(f"Phase 2: Coil with κ={coil_curvature}")

    history = []

    # Phase 1: Approach with serpenoid-like motion
    for step in range(approach_steps):
        # Use oscillating curvature for locomotion
        t = step * config.dt
        freq = 2.0
        wave_num = 2.0

        # Serpenoid pattern with slight bias toward prey direction
        curvatures = np.zeros(num_joints)
        for i in range(num_joints):
            s = i / (num_joints - 1)  # Normalized position along body
            curvatures[i] = approach_curvature * np.sin(
                wave_num * 2 * np.pi * s - freq * 2 * np.pi * t
            )

        robot.set_curvature_control(curvatures)
        state = robot.step()
        history.append(state)

        if verbose and step % 50 == 0:
            print(f"Approach step {step}: distance={state['prey_distance']:.3f}, "
                  f"contact={state['contact_fraction']:.1%}")

    # Phase 2: Coil with high uniform curvature
    for step in range(coil_steps):
        # Gradually transition to uniform high curvature
        t = min(step / 50.0, 1.0)  # Transition over 50 steps
        current_curvature = approach_curvature * (1 - t) + coil_curvature * t
        curvatures = np.full(num_joints, current_curvature)

        robot.set_curvature_control(curvatures)
        state = robot.step()
        history.append(state)

        if verbose and step % 50 == 0:
            actual_step = approach_steps + step
            print(f"Coil step {actual_step}: κ={current_curvature:.2f}, "
                  f"wrap={state['wrap_count']:.2f}, "
                  f"contact={state['contact_fraction']:.1%}")

    final = history[-1]
    success = final['wrap_count'] >= 1.0 and final['contact_fraction'] >= 0.3

    if verbose:
        print(f"\nFinal: wrap_count={final['wrap_count']:.2f}, "
              f"contact={final['contact_fraction']:.1%}")
        print(f"Success: {success}")

    return history, success


def test_curvature_sweep(
    curvatures_to_test: list = None,
    steps_per_curvature: int = 100,
    config: PhysicsConfig = None,
):
    """Sweep through different curvature values and measure outcomes.

    Args:
        curvatures_to_test: List of curvature values to test
        steps_per_curvature: Steps to run for each curvature
        config: Physics configuration

    Returns:
        results: Dict mapping curvature to metrics
    """
    if config is None:
        config = PhysicsConfig()

    if curvatures_to_test is None:
        curvatures_to_test = [2.0, 4.0, 6.0, 8.0, 10.0]

    num_joints = config.num_segments - 1

    print(f"\n{'=' * 60}")
    print("CURVATURE SWEEP TEST")
    print(f"{'=' * 60}")

    results = {}

    for target_kappa in curvatures_to_test:
        # Fresh robot for each test
        prey_center = np.array([0.3, 0.0, config.prey_length / 2])
        snake_start = np.array([0.0, 0.0, 0.0])

        robot = SnakeRobot(
            config=config,
            initial_snake_position=snake_start,
            initial_prey_position=prey_center,
        )

        # Apply constant curvature
        curvatures = np.full(num_joints, target_kappa)
        robot.set_curvature_control(curvatures)

        for _ in range(steps_per_curvature):
            state = robot.step()

        final_state = state
        actual_curvature = np.mean(np.abs(final_state['curvatures']))
        theoretical_radius = 1.0 / target_kappa if target_kappa > 0 else float('inf')

        results[target_kappa] = {
            'target_curvature': target_kappa,
            'actual_curvature': actual_curvature,
            'theoretical_radius': theoretical_radius,
            'wrap_count': final_state['wrap_count'],
            'contact_fraction': final_state['contact_fraction'],
            'prey_distance': final_state['prey_distance'],
            'final_positions': final_state['positions'].copy(),
        }

        print(f"κ={target_kappa}: actual={actual_curvature:.2f}, "
              f"wrap={final_state['wrap_count']:.2f}, "
              f"contact={final_state['contact_fraction']:.1%}")

    return results


def visualize_sweep_results(results: dict, save_path: str = None):
    """Visualize curvature sweep results.

    Args:
        results: Dict from test_curvature_sweep
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(15, 10))

    # Extract data
    curvatures = sorted(results.keys())
    actual_curv = [results[k]['actual_curvature'] for k in curvatures]
    wrap_counts = [results[k]['wrap_count'] for k in curvatures]
    contact_fracs = [results[k]['contact_fraction'] for k in curvatures]

    # 1. Target vs Actual Curvature
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(curvatures, actual_curv, 'bo-', markersize=8, label='Actual')
    ax1.plot(curvatures, curvatures, 'r--', label='Ideal (target=actual)')
    ax1.set_xlabel('Target Curvature')
    ax1.set_ylabel('Actual Curvature')
    ax1.set_title('Target vs Actual Curvature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Wrap Count vs Curvature
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(curvatures, wrap_counts, width=0.8, alpha=0.7)
    ax2.axhline(y=1.5, color='r', linestyle='--', label='Success threshold')
    ax2.set_xlabel('Target Curvature')
    ax2.set_ylabel('Wrap Count')
    ax2.set_title('Wrap Count vs Curvature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Contact Fraction vs Curvature
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(curvatures, contact_fracs, width=0.8, alpha=0.7, color='green')
    ax3.axhline(y=0.6, color='r', linestyle='--', label='Success threshold')
    ax3.set_xlabel('Target Curvature')
    ax3.set_ylabel('Contact Fraction')
    ax3.set_title('Contact Fraction vs Curvature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4-6. Snake configurations for different curvatures
    config = PhysicsConfig()
    theta = np.linspace(0, 2 * np.pi, 100)
    prey_x = config.prey_radius * np.cos(theta) + 0.3  # Prey at x=0.3
    prey_y = config.prey_radius * np.sin(theta)

    for idx, (kappa, ax_idx) in enumerate(zip([min(curvatures), curvatures[len(curvatures)//2], max(curvatures)], [4, 5, 6])):
        ax = fig.add_subplot(2, 3, ax_idx)
        ax.set_aspect('equal')

        # Draw prey
        ax.fill(prey_x, prey_y, alpha=0.3, color='green')
        ax.plot(prey_x, prey_y, 'g-', linewidth=2)

        # Draw snake
        if kappa in results:
            pos = results[kappa]['final_positions']
            ax.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=3)
            ax.plot(pos[0, 0], pos[0, 1], 'ro', markersize=8)  # Head

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'κ = {kappa}, wrap = {results[kappa]["wrap_count"]:.2f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def analyze_achievable_coil_radius():
    """Analyze what coil radii are achievable given physics constraints."""
    config = PhysicsConfig()

    print(f"\n{'=' * 60}")
    print("ACHIEVABLE COIL RADIUS ANALYSIS")
    print(f"{'=' * 60}")

    print(f"\nPhysics parameters:")
    print(f"  Snake length: {config.snake_length} m")
    print(f"  Snake radius: {config.snake_radius} m")
    print(f"  Prey radius: {config.prey_radius} m")
    print(f"  Max curvature control: 10.0")

    min_coil_radius = 1.0 / 10.0  # At max curvature
    print(f"\nCoil radius analysis:")
    print(f"  Minimum coil radius (at κ_max=10): {min_coil_radius} m")
    print(f"  Prey radius: {config.prey_radius} m")

    if min_coil_radius <= config.prey_radius:
        print(f"  ✓ CAN coil around prey (R_coil ≤ R_prey)")
    else:
        print(f"  ✗ CANNOT coil around prey (need R_coil > R_prey)")
        required_kappa = 1.0 / config.prey_radius
        print(f"  Required κ for prey: {required_kappa:.2f}")

    # Wrap analysis
    print(f"\nWrap analysis at different curvatures:")
    for kappa in [5.0, 8.0, 10.0]:
        R = 1.0 / kappa
        circumference = 2 * np.pi * R
        max_wraps = config.snake_length / circumference
        fits_prey = R >= config.prey_radius

        print(f"  κ={kappa}: R={R:.3f}m, max_wraps={max_wraps:.2f}, "
              f"fits_prey={'Yes' if fits_prey else 'No'}")

    # Recommendations
    print(f"\nRecommendations:")
    print(f"  1. Use curvature κ ≈ 10 for tightest coil")
    print(f"  2. Expected wrap count: ~{config.snake_length / (2 * np.pi * 0.1):.1f}")
    print(f"  3. For RL: normalize action such that action=1 → κ=10")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Test coil feasibility')
    parser.add_argument('--test', type=str, default='all',
                       choices=['gradual', 'approach', 'sweep', 'analysis', 'all'],
                       help='Which test to run')
    parser.add_argument('--save-fig', type=str, default=None,
                       help='Path to save figure')
    args = parser.parse_args()

    if args.test in ['analysis', 'all']:
        analyze_achievable_coil_radius()

    if args.test in ['sweep', 'all']:
        results = test_curvature_sweep()
        save_path = args.save_fig or os.path.join(
            os.path.dirname(__file__), '..', 'figures', 'curvature_sweep.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_sweep_results(results, save_path)

    if args.test in ['gradual', 'all']:
        test_gradual_coiling(target_curvature=8.0, ramp_steps=100, hold_steps=200)

    if args.test in ['approach', 'all']:
        test_approach_then_coil()


if __name__ == '__main__':
    main()
