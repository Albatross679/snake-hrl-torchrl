#!/usr/bin/env python3
"""Quick high curvature stability test with optimized parameters.

Tests snake stability at high curvature with:
1. Smaller timestep for better convergence
2. More iterations for the implicit solver
3. Comparison across curvature values
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snake_hrl.configs.env import PhysicsConfig
from snake_hrl.physics.snake_robot import SnakeRobot


def run_quick_stability_test(
    target_curvature: float,
    num_steps: int = 50,
    dt: float = 0.02,  # Smaller timestep
    max_iter: int = 100,  # More iterations
    verbose: bool = True,
) -> dict:
    """Run a quick stability test.

    Args:
        target_curvature: Target curvature
        num_steps: Number of steps
        dt: Timestep
        max_iter: Max Newton iterations
        verbose: Print progress

    Returns:
        Dict with results
    """
    config = PhysicsConfig(
        dt=dt,
        max_iter=max_iter,
        tol=1e-3,  # Looser tolerance for faster convergence
        ftol=1e-3,
        dtol=1e-2,
    )

    num_joints = config.num_segments - 1

    robot = SnakeRobot(
        config=config,
        initial_snake_position=np.array([0.0, 0.0, 0.0]),
    )

    # Set target curvatures
    target_curvatures = np.full(num_joints, target_curvature)
    robot.set_curvature_control(target_curvatures)

    mean_curvatures = []
    std_curvatures = []
    head_positions = []

    start_time = time.time()

    for step in range(num_steps):
        state = robot.step()

        curvs = state['curvatures']
        mean_curvatures.append(np.mean(np.abs(curvs)))
        std_curvatures.append(np.std(curvs))
        head_positions.append(state['positions'][0].copy())

        if verbose and step % 10 == 0:
            print(f"  Step {step:3d}: κ_mean={mean_curvatures[-1]:6.2f}, κ_std={std_curvatures[-1]:6.2f}")

    elapsed = time.time() - start_time

    final_state = robot.get_state()

    result = {
        'target': target_curvature,
        'final_mean_curv': mean_curvatures[-1],
        'final_std_curv': std_curvatures[-1],
        'mean_curvatures': mean_curvatures,
        'std_curvatures': std_curvatures,
        'head_positions': np.array(head_positions),
        'final_positions': final_state['positions'],
        'final_curvatures': final_state['curvatures'],
        'elapsed_time': elapsed,
        'tracking_error': np.mean(np.abs(np.array(mean_curvatures) - target_curvature)),
    }

    # Check stability
    result['is_stable'] = (
        not np.any(np.isnan(mean_curvatures)) and
        max(std_curvatures) < 50 and
        np.max(np.abs(np.array(head_positions))) < 5
    )

    return result


def main():
    """Run stability tests."""
    print("="*70)
    print("HIGH CURVATURE STABILITY TEST")
    print("="*70)
    print("\nParameters: dt=0.02s, max_iter=100, 50 steps")
    print("\nTesting curvatures: 2, 4, 6, 8, 10")

    curvatures = [2.0, 4.0, 6.0, 8.0, 10.0]
    results = {}

    for kappa in curvatures:
        print(f"\n{'='*50}")
        print(f"Testing κ = {kappa}")
        print('='*50)
        results[kappa] = run_quick_stability_test(kappa, num_steps=50, verbose=True)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Target κ':>10} {'Final κ':>10} {'Std':>10} {'Error':>10} {'Stable':>8} {'Time':>8}")
    print("-"*70)

    for kappa in curvatures:
        r = results[kappa]
        stable = "YES" if r['is_stable'] else "NO"
        print(f"{kappa:>10.1f} {r['final_mean_curv']:>10.2f} {r['final_std_curv']:>10.2f} "
              f"{r['tracking_error']:>10.2f} {stable:>8} {r['elapsed_time']:>7.1f}s")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Curvature evolution
    ax1 = axes[0, 0]
    for kappa in curvatures:
        r = results[kappa]
        ax1.plot(r['mean_curvatures'], label=f'κ={kappa}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Curvature')
    ax1.set_title('Curvature Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Curvature std (uniformity)
    ax2 = axes[0, 1]
    for kappa in curvatures:
        r = results[kappa]
        ax2.plot(r['std_curvatures'], label=f'κ={kappa}')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Curvature Std')
    ax2.set_title('Curvature Uniformity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Target vs Achieved
    ax3 = axes[1, 0]
    final_curvs = [results[k]['final_mean_curv'] for k in curvatures]
    ax3.scatter(curvatures, final_curvs, s=100, c='blue', label='Achieved')
    ax3.plot([0, 12], [0, 12], 'r--', label='Ideal')
    ax3.set_xlabel('Target κ')
    ax3.set_ylabel('Achieved κ')
    ax3.set_title('Target vs Achieved Curvature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, max(final_curvs) * 1.1)

    # 4. Final snake configurations
    ax4 = axes[1, 1]
    ax4.set_aspect('equal')
    colors = plt.cm.viridis(np.linspace(0, 1, len(curvatures)))
    for kappa, color in zip(curvatures, colors):
        r = results[kappa]
        pos = r['final_positions']
        ax4.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, label=f'κ={kappa}')
        ax4.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Final Snake Configurations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'stability_quick_test.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    plt.show()

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Check κ=10 specifically
    r10 = results[10.0]
    print(f"\nAt κ = 10.0 (required for coiling):")
    print(f"  Achieved curvature: {r10['final_mean_curv']:.2f}")
    print(f"  Curvature std: {r10['final_std_curv']:.2f}")
    print(f"  Tracking error: {r10['tracking_error']:.2f}")
    print(f"  Stable: {'YES' if r10['is_stable'] else 'NO'}")

    if r10['is_stable'] and r10['tracking_error'] < 5:
        print("\n✓ CONCLUSION: Snake CAN remain stable at κ = 10")
    else:
        print("\n✗ CONCLUSION: Snake has stability issues at κ = 10")


if __name__ == '__main__':
    main()
