#!/usr/bin/env python3
"""High curvature stability test starting from pre-coiled configuration.

Tests snake stability at high curvature when starting from a configuration
that already has some curvature (more realistic for coiling task).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snake_hrl.configs.env import PhysicsConfig

# DisMech imports
import dismech
from dismech import (
    SoftRobot,
    Geometry,
    GeomParams,
    Material,
    SimParams,
    Environment,
    ImplicitEulerTimeStepper,
)


def create_curved_geometry(
    num_segments: int,
    snake_length: float,
    initial_curvature: float,
) -> Geometry:
    """Create a curved rod geometry.

    Args:
        num_segments: Number of segments
        snake_length: Total arc length
        initial_curvature: Initial curvature (1/radius)

    Returns:
        DisMech Geometry with curved node positions
    """
    num_nodes = num_segments + 1
    segment_length = snake_length / num_segments

    if abs(initial_curvature) < 1e-6:
        # Straight line
        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            nodes[i, 0] = i * segment_length
        return Geometry(nodes, np.array([[i, i+1] for i in range(num_segments)], dtype=np.int64),
                       np.empty((0, 3), dtype=np.int64), plot_from_txt=False)

    # Curved: place nodes along arc
    radius = 1.0 / initial_curvature
    angle_per_segment = segment_length / radius

    nodes = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        angle = i * angle_per_segment
        nodes[i, 0] = radius * np.sin(angle)
        nodes[i, 1] = radius * (1 - np.cos(angle))
        nodes[i, 2] = 0.0

    edges = np.array([[i, i + 1] for i in range(num_segments)], dtype=np.int64)
    face_nodes = np.empty((0, 3), dtype=np.int64)

    return Geometry(nodes, edges, face_nodes, plot_from_txt=False)


def get_curvatures(positions: np.ndarray) -> np.ndarray:
    """Compute curvatures from node positions."""
    num_nodes = len(positions)
    curvatures = []

    for i in range(1, num_nodes - 1):
        v1 = positions[i] - positions[i - 1]
        v2 = positions[i + 1] - positions[i]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-8 or v2_norm < 1e-8:
            curvatures.append(0.0)
            continue

        cos_angle = np.clip(np.dot(v1/v1_norm, v2/v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        avg_length = (v1_norm + v2_norm) / 2

        curvatures.append(angle / avg_length if avg_length > 1e-8 else 0.0)

    return np.array(curvatures)


def run_stability_from_curved(
    initial_curvature: float,
    target_curvature: float,
    num_steps: int = 100,
    verbose: bool = True,
) -> dict:
    """Run stability test starting from curved configuration.

    Args:
        initial_curvature: Initial curvature of the snake
        target_curvature: Target curvature for control
        num_steps: Number of steps
        verbose: Print progress

    Returns:
        Dict with results
    """
    config = PhysicsConfig(
        dt=0.02,
        max_iter=100,
        enable_gravity=False,  # Simplify by disabling gravity
        use_rft=True,
    )

    num_segments = config.num_segments
    num_joints = num_segments - 1
    num_nodes = num_segments + 1

    # Create curved geometry
    geometry = create_curved_geometry(
        num_segments=num_segments,
        snake_length=config.snake_length,
        initial_curvature=initial_curvature,
    )

    # Create robot
    geom_params = GeomParams(rod_r0=config.snake_radius, shell_h=0)
    material = Material(
        density=config.density,
        youngs_rod=config.youngs_modulus,
        youngs_shell=0,
        poisson_rod=config.poisson_ratio,
        poisson_shell=0,
    )
    sim_params = SimParams(
        static_sim=False,
        two_d_sim=True,
        use_mid_edge=False,
        use_line_search=True,
        log_data=False,
        log_step=1,
        show_floor=False,
        dt=config.dt,
        max_iter=config.max_iter,
        total_time=1000.0,
        plot_step=1,
        tol=1e-3,
        ftol=1e-3,
        dtol=1e-2,
    )

    env = Environment()
    if config.use_rft:
        env.add_force('rft', ct=config.rft_ct, cn=config.rft_cn)

    robot = SoftRobot(geom_params, material, geometry, sim_params, env)
    time_stepper = ImplicitEulerTimeStepper(robot)

    # Get initial positions
    q = robot.state.q
    initial_positions = q[:3*num_nodes].reshape(num_nodes, 3)
    initial_curvs = get_curvatures(initial_positions)

    if verbose:
        print(f"\nInitial state:")
        print(f"  Initial curvature (geometric): {initial_curvature:.2f}")
        print(f"  Measured mean curvature: {np.mean(initial_curvs):.2f}")
        print(f"  Target curvature: {target_curvature:.2f}")

    # Set target curvatures in bend springs
    bend_springs = robot.bend_springs
    target_curvatures = np.full(num_joints, target_curvature)
    if bend_springs.N > 0 and hasattr(bend_springs, 'nat_strain'):
        for i in range(min(len(target_curvatures), bend_springs.N)):
            bend_springs.nat_strain[i, 0] = target_curvatures[i]
            bend_springs.nat_strain[i, 1] = 0.0

    mean_curvatures = [np.mean(initial_curvs)]
    std_curvatures = [np.std(initial_curvs)]
    positions_history = [initial_positions.copy()]

    start_time = time.time()

    for step in range(num_steps):
        try:
            robot, _ = time_stepper.step(robot, debug=False)
        except ValueError:
            pass  # Ignore convergence warnings

        q = robot.state.q
        positions = q[:3*num_nodes].reshape(num_nodes, 3)
        curvs = get_curvatures(positions)

        mean_curvatures.append(np.mean(np.abs(curvs)))
        std_curvatures.append(np.std(curvs))
        positions_history.append(positions.copy())

        if verbose and step % 20 == 0:
            print(f"  Step {step:3d}: κ_mean={mean_curvatures[-1]:6.2f}, κ_std={std_curvatures[-1]:6.2f}")

    elapsed = time.time() - start_time

    final_positions = positions_history[-1]
    final_curvatures = get_curvatures(final_positions)

    result = {
        'initial_curvature': initial_curvature,
        'target_curvature': target_curvature,
        'initial_measured': np.mean(initial_curvs),
        'final_mean_curv': mean_curvatures[-1],
        'final_std_curv': std_curvatures[-1],
        'mean_curvatures': mean_curvatures,
        'std_curvatures': std_curvatures,
        'final_positions': final_positions,
        'final_curvatures': final_curvatures,
        'elapsed_time': elapsed,
        'tracking_error': abs(mean_curvatures[-1] - target_curvature),
    }

    # Stability check
    result['is_stable'] = (
        not np.any(np.isnan(mean_curvatures)) and
        max(std_curvatures) < 20 and
        np.max(np.abs(final_positions)) < 5
    )

    if verbose:
        print(f"\nFinal state:")
        print(f"  Final mean κ: {result['final_mean_curv']:.2f}")
        print(f"  Tracking error: {result['tracking_error']:.2f}")
        print(f"  Stable: {'YES' if result['is_stable'] else 'NO'}")
        print(f"  Time: {elapsed:.1f}s")

    return result


def main():
    """Main function."""
    print("="*70)
    print("HIGH CURVATURE STABILITY TEST (PRE-CURVED INITIALIZATION)")
    print("="*70)

    # Test different scenarios
    tests = [
        # (initial_curvature, target_curvature, description)
        (0.0, 5.0, "Straight → κ=5"),
        (0.0, 10.0, "Straight → κ=10"),
        (5.0, 5.0, "Curved@5 → κ=5 (maintain)"),
        (5.0, 10.0, "Curved@5 → κ=10 (increase)"),
        (10.0, 10.0, "Curved@10 → κ=10 (maintain)"),
        (8.0, 10.0, "Curved@8 → κ=10 (increase)"),
    ]

    results = {}

    for init_k, target_k, desc in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {desc}")
        print(f"  Initial κ: {init_k}, Target κ: {target_k}")
        print('='*60)

        key = f"{init_k}->{target_k}"
        results[key] = run_stability_from_curved(
            initial_curvature=init_k,
            target_curvature=target_k,
            num_steps=100,
            verbose=True,
        )
        results[key]['description'] = desc

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Test':30} {'Initial κ':>10} {'Target κ':>10} {'Final κ':>10} {'Error':>8} {'Stable':>8}")
    print("-"*70)

    for key in results:
        r = results[key]
        stable = "YES" if r['is_stable'] else "NO"
        print(f"{r['description']:30} {r['initial_curvature']:>10.1f} {r['target_curvature']:>10.1f} "
              f"{r['final_mean_curv']:>10.2f} {r['tracking_error']:>8.2f} {stable:>8}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Curvature evolution - from straight
    ax1 = axes[0, 0]
    for key in results:
        if results[key]['initial_curvature'] == 0:
            r = results[key]
            ax1.plot(r['mean_curvatures'], label=f"→κ={r['target_curvature']}")
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Curvature')
    ax1.set_title('From Straight (κ_init=0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Curvature evolution - from curved
    ax2 = axes[0, 1]
    for key in results:
        if results[key]['initial_curvature'] > 0:
            r = results[key]
            ax2.plot(r['mean_curvatures'], label=f"κ={r['initial_curvature']}→{r['target_curvature']}")
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Mean Curvature')
    ax2.set_title('From Pre-Curved')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Target vs Achieved
    ax3 = axes[0, 2]
    targets = [results[k]['target_curvature'] for k in results]
    achieved = [results[k]['final_mean_curv'] for k in results]
    init_curvs = [results[k]['initial_curvature'] for k in results]

    colors = ['red' if ic == 0 else 'blue' for ic in init_curvs]
    ax3.scatter(targets, achieved, c=colors, s=100)
    ax3.plot([0, 15], [0, 15], 'k--', alpha=0.5, label='Ideal')
    ax3.set_xlabel('Target κ')
    ax3.set_ylabel('Achieved κ')
    ax3.set_title('Target vs Achieved\n(Red=from straight, Blue=from curved)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4-6. Final configurations
    for idx, (key, title) in enumerate([
        ("0.0->10.0", "Straight → κ=10"),
        ("10.0->10.0", "Curved@10 → κ=10"),
        ("5.0->10.0", "Curved@5 → κ=10"),
    ]):
        ax = axes[1, idx]
        ax.set_aspect('equal')

        if key in results:
            r = results[key]
            pos = r['final_positions']
            ax.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=3)
            ax.plot(pos[0, 0], pos[0, 1], 'ro', markersize=10, label='Head')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{title}\nFinal κ={results[key]["final_mean_curv"]:.1f}' if key in results else title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'stability_curved_init.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    plt.show()

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Check the critical case: maintaining κ=10 from κ=10
    if "10.0->10.0" in results:
        r = results["10.0->10.0"]
        print(f"\nMaintaining κ=10 from pre-curved κ=10:")
        print(f"  Final curvature: {r['final_mean_curv']:.2f}")
        print(f"  Tracking error: {r['tracking_error']:.2f}")
        print(f"  Stable: {'YES' if r['is_stable'] else 'NO'}")

        if r['is_stable'] and r['tracking_error'] < 3:
            print("\n✓ Snake CAN remain stable at κ=10 when starting from curved position")
        else:
            print("\n✗ Snake has issues maintaining κ=10")

    # Check transition from κ=5 to κ=10
    if "5.0->10.0" in results:
        r = results["5.0->10.0"]
        print(f"\nIncreasing from κ=5 to κ=10:")
        print(f"  Final curvature: {r['final_mean_curv']:.2f}")
        print(f"  Stable: {'YES' if r['is_stable'] else 'NO'}")


if __name__ == '__main__':
    main()
