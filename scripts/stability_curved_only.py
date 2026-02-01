#!/usr/bin/env python3
"""Quick test of stability from pre-curved positions only."""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snake_hrl.configs.env import PhysicsConfig

import dismech
from dismech import (
    SoftRobot, Geometry, GeomParams, Material, SimParams, Environment, ImplicitEulerTimeStepper,
)


def create_curved_geometry(num_segments, snake_length, initial_curvature):
    """Create curved rod geometry."""
    num_nodes = num_segments + 1
    segment_length = snake_length / num_segments

    if abs(initial_curvature) < 1e-6:
        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            nodes[i, 0] = i * segment_length
    else:
        radius = 1.0 / initial_curvature
        angle_per_segment = segment_length / radius
        nodes = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            angle = i * angle_per_segment
            nodes[i, 0] = radius * np.sin(angle)
            nodes[i, 1] = radius * (1 - np.cos(angle))

    edges = np.array([[i, i + 1] for i in range(num_segments)], dtype=np.int64)
    return Geometry(nodes, edges, np.empty((0, 3), dtype=np.int64), plot_from_txt=False)


def get_curvatures(positions):
    """Compute curvatures from positions."""
    num_nodes = len(positions)
    curvatures = []
    for i in range(1, num_nodes - 1):
        v1 = positions[i] - positions[i - 1]
        v2 = positions[i + 1] - positions[i]
        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            curvatures.append(0.0)
            continue
        cos_angle = np.clip(np.dot(v1/v1_norm, v2/v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        curvatures.append(angle / ((v1_norm + v2_norm) / 2))
    return np.array(curvatures)


def test_maintain_curvature(initial_curvature, target_curvature, num_steps=50):
    """Test maintaining curvature from pre-curved state."""
    config = PhysicsConfig(dt=0.02, max_iter=100, enable_gravity=False, use_rft=True)
    num_segments = config.num_segments
    num_joints = num_segments - 1
    num_nodes = num_segments + 1

    geometry = create_curved_geometry(num_segments, config.snake_length, initial_curvature)

    geom_params = GeomParams(rod_r0=config.snake_radius, shell_h=0)
    material = Material(density=config.density, youngs_rod=config.youngs_modulus,
                       youngs_shell=0, poisson_rod=config.poisson_ratio, poisson_shell=0)
    sim_params = SimParams(static_sim=False, two_d_sim=True, use_mid_edge=False,
                          use_line_search=True, log_data=False, log_step=1, show_floor=False,
                          dt=config.dt, max_iter=config.max_iter, total_time=1000.0,
                          plot_step=1, tol=1e-3, ftol=1e-3, dtol=1e-2)
    env = Environment()
    env.add_force('rft', ct=config.rft_ct, cn=config.rft_cn)

    robot = SoftRobot(geom_params, material, geometry, sim_params, env)
    time_stepper = ImplicitEulerTimeStepper(robot)

    # Set target curvatures
    bend_springs = robot.bend_springs
    if bend_springs.N > 0 and hasattr(bend_springs, 'nat_strain'):
        for i in range(min(num_joints, bend_springs.N)):
            bend_springs.nat_strain[i, 0] = target_curvature
            bend_springs.nat_strain[i, 1] = 0.0

    # Get initial curvature
    q = robot.state.q
    initial_positions = q[:3*num_nodes].reshape(num_nodes, 3)
    initial_curvs = get_curvatures(initial_positions)

    mean_curvatures = [np.mean(np.abs(initial_curvs))]

    print(f"\nκ_init={initial_curvature:.1f} → κ_target={target_curvature:.1f}")
    print(f"  Initial measured κ: {mean_curvatures[0]:.2f}")

    for step in range(num_steps):
        try:
            robot, _ = time_stepper.step(robot, debug=False)
        except:
            pass
        q = robot.state.q
        positions = q[:3*num_nodes].reshape(num_nodes, 3)
        curvs = get_curvatures(positions)
        mean_curvatures.append(np.mean(np.abs(curvs)))

        if step % 10 == 0:
            print(f"  Step {step:3d}: κ_mean={mean_curvatures[-1]:.2f}")

    final_curv = mean_curvatures[-1]
    error = abs(final_curv - target_curvature)
    is_stable = not np.isnan(final_curv) and error < 10 and final_curv < 50

    print(f"  Final κ: {final_curv:.2f}, Error: {error:.2f}, Stable: {'YES' if is_stable else 'NO'}")

    return {
        'initial': initial_curvature,
        'target': target_curvature,
        'final': final_curv,
        'error': error,
        'stable': is_stable,
        'history': mean_curvatures,
    }


def main():
    print("="*70)
    print("STABILITY TEST: PRE-CURVED INITIALIZATION")
    print("="*70)

    # Key tests: maintaining and increasing curvature from curved state
    tests = [
        (5.0, 5.0),   # Maintain κ=5
        (8.0, 8.0),   # Maintain κ=8
        (10.0, 10.0), # Maintain κ=10 (critical test!)
        (5.0, 10.0),  # Increase from 5 to 10
        (8.0, 10.0),  # Increase from 8 to 10
    ]

    results = []
    for init_k, target_k in tests:
        r = test_maintain_curvature(init_k, target_k, num_steps=50)
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Initial κ':>12} {'Target κ':>12} {'Final κ':>12} {'Error':>10} {'Stable':>8}")
    print("-"*70)
    for r in results:
        stable = "YES" if r['stable'] else "NO"
        print(f"{r['initial']:>12.1f} {r['target']:>12.1f} {r['final']:>12.2f} {r['error']:>10.2f} {stable:>8}")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS FOR κ=10 (COILING REQUIREMENT)")
    print("="*70)

    # Find the κ=10 → κ=10 test
    k10_test = next((r for r in results if r['initial'] == 10.0 and r['target'] == 10.0), None)
    if k10_test:
        print(f"\nMaintaining κ=10 from pre-curved κ=10 position:")
        print(f"  Final curvature: {k10_test['final']:.2f}")
        print(f"  Tracking error: {k10_test['error']:.2f}")
        print(f"  Stable: {'YES ✓' if k10_test['stable'] else 'NO ✗'}")

        if k10_test['stable']:
            print("\n✓ CONCLUSION: Snake CAN remain stable at κ=10 when pre-curved")
        else:
            print("\n✗ CONCLUSION: Snake CANNOT remain stable at κ=10")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    for r in results:
        label = f"κ={r['initial']}→{r['target']}"
        ax1.plot(r['history'], label=label)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Curvature')
    ax1.set_title('Curvature Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    targets = [r['target'] for r in results]
    finals = [r['final'] for r in results]
    colors = ['green' if r['stable'] else 'red' for r in results]
    ax2.scatter(targets, finals, c=colors, s=100)
    ax2.plot([0, 15], [0, 15], 'k--', alpha=0.5)
    ax2.set_xlabel('Target κ')
    ax2.set_ylabel('Final κ')
    ax2.set_title('Target vs Achieved (Green=stable, Red=unstable)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'figures/stability_curved_only.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
