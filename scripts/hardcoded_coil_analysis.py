#!/usr/bin/env python3
"""Hardcoded coil analysis script.

This script creates a geometrically ideal coiled snake configuration around a cylinder,
extracts curvature information, and verifies that the snake can maintain the coil
using curvature control.

The goal is to:
1. Prove that curvature control CAN achieve coiling
2. Extract the target curvature profile for a coiled configuration
3. Use this as a reference for RL policy initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snake_hrl.configs.env import PhysicsConfig, CoilEnvConfig
from snake_hrl.physics.snake_robot import SnakeRobot
from snake_hrl.physics.geometry import compute_wrap_angle, compute_contact_points


@dataclass
class CoilAnalysisResult:
    """Results from coil analysis."""
    # Geometric configuration
    positions: np.ndarray  # (n_nodes, 3) - node positions
    curvatures: np.ndarray  # (n_joints,) - curvatures at joints
    target_curvatures: np.ndarray  # (n_joints,) - target curvatures for control

    # Metrics
    wrap_angle: float  # Total wrap angle in radians
    wrap_count: float  # Number of full wraps
    contact_fraction: float  # Fraction of nodes in contact

    # Parameters used
    coil_radius: float
    prey_radius: float
    snake_length: float


def create_coiled_positions(
    num_nodes: int = 21,
    snake_length: float = 1.0,
    coil_radius: float = 0.1,
    prey_center: np.ndarray = None,
    start_angle: float = 0.0,
    coil_direction: int = 1,  # 1 for CCW, -1 for CW
) -> np.ndarray:
    """Create node positions for a snake coiled around a cylinder.

    Args:
        num_nodes: Number of nodes in the snake
        snake_length: Total arc length of the snake
        coil_radius: Radius of the coil (prey_radius + snake_radius)
        prey_center: Center of the prey cylinder (x, y, z)
        start_angle: Starting angle in radians
        coil_direction: 1 for counter-clockwise, -1 for clockwise

    Returns:
        positions: (num_nodes, 3) array of node positions
    """
    if prey_center is None:
        prey_center = np.array([0.0, 0.0, 0.0])

    # Arc length per segment
    segment_length = snake_length / (num_nodes - 1)

    # For a circle: arc_length = radius * angle
    # So angle = arc_length / radius
    total_angle = snake_length / coil_radius
    angle_per_segment = segment_length / coil_radius

    positions = np.zeros((num_nodes, 3))

    for i in range(num_nodes):
        angle = start_angle + coil_direction * i * angle_per_segment
        positions[i, 0] = prey_center[0] + coil_radius * np.cos(angle)
        positions[i, 1] = prey_center[1] + coil_radius * np.sin(angle)
        positions[i, 2] = prey_center[2]  # Keep z constant (2D coil)

    return positions


def compute_curvatures_from_positions(positions: np.ndarray) -> np.ndarray:
    """Compute discrete curvatures at internal nodes.

    This matches the computation in SnakeGeometryAdapter.get_curvatures()

    Args:
        positions: (n_nodes, 3) array of node positions

    Returns:
        curvatures: (n_nodes - 2,) array of curvatures at internal nodes
    """
    num_nodes = len(positions)
    if num_nodes < 3:
        return np.array([])

    curvatures = []
    for i in range(1, num_nodes - 1):
        v1 = positions[i] - positions[i - 1]
        v2 = positions[i + 1] - positions[i]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-8 or v2_norm < 1e-8:
            curvatures.append(0.0)
            continue

        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm

        # Curvature from angle between consecutive segments
        cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        avg_length = (v1_norm + v2_norm) / 2

        curvature = angle / avg_length if avg_length > 1e-8 else 0.0
        curvatures.append(curvature)

    return np.array(curvatures)


def compute_signed_curvatures(positions: np.ndarray) -> np.ndarray:
    """Compute signed curvatures (positive = CCW, negative = CW).

    Args:
        positions: (n_nodes, 3) array of node positions

    Returns:
        curvatures: (n_nodes - 2,) array of signed curvatures
    """
    num_nodes = len(positions)
    if num_nodes < 3:
        return np.array([])

    curvatures = []
    for i in range(1, num_nodes - 1):
        v1 = positions[i] - positions[i - 1]
        v2 = positions[i + 1] - positions[i]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-8 or v2_norm < 1e-8:
            curvatures.append(0.0)
            continue

        # Use cross product for sign (in 2D, z-component gives rotation direction)
        cross = np.cross(v1, v2)
        sign = np.sign(cross[2]) if abs(cross[2]) > 1e-10 else 1.0

        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm

        cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        avg_length = (v1_norm + v2_norm) / 2

        curvature = sign * angle / avg_length if avg_length > 1e-8 else 0.0
        curvatures.append(curvature)

    return np.array(curvatures)


def analyze_ideal_coil(config: PhysicsConfig = None) -> CoilAnalysisResult:
    """Analyze the ideal coiled configuration.

    Args:
        config: Physics configuration (uses defaults if None)

    Returns:
        CoilAnalysisResult with positions, curvatures, and metrics
    """
    if config is None:
        config = PhysicsConfig()

    # Parameters
    num_nodes = config.num_segments + 1  # 21
    num_joints = config.num_segments - 1  # 19
    snake_length = config.snake_length  # 1.0
    prey_radius = config.prey_radius  # 0.1
    snake_radius = config.snake_radius  # 0.001

    # Coil radius = prey_radius + snake_radius (snake wraps around outside)
    coil_radius = prey_radius + snake_radius

    # Prey center - place it appropriately
    prey_center = np.array([0.0, 0.0, 0.0])

    # Theoretical curvature for a circular arc: κ = 1/R
    theoretical_curvature = 1.0 / coil_radius

    print("=" * 60)
    print("IDEAL COIL ANALYSIS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Snake length: {snake_length} m")
    print(f"  Snake radius: {snake_radius} m")
    print(f"  Num segments: {config.num_segments}")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Num joints (curvature control points): {num_joints}")
    print(f"  Prey radius: {prey_radius} m")
    print(f"  Coil radius: {coil_radius} m")

    # Theoretical analysis
    circumference = 2 * np.pi * coil_radius
    max_wraps = snake_length / circumference
    total_angle_rad = snake_length / coil_radius
    total_angle_deg = np.degrees(total_angle_rad)

    print(f"\nTheoretical:")
    print(f"  Target curvature (1/R): {theoretical_curvature:.4f}")
    print(f"  Prey circumference: {circumference:.4f} m")
    print(f"  Max possible wraps: {max_wraps:.2f}")
    print(f"  Total wrap angle: {total_angle_deg:.1f}° ({total_angle_rad:.2f} rad)")

    # Check against curvature limits
    max_curvature = 10.0  # From set_curvature_control clipping
    print(f"\nCurvature limits:")
    print(f"  Max controllable curvature: {max_curvature}")
    print(f"  Required curvature: {theoretical_curvature:.4f}")
    if theoretical_curvature > max_curvature:
        print(f"  WARNING: Required curvature exceeds control limit!")
        print(f"  Minimum coilable radius: {1.0/max_curvature:.4f} m")
    else:
        print(f"  OK: Curvature is within control limits")

    # Create ideal coiled positions
    positions = create_coiled_positions(
        num_nodes=num_nodes,
        snake_length=snake_length,
        coil_radius=coil_radius,
        prey_center=prey_center,
        start_angle=0.0,
        coil_direction=1,  # Counter-clockwise
    )

    # Compute curvatures from positions
    curvatures = compute_curvatures_from_positions(positions)
    signed_curvatures = compute_signed_curvatures(positions)

    print(f"\nComputed curvatures from ideal coil:")
    print(f"  Mean unsigned curvature: {np.mean(curvatures):.4f}")
    print(f"  Std unsigned curvature: {np.std(curvatures):.6f}")
    print(f"  Mean signed curvature: {np.mean(signed_curvatures):.4f}")
    print(f"  Min/Max signed: [{np.min(signed_curvatures):.4f}, {np.max(signed_curvatures):.4f}]")

    # Target curvatures for control (uniform for perfect circle)
    # Use signed curvatures for control
    target_curvatures = np.full(num_joints, theoretical_curvature)

    # Compute wrap metrics
    wrap_angle = total_angle_rad  # For ideal coil, this is exact
    wrap_count = wrap_angle / (2 * np.pi)

    # Contact: all nodes should be at exactly coil_radius from center
    distances = np.linalg.norm(positions[:, :2] - prey_center[:2], axis=1)
    contact_mask = np.abs(distances - coil_radius) < 0.01 + snake_radius
    contact_fraction = np.mean(contact_mask)

    print(f"\nCoil metrics:")
    print(f"  Wrap angle: {np.degrees(wrap_angle):.1f}° ({wrap_angle:.2f} rad)")
    print(f"  Wrap count: {wrap_count:.2f}")
    print(f"  Contact fraction: {contact_fraction:.2%}")

    return CoilAnalysisResult(
        positions=positions,
        curvatures=curvatures,
        target_curvatures=target_curvatures,
        wrap_angle=wrap_angle,
        wrap_count=wrap_count,
        contact_fraction=contact_fraction,
        coil_radius=coil_radius,
        prey_radius=prey_radius,
        snake_length=snake_length,
    )


def simulate_coil_with_curvature_control(
    target_curvatures: np.ndarray,
    num_steps: int = 200,
    config: PhysicsConfig = None,
) -> Tuple[dict, list]:
    """Simulate the snake with constant curvature control.

    Args:
        target_curvatures: Target curvatures for all joints
        num_steps: Number of simulation steps
        config: Physics configuration

    Returns:
        final_state: Final simulation state
        history: List of states over time
    """
    if config is None:
        config = PhysicsConfig()

    # Position snake near the prey for coiling
    # Start with head at angle 0 on the coil circle
    prey_center = np.array([config.snake_length + config.prey_radius * 2, 0.0, 0.0])
    coil_radius = config.prey_radius + config.snake_radius

    # Create ideal starting positions (snake already coiled)
    initial_positions = create_coiled_positions(
        num_nodes=config.num_segments + 1,
        snake_length=config.snake_length,
        coil_radius=coil_radius,
        prey_center=prey_center,
        start_angle=0.0,
        coil_direction=1,
    )

    # Create robot with initial position at head
    robot = SnakeRobot(
        config=config,
        initial_snake_position=initial_positions[0],
        initial_prey_position=prey_center,
    )

    # Set the positions directly in DisMech (this is a hack for initialization)
    # Actually, we can't do this easily - DisMech manages positions internally
    # Instead, we'll start from a straight snake and apply curvature control

    print("\n" + "=" * 60)
    print("SIMULATION WITH CURVATURE CONTROL")
    print("=" * 60)

    # Apply target curvatures
    robot.set_curvature_control(target_curvatures)

    history = []

    for step in range(num_steps):
        state = robot.step()
        history.append(state)

        if step % 50 == 0:
            print(f"Step {step}: wrap_count={state['wrap_count']:.2f}, "
                  f"contact={state['contact_fraction']:.2%}, "
                  f"curvature_mean={np.mean(np.abs(state['curvatures'])):.4f}")

    final_state = history[-1]

    print(f"\nFinal state:")
    print(f"  Wrap count: {final_state['wrap_count']:.2f}")
    print(f"  Contact fraction: {final_state['contact_fraction']:.2%}")
    print(f"  Mean curvature: {np.mean(np.abs(final_state['curvatures'])):.4f}")

    return final_state, history


def visualize_coil_analysis(
    result: CoilAnalysisResult,
    simulation_history: list = None,
    save_path: str = None,
):
    """Visualize the coil analysis results.

    Args:
        result: CoilAnalysisResult from analyze_ideal_coil
        simulation_history: Optional simulation history
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))

    # 1. Ideal coil configuration (top-down view)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_aspect('equal')

    # Draw prey cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    prey_x = result.prey_radius * np.cos(theta)
    prey_y = result.prey_radius * np.sin(theta)
    ax1.fill(prey_x, prey_y, alpha=0.3, color='green', label='Prey')
    ax1.plot(prey_x, prey_y, 'g-', linewidth=2)

    # Draw snake
    ax1.plot(result.positions[:, 0], result.positions[:, 1], 'b-', linewidth=3, label='Snake')
    ax1.plot(result.positions[0, 0], result.positions[0, 1], 'ro', markersize=10, label='Head')
    ax1.plot(result.positions[-1, 0], result.positions[-1, 1], 'ko', markersize=8, label='Tail')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Ideal Coiled Configuration\n(Top-Down View)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Curvature profile along snake
    ax2 = fig.add_subplot(2, 3, 2)

    # Node indices for curvatures (internal nodes only)
    joint_indices = np.arange(1, len(result.curvatures) + 1)

    ax2.bar(joint_indices, result.curvatures, alpha=0.7, label='Computed κ')
    ax2.axhline(y=1.0/result.coil_radius, color='r', linestyle='--',
                label=f'Theoretical κ = {1.0/result.coil_radius:.2f}')
    ax2.axhline(y=10.0, color='orange', linestyle=':', label='Max control limit')

    ax2.set_xlabel('Joint Index')
    ax2.set_ylabel('Curvature (1/m)')
    ax2.set_title('Curvature Profile for Ideal Coil')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Target curvatures for RL initialization
    ax3 = fig.add_subplot(2, 3, 3)

    ax3.bar(joint_indices, result.target_curvatures, alpha=0.7, color='green')
    ax3.axhline(y=10.0, color='orange', linestyle=':', label='Max control limit')

    ax3.set_xlabel('Joint Index')
    ax3.set_ylabel('Target Curvature (1/m)')
    ax3.set_title('Target Curvatures for Control\n(Use for RL initialization)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4-6: Simulation results if available
    if simulation_history:
        # 4. Trajectory of snake over time
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_aspect('equal')

        # Draw prey
        prey_center = simulation_history[0]['prey_position']
        prey_x = prey_center[0] + result.prey_radius * np.cos(theta)
        prey_y = prey_center[1] + result.prey_radius * np.sin(theta)
        ax4.fill(prey_x, prey_y, alpha=0.3, color='green')
        ax4.plot(prey_x, prey_y, 'g-', linewidth=2)

        # Draw snake at different times
        n_snapshots = min(5, len(simulation_history))
        indices = np.linspace(0, len(simulation_history) - 1, n_snapshots, dtype=int)
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_snapshots))

        for idx, color in zip(indices, colors):
            pos = simulation_history[idx]['positions']
            ax4.plot(pos[:, 0], pos[:, 1], '-', color=color, linewidth=2,
                    alpha=0.7, label=f't={idx}')

        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('Snake Evolution During Simulation')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. Wrap count over time
        ax5 = fig.add_subplot(2, 3, 5)

        times = [s['time'] for s in simulation_history]
        wrap_counts = [s['wrap_count'] for s in simulation_history]
        contact_fractions = [s['contact_fraction'] for s in simulation_history]

        ax5.plot(times, wrap_counts, 'b-', linewidth=2, label='Wrap count')
        ax5.axhline(y=1.5, color='r', linestyle='--', label='Success threshold')

        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Wrap Count')
        ax5.set_title('Wrap Progress Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Contact fraction over time
        ax6 = fig.add_subplot(2, 3, 6)

        ax6.plot(times, contact_fractions, 'g-', linewidth=2, label='Contact fraction')
        ax6.axhline(y=0.6, color='r', linestyle='--', label='Success threshold')

        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Contact Fraction')
        ax6.set_title('Contact Progress Over Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        # Placeholder for simulation results
        for i, ax_idx in enumerate([4, 5, 6]):
            ax = fig.add_subplot(2, 3, ax_idx)
            ax.text(0.5, 0.5, 'Simulation not run\n(Run with --simulate)',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(['Snake Trajectory', 'Wrap Progress', 'Contact Progress'][i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def save_curvature_sequence(
    result: CoilAnalysisResult,
    save_path: str,
):
    """Save curvature sequence for RL initialization.

    Args:
        result: CoilAnalysisResult
        save_path: Path to save numpy file
    """
    data = {
        'target_curvatures': result.target_curvatures,
        'computed_curvatures': result.curvatures,
        'positions': result.positions,
        'coil_radius': result.coil_radius,
        'prey_radius': result.prey_radius,
        'snake_length': result.snake_length,
        'wrap_count': result.wrap_count,
        'contact_fraction': result.contact_fraction,
    }

    np.savez(save_path, **data)
    print(f"\nCurvature sequence saved to: {save_path}")
    print(f"To load: data = np.load('{save_path}')")
    print(f"         target_curvatures = data['target_curvatures']")


def test_varying_curvatures():
    """Test coiling with varying curvature values."""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT CURVATURE VALUES")
    print("=" * 60)

    config = PhysicsConfig()
    num_joints = config.num_segments - 1

    # Test different uniform curvatures
    test_curvatures = [2.0, 5.0, 8.0, 10.0]

    for kappa in test_curvatures:
        target = np.full(num_joints, kappa)

        # Theoretical coil radius for this curvature
        R = 1.0 / kappa
        max_wraps = config.snake_length / (2 * np.pi * R)

        print(f"\nCurvature κ = {kappa}:")
        print(f"  Coil radius: {R:.4f} m")
        print(f"  Max wraps: {max_wraps:.2f}")
        print(f"  Fits prey (R={config.prey_radius}m)? {'Yes' if R >= config.prey_radius else 'No'}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze hardcoded coil configuration')
    parser.add_argument('--simulate', action='store_true',
                       help='Run simulation with curvature control')
    parser.add_argument('--num-steps', type=int, default=200,
                       help='Number of simulation steps')
    parser.add_argument('--save-fig', type=str, default=None,
                       help='Path to save figure')
    parser.add_argument('--save-curvatures', type=str, default=None,
                       help='Path to save curvature sequence (.npz)')
    parser.add_argument('--test-curvatures', action='store_true',
                       help='Test different curvature values')
    args = parser.parse_args()

    # Analyze ideal coil
    result = analyze_ideal_coil()

    # Test different curvatures if requested
    if args.test_curvatures:
        test_varying_curvatures()

    # Run simulation if requested
    simulation_history = None
    if args.simulate:
        final_state, simulation_history = simulate_coil_with_curvature_control(
            target_curvatures=result.target_curvatures,
            num_steps=args.num_steps,
        )

    # Save curvature sequence if requested
    if args.save_curvatures:
        save_curvature_sequence(result, args.save_curvatures)
    else:
        # Default save location
        default_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'ideal_coil_curvatures.npz'
        )
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        save_curvature_sequence(result, default_path)

    # Visualize
    save_path = args.save_fig or os.path.join(
        os.path.dirname(__file__), '..', 'figures', 'coil_analysis.png'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    visualize_coil_analysis(result, simulation_history, save_path)

    # Print summary for RL initialization
    print("\n" + "=" * 60)
    print("SUMMARY FOR RL POLICY INITIALIZATION")
    print("=" * 60)
    print(f"\nTarget curvature profile (19-dim action):")
    print(f"  All joints: κ = {result.target_curvatures[0]:.4f}")
    print(f"  Range: [{result.target_curvatures.min():.4f}, {result.target_curvatures.max():.4f}]")
    print(f"\nAs normalized action (assuming action_scale=1, curvature=5*action):")
    normalized_action = result.target_curvatures / 5.0  # Reverse the scaling
    print(f"  Action values: {normalized_action[0]:.4f}")
    print(f"  Clipped to [-1, 1]: {np.clip(normalized_action, -1, 1)[0]:.4f}")
    print(f"\nUsage in RL:")
    print(f"  1. Initialize policy bias to output {normalized_action[0]:.4f} for all joints")
    print(f"  2. Or use imitation learning from this demonstration")
    print(f"  3. Or use as reference for reward shaping")


if __name__ == '__main__':
    main()
