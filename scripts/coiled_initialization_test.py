#!/usr/bin/env python3
"""Test snake initialization in a coiled configuration.

This script tests whether DisMech can:
1. Initialize a snake in an already-coiled configuration
2. Maintain the coil with appropriate curvature control
3. Extract the resulting curvature profile for RL initialization

This bypasses the problem of trying to achieve a coil from a straight position.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
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


def create_coiled_geometry(
    num_segments: int = 20,
    snake_length: float = 1.0,
    coil_radius: float = 0.1,
    prey_center: np.ndarray = None,
    start_angle: float = 0.0,
    coil_direction: int = 1,  # 1 for CCW, -1 for CW
) -> Geometry:
    """Create a coiled rod geometry for DisMech.

    Args:
        num_segments: Number of segments in the rod
        snake_length: Total arc length of the rod
        coil_radius: Radius of the coil
        prey_center: Center of the coil
        start_angle: Starting angle in radians
        coil_direction: 1 for counter-clockwise, -1 for clockwise

    Returns:
        DisMech Geometry object with coiled node positions
    """
    if prey_center is None:
        prey_center = np.array([0.0, 0.0, 0.0])

    num_nodes = num_segments + 1
    segment_length = snake_length / num_segments

    # For a circle: arc_length = radius * angle
    angle_per_segment = segment_length / coil_radius

    nodes = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        angle = start_angle + coil_direction * i * angle_per_segment
        nodes[i, 0] = prey_center[0] + coil_radius * np.cos(angle)
        nodes[i, 1] = prey_center[1] + coil_radius * np.sin(angle)
        nodes[i, 2] = prey_center[2]

    # Create edges (connectivity)
    edges = np.array([[i, i + 1] for i in range(num_segments)], dtype=np.int64)

    # No faces (rod only, no shell)
    face_nodes = np.empty((0, 3), dtype=np.int64)

    # Create Geometry without plotting
    return Geometry(nodes, edges, face_nodes, plot_from_txt=False)


def create_straight_geometry(
    num_segments: int = 20,
    snake_length: float = 1.0,
    initial_position: np.ndarray = None,
    initial_direction: np.ndarray = None,
) -> Geometry:
    """Create a straight rod geometry (for comparison)."""
    if initial_position is None:
        initial_position = np.array([0.0, 0.0, 0.0])
    if initial_direction is None:
        initial_direction = np.array([1.0, 0.0, 0.0])

    direction = initial_direction / np.linalg.norm(initial_direction)
    num_nodes = num_segments + 1
    segment_length = snake_length / num_segments

    nodes = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        nodes[i] = initial_position + i * segment_length * direction

    edges = np.array([[i, i + 1] for i in range(num_segments)], dtype=np.int64)
    face_nodes = np.empty((0, 3), dtype=np.int64)

    return Geometry(nodes, edges, face_nodes, plot_from_txt=False)


def create_dismech_robot(geometry: Geometry, config: PhysicsConfig) -> tuple:
    """Create DisMech robot with given geometry.

    Args:
        geometry: DisMech Geometry object
        config: Physics configuration

    Returns:
        (robot, time_stepper) tuple
    """
    # Geometry parameters
    geom_params = GeomParams(
        rod_r0=config.snake_radius,
        shell_h=0,
    )

    # Material parameters
    material = Material(
        density=config.density,
        youngs_rod=config.youngs_modulus,
        youngs_shell=0,
        poisson_rod=config.poisson_ratio,
        poisson_shell=0,
    )

    # Simulation parameters - use smaller timestep for stability
    sim_params = SimParams(
        static_sim=False,
        two_d_sim=True,
        use_mid_edge=False,
        use_line_search=True,
        log_data=False,
        log_step=1,
        show_floor=False,
        dt=config.dt,
        max_iter=50,  # More iterations for convergence
        total_time=1000.0,
        plot_step=1,
        tol=config.tol,
        ftol=config.ftol,
        dtol=config.dtol,
    )

    # Environment with forces
    env = Environment()
    if config.enable_gravity:
        env.add_force('gravity', g=np.array(config.gravity))
    if config.use_rft:
        env.add_force('rft', ct=config.rft_ct, cn=config.rft_cn)

    # Create DisMech SoftRobot
    robot = SoftRobot(geom_params, material, geometry, sim_params, env)
    time_stepper = ImplicitEulerTimeStepper(robot)

    return robot, time_stepper


def get_positions(robot: SoftRobot, num_nodes: int) -> np.ndarray:
    """Extract positions from DisMech state."""
    q = robot.state.q
    return q[:3 * num_nodes].reshape(num_nodes, 3)


def get_curvatures(positions: np.ndarray) -> np.ndarray:
    """Compute curvatures from positions."""
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

        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm

        cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        avg_length = (v1_norm + v2_norm) / 2

        curvature = angle / avg_length if avg_length > 1e-8 else 0.0
        curvatures.append(curvature)

    return np.array(curvatures)


def apply_curvature_control(robot: SoftRobot, target_curvatures: np.ndarray) -> None:
    """Apply target curvatures to bend springs."""
    bend_springs = robot.bend_springs

    if bend_springs.N > 0 and hasattr(bend_springs, 'nat_strain'):
        num_springs = min(len(target_curvatures), bend_springs.N)
        for i in range(num_springs):
            bend_springs.nat_strain[i, 0] = target_curvatures[i]
            bend_springs.nat_strain[i, 1] = 0.0


def test_coiled_initialization(
    config: PhysicsConfig = None,
    num_steps: int = 100,
    apply_control: bool = True,
    verbose: bool = True,
):
    """Test initializing the snake in a coiled configuration.

    Args:
        config: Physics configuration
        num_steps: Number of simulation steps
        apply_control: Whether to apply curvature control to maintain coil
        verbose: Print progress

    Returns:
        history: List of (positions, curvatures) tuples
    """
    if config is None:
        config = PhysicsConfig()

    num_nodes = config.num_segments + 1
    num_joints = config.num_segments - 1

    # Coil parameters
    coil_radius = config.prey_radius + config.snake_radius  # ~0.101
    prey_center = np.array([0.0, 0.0, 0.0])
    target_curvature = 1.0 / coil_radius  # ~9.9

    if verbose:
        print("=" * 60)
        print("COILED INITIALIZATION TEST")
        print("=" * 60)
        print(f"\nCoil parameters:")
        print(f"  Coil radius: {coil_radius:.4f} m")
        print(f"  Prey center: {prey_center}")
        print(f"  Target curvature: {target_curvature:.4f}")
        print(f"  Apply control: {apply_control}")

    # Create coiled geometry
    geometry = create_coiled_geometry(
        num_segments=config.num_segments,
        snake_length=config.snake_length,
        coil_radius=coil_radius,
        prey_center=prey_center,
    )

    # Create robot
    robot, time_stepper = create_dismech_robot(geometry, config)

    # Get initial state
    initial_positions = get_positions(robot, num_nodes)
    initial_curvatures = get_curvatures(initial_positions)

    if verbose:
        print(f"\nInitial state:")
        print(f"  Mean curvature: {np.mean(initial_curvatures):.4f}")
        print(f"  Curvature std: {np.std(initial_curvatures):.6f}")

    # Target curvatures for control
    target_curvatures = np.full(num_joints, target_curvature)

    history = [(initial_positions.copy(), initial_curvatures.copy())]

    # Simulate
    for step in range(num_steps):
        if apply_control:
            apply_curvature_control(robot, target_curvatures)

        try:
            robot, _ = time_stepper.step(robot, debug=False)
        except ValueError as e:
            if verbose:
                print(f"Step {step}: Convergence warning - {e}")

        positions = get_positions(robot, num_nodes)
        curvatures = get_curvatures(positions)
        history.append((positions.copy(), curvatures.copy()))

        if verbose and step % 25 == 0:
            mean_curv = np.mean(curvatures)
            std_curv = np.std(curvatures)
            print(f"Step {step}: κ_mean={mean_curv:.4f}, κ_std={std_curv:.4f}")

    final_positions, final_curvatures = history[-1]

    if verbose:
        print(f"\nFinal state:")
        print(f"  Mean curvature: {np.mean(final_curvatures):.4f}")
        print(f"  Target curvature: {target_curvature:.4f}")
        print(f"  Curvature std: {np.std(final_curvatures):.6f}")

        # Check if coil is maintained
        mean_error = abs(np.mean(final_curvatures) - target_curvature)
        print(f"  Mean error: {mean_error:.4f}")
        if mean_error < 1.0:
            print("  ✓ Coil approximately maintained!")
        else:
            print("  ✗ Coil NOT maintained")

    return history


def test_straight_to_coil_transition(
    config: PhysicsConfig = None,
    transition_steps: int = 100,
    hold_steps: int = 100,
    verbose: bool = True,
):
    """Test transitioning from straight to coiled using gradual curvature increase.

    This uses smaller timestep and gradual ramping for stability.
    """
    if config is None:
        config = PhysicsConfig()
        # Use smaller timestep for stability during transition
        config.dt = 0.01

    num_nodes = config.num_segments + 1
    num_joints = config.num_segments - 1

    coil_radius = config.prey_radius + config.snake_radius
    target_curvature = 1.0 / coil_radius

    if verbose:
        print("=" * 60)
        print("STRAIGHT TO COIL TRANSITION TEST")
        print("=" * 60)
        print(f"\nUsing dt={config.dt} for stability")
        print(f"Target curvature: {target_curvature:.4f}")

    # Create straight geometry
    geometry = create_straight_geometry(
        num_segments=config.num_segments,
        snake_length=config.snake_length,
    )

    robot, time_stepper = create_dismech_robot(geometry, config)

    history = []
    total_steps = transition_steps + hold_steps

    for step in range(total_steps):
        # Gradual curvature ramp
        if step < transition_steps:
            t = step / transition_steps
            # Use smooth easing
            t_smooth = (1 - np.cos(t * np.pi)) / 2
            current_target = target_curvature * t_smooth
        else:
            current_target = target_curvature

        target_curvatures = np.full(num_joints, current_target)
        apply_curvature_control(robot, target_curvatures)

        try:
            robot, _ = time_stepper.step(robot, debug=False)
        except ValueError as e:
            pass  # Ignore convergence warnings

        positions = get_positions(robot, num_nodes)
        curvatures = get_curvatures(positions)
        history.append((positions.copy(), curvatures.copy(), current_target))

        if verbose and step % 50 == 0:
            mean_curv = np.mean(curvatures)
            print(f"Step {step}: target={current_target:.2f}, actual={mean_curv:.2f}")

    return history


def visualize_coil_test(history: list, config: PhysicsConfig = None, save_path: str = None):
    """Visualize results from coiled initialization test."""
    if config is None:
        config = PhysicsConfig()

    fig = plt.figure(figsize=(15, 10))

    # 1. Initial configuration
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_aspect('equal')

    pos0, curv0 = history[0]
    ax1.plot(pos0[:, 0], pos0[:, 1], 'b-', linewidth=3)
    ax1.plot(pos0[0, 0], pos0[0, 1], 'ro', markersize=10, label='Head')

    # Draw prey cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    prey_x = config.prey_radius * np.cos(theta)
    prey_y = config.prey_radius * np.sin(theta)
    ax1.plot(prey_x, prey_y, 'g-', linewidth=2, label='Prey')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Initial Configuration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final configuration
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_aspect('equal')

    pos_final, curv_final = history[-1][:2] if len(history[-1]) == 3 else history[-1]
    ax2.plot(pos_final[:, 0], pos_final[:, 1], 'b-', linewidth=3)
    ax2.plot(pos_final[0, 0], pos_final[0, 1], 'ro', markersize=10)
    ax2.plot(prey_x, prey_y, 'g-', linewidth=2)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Final Configuration')
    ax2.grid(True, alpha=0.3)

    # 3. Curvature profile comparison
    ax3 = fig.add_subplot(2, 3, 3)

    joint_indices = np.arange(1, len(curv0) + 1)
    ax3.plot(joint_indices, curv0, 'b--', label='Initial', linewidth=2)
    ax3.plot(joint_indices, curv_final, 'r-', label='Final', linewidth=2)

    coil_radius = config.prey_radius + config.snake_radius
    target = 1.0 / coil_radius
    ax3.axhline(y=target, color='g', linestyle=':', label=f'Target ({target:.2f})')

    ax3.set_xlabel('Joint Index')
    ax3.set_ylabel('Curvature')
    ax3.set_title('Curvature Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Mean curvature over time
    ax4 = fig.add_subplot(2, 3, 4)

    mean_curvatures = [np.mean(h[1]) for h in history]
    ax4.plot(mean_curvatures, 'b-', linewidth=2)
    ax4.axhline(y=target, color='r', linestyle='--', label='Target')

    ax4.set_xlabel('Step')
    ax4.set_ylabel('Mean Curvature')
    ax4.set_title('Mean Curvature Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Curvature std over time
    ax5 = fig.add_subplot(2, 3, 5)

    std_curvatures = [np.std(h[1]) for h in history]
    ax5.plot(std_curvatures, 'g-', linewidth=2)

    ax5.set_xlabel('Step')
    ax5.set_ylabel('Curvature Std')
    ax5.set_title('Curvature Uniformity')
    ax5.grid(True, alpha=0.3)

    # 6. Snake evolution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_aspect('equal')

    # Draw prey
    ax6.fill(prey_x, prey_y, alpha=0.3, color='green')
    ax6.plot(prey_x, prey_y, 'g-', linewidth=2)

    # Draw snake at different times
    n_snapshots = min(6, len(history))
    indices = np.linspace(0, len(history) - 1, n_snapshots, dtype=int)
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_snapshots))

    for idx, color in zip(indices, colors):
        pos = history[idx][0]
        ax6.plot(pos[:, 0], pos[:, 1], '-', color=color, linewidth=2, alpha=0.8)

    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('Snake Evolution')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def extract_curvature_for_rl(history: list, config: PhysicsConfig = None) -> dict:
    """Extract curvature information for RL policy initialization.

    Args:
        history: Simulation history
        config: Physics configuration

    Returns:
        Dictionary with curvature data for RL
    """
    if config is None:
        config = PhysicsConfig()

    final_positions, final_curvatures = history[-1][:2]

    coil_radius = config.prey_radius + config.snake_radius
    theoretical_curvature = 1.0 / coil_radius

    # Action normalization
    # In the environment: curvatures = 5 * action_scale * action
    # Default action_scale = 1, so curvatures = 5 * action
    # To get curvature = 10 (max), need action = 2 (but clipped to 1)
    # For this coil, we need curvature ≈ 9.9, so action ≈ 1.98

    data = {
        'final_curvatures': final_curvatures,
        'theoretical_curvature': theoretical_curvature,
        'coil_radius': coil_radius,
        'mean_curvature': np.mean(final_curvatures),
        'curvature_std': np.std(final_curvatures),
        # For RL initialization
        'normalized_action': theoretical_curvature / 5.0,  # Assuming curvature = 5 * action
        'normalized_action_clipped': min(theoretical_curvature / 5.0, 1.0),
        # Per-joint actions (if you want non-uniform initialization)
        'per_joint_actions': final_curvatures / 5.0,
    }

    print("\n" + "=" * 60)
    print("CURVATURE DATA FOR RL INITIALIZATION")
    print("=" * 60)
    print(f"\nTheoretical curvature: {data['theoretical_curvature']:.4f}")
    print(f"Achieved mean curvature: {data['mean_curvature']:.4f}")
    print(f"Curvature uniformity (std): {data['curvature_std']:.6f}")
    print(f"\nFor RL policy:")
    print(f"  Ideal action value: {data['normalized_action']:.4f}")
    print(f"  Clipped to [-1,1]: {data['normalized_action_clipped']:.4f}")
    print(f"\nNote: Current action scaling (curvature = 5*action) is insufficient!")
    print(f"  Max curvature with action=1: 5.0")
    print(f"  Required curvature for coil: {data['theoretical_curvature']:.2f}")
    print(f"\nRecommendation: Increase action_scale to 2.0 or higher")
    print(f"  With action_scale=2: max curvature = 10.0")

    return data


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Test coiled initialization')
    parser.add_argument('--test', type=str, default='coiled',
                       choices=['coiled', 'transition', 'both'],
                       help='Which test to run')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of simulation steps')
    parser.add_argument('--no-control', action='store_true',
                       help='Disable curvature control (test natural relaxation)')
    parser.add_argument('--save-fig', type=str, default=None,
                       help='Path to save figure')
    args = parser.parse_args()

    config = PhysicsConfig()

    if args.test in ['coiled', 'both']:
        print("\n" + "=" * 60)
        print("TEST 1: Starting from coiled configuration")
        print("=" * 60)
        history = test_coiled_initialization(
            config=config,
            num_steps=args.num_steps,
            apply_control=not args.no_control,
        )

        save_path = args.save_fig or os.path.join(
            os.path.dirname(__file__), '..', 'figures', 'coiled_init_test.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_coil_test(history, config, save_path)

        # Extract RL initialization data
        rl_data = extract_curvature_for_rl(history, config)

        # Save data
        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'coil_rl_init.npz'
        )
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez(data_path, **rl_data)
        print(f"\nRL initialization data saved to: {data_path}")

    if args.test in ['transition', 'both']:
        print("\n" + "=" * 60)
        print("TEST 2: Transition from straight to coiled")
        print("=" * 60)
        history = test_straight_to_coil_transition(
            num_steps=200,
        )


if __name__ == '__main__':
    main()
