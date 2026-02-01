#!/usr/bin/env python3
"""Verify that DIRECT control (19-dim curvature) can achieve successful coiling.

This script tests whether the physics simulation and control interface support
the coiling task by:
1. Initializing the snake in a pre-coiled configuration around prey
2. Applying constant high curvature to maintain/tighten the coil
3. Tracking contact_fraction and wrap_count over time
4. Reporting success/failure with detailed metrics

Success criteria:
- contact_fraction >= 0.6 (60% of snake body touching prey)
- abs(wrap_count) >= 1.5 (at least 1.5 complete wraps)
- Both conditions held for at least 10 consecutive timesteps
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake_hrl.physics.snake_robot import SnakeRobot
from snake_hrl.configs.env import PhysicsConfig


def create_coiled_positions(
    num_nodes: int,
    prey_center: np.ndarray,
    prey_radius: float,
    snake_length: float,
    initial_wraps: float = 1.0,
    offset: float = 0.005,
    direction: int = 1,
) -> np.ndarray:
    """Generate snake node positions forming a spiral around prey.

    The snake is positioned in the XY plane (z=0) forming a spiral
    around the prey cylinder.

    Args:
        num_nodes: Number of snake nodes (21 for 20 segments)
        prey_center: Center position of prey [x, y, z]
        prey_radius: Radius of cylindrical prey
        snake_length: Total length of snake
        initial_wraps: Number of complete wraps to form (default 1.0)
        offset: Gap between snake and prey surface for contact detection
        direction: 1 for counterclockwise, -1 for clockwise

    Returns:
        positions: (num_nodes, 3) array of node positions
    """
    # Compute arc length parameters
    coil_radius = prey_radius + offset
    total_angle = 2 * np.pi * initial_wraps * direction
    segment_length = snake_length / (num_nodes - 1)

    positions = np.zeros((num_nodes, 3))

    # Start angle (snake head position)
    start_angle = 0.0

    # Arc length per radian at coil_radius
    arc_per_radian = coil_radius

    # Current angle and position
    current_angle = start_angle
    arc_traveled = 0.0

    for i in range(num_nodes):
        # Position on circle
        x = prey_center[0] + coil_radius * np.cos(current_angle)
        y = prey_center[1] + coil_radius * np.sin(current_angle)
        z = 0.0  # In XY plane

        positions[i] = [x, y, z]

        if i < num_nodes - 1:
            # Move along arc by segment_length
            # d_theta = arc_length / radius
            d_theta = segment_length / coil_radius * direction
            current_angle += d_theta
            arc_traveled += segment_length

    return positions


def inject_coiled_state(
    robot: SnakeRobot,
    coiled_positions: np.ndarray,
) -> None:
    """Manually inject coiled positions into DisMech state.

    This bypasses the normal API which creates a straight snake.

    Args:
        robot: SnakeRobot instance
        coiled_positions: (num_nodes, 3) array of target positions
    """
    num_nodes = len(coiled_positions)

    # Get current state
    q = robot._dismech_robot.state.q.copy()

    # Replace positions (first 3*num_nodes elements of q)
    q[:3 * num_nodes] = coiled_positions.flatten()

    # Zero velocities
    u = np.zeros_like(robot._dismech_robot.state.u)

    # Create new state with injected positions
    # Note: DisMech state is typically immutable, but we can modify the arrays
    robot._dismech_robot.state.q[:] = q
    robot._dismech_robot.state.u[:] = u

    # Update the snake adapter to reflect new positions
    robot._update_snake_adapter()

    # Update contact state
    robot._update_contact_state()


def run_coiling_experiment(
    strategy: str,
    initial_wraps: float,
    max_steps: int = 500,
    success_steps: int = 10,
    contact_threshold: float = 0.6,
    wrap_threshold: float = 1.5,
    verbose: bool = True,
) -> Dict:
    """Run a single coiling experiment with given strategy and initial configuration.

    Args:
        strategy: Curvature strategy name
        initial_wraps: Initial number of wraps in coiled configuration
        max_steps: Maximum simulation steps
        success_steps: Required consecutive steps at success threshold
        contact_threshold: Minimum contact_fraction for success
        wrap_threshold: Minimum abs(wrap_count) for success
        verbose: Print progress

    Returns:
        Dictionary with experiment results
    """
    # Physics configuration
    config = PhysicsConfig(
        snake_length=1.0,
        snake_radius=0.001,
        num_segments=20,
        prey_radius=0.1,
        prey_length=0.3,
        dt=0.05,
        max_iter=50,  # Increased for high curvature stability
        tol=1e-3,
        use_rft=True,
        enable_gravity=False,  # Simplified 2D test
    )

    num_nodes = config.num_segments + 1  # 21
    num_joints = config.num_segments - 1  # 19

    # Prey at origin
    prey_center = np.array([0.0, 0.0, 0.0])

    # Create robot (will be reset with coiled state)
    robot = SnakeRobot(config, initial_prey_position=prey_center)

    # Generate coiled initial positions
    coiled_positions = create_coiled_positions(
        num_nodes=num_nodes,
        prey_center=prey_center,
        prey_radius=config.prey_radius,
        snake_length=config.snake_length,
        initial_wraps=initial_wraps,
        offset=0.005,  # Small gap for contact detection
        direction=1,  # Counterclockwise
    )

    # Inject coiled state
    inject_coiled_state(robot, coiled_positions)

    # Define curvature strategy
    curvature_target = 1.0 / config.prey_radius  # 10.0 for R=0.1
    curvatures = get_curvature_strategy(strategy, num_joints, curvature_target)

    # Track metrics
    history = {
        "contact_fraction": [],
        "wrap_count": [],
        "positions": [],
        "curvatures": [],
    }

    consecutive_success = 0
    success_step = None

    if verbose:
        print(f"\nStrategy: {strategy}, Initial wraps: {initial_wraps}")
        print("-" * 50)

    for step in range(max_steps):
        # Apply curvature control
        current_curvatures = get_curvature_at_step(
            strategy, num_joints, curvature_target, step
        )
        robot.set_curvature_control(current_curvatures)

        # Step simulation
        try:
            state = robot.step()
        except Exception as e:
            if verbose:
                print(f"  Step {step}: Simulation error: {e}")
            break

        # Record metrics
        contact_fraction = state["contact_fraction"]
        wrap_count = state["wrap_count"]
        history["contact_fraction"].append(contact_fraction)
        history["wrap_count"].append(wrap_count)
        history["positions"].append(state["positions"].copy())
        history["curvatures"].append(state["curvatures"].copy())

        # Check success criteria
        is_success = (
            contact_fraction >= contact_threshold
            and abs(wrap_count) >= wrap_threshold
        )

        if is_success:
            consecutive_success += 1
        else:
            consecutive_success = 0

        # Print progress
        if verbose and step % 50 == 0:
            print(
                f"  Step {step:3d}: contact_fraction={contact_fraction:.3f}, "
                f"wrap_count={wrap_count:.3f}, consecutive={consecutive_success}"
            )

        # Check for sustained success
        if consecutive_success >= success_steps and success_step is None:
            success_step = step - success_steps + 1
            if verbose:
                print(f"  Step {step}: SUCCESS! Held for {success_steps} steps")
            break

    # Compile results
    success = success_step is not None
    final_contact = history["contact_fraction"][-1] if history["contact_fraction"] else 0.0
    final_wrap = history["wrap_count"][-1] if history["wrap_count"] else 0.0

    result = {
        "strategy": strategy,
        "initial_wraps": initial_wraps,
        "success": success,
        "success_step": success_step,
        "steps_run": len(history["contact_fraction"]),
        "final_contact_fraction": final_contact,
        "final_wrap_count": final_wrap,
        "max_contact_fraction": max(history["contact_fraction"]) if history["contact_fraction"] else 0.0,
        "max_wrap_count": max(abs(w) for w in history["wrap_count"]) if history["wrap_count"] else 0.0,
        "history": history,
    }

    return result


def get_curvature_strategy(
    strategy: str,
    num_joints: int,
    curvature_target: float,
) -> np.ndarray:
    """Get initial curvature array for a strategy.

    Args:
        strategy: Strategy name
        num_joints: Number of controllable joints (19)
        curvature_target: Target curvature (1/prey_radius)

    Returns:
        curvatures: (num_joints,) array
    """
    if strategy == "constant_max":
        # Maximum curvature (just under control limit)
        return np.full(num_joints, 9.5)

    elif strategy == "constant_match":
        # Match prey radius exactly
        return np.full(num_joints, curvature_target)

    elif strategy == "gradient":
        # Tighter at head, looser at tail
        return np.linspace(curvature_target, curvature_target * 0.6, num_joints)

    elif strategy == "reverse_gradient":
        # Looser at head, tighter at tail
        return np.linspace(curvature_target * 0.6, curvature_target, num_joints)

    elif strategy == "progressive":
        # Start lower, will ramp up over time
        return np.full(num_joints, 5.0)

    elif strategy == "moderate":
        # Moderate curvature
        return np.full(num_joints, 7.5)

    elif strategy == "low":
        # Low curvature
        return np.full(num_joints, 5.0)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_curvature_at_step(
    strategy: str,
    num_joints: int,
    curvature_target: float,
    step: int,
) -> np.ndarray:
    """Get curvature array for a strategy at a given step.

    Allows time-varying strategies like 'progressive'.

    Args:
        strategy: Strategy name
        num_joints: Number of controllable joints
        curvature_target: Target curvature
        step: Current simulation step

    Returns:
        curvatures: (num_joints,) array
    """
    base = get_curvature_strategy(strategy, num_joints, curvature_target)

    if strategy == "progressive":
        # Ramp up from 5.0 to target over 200 steps
        ramp = min(1.0, step / 200.0)
        base = 5.0 + ramp * (curvature_target - 5.0)
        return np.full(num_joints, base)

    return base


def sanity_check_contact(verbose: bool = True) -> bool:
    """Verify contact detection works with a simple tangent configuration.

    Places snake tangent to prey and checks that contact is detected.

    Returns:
        True if contact detection works
    """
    if verbose:
        print("\n=== Sanity Check: Contact Detection ===")

    config = PhysicsConfig(
        snake_length=1.0,
        num_segments=20,
        prey_radius=0.1,
        enable_gravity=False,
    )

    num_nodes = config.num_segments + 1
    prey_center = np.array([0.0, 0.0, 0.0])

    robot = SnakeRobot(config, initial_prey_position=prey_center)

    # Place snake tangent to prey (along x-axis, touching at y=prey_radius)
    positions = np.zeros((num_nodes, 3))
    segment_length = config.snake_length / config.num_segments

    # Snake runs along x-axis at y = prey_radius (just touching)
    for i in range(num_nodes):
        positions[i] = [
            -config.snake_length / 2 + i * segment_length,
            config.prey_radius + 0.005,  # Just at contact threshold
            0.0,
        ]

    inject_coiled_state(robot, positions)
    state = robot.get_state()

    contact_fraction = state["contact_fraction"]
    if verbose:
        print(f"  Snake tangent to prey at y={config.prey_radius + 0.005}")
        print(f"  Contact fraction: {contact_fraction:.3f}")
        print(f"  Contact mask: {state['contact_mask']}")

    passed = contact_fraction > 0
    if verbose:
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def sanity_check_wrap_angle(verbose: bool = True) -> bool:
    """Verify wrap angle computation with a 1-wrap spiral configuration.

    Returns:
        True if wrap angle computation is reasonable
    """
    if verbose:
        print("\n=== Sanity Check: Wrap Angle Computation ===")

    config = PhysicsConfig(
        snake_length=1.0,
        num_segments=20,
        prey_radius=0.1,
        enable_gravity=False,
    )

    num_nodes = config.num_segments + 1
    prey_center = np.array([0.0, 0.0, 0.0])

    robot = SnakeRobot(config, initial_prey_position=prey_center)

    # Create a 1-wrap spiral
    coiled_positions = create_coiled_positions(
        num_nodes=num_nodes,
        prey_center=prey_center,
        prey_radius=config.prey_radius,
        snake_length=config.snake_length,
        initial_wraps=1.0,
        offset=0.005,
    )

    inject_coiled_state(robot, coiled_positions)
    state = robot.get_state()

    wrap_count = state["wrap_count"]
    wrap_angle = state["wrap_angle"]

    if verbose:
        print(f"  1-wrap spiral configuration")
        print(f"  Wrap angle: {wrap_angle:.3f} rad ({np.degrees(wrap_angle):.1f} deg)")
        print(f"  Wrap count: {wrap_count:.3f}")

    # Should be close to 1.0 (may not be exact due to discrete segments)
    # For snake_length=1.0 and prey_radius=0.1, circumference=0.628
    # So 1 wrap uses 62.8% of snake, remaining 37.2% extends beyond
    # Geometric wrap count should be approximately snake_length / circumference
    expected_max_wraps = config.snake_length / (2 * np.pi * (config.prey_radius + 0.005))
    passed = abs(wrap_count) > 0.5 and abs(wrap_count) < expected_max_wraps + 0.2

    if verbose:
        print(f"  Expected max wraps (geometric): {expected_max_wraps:.3f}")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def sanity_check_curvature_control(verbose: bool = True) -> bool:
    """Verify curvature control affects snake shape.

    Returns:
        True if curvature control works
    """
    if verbose:
        print("\n=== Sanity Check: Curvature Control ===")

    config = PhysicsConfig(
        snake_length=1.0,
        num_segments=20,
        enable_gravity=False,
        dt=0.05,
    )

    num_joints = config.num_segments - 1

    robot = SnakeRobot(config)

    # Get initial curvatures (should be ~0 for straight snake)
    initial_state = robot.get_state()
    initial_curvatures = initial_state["curvatures"]

    if verbose:
        print(f"  Initial curvatures (mean): {np.mean(initial_curvatures):.4f}")

    # Apply high curvature
    robot.set_curvature_control(np.full(num_joints, 5.0))

    # Step a few times to let the snake bend
    for _ in range(20):
        robot.step()

    final_state = robot.get_state()
    final_curvatures = final_state["curvatures"]

    if verbose:
        print(f"  Final curvatures (mean): {np.mean(final_curvatures):.4f}")

    # Curvatures should have increased
    passed = np.mean(final_curvatures) > np.mean(initial_curvatures) + 0.1

    if verbose:
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify DIRECT control can achieve coiling"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["constant_match", "constant_max", "moderate", "progressive", "gradient"],
        help="Curvature strategies to test",
    )
    parser.add_argument(
        "--initial-wraps",
        nargs="+",
        type=float,
        default=[0.8, 1.0, 1.2],
        help="Initial wrap counts to test",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum simulation steps per experiment",
    )
    parser.add_argument(
        "--success-steps",
        type=int,
        default=10,
        help="Required consecutive steps at success threshold",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.6,
        help="Minimum contact_fraction for success",
    )
    parser.add_argument(
        "--wrap-threshold",
        type=float,
        default=1.5,
        help="Minimum abs(wrap_count) for success",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/coil_verification_trajectory.npz"),
        help="Output file for successful trajectory",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip sanity checks",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DIRECT Control Coiling Verification")
    print("=" * 60)

    # Run sanity checks first
    if not args.skip_sanity:
        sanity_results = []
        sanity_results.append(("Contact detection", sanity_check_contact(args.verbose)))
        sanity_results.append(("Wrap angle computation", sanity_check_wrap_angle(args.verbose)))
        sanity_results.append(("Curvature control", sanity_check_curvature_control(args.verbose)))

        print("\n=== Sanity Check Summary ===")
        all_passed = True
        for name, passed in sanity_results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        if not all_passed:
            print("\nSanity checks failed. Fix issues before proceeding.")
            return 1

    # Run main experiments
    print("\n" + "=" * 60)
    print("Running Coiling Experiments")
    print("=" * 60)
    print(f"\nSuccess criteria:")
    print(f"  - contact_fraction >= {args.contact_threshold}")
    print(f"  - abs(wrap_count) >= {args.wrap_threshold}")
    print(f"  - Held for {args.success_steps} consecutive steps")

    results: List[Dict] = []
    successful_result = None

    for strategy in args.strategies:
        for initial_wraps in args.initial_wraps:
            result = run_coiling_experiment(
                strategy=strategy,
                initial_wraps=initial_wraps,
                max_steps=args.max_steps,
                success_steps=args.success_steps,
                contact_threshold=args.contact_threshold,
                wrap_threshold=args.wrap_threshold,
                verbose=args.verbose,
            )
            results.append(result)

            if result["success"] and successful_result is None:
                successful_result = result
                print(f"\n>>> Found successful configuration: {strategy}, wraps={initial_wraps}")

            # If we found a successful configuration, we can optionally continue
            # to test all configurations or break early
            # For comprehensive testing, we continue

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\n{'Strategy':<20} {'Wraps':>6} {'Success':>8} {'Steps':>6} {'Contact':>8} {'Wrap':>8}")
    print("-" * 60)

    for r in results:
        success_str = "YES" if r["success"] else "no"
        step_str = str(r["success_step"]) if r["success"] else "-"
        print(
            f"{r['strategy']:<20} {r['initial_wraps']:>6.1f} {success_str:>8} "
            f"{step_str:>6} {r['final_contact_fraction']:>8.3f} {r['final_wrap_count']:>8.3f}"
        )

    # Save successful trajectory
    if successful_result is not None:
        print(f"\n=== SUCCESS ===")
        print(f"Strategy: {successful_result['strategy']}")
        print(f"Initial wraps: {successful_result['initial_wraps']}")
        print(f"Steps to success: {successful_result['success_step']}")
        print(f"Final contact_fraction: {successful_result['final_contact_fraction']:.3f}")
        print(f"Final wrap_count: {successful_result['final_wrap_count']:.3f}")

        # Save trajectory
        args.output.parent.mkdir(parents=True, exist_ok=True)
        history = successful_result["history"]
        np.savez(
            args.output,
            contact_fraction=np.array(history["contact_fraction"]),
            wrap_count=np.array(history["wrap_count"]),
            positions=np.array(history["positions"]),
            curvatures=np.array(history["curvatures"]),
            strategy=successful_result["strategy"],
            initial_wraps=successful_result["initial_wraps"],
            success_step=successful_result["success_step"],
        )
        print(f"\nTrajectory saved to: {args.output}")

        return 0
    else:
        print(f"\n=== FAILURE ===")
        print("No configuration achieved sustained success criteria.")
        print("\nBest results:")
        best_contact = max(results, key=lambda r: r["max_contact_fraction"])
        best_wrap = max(results, key=lambda r: r["max_wrap_count"])
        print(f"  Best contact_fraction: {best_contact['max_contact_fraction']:.3f} "
              f"({best_contact['strategy']}, wraps={best_contact['initial_wraps']})")
        print(f"  Best wrap_count: {best_wrap['max_wrap_count']:.3f} "
              f"({best_wrap['strategy']}, wraps={best_wrap['initial_wraps']})")

        return 1


if __name__ == "__main__":
    exit(main())
