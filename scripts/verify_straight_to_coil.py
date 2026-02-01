#!/usr/bin/env python3
"""Verify that snake can bend from STRAIGHT to COILED configuration.

This is the key test - can the RL agent learn to coil starting from
a straight snake near the prey?
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snake_hrl.physics.snake_robot import SnakeRobot
from snake_hrl.configs.env import PhysicsConfig


def run_straight_to_coil_experiment(
    curvature_strategy: str,
    initial_distance: float,
    max_steps: int = 500,
    dt: float = 0.01,  # Smaller timestep for stability
    verbose: bool = True,
) -> dict:
    """
    Run experiment starting from straight snake near prey.

    Args:
        curvature_strategy: How to apply curvature ('instant', 'ramp', 'slow_ramp')
        initial_distance: Distance from snake head to prey center
        max_steps: Maximum simulation steps
        dt: Timestep (smaller = more stable)
        verbose: Print progress

    Returns:
        Dictionary with trajectory data and success metrics
    """
    config = PhysicsConfig(
        snake_length=1.0,
        snake_radius=0.001,
        num_segments=20,
        prey_radius=0.1,
        prey_length=0.3,
        dt=dt,
        max_iter=100,  # More iterations for convergence
        tol=1e-4,
        enable_gravity=False,
        use_rft=True,
    )

    num_joints = config.num_segments - 1  # 19

    # Position prey at origin
    prey_center = np.array([0.0, 0.0, 0.0])

    # Position snake head near prey, pointing toward it
    # Snake extends in -x direction from head
    snake_head_pos = np.array([initial_distance, 0.0, 0.0])

    # Create robot
    robot = SnakeRobot(
        config,
        initial_snake_position=snake_head_pos,
        initial_prey_position=prey_center
    )

    # Target curvature for tight coil
    target_curvature = 1.0 / (config.prey_radius + 0.005)  # ~9.5

    # Tracking
    history = {
        'positions': [],
        'curvatures': [],
        'contact_fraction': [],
        'wrap_count': [],
        'applied_curvature': [],
    }

    # Success tracking
    success_steps = 0
    success_achieved = False
    success_step = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"Strategy: {curvature_strategy}, Initial distance: {initial_distance}")
        print(f"Target curvature: {target_curvature:.2f}")
        print(f"{'='*60}")

    for step in range(max_steps):
        # Determine curvature to apply based on strategy
        if curvature_strategy == 'instant':
            applied_curv = target_curvature
        elif curvature_strategy == 'ramp':
            # Ramp up over 100 steps
            ramp = min(1.0, step / 100.0)
            applied_curv = ramp * target_curvature
        elif curvature_strategy == 'slow_ramp':
            # Ramp up over 200 steps
            ramp = min(1.0, step / 200.0)
            applied_curv = ramp * target_curvature
        elif curvature_strategy == 'very_slow_ramp':
            # Ramp up over 300 steps
            ramp = min(1.0, step / 300.0)
            applied_curv = ramp * target_curvature
        elif curvature_strategy == 'stepped':
            # Step increases
            if step < 50:
                applied_curv = 2.0
            elif step < 100:
                applied_curv = 4.0
            elif step < 150:
                applied_curv = 6.0
            elif step < 200:
                applied_curv = 8.0
            else:
                applied_curv = target_curvature
        elif curvature_strategy == 'low_constant':
            # Try lower curvature
            applied_curv = 5.0
        elif curvature_strategy == 'medium_constant':
            applied_curv = 7.0
        else:
            applied_curv = target_curvature

        # Apply curvature control
        curvature_cmd = np.full(num_joints, applied_curv)
        robot.set_curvature_control(curvature_cmd)

        # Step simulation
        try:
            state = robot.step()
        except Exception as e:
            if verbose:
                print(f"  Step {step}: Simulation error: {e}")
            break

        # Record data
        history['positions'].append(state['positions'].copy())
        history['curvatures'].append(state['curvatures'].copy())
        history['contact_fraction'].append(state['contact_fraction'])
        history['wrap_count'].append(state['wrap_count'])
        history['applied_curvature'].append(applied_curv)

        # Check success criteria
        contact_ok = state['contact_fraction'] >= 0.6
        wrap_ok = abs(state['wrap_count']) >= 1.5

        if contact_ok and wrap_ok:
            success_steps += 1
        else:
            success_steps = 0

        if success_steps >= 10 and not success_achieved:
            success_achieved = True
            success_step = step - 9
            if verbose:
                print(f"  Step {step}: SUCCESS! Criteria held for 10 steps")

        # Progress update
        if verbose and step % 50 == 0:
            print(f"  Step {step:3d}: κ_applied={applied_curv:.2f}, "
                  f"contact={state['contact_fraction']:.3f}, "
                  f"wrap={state['wrap_count']:.3f}")

    # Convert to arrays
    for key in ['positions', 'curvatures', 'contact_fraction', 'wrap_count', 'applied_curvature']:
        history[key] = np.array(history[key])

    result = {
        'strategy': curvature_strategy,
        'initial_distance': initial_distance,
        'success': success_achieved,
        'success_step': success_step,
        'final_contact': history['contact_fraction'][-1] if len(history['contact_fraction']) > 0 else 0,
        'final_wrap': history['wrap_count'][-1] if len(history['wrap_count']) > 0 else 0,
        'max_contact': np.max(history['contact_fraction']) if len(history['contact_fraction']) > 0 else 0,
        'max_wrap': np.max(np.abs(history['wrap_count'])) if len(history['wrap_count']) > 0 else 0,
        'history': history,
        'config': {
            'snake_length': config.snake_length,
            'snake_radius': config.snake_radius,
            'prey_radius': config.prey_radius,
            'dt': dt,
        }
    }

    return result


def create_visualization(result: dict, output_dir: Path):
    """Create visualization of straight-to-coil trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    history = result['history']
    positions = history['positions']
    contact_fraction = history['contact_fraction']
    wrap_count = history['wrap_count']
    applied_curv = history['applied_curvature']

    num_steps = len(positions)
    prey_radius = result['config']['prey_radius']
    prey_center = np.array([0.0, 0.0])

    # Static plot showing key frames
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Select frames to show
    frames = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
    frames = [min(f, num_steps-1) for f in frames[:4]]

    for i, frame in enumerate(frames):
        ax = axes[0, i]
        pos = positions[frame]

        # Plot prey
        circle = Circle(prey_center, prey_radius, fill=True,
                       facecolor='lightcoral', edgecolor='darkred',
                       linewidth=2, alpha=0.7)
        ax.add_patch(circle)

        # Plot snake
        ax.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2)
        ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=20, zorder=3)
        ax.scatter(pos[0, 0], pos[0, 1], c='red', s=80, marker='^', zorder=4)
        ax.scatter(pos[-1, 0], pos[-1, 1], c='purple', s=80, marker='s', zorder=4)

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"t={frame}: κ={applied_curv[frame]:.1f}\n"
                    f"contact={contact_fraction[frame]:.2f}, wrap={wrap_count[frame]:.2f}")

    # Metrics over time
    t = np.arange(num_steps)

    axes[1, 0].plot(t, applied_curv, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Applied Curvature')
    axes[1, 0].set_title('Curvature Command Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t, contact_fraction, 'g-', linewidth=2)
    axes[1, 1].axhline(y=0.6, color='r', linestyle='--', label='Threshold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Contact Fraction')
    axes[1, 1].set_title('Contact Fraction Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(t, wrap_count, 'purple', linewidth=2)
    axes[1, 2].axhline(y=1.5, color='r', linestyle='--', label='Threshold')
    axes[1, 2].axhline(y=-1.5, color='r', linestyle='--')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Wrap Count')
    axes[1, 2].set_title('Wrap Count Over Time')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Curvature heatmap
    curv_data = history['curvatures']
    if len(curv_data.shape) == 2:
        im = axes[1, 3].imshow(curv_data.T, aspect='auto', cmap='viridis',
                               extent=[0, num_steps, 0, curv_data.shape[1]])
        axes[1, 3].set_xlabel('Step')
        axes[1, 3].set_ylabel('Joint Index')
        axes[1, 3].set_title('Actual Curvature Profile')
        plt.colorbar(im, ax=axes[1, 3], label='κ')

    success_str = "SUCCESS" if result['success'] else "FAILED"
    fig.suptitle(f"Straight to Coil: {result['strategy']} (dist={result['initial_distance']}) - {success_str}",
                 fontsize=14)
    plt.tight_layout()

    filename = f"straight_to_coil_{result['strategy']}_{result['initial_distance']}.png"
    plt.savefig(output_dir / filename, dpi=150)
    print(f"Saved: {output_dir / filename}")
    plt.close()

    return output_dir / filename


def create_video(result: dict, output_path: Path):
    """Create video animation of straight-to-coil trajectory."""
    history = result['history']
    positions = history['positions']
    contact_fraction = history['contact_fraction']
    wrap_count = history['wrap_count']
    applied_curv = history['applied_curvature']

    num_steps = len(positions)
    prey_radius = result['config']['prey_radius']
    prey_center = np.array([0.0, 0.0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def animate(frame):
        for ax in axes:
            ax.clear()

        pos = positions[frame]

        # Left: Snake configuration
        circle = Circle(prey_center, prey_radius, fill=True,
                       facecolor='lightcoral', edgecolor='darkred',
                       linewidth=2, alpha=0.7)
        axes[0].add_patch(circle)

        axes[0].plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2)
        axes[0].scatter(pos[:, 0], pos[:, 1], c='blue', s=20, zorder=3)
        axes[0].scatter(pos[0, 0], pos[0, 1], c='red', s=80, marker='^', zorder=4)
        axes[0].scatter(pos[-1, 0], pos[-1, 1], c='purple', s=80, marker='s', zorder=4)

        axes[0].set_xlim(-0.4, 0.4)
        axes[0].set_ylim(-0.4, 0.4)
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f"Step {frame}: κ={applied_curv[frame]:.1f}\n"
                         f"contact={contact_fraction[frame]:.2f}, wrap={wrap_count[frame]:.2f}")

        # Right: Metrics history
        t = np.arange(frame + 1)
        axes[1].plot(t, contact_fraction[:frame+1], 'g-', linewidth=2, label='Contact')
        axes[1].plot(t, wrap_count[:frame+1] / 1.5, 'purple', linewidth=2, label='Wrap/1.5')
        axes[1].axhline(y=0.6, color='g', linestyle='--', alpha=0.5)
        axes[1].axhline(y=1.0, color='purple', linestyle='--', alpha=0.5)
        axes[1].set_xlim(0, num_steps)
        axes[1].set_ylim(-0.5, 1.5)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Progress')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)

        return axes

    # Sample every few frames for reasonable video length
    step_size = max(1, num_steps // 100)
    frame_indices = list(range(0, num_steps, step_size))
    if frame_indices[-1] != num_steps - 1:
        frame_indices.append(num_steps - 1)

    anim = FuncAnimation(fig, animate, frames=frame_indices,
                        interval=100, blit=False)

    writer = FFMpegWriter(fps=10, metadata=dict(artist='snake-hrl'))
    anim.save(output_path, writer=writer)
    print(f"Saved video: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("VERIFICATION: Straight to Coil")
    print("="*60)
    print("\nDimensions:")
    print("  Snake length: 1.0 m")
    print("  Snake radius: 0.001 m (1 mm)")
    print("  Prey radius: 0.1 m (10 cm)")
    print("  Target curvature: ~9.5 (to coil around prey)")
    print()

    output_dir = Path("Media/straight_to_coil")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test different strategies and starting distances
    strategies = [
        'slow_ramp',
        'very_slow_ramp',
        'stepped',
        'ramp',
    ]

    distances = [0.15, 0.12]  # Close to prey (prey radius = 0.1)

    results = []
    successful_result = None

    for strategy in strategies:
        for dist in distances:
            result = run_straight_to_coil_experiment(
                curvature_strategy=strategy,
                initial_distance=dist,
                max_steps=500,
                dt=0.01,  # Smaller timestep
                verbose=True,
            )
            results.append(result)

            # Create visualization
            create_visualization(result, output_dir)

            if result['success'] and successful_result is None:
                successful_result = result
                print(f"\n>>> FOUND SUCCESSFUL CONFIG: {strategy}, dist={dist}")

                # Save successful trajectory
                np.savez(
                    output_dir / "successful_straight_to_coil.npz",
                    **result['history'],
                    strategy=strategy,
                    initial_distance=dist,
                    success_step=result['success_step'],
                )

                # Create video for successful case
                create_video(result, output_dir / "successful_coil.mp4")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Strategy':<20} {'Dist':>6} {'Success':>8} {'MaxContact':>10} {'MaxWrap':>10}")
    print("-"*60)

    for r in results:
        success_str = "YES" if r['success'] else "no"
        print(f"{r['strategy']:<20} {r['initial_distance']:>6.2f} {success_str:>8} "
              f"{r['max_contact']:>10.3f} {r['max_wrap']:>10.3f}")

    if successful_result:
        print(f"\nSUCCESS! Saved trajectory to {output_dir / 'successful_straight_to_coil.npz'}")
        print(f"Video saved to {output_dir / 'successful_coil.mp4'}")
    else:
        print("\nNo configuration achieved full success criteria.")
        print("Best results:")
        best = max(results, key=lambda r: r['max_contact'] + r['max_wrap'])
        print(f"  Strategy: {best['strategy']}, Distance: {best['initial_distance']}")
        print(f"  Max contact: {best['max_contact']:.3f}, Max wrap: {best['max_wrap']:.3f}")


if __name__ == "__main__":
    main()
