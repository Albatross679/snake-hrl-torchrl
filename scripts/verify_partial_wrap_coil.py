#!/usr/bin/env python3
"""Verify coiling starting from PARTIAL wrap around prey.

The snake starts with part of its body already curved around the prey,
then curvature control completes the wrap.
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


def create_partial_wrap_positions(
    num_nodes: int,
    prey_center: np.ndarray,
    prey_radius: float,
    snake_length: float,
    wrap_fraction: float = 0.5,  # How much of snake is wrapped (0 to 1)
    offset: float = 0.005,
) -> np.ndarray:
    """
    Create snake with partial wrap around prey.

    The first `wrap_fraction` of the snake follows a circular arc around prey,
    the rest extends straight tangent to the circle.
    """
    coil_radius = prey_radius + offset
    segment_length = snake_length / (num_nodes - 1)

    # How many nodes are in the wrapped portion
    wrapped_length = wrap_fraction * snake_length
    wrapped_nodes = int(wrap_fraction * (num_nodes - 1)) + 1

    positions = np.zeros((num_nodes, 3))

    # Start at angle 0, wrap counterclockwise
    current_angle = 0.0
    d_theta = segment_length / coil_radius

    for i in range(num_nodes):
        if i < wrapped_nodes:
            # Wrapped portion - on circular arc
            x = prey_center[0] + coil_radius * np.cos(current_angle)
            y = prey_center[1] + coil_radius * np.sin(current_angle)
            positions[i] = [x, y, 0.0]
            current_angle += d_theta
        else:
            # Straight portion - tangent to the end of wrapped section
            # Tangent direction at end of wrap
            end_angle = current_angle - d_theta  # Last wrapped angle
            tangent_dir = np.array([-np.sin(end_angle), np.cos(end_angle)])

            # Position along tangent
            dist_from_wrap_end = (i - wrapped_nodes + 1) * segment_length
            end_pos = positions[wrapped_nodes - 1, :2]
            pos_2d = end_pos + dist_from_wrap_end * tangent_dir
            positions[i] = [pos_2d[0], pos_2d[1], 0.0]

    return positions


def inject_positions(robot: SnakeRobot, positions: np.ndarray) -> None:
    """Inject custom positions into robot state."""
    num_nodes = len(positions)
    robot._dismech_robot.state.q[:3 * num_nodes] = positions.flatten()
    robot._dismech_robot.state.u[:] = 0
    robot._update_snake_adapter()
    robot._update_contact_state()


def run_partial_wrap_experiment(
    wrap_fraction: float,
    curvature_value: float,
    max_steps: int = 200,
    verbose: bool = True,
) -> dict:
    """Run coiling experiment from partial wrap position."""

    config = PhysicsConfig(
        snake_length=1.0,
        snake_radius=0.001,
        num_segments=20,
        prey_radius=0.1,
        prey_length=0.3,
        dt=0.02,
        max_iter=100,
        tol=1e-4,
        enable_gravity=False,
        use_rft=True,
    )

    num_nodes = config.num_segments + 1
    num_joints = config.num_segments - 1
    prey_center = np.array([0.0, 0.0, 0.0])

    # Create robot
    robot = SnakeRobot(config, initial_prey_position=prey_center)

    # Create partial wrap positions
    partial_positions = create_partial_wrap_positions(
        num_nodes=num_nodes,
        prey_center=prey_center,
        prey_radius=config.prey_radius,
        snake_length=config.snake_length,
        wrap_fraction=wrap_fraction,
        offset=0.005,
    )

    # Inject positions
    inject_positions(robot, partial_positions)

    # Get initial state
    initial_state = robot.get_state()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Wrap fraction: {wrap_fraction:.1%}, Curvature: {curvature_value:.2f}")
        print(f"Initial contact: {initial_state['contact_fraction']:.3f}")
        print(f"Initial wrap: {initial_state['wrap_count']:.3f}")
        print(f"{'='*60}")

    # Tracking
    history = {
        'positions': [partial_positions.copy()],
        'curvatures': [initial_state['curvatures'].copy()],
        'contact_fraction': [initial_state['contact_fraction']],
        'wrap_count': [initial_state['wrap_count']],
    }

    success_steps = 0
    success_achieved = False
    success_step = None

    # Apply constant curvature
    curvature_cmd = np.full(num_joints, curvature_value)
    robot.set_curvature_control(curvature_cmd)

    for step in range(max_steps):
        try:
            state = robot.step()
        except Exception as e:
            if verbose:
                print(f"  Step {step}: Error: {e}")
            break

        history['positions'].append(state['positions'].copy())
        history['curvatures'].append(state['curvatures'].copy())
        history['contact_fraction'].append(state['contact_fraction'])
        history['wrap_count'].append(state['wrap_count'])

        # Check success
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
                print(f"  Step {step}: SUCCESS!")

        if verbose and step % 20 == 0:
            print(f"  Step {step:3d}: contact={state['contact_fraction']:.3f}, "
                  f"wrap={state['wrap_count']:.3f}")

    # Convert to arrays
    for key in history:
        history[key] = np.array(history[key])

    return {
        'wrap_fraction': wrap_fraction,
        'curvature': curvature_value,
        'success': success_achieved,
        'success_step': success_step,
        'final_contact': history['contact_fraction'][-1],
        'final_wrap': history['wrap_count'][-1],
        'max_contact': np.max(history['contact_fraction']),
        'max_wrap': np.max(np.abs(history['wrap_count'])),
        'history': history,
        'config': {
            'snake_length': config.snake_length,
            'prey_radius': config.prey_radius,
        }
    }


def create_visualization(result: dict, output_dir: Path):
    """Create visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    history = result['history']
    positions = history['positions']
    contact = history['contact_fraction']
    wrap = history['wrap_count']

    num_steps = len(positions)
    prey_radius = result['config']['prey_radius']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Key frames
    frames = [0, num_steps//4, num_steps//2, num_steps-1]

    for i, frame in enumerate(frames):
        ax = axes[0, i]
        pos = positions[frame]

        circle = Circle((0, 0), prey_radius, fill=True,
                        facecolor='lightcoral', edgecolor='darkred',
                        linewidth=2, alpha=0.7)
        ax.add_patch(circle)

        ax.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2)
        ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=20, zorder=3)
        ax.scatter(pos[0, 0], pos[0, 1], c='red', s=80, marker='^', zorder=4)
        ax.scatter(pos[-1, 0], pos[-1, 1], c='purple', s=80, marker='s', zorder=4)

        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"t={frame}\ncontact={contact[frame]:.2f}, wrap={wrap[frame]:.2f}")

    # Metrics
    t = np.arange(num_steps)

    axes[1, 0].plot(t, contact, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0.6, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Contact Fraction')
    axes[1, 0].set_title('Contact Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t, wrap, 'purple', linewidth=2)
    axes[1, 1].axhline(y=1.5, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Wrap Count')
    axes[1, 1].set_title('Wrap Count Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    # Curvature heatmap
    curv = history['curvatures']
    if len(curv.shape) == 2:
        im = axes[1, 2].imshow(curv.T, aspect='auto', cmap='viridis')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Joint')
        axes[1, 2].set_title('Curvature Profile')
        plt.colorbar(im, ax=axes[1, 2])

    axes[1, 3].text(0.5, 0.5,
                   f"Initial wrap: {result['wrap_fraction']:.0%}\n"
                   f"Curvature: {result['curvature']:.2f}\n"
                   f"Success: {result['success']}\n"
                   f"Max contact: {result['max_contact']:.3f}\n"
                   f"Max wrap: {result['max_wrap']:.3f}",
                   transform=axes[1, 3].transAxes,
                   fontsize=12, verticalalignment='center',
                   horizontalalignment='center')
    axes[1, 3].axis('off')

    success_str = "SUCCESS" if result['success'] else "FAILED"
    fig.suptitle(f"Partial Wrap to Full Coil - {success_str}", fontsize=14)
    plt.tight_layout()

    filename = f"partial_wrap_{int(result['wrap_fraction']*100)}pct_k{result['curvature']:.1f}.png"
    plt.savefig(output_dir / filename, dpi=150)
    print(f"Saved: {output_dir / filename}")
    plt.close()


def create_video(result: dict, output_path: Path):
    """Create video of coiling."""
    history = result['history']
    positions = history['positions']
    contact = history['contact_fraction']
    wrap = history['wrap_count']

    num_steps = len(positions)
    prey_radius = result['config']['prey_radius']

    fig, ax = plt.subplots(figsize=(8, 8))

    def animate(frame):
        ax.clear()
        pos = positions[frame]

        circle = Circle((0, 0), prey_radius, fill=True,
                        facecolor='lightcoral', edgecolor='darkred',
                        linewidth=2, alpha=0.7)
        ax.add_patch(circle)

        ax.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=3)
        ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=30, zorder=3)
        ax.scatter(pos[0, 0], pos[0, 1], c='red', s=100, marker='^', zorder=4, label='Head')
        ax.scatter(pos[-1, 0], pos[-1, 1], c='purple', s=100, marker='s', zorder=4, label='Tail')

        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f"Step {frame}: contact={contact[frame]:.2f}, wrap={wrap[frame]:.2f}")

        return [ax]

    anim = FuncAnimation(fig, animate, frames=range(0, num_steps, 2),
                        interval=50, blit=False)

    writer = FFMpegWriter(fps=20, metadata=dict(artist='snake-hrl'))
    anim.save(output_path, writer=writer)
    print(f"Saved video: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("VERIFICATION: Partial Wrap to Full Coil")
    print("="*60)
    print("\nStrategy: Start snake with partial wrap, complete with curvature")
    print()

    output_dir = Path("Media/partial_wrap_coil")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test different initial wrap fractions and curvatures
    wrap_fractions = [0.3, 0.5, 0.7, 0.9]  # 30%, 50%, 70%, 90%
    curvatures = [9.5, 10.0]

    results = []
    successful_result = None

    for wrap_frac in wrap_fractions:
        for curv in curvatures:
            result = run_partial_wrap_experiment(
                wrap_fraction=wrap_frac,
                curvature_value=curv,
                max_steps=200,
                verbose=True,
            )
            results.append(result)

            create_visualization(result, output_dir)

            if result['success'] and successful_result is None:
                successful_result = result
                print(f"\n>>> SUCCESS FOUND!")

                # Save trajectory
                np.savez(
                    output_dir / "successful_partial_wrap.npz",
                    **result['history'],
                    wrap_fraction=result['wrap_fraction'],
                    curvature=result['curvature'],
                )

                # Create video
                create_video(result, output_dir / "successful_partial_wrap.mp4")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'WrapFrac':>8} {'κ':>6} {'Success':>8} {'MaxContact':>10} {'MaxWrap':>10}")
    print("-"*50)

    for r in results:
        wrap_pct = r['wrap_fraction'] * 100
        success_str = "YES" if r['success'] else "no"
        print(f"{wrap_pct:>7.0f}% {r['curvature']:>6.1f} {success_str:>8} "
              f"{r['max_contact']:>10.3f} {r['max_wrap']:>10.3f}")

    if successful_result:
        print(f"\nSUCCESS! Trajectory saved.")
    else:
        print("\nNo success. Best result:")
        best = max(results, key=lambda r: r['max_contact'] + r['max_wrap'])
        print(f"  Wrap fraction: {best['wrap_fraction']:.0%}")
        print(f"  Curvature: {best['curvature']}")
        print(f"  Max contact: {best['max_contact']:.3f}")
        print(f"  Max wrap: {best['max_wrap']:.3f}")


if __name__ == "__main__":
    main()
