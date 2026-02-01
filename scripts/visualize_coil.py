#!/usr/bin/env python3
"""Visualize coiled snake configurations and generate videos.

This script creates visualizations of the coiled snake around prey,
including:
1. Static plot of the coiled configuration
2. Video showing the coil over time
3. Curvature profile visualization
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
import sys

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
    """Generate snake node positions forming a spiral around prey."""
    coil_radius = prey_radius + offset
    segment_length = snake_length / (num_nodes - 1)
    positions = np.zeros((num_nodes, 3))

    current_angle = 0.0
    for i in range(num_nodes):
        x = prey_center[0] + coil_radius * np.cos(current_angle)
        y = prey_center[1] + coil_radius * np.sin(current_angle)
        positions[i] = [x, y, 0.0]

        if i < num_nodes - 1:
            d_theta = segment_length / coil_radius * direction
            current_angle += d_theta

    return positions


def inject_coiled_state(robot: SnakeRobot, coiled_positions: np.ndarray) -> None:
    """Manually inject coiled positions into DisMech state."""
    num_nodes = len(coiled_positions)
    q = robot._dismech_robot.state.q.copy()
    q[:3 * num_nodes] = coiled_positions.flatten()
    robot._dismech_robot.state.q[:] = q
    robot._dismech_robot.state.u[:] = 0
    robot._update_snake_adapter()
    robot._update_contact_state()


def plot_snake_configuration(ax, positions, prey_center, prey_radius,
                             contact_mask=None, title=None):
    """Plot snake and prey configuration."""
    ax.clear()

    # Plot prey cylinder (as circle in XY plane)
    prey_circle = Circle(prey_center[:2], prey_radius,
                         fill=True, facecolor='lightcoral',
                         edgecolor='darkred', linewidth=2, alpha=0.7,
                         label='Prey')
    ax.add_patch(prey_circle)

    # Plot snake body as connected line
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=3,
            label='Snake body', zorder=2)

    # Plot nodes with contact coloring
    if contact_mask is not None:
        # Contact nodes in green, non-contact in blue
        for i, (pos, in_contact) in enumerate(zip(positions, contact_mask)):
            color = 'lime' if in_contact else 'blue'
            size = 80 if in_contact else 40
            ax.scatter(pos[0], pos[1], c=color, s=size, zorder=3)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=40, zorder=3)

    # Mark head and tail
    ax.scatter(positions[0, 0], positions[0, 1], c='red', s=120,
               marker='^', label='Head', zorder=4)
    ax.scatter(positions[-1, 0], positions[-1, 1], c='purple', s=120,
               marker='s', label='Tail', zorder=4)

    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    if title:
        ax.set_title(title)


def visualize_saved_trajectory(data_path: Path, output_dir: Path,
                               create_video: bool = True):
    """Visualize a saved verification trajectory."""
    data = np.load(data_path, allow_pickle=True)

    positions = data['positions']  # (timesteps, 21, 3)
    curvatures = data['curvatures']  # (timesteps, 19)
    contact_fraction = data['contact_fraction']
    wrap_count = data['wrap_count']
    strategy = str(data['strategy']) if 'strategy' in data else 'simulation'

    num_timesteps = len(positions)

    # Get prey info from config
    config = PhysicsConfig()
    prey_center = np.array([0.0, 0.0, 0.0])
    prey_radius = config.prey_radius

    print(f"Loaded trajectory: {strategy}")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Contact fraction range: [{contact_fraction.min():.3f}, {contact_fraction.max():.3f}]")
    print(f"  Wrap count range: [{wrap_count.min():.3f}, {wrap_count.max():.3f}]")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create static visualization of first and last frames
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Frame 0
    plot_snake_configuration(
        axes[0], positions[0], prey_center, prey_radius,
        title=f"t=0: contact={contact_fraction[0]:.2f}, wraps={wrap_count[0]:.2f}"
    )

    # Middle frame
    mid = num_timesteps // 2
    plot_snake_configuration(
        axes[1], positions[mid], prey_center, prey_radius,
        title=f"t={mid}: contact={contact_fraction[mid]:.2f}, wraps={wrap_count[mid]:.2f}"
    )

    # Last frame
    plot_snake_configuration(
        axes[2], positions[-1], prey_center, prey_radius,
        title=f"t={num_timesteps-1}: contact={contact_fraction[-1]:.2f}, wraps={wrap_count[-1]:.2f}"
    )

    fig.suptitle(f"Coiled Snake Configuration - Strategy: {strategy}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "coil_frames.png", dpi=150)
    print(f"Saved static frames to {output_dir / 'coil_frames.png'}")
    plt.close()

    # Create curvature profile plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Curvature over time (heatmap)
    im = axes[0, 0].imshow(curvatures.T, aspect='auto', cmap='viridis',
                           extent=[0, num_timesteps, 0, 19])
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Joint index')
    axes[0, 0].set_title('Curvature Profile Over Time')
    plt.colorbar(im, ax=axes[0, 0], label='Curvature (κ)')

    # Mean curvature over time
    mean_curv = np.mean(curvatures, axis=1)
    axes[0, 1].plot(range(num_timesteps), mean_curv, 'b-', linewidth=2)
    axes[0, 1].axhline(y=10.0, color='r', linestyle='--', label='κ=10 (ideal)')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Mean Curvature')
    axes[0, 1].set_title('Mean Curvature Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Contact fraction and wrap count
    axes[1, 0].plot(range(num_timesteps), contact_fraction, 'g-',
                    linewidth=2, label='Contact fraction')
    axes[1, 0].axhline(y=0.6, color='r', linestyle='--', label='Threshold (0.6)')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Contact Fraction')
    axes[1, 0].set_title('Contact Fraction Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.1)

    axes[1, 1].plot(range(num_timesteps), wrap_count, 'purple',
                    linewidth=2, label='Wrap count')
    axes[1, 1].axhline(y=1.5, color='r', linestyle='--', label='Threshold (1.5)')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Wrap Count')
    axes[1, 1].set_title('Wrap Count Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Coiling Metrics - Strategy: {strategy}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "coil_metrics.png", dpi=150)
    print(f"Saved metrics plot to {output_dir / 'coil_metrics.png'}")
    plt.close()

    # Create video if requested
    if create_video:
        print("Creating video animation...")
        create_coil_video(positions, prey_center, prey_radius,
                          contact_fraction, wrap_count, curvatures,
                          output_dir / "coil_animation.mp4")


def create_coil_video(positions, prey_center, prey_radius,
                      contact_fraction, wrap_count, curvatures, output_path):
    """Create animated video of coiled snake."""
    num_timesteps = len(positions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def animate(frame):
        # Left: Snake configuration
        axes[0].clear()
        pos = positions[frame]

        # Plot prey
        prey_circle = Circle(prey_center[:2], prey_radius,
                             fill=True, facecolor='lightcoral',
                             edgecolor='darkred', linewidth=2, alpha=0.7)
        axes[0].add_patch(prey_circle)

        # Plot snake
        axes[0].plot(pos[:, 0], pos[:, 1], 'b-', linewidth=3)
        axes[0].scatter(pos[:, 0], pos[:, 1], c='blue', s=30, zorder=3)
        axes[0].scatter(pos[0, 0], pos[0, 1], c='red', s=100, marker='^', zorder=4)
        axes[0].scatter(pos[-1, 0], pos[-1, 1], c='purple', s=100, marker='s', zorder=4)

        axes[0].set_xlim(-0.2, 0.2)
        axes[0].set_ylim(-0.2, 0.2)
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f"Step {frame}: contact={contact_fraction[frame]:.2f}, wraps={wrap_count[frame]:.2f}")

        # Right: Curvature profile
        axes[1].clear()
        axes[1].bar(range(19), curvatures[frame], color='steelblue', alpha=0.7)
        axes[1].axhline(y=10.0, color='r', linestyle='--', label='κ=10 target')
        axes[1].set_xlabel('Joint Index')
        axes[1].set_ylabel('Curvature (κ)')
        axes[1].set_title(f'Curvature Profile (mean={np.mean(curvatures[frame]):.2f})')
        axes[1].set_ylim(0, 12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        return axes

    anim = FuncAnimation(fig, animate, frames=num_timesteps,
                         interval=200, blit=False)

    # Save as MP4
    writer = FFMpegWriter(fps=5, metadata=dict(artist='snake-hrl'))
    anim.save(output_path, writer=writer)
    print(f"Saved video to {output_path}")
    plt.close()


def run_and_visualize_coiling(output_dir: Path, num_steps: int = 100,
                              curvature_target: float = 10.0):
    """Run a coiling simulation and visualize it."""
    config = PhysicsConfig(
        snake_length=1.0,
        num_segments=20,
        prey_radius=0.1,
        enable_gravity=False,
        dt=0.05,
        max_iter=50,
    )

    num_nodes = config.num_segments + 1
    num_joints = config.num_segments - 1
    prey_center = np.array([0.0, 0.0, 0.0])

    # Create robot with coiled initial position
    robot = SnakeRobot(config, initial_prey_position=prey_center)

    coiled_positions = create_coiled_positions(
        num_nodes=num_nodes,
        prey_center=prey_center,
        prey_radius=config.prey_radius,
        snake_length=config.snake_length,
        initial_wraps=1.0,
        offset=0.005,
    )

    inject_coiled_state(robot, coiled_positions)

    # Run simulation collecting data
    all_positions = []
    all_curvatures = []
    all_contact = []
    all_wrap = []

    # Set constant curvature control
    target_curv = np.full(num_joints, curvature_target)
    robot.set_curvature_control(target_curv)

    print(f"Running coiling simulation for {num_steps} steps...")

    for step in range(num_steps):
        state = robot.step()
        all_positions.append(state['positions'].copy())
        all_curvatures.append(state['curvatures'].copy())
        all_contact.append(state['contact_fraction'])
        all_wrap.append(state['wrap_count'])

        if step % 20 == 0:
            print(f"  Step {step}: contact={state['contact_fraction']:.3f}, wrap={state['wrap_count']:.3f}")

    # Convert to arrays
    positions = np.array(all_positions)
    curvatures = np.array(all_curvatures)
    contact_fraction = np.array(all_contact)
    wrap_count = np.array(all_wrap)

    # Save data
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "coil_simulation.npz",
        positions=positions,
        curvatures=curvatures,
        contact_fraction=contact_fraction,
        wrap_count=wrap_count,
        curvature_target=curvature_target,
    )
    print(f"Saved simulation data to {output_dir / 'coil_simulation.npz'}")

    # Create visualizations
    visualize_saved_trajectory(
        output_dir / "coil_simulation.npz",
        output_dir,
        create_video=True
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize coiled snake configurations")
    parser.add_argument("--data", type=Path,
                        default=Path("data/coil_verification_trajectory.npz"),
                        help="Path to saved trajectory data")
    parser.add_argument("--output", type=Path, default=Path("Media/coil_viz"),
                        help="Output directory for visualizations")
    parser.add_argument("--run-simulation", action="store_true",
                        help="Run new simulation instead of loading saved data")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of simulation steps (if running)")
    parser.add_argument("--curvature", type=float, default=10.0,
                        help="Target curvature (if running)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video creation")

    args = parser.parse_args()

    if args.run_simulation:
        run_and_visualize_coiling(args.output, args.steps, args.curvature)
    else:
        if not args.data.exists():
            print(f"Data file not found: {args.data}")
            print("Running new simulation instead...")
            run_and_visualize_coiling(args.output, args.steps, args.curvature)
        else:
            visualize_saved_trajectory(args.data, args.output,
                                       create_video=not args.no_video)


if __name__ == "__main__":
    main()
