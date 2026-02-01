#!/usr/bin/env python3
"""
Standalone snake simulation matching the dismech-python reference notebook.
https://github.com/StructuresComp/dismech-python/blob/main/experiments/snake.ipynb

This script reproduces the exact simulation parameters from the reference notebook
for validation and comparison purposes.

Usage:
    python scripts/simulate_snake_reference.py --total-time 10.0
    python scripts/simulate_snake_reference.py --total-time 10.0 --visualize
    python scripts/simulate_snake_reference.py --total-time 10.0 --output reference_traj.npy

Parameters (matching snake.ipynb exactly):
    - Rod radius: 0.001m
    - Density: 1200 kg/m^3
    - Young's modulus: 2e6 Pa
    - Poisson's ratio: 0.5
    - Timestep: 0.05s
    - RFT coefficients: ct=0.01, cn=0.1
    - Actuation: amplitude=0.2, frequency=2.0Hz, wavelength=1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dismech


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Snake simulation with reference parameters from dismech-python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=10.0,
        help="Total simulation time in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.2,
        help="Actuation amplitude (default: 0.2)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=2.0,
        help="Actuation frequency in Hz (default: 2.0)",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=1.0,
        help="Spatial wavelength in normalized units (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save trajectory to .npy file",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show interactive animation after simulation",
    )
    parser.add_argument(
        "--plot-position",
        action="store_true",
        help="Plot x-position of node 0 over time",
    )
    return parser.parse_args()


def create_robot(total_time: float = 10.0):
    """Create DisMech robot with reference parameters from snake.ipynb.

    Returns:
        Tuple of (robot, sim_params)
    """
    # Geometry parameters (matching snake.ipynb)
    geom = dismech.GeomParams(
        rod_r0=0.001,  # Rod radius: 1mm
        shell_h=0,     # No shell
    )

    # Material properties (matching snake.ipynb)
    material = dismech.Material(
        density=1200,       # kg/m^3
        youngs_rod=2e6,     # 2 MPa
        youngs_shell=0,
        poisson_rod=0.5,
        poisson_shell=0,
    )

    # Simulation parameters (matching snake.ipynb exactly)
    sim_params = dismech.SimParams(
        static_sim=False,
        two_d_sim=False,       # Enable full 3D (with twisting)
        use_mid_edge=False,
        use_line_search=False,
        show_floor=False,
        log_data=True,
        log_step=1,
        dt=5e-2,               # 50ms timestep
        max_iter=25,
        total_time=total_time,
        plot_step=1,
        tol=1e-4,
        ftol=1e-4,
        dtol=1e-2,
    )

    # Environment: RFT only, no gravity (matching snake.ipynb)
    env = dismech.Environment()
    env.add_force('rft', ct=0.01, cn=0.1)
    # Note: The following are commented out in snake.ipynb:
    # env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
    # env.add_force('floorContact', ground_z=0, stiffness=1e3, delta=5e-3, h=1e-3)
    # env.add_force('floorFriction', mu=0.75, vel_tol=1e-3)

    # Load geometry from file (21 nodes, 0.1m total length)
    script_dir = Path(__file__).parent
    geo_path = script_dir.parent / "dismech-python/tests/resources/rod_cantilever/horizontal_rod_n21.txt"

    if not geo_path.exists():
        raise FileNotFoundError(
            f"Geometry file not found: {geo_path}\n"
            "Make sure dismech-python is cloned in the project root."
        )

    geo = dismech.Geometry.from_txt(str(geo_path))

    # Create robot
    robot = dismech.SoftRobot(geom, material, geo, sim_params, env)

    return robot, sim_params


def make_actuation_fn(amplitude: float, frequency: float, wavelength: float, phase_offset: float = np.pi / 2):
    """Create actuation callback for traveling wave (matching snake.ipynb).

    The actuation applies a sinusoidal traveling wave to the tangential
    bending component (inc_strain[:, 1]).

    Args:
        amplitude: Maximum strain applied to each bend
        frequency: Oscillation frequency in Hz
        wavelength: Wavelength along the body in normalized units
        phase_offset: Phase offset between vertical and tangential components

    Returns:
        Callback function for stepper.before_step
    """
    def actuate_snake(robot, t):
        # Get number of bending elements
        n_bends = robot.bend_springs.inc_strain.shape[0]
        s = np.linspace(0, 1, n_bends)  # Normalized position along body

        # Compute traveling wave parameters
        omega = 2 * np.pi * frequency
        k = 2 * np.pi / wavelength

        # Generate sinusoidal waves
        # vertical_wave = amplitude * np.sin(omega * t - k * s)  # Not used in snake.ipynb
        tangential_wave = amplitude * np.sin(omega * t - k * s + phase_offset)

        # Update robot bending strain (shape [n_bends, 2])
        # robot.bend_springs.inc_strain[:, 0] = vertical_wave   # normal bending (commented in snake.ipynb)
        robot.bend_springs.inc_strain[:, 1] = tangential_wave   # tangential bending

        return robot

    return actuate_snake


def main():
    """Main entry point."""
    args = parse_args()

    # Create robot with reference parameters
    print(f"Creating robot with reference parameters...")
    robot, sim_params = create_robot(total_time=args.total_time)

    print(f"  Geometry: 21 nodes, 0.1m total length")
    print(f"  Material: density=1200, E=2e6, nu=0.5")
    print(f"  Timestep: {sim_params.dt}s")
    print(f"  Environment: RFT (ct=0.01, cn=0.1)")

    # Create time stepper
    stepper = dismech.ImplicitEulerTimeStepper(robot)

    # Set up actuation callback
    stepper.before_step = make_actuation_fn(
        amplitude=args.amplitude,
        frequency=args.frequency,
        wavelength=args.wavelength,
        phase_offset=np.pi / 2,
    )

    print(f"\nActuation parameters:")
    print(f"  Amplitude: {args.amplitude}")
    print(f"  Frequency: {args.frequency} Hz")
    print(f"  Wavelength: {args.wavelength}")
    print(f"  Phase offset: pi/2")

    # Run simulation
    print(f"\nRunning simulation for {args.total_time}s...")
    robots, time_array, f_norms = stepper.simulate(robot)

    # Extract trajectory (state vectors for each timestep)
    qs = np.stack([r.state.q for r in robots])
    # Match time array length to trajectory length
    t = np.array(time_array[:len(qs)])

    # Print results
    print(f"\nSimulation complete!")
    print(f"  Trajectory shape: {qs.shape}")
    print(f"  Number of timesteps: {len(t)}")
    print(f"  Final head position (node 0): [{qs[-1, 0]:.6f}, {qs[-1, 1]:.6f}, {qs[-1, 2]:.6f}]")
    print(f"  Head x-displacement: {qs[-1, 0] - qs[0, 0]:.6f}m")

    # Save trajectory if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, qs)
        print(f"\nSaved trajectory to: {args.output}")

    # Plot position if requested
    if args.plot_position:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.title("x Position of Node 0")
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.plot(t, qs[:, 0])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Warning: matplotlib not available for plotting")

    # Visualize if requested
    if args.visualize:
        print("\nLaunching interactive visualization...")
        options = dismech.AnimationOptions(title='Snake Reference')
        fig = dismech.get_interactive_animation_plotly(robot, t, qs, options)
        fig.show()

    return qs, t


if __name__ == "__main__":
    main()
