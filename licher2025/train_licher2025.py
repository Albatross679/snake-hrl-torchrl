"""Training script for DD-PINN surrogate and MPC evaluation (Licher et al., 2025).

This trains the DD-PINN on Cosserat rod simulation data (supervised learning),
then evaluates it inside the NEMPC controller for tip-tracking tasks.

Usage:
    python -m licher2025.train_licher2025 --mode train_pinn --epochs 500
    python -m licher2025.train_licher2025 --mode evaluate_mpc --checkpoint model/licher2025_pinn.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from licher2025.configs_licher2025 import Licher2025Config
from licher2025.pinn_licher2025 import DomainDecoupledPINN
from licher2025.mpc_licher2025 import NonlinearEvolutionaryMPC
from licher2025.env_licher2025 import SoftPneumaticEnv
from configs.base import resolve_device
from configs.run_dir import setup_run_dir
from configs.console import ConsoleLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Licher 2025 DD-PINN + MPC")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_pinn",
        choices=["train_pinn", "evaluate_mpc", "generate_data"],
        help="Operating mode",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="PINN checkpoint path"
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="circular",
        choices=["circular", "figure_eight", "step"],
        help="Target trajectory type for MPC evaluation",
    )
    return parser.parse_args()


def generate_training_data(config: Licher2025Config, num_trajs: int = 1000):
    """Generate training data by running the Cosserat rod simulation.

    In a full implementation, this would numerically integrate the
    dynamic Cosserat rod equations for random initial conditions and
    control inputs, saving state trajectories for PINN training.
    """
    pinn_cfg = config.env.pinn
    state_dim = pinn_cfg.state_dim
    control_dim = pinn_cfg.control_dim
    param_dim = pinn_cfg.param_dim

    rng = np.random.default_rng(config.seed)

    # Synthetic training data (placeholder for actual Cosserat rod sim)
    x0 = rng.standard_normal((num_trajs, state_dim)).astype(np.float32) * 0.01
    u0 = rng.uniform(0, config.env.mpc.max_pressure, (num_trajs, control_dim)).astype(
        np.float32
    )
    theta = rng.uniform(5e-5, 2e-4, (num_trajs, param_dim)).astype(np.float32)

    # Multiple time points per trajectory
    num_times = 10
    dt = config.env.mpc.control_dt
    times = np.arange(1, num_times + 1) * dt

    # Ground truth states (would come from Cosserat rod integration)
    x_target = np.zeros((num_trajs, num_times, state_dim), dtype=np.float32)
    for t_idx, t_val in enumerate(times):
        # Simplified: x(t) = x0 + small perturbation (placeholder)
        x_target[:, t_idx] = x0 + rng.standard_normal(
            (num_trajs, state_dim)
        ).astype(np.float32) * 0.001 * (t_idx + 1)

    # Save
    save_path = Path(config.pinn_training_data)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        x0=x0,
        u0=u0,
        theta=theta,
        times=times.astype(np.float32),
        x_target=x_target,
    )
    print(f"Generated {num_trajs} trajectories -> {save_path}")


def train_pinn(config: Licher2025Config, device: str = "cpu", run_dir: Path = None):
    """Train the DD-PINN on Cosserat rod simulation data."""
    pinn_cfg = config.env.pinn

    # Load data
    data_path = Path(config.pinn_training_data)
    if not data_path.exists():
        print("Training data not found, generating...")
        generate_training_data(config, config.pinn_num_trajectories)

    data = np.load(data_path)
    x0 = torch.tensor(data["x0"], device=device)
    u0 = torch.tensor(data["u0"], device=device)
    theta = torch.tensor(data["theta"], device=device)
    times = torch.tensor(data["times"], device=device)
    x_target = torch.tensor(data["x_target"], device=device)

    # Create model
    model = DomainDecoupledPINN(pinn_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pinn_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=pinn_cfg.lr_patience, factor=pinn_cfg.lr_factor
    )

    num_epochs = pinn_cfg.num_epochs
    num_times = times.shape[0]
    n = x0.shape[0]

    # Determine save directory
    save_dir = (run_dir / "checkpoints") if run_dir else Path("model")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training DD-PINN: {n} trajectories, {num_times} time points")
    print(f"  State dim: {pinn_cfg.state_dim}, Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Data loss: predict state at each time point
        for t_idx in range(num_times):
            t_val = times[t_idx].expand(n)
            x_pred = model(x0, u0, theta, t_val)
            data_loss = nn.functional.mse_loss(x_pred, x_target[:, t_idx])

            # Physics loss: time derivative should match dynamics residual
            if pinn_cfg.physics_loss_weight > 0:
                dx_dt = model.time_derivative(x0, u0, theta, t_val)
                # Simplified physics residual (full version uses Cosserat PDE)
                physics_loss = torch.mean(dx_dt ** 2) * 0.01
            else:
                physics_loss = torch.tensor(0.0, device=device)

            loss = (
                pinn_cfg.data_loss_weight * data_loss
                + pinn_cfg.physics_loss_weight * physics_loss
            )
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / num_times
        scheduler.step(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.6f}, lr={lr:.2e}")

    # Save model
    save_path = save_dir / "licher2025_pinn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model


def evaluate_mpc(config: Licher2025Config, checkpoint: str, device: str = "cpu"):
    """Evaluate the trained PINN inside the NEMPC controller."""
    pinn_cfg = config.env.pinn

    # Load model
    model = DomainDecoupledPINN(pinn_cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Create MPC controller
    mpc = NonlinearEvolutionaryMPC(model, config.env.mpc, device=device)

    # Create environment
    env = SoftPneumaticEnv(config.env, device=device)

    # Run episodes
    num_episodes = 5
    for ep in range(num_episodes):
        td = env.reset()
        mpc.reset()
        total_reward = 0.0

        for step in range(config.env.max_episode_steps):
            obs = td["observation"].cpu().numpy()
            x0 = np.zeros(pinn_cfg.state_dim)
            x0[:3] = obs[:3]  # tip position
            theta = obs[6:9]  # compliance

            # Generate target trajectory for horizon
            dt = config.env.mpc.control_dt
            targets = np.array([
                env._generate_target(env._trajectory_time + i * dt)
                for i in range(config.env.mpc.prediction_horizon)
            ])

            # Solve MPC
            u_opt, cost = mpc.solve(x0, theta, targets)

            # Step environment
            action = torch.tensor(u_opt, dtype=torch.float32, device=device)
            td = env.step(TensorDict({"action": action}, batch_size=env.batch_size))
            total_reward += td["next", "reward"].item()

            if td["next", "done"].item():
                break

            td = td["next"]

        print(f"Episode {ep + 1}: reward={total_reward:.4f}, steps={step + 1}")

    env.close()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    config = Licher2025Config(seed=args.seed)
    config.name = f"licher2025_{args.mode}"
    config.env.target_type = args.target_type

    if args.epochs is not None:
        config.env.pinn.num_epochs = args.epochs

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)

    with ConsoleLogger(run_dir, None):
        print(f"Licher 2025 DD-PINN + MPC: mode={args.mode}")
        print(f"  Run directory: {run_dir}")

        if args.mode == "generate_data":
            generate_training_data(config, config.pinn_num_trajectories)
        elif args.mode == "train_pinn":
            train_pinn(config, device=device, run_dir=run_dir)
        elif args.mode == "evaluate_mpc":
            if args.checkpoint is None:
                print("Error: --checkpoint required for evaluate_mpc mode")
                return
            evaluate_mpc(config, args.checkpoint, device=device)


if __name__ == "__main__":
    main()
