"""Nonlinear Evolutionary MPC using DD-PINN surrogate (Licher et al., 2025).

The NEMPC uses CMA-ES to optimize pressure sequences over a finite horizon,
evaluating candidate trajectories with the trained DD-PINN instead of the
numerical Cosserat rod solver. Runs at 70 Hz on GPU.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from licher2025.configs_licher2025 import MPCConfig, UKFConfig


class UnscentedKalmanFilter:
    """UKF for Cosserat rod state and parameter estimation.

    Estimates the full 72-dim state plus bending compliance parameters
    from tip position/orientation measurements (motion capture markers).
    """

    def __init__(self, state_dim: int, param_dim: int, config: UKFConfig = None):
        self.config = config or UKFConfig()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.augmented_dim = state_dim + param_dim

        # State estimate
        self.x_hat = np.zeros(self.augmented_dim)
        self.P = np.eye(self.augmented_dim) * 1e-2

        # UKF weights
        n = self.augmented_dim
        alpha = self.config.alpha
        beta = self.config.beta
        kappa = self.config.kappa
        lam = alpha ** 2 * (n + kappa) - n

        self.num_sigma = 2 * n + 1
        self.wm = np.full(self.num_sigma, 1.0 / (2 * (n + lam)))
        self.wc = np.full(self.num_sigma, 1.0 / (2 * (n + lam)))
        self.wm[0] = lam / (n + lam)
        self.wc[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)
        self._sqrt_factor = np.sqrt(n + lam)

    def reset(self, x0: np.ndarray, theta0: np.ndarray) -> None:
        """Initialize state estimate.

        Args:
            x0: Initial state estimate, shape (state_dim,).
            theta0: Initial parameter estimate, shape (param_dim,).
        """
        self.x_hat[:self.state_dim] = x0
        self.x_hat[self.state_dim:] = theta0
        self.P = np.eye(self.augmented_dim) * 1e-2

    def predict(self, dynamics_fn, u: np.ndarray, dt: float) -> None:
        """UKF predict step using the PINN dynamics model.

        Args:
            dynamics_fn: Callable (x, u, theta, dt) -> x_next.
            u: Control input, shape (control_dim,).
            dt: Time step.
        """
        # Generate sigma points
        sigma = self._sigma_points()

        # Propagate through dynamics
        sigma_pred = np.zeros_like(sigma)
        for i in range(self.num_sigma):
            x_i = sigma[i, :self.state_dim]
            theta_i = sigma[i, self.state_dim:]
            x_next = dynamics_fn(x_i, u, theta_i, dt)
            sigma_pred[i, :self.state_dim] = x_next
            sigma_pred[i, self.state_dim:] = theta_i  # Params unchanged

        # Weighted mean and covariance
        self.x_hat = np.sum(self.wm[:, None] * sigma_pred, axis=0)
        diff = sigma_pred - self.x_hat
        self.P = np.sum(
            self.wc[:, None, None] * (diff[:, :, None] @ diff[:, None, :]),
            axis=0,
        )

        # Add process noise
        Q = np.eye(self.augmented_dim)
        Q[:self.state_dim, :self.state_dim] *= self.config.process_noise_state
        Q[self.state_dim:, self.state_dim:] *= self.config.process_noise_param
        self.P += Q

    def update(self, z: np.ndarray, measurement_fn) -> None:
        """UKF update step with measurement.

        Args:
            z: Measurement vector (tip pos + orientation).
            measurement_fn: Callable (augmented_state) -> measurement.
        """
        sigma = self._sigma_points()
        meas_dim = z.shape[0]

        # Predicted measurements
        z_pred = np.zeros((self.num_sigma, meas_dim))
        for i in range(self.num_sigma):
            z_pred[i] = measurement_fn(sigma[i])

        z_mean = np.sum(self.wm[:, None] * z_pred, axis=0)

        # Innovation covariance
        z_diff = z_pred - z_mean
        S = np.sum(
            self.wc[:, None, None] * (z_diff[:, :, None] @ z_diff[:, None, :]),
            axis=0,
        )
        R = np.eye(meas_dim)
        R[:3, :3] *= self.config.measurement_noise_pos ** 2
        if meas_dim > 3:
            R[3:, 3:] *= self.config.measurement_noise_orient ** 2
        S += R

        # Cross-covariance
        x_diff = sigma - self.x_hat
        Pxz = np.sum(
            self.wc[:, None, None] * (x_diff[:, :, None] @ z_diff[:, None, :]),
            axis=0,
        )

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # Update
        self.x_hat += K @ (z - z_mean)
        self.P -= K @ S @ K.T

    @property
    def state_estimate(self) -> np.ndarray:
        return self.x_hat[:self.state_dim].copy()

    @property
    def param_estimate(self) -> np.ndarray:
        return self.x_hat[self.state_dim:].copy()

    def _sigma_points(self) -> np.ndarray:
        """Generate sigma points around current estimate."""
        n = self.augmented_dim
        sigma = np.zeros((self.num_sigma, n))
        sigma[0] = self.x_hat

        try:
            L = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(self.P + 1e-6 * np.eye(n))

        scaled_L = self._sqrt_factor * L
        for i in range(n):
            sigma[1 + i] = self.x_hat + scaled_L[:, i]
            sigma[1 + n + i] = self.x_hat - scaled_L[:, i]

        return sigma


class NonlinearEvolutionaryMPC:
    """NEMPC controller using DD-PINN for trajectory evaluation.

    Optimizes pressure sequences u_{0:H} over a prediction horizon
    using a simple evolutionary strategy (truncation selection + mutation).

    At each control step:
    1. Generate population of candidate control sequences
    2. Roll out each using the DD-PINN (batched, single forward pass per step)
    3. Evaluate cost (tracking error + effort + smoothness)
    4. Select elite, mutate, repeat for num_generations
    5. Apply first control of best sequence
    """

    def __init__(
        self,
        pinn_model: "DomainDecoupledPINN",
        config: MPCConfig = None,
        device: str = "cpu",
    ):
        self.pinn = pinn_model
        self.config = config or MPCConfig()
        self._device = device

        self._prev_u = None  # Warm-start from previous solution

    def solve(
        self,
        x0: np.ndarray,
        theta: np.ndarray,
        target_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Solve for optimal control sequence.

        Args:
            x0: Current state estimate, shape (state_dim,).
            theta: Current parameter estimate, shape (param_dim,).
            target_trajectory: Desired states over horizon,
                shape (prediction_horizon, target_dim).

        Returns:
            (optimal_u, cost): First control action and associated cost.
        """
        cfg = self.config
        H = cfg.control_horizon
        pop_size = cfg.population_size
        n_elite = max(1, int(pop_size * cfg.elite_fraction))
        control_dim = 3  # 3 pressure inputs

        # Initialize population (warm-start or random)
        if self._prev_u is not None:
            # Shift previous solution by 1 step
            mean = np.zeros((H, control_dim))
            mean[:-1] = self._prev_u[1:]
            mean[-1] = self._prev_u[-1]
        else:
            mean = np.full((H, control_dim), cfg.max_pressure / 2)

        sigma = np.full((H, control_dim), cfg.mutation_sigma * cfg.max_pressure)

        best_u = mean.copy()
        best_cost = float("inf")

        for gen in range(cfg.num_generations):
            # Sample population
            population = np.random.randn(pop_size, H, control_dim) * sigma + mean

            # Clip to constraints
            population = np.clip(population, cfg.min_pressure, cfg.max_pressure)

            # Enforce rate constraints
            if self._prev_u is not None:
                rate = population[:, 0] - self._prev_u[-1]
                rate = np.clip(rate, -cfg.max_pressure_rate, cfg.max_pressure_rate)
                population[:, 0] = self._prev_u[-1] + rate

            for t in range(1, H):
                rate = population[:, t] - population[:, t - 1]
                rate = np.clip(rate, -cfg.max_pressure_rate, cfg.max_pressure_rate)
                population[:, t] = population[:, t - 1] + rate

            # Evaluate costs (batched PINN rollout)
            costs = self._evaluate_batch(population, x0, theta, target_trajectory)

            # Select elite
            elite_idx = np.argsort(costs)[:n_elite]
            elite = population[elite_idx]

            # Update mean and sigma
            mean = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 1e-6

            if costs[elite_idx[0]] < best_cost:
                best_cost = costs[elite_idx[0]]
                best_u = elite[0].copy()

        self._prev_u = best_u

        return best_u[0], float(best_cost)

    def _evaluate_batch(
        self,
        population: np.ndarray,
        x0: np.ndarray,
        theta: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Evaluate cost for a batch of control sequences using the PINN.

        Args:
            population: Control sequences, shape (pop, H, control_dim).
            x0: Current state, shape (state_dim,).
            theta: Parameters, shape (param_dim,).
            targets: Target trajectory, shape (pred_horizon, target_dim).

        Returns:
            Costs for each member, shape (pop,).
        """
        cfg = self.config
        pop_size = population.shape[0]
        H = population.shape[1]
        dt = cfg.control_dt

        # Convert to torch
        x = torch.tensor(
            np.tile(x0, (pop_size, 1)), dtype=torch.float32, device=self._device
        )
        th = torch.tensor(
            np.tile(theta, (pop_size, 1)), dtype=torch.float32, device=self._device
        )

        costs = np.zeros(pop_size)

        with torch.no_grad():
            for t in range(min(H, cfg.prediction_horizon)):
                u_t = torch.tensor(
                    population[:, min(t, H - 1)],
                    dtype=torch.float32,
                    device=self._device,
                )
                t_tensor = torch.full(
                    (pop_size,), dt, dtype=torch.float32, device=self._device
                )

                # Predict next state
                x = self.pinn(x, u_t, th, t_tensor)

                # Tip position is first 3 elements of state (simplified)
                tip_pos = x[:, :3].cpu().numpy()
                target_t = targets[min(t, len(targets) - 1)]

                # Position tracking cost
                pos_err = np.linalg.norm(tip_pos - target_t[:3], axis=1)
                costs += cfg.position_weight * pos_err ** 2

                # Control effort
                u_np = population[:, min(t, H - 1)]
                costs += cfg.control_effort_weight * np.sum(u_np ** 2, axis=1)

                # Smoothness (rate of change)
                if t > 0:
                    du = population[:, min(t, H - 1)] - population[:, min(t - 1, H - 1)]
                    costs += cfg.smoothness_weight * np.sum(du ** 2, axis=1)

        return costs

    def reset(self) -> None:
        """Reset warm-start (call at episode boundaries)."""
        self._prev_u = None
