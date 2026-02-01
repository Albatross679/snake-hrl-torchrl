"""Behavioral Cloning pretrainer for approach policy.

This module provides a trainer for pretraining the approach worker policy
using supervised learning (behavioral cloning) from demonstration experiences.

The pretrainer uses MSE loss to train the policy to reproduce the actions
from successful locomotion experiences generated via grid search.

Example:
    >>> from snake_hrl.trainers.behavioral_cloning import BehavioralCloningPretrainer
    >>> from snake_hrl.demonstrations.approach_experiences import ApproachExperienceBuffer
    >>> import torch.nn as nn
    >>>
    >>> # Load experiences
    >>> buffer = ApproachExperienceBuffer()
    >>> buffer.load("data/approach_experiences.npz")
    >>>
    >>> # Create policy network
    >>> policy = nn.Sequential(
    ...     nn.Linear(13, 64),
    ...     nn.ReLU(),
    ...     nn.Linear(64, 64),
    ...     nn.ReLU(),
    ...     nn.Linear(64, 4),
    ... )
    >>>
    >>> # Train
    >>> pretrainer = BehavioralCloningPretrainer(policy, learning_rate=1e-3)
    >>> metrics = pretrainer.train(buffer, num_epochs=100)
    >>>
    >>> # Save pretrained weights
    >>> torch.save(policy.state_dict(), "checkpoints/approach_policy_pretrained.pt")
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from snake_hrl.demonstrations.approach_experiences import ApproachExperienceBuffer


class BehavioralCloningPretrainer:
    """Pretrain policy networks using behavioral cloning (supervised learning).

    Uses MSE loss to train a policy to reproduce expert actions:
        loss = ||policy(state) - action||^2

    Supports train/validation splits for monitoring overfitting.
    """

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        """Initialize the behavioral cloning pretrainer.

        Args:
            policy: Neural network mapping states to actions
            learning_rate: Learning rate for Adam optimizer
            batch_size: Mini-batch size for training
            weight_decay: L2 regularization coefficient
            device: Device for training ("cpu" or "cuda")
        """
        self.policy = policy.to(device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function (MSE)
        self.criterion = nn.MSELoss()

    def train(
        self,
        experience_buffer: ApproachExperienceBuffer,
        num_epochs: int = 100,
        validation_split: float = 0.1,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train policy using behavioral cloning.

        Args:
            experience_buffer: Buffer containing (state, action) experiences
            num_epochs: Number of training epochs
            validation_split: Fraction of data for validation (0 to disable)
            early_stopping_patience: Stop if val loss doesn't improve for N epochs
            verbose: If True, show progress bar and metrics

        Returns:
            Dictionary with training metrics:
            - train_loss: List of training losses per epoch
            - val_loss: List of validation losses per epoch (if validation_split > 0)
            - best_epoch: Epoch with best validation loss
        """
        # Convert buffer to tensors
        states, actions = experience_buffer.to_dataset()
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

        # Create dataset
        dataset = TensorDataset(states_tensor, actions_tensor)

        # Split into train/validation
        if validation_split > 0:
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            train_dataset = dataset
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training metrics
        metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
        }

        best_val_loss = float("inf")
        best_state_dict = None
        patience_counter = 0

        # Training loop
        epoch_iter = tqdm(range(num_epochs), desc="Training", disable=not verbose)
        for epoch in epoch_iter:
            # Training phase
            self.policy.train()
            train_losses = []

            for batch_states, batch_actions in train_loader:
                self.optimizer.zero_grad()

                # Forward pass
                predicted_actions = self.policy(batch_states)

                # Compute loss
                loss = self.criterion(predicted_actions, batch_actions)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            metrics["train_loss"].append(avg_train_loss)

            # Validation phase
            if val_loader is not None:
                self.policy.eval()
                val_losses = []

                with torch.no_grad():
                    for batch_states, batch_actions in val_loader:
                        predicted_actions = self.policy(batch_states)
                        loss = self.criterion(predicted_actions, batch_actions)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                metrics["val_loss"].append(avg_val_loss)

                # Track best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state_dict = {
                        k: v.clone() for k, v in self.policy.state_dict().items()
                    }
                    metrics["best_epoch"] = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Update progress bar
                epoch_iter.set_postfix(
                    train_loss=f"{avg_train_loss:.4f}",
                    val_loss=f"{avg_val_loss:.4f}",
                    best=f"{best_val_loss:.4f}",
                )

                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            else:
                epoch_iter.set_postfix(train_loss=f"{avg_train_loss:.4f}")

        # Restore best model if we did validation
        if best_state_dict is not None:
            self.policy.load_state_dict(best_state_dict)
            if verbose:
                print(f"\nRestored best model from epoch {metrics['best_epoch'] + 1}")

        return metrics

    def evaluate(
        self,
        experience_buffer: ApproachExperienceBuffer,
    ) -> Dict[str, float]:
        """Evaluate policy on experience buffer.

        Args:
            experience_buffer: Buffer containing (state, action) experiences

        Returns:
            Dictionary with evaluation metrics:
            - mse: Mean squared error
            - mae: Mean absolute error
            - action_error_per_dim: Per-dimension average error
        """
        states, actions = experience_buffer.to_dataset()
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

        self.policy.eval()
        with torch.no_grad():
            predicted = self.policy(states_tensor)

            mse = torch.mean((predicted - actions_tensor) ** 2).item()
            mae = torch.mean(torch.abs(predicted - actions_tensor)).item()

            # Per-dimension errors
            per_dim_error = torch.mean(
                torch.abs(predicted - actions_tensor), dim=0
            ).cpu().numpy()

        return {
            "mse": mse,
            "mae": mae,
            "action_error_per_dim": per_dim_error.tolist(),
        }

    def save_policy(self, path: str) -> None:
        """Save policy weights to file.

        Args:
            path: Path to save weights (typically .pt file)
        """
        torch.save(self.policy.state_dict(), path)

    def load_policy(self, path: str) -> None:
        """Load policy weights from file.

        Args:
            path: Path to load weights from
        """
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device)
        )


def create_mlp_policy(
    state_dim: int = 13,
    action_dim: int = 4,
    hidden_dims: Tuple[int, ...] = (64, 64),
    activation: str = "relu",
    output_activation: Optional[str] = None,
) -> nn.Module:
    """Create a simple MLP policy network.

    Args:
        state_dim: Input dimension (default: 13 for REDUCED_APPROACH)
        action_dim: Output dimension (default: 4 for serpenoid params)
        hidden_dims: Hidden layer sizes
        activation: Activation function ("relu", "tanh", "elu")
        output_activation: Output activation (None, "tanh", "sigmoid")

    Returns:
        MLP neural network module
    """
    # Activation functions
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }

    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")

    act_fn = activations[activation]

    # Build layers
    layers = []
    in_dim = state_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_fn())
        in_dim = hidden_dim

    # Output layer
    layers.append(nn.Linear(in_dim, action_dim))

    # Optional output activation
    if output_activation == "tanh":
        layers.append(nn.Tanh())
    elif output_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif output_activation is not None:
        raise ValueError(f"Unknown output activation: {output_activation}")

    return nn.Sequential(*layers)
