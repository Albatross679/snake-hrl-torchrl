"""Base TorchRL environment for snake simulation."""

from typing import Optional, Dict, Any
import torch
import numpy as np

from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)
from tensordict import TensorDict, TensorDictBase

from snake_hrl.configs.env import EnvConfig, PhysicsConfig, StateRepresentation
from snake_hrl.physics import create_snake_robot


class BaseSnakeEnv(EnvBase):
    """Base TorchRL environment for snake robot simulation.

    This environment wraps the physics simulation and provides
    a standard TorchRL interface with TensorDict observations and actions.
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        device: str = "cpu",
        batch_size: Optional[torch.Size] = None,
    ):
        """Initialize base snake environment.

        Args:
            config: Environment configuration
            device: Device for tensors
            batch_size: Batch size for vectorized environments
        """
        self.config = config or EnvConfig()
        self._device = device

        # Initialize parent class
        super().__init__(device=device, batch_size=batch_size or torch.Size([]))

        # Create physics simulation
        self.sim = create_snake_robot(self.config.physics)

        # Episode tracking
        self._step_count = 0
        self._episode_reward = 0.0

        # Build specs
        self._make_spec()

        # Cache for observations
        self._current_state: Optional[Dict[str, Any]] = None

    def _make_spec(self) -> None:
        """Define observation, action, and reward specs."""
        num_segments = self.config.physics.num_segments
        num_nodes = num_segments + 1
        action_dim = num_segments - 1  # Curvature control at joints

        # Observation spec
        obs_shape = (self.config.obs_dim,)
        num_joints = num_segments - 1  # Internal joints where curvature is defined

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=obs_shape,
                dtype=torch.float32,
                device=self._device,
            ),
            # Additional observation components for convenience
            snake_positions=UnboundedContinuousTensorSpec(
                shape=(num_nodes, 3),
                dtype=torch.float32,
                device=self._device,
            ),
            snake_velocities=UnboundedContinuousTensorSpec(
                shape=(num_nodes, 3),
                dtype=torch.float32,
                device=self._device,
            ),
            snake_curvatures=UnboundedContinuousTensorSpec(
                shape=(num_joints,),
                dtype=torch.float32,
                device=self._device,
            ),
            prey_position=UnboundedContinuousTensorSpec(
                shape=(3,),
                dtype=torch.float32,
                device=self._device,
            ),
            prey_distance=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self._device,
            ),
            shape=self.batch_size,
            device=self._device,
        )

        # Action spec (continuous curvature control)
        self.action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=torch.float32,
            device=self._device,
        )

        # Reward spec
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
            device=self._device,
        )

        # Done spec
        self.done_spec = CompositeSpec(
            done=DiscreteTensorSpec(
                n=2,
                shape=(1,),
                dtype=torch.bool,
                device=self._device,
            ),
            terminated=DiscreteTensorSpec(
                n=2,
                shape=(1,),
                dtype=torch.bool,
                device=self._device,
            ),
            truncated=DiscreteTensorSpec(
                n=2,
                shape=(1,),
                dtype=torch.bool,
                device=self._device,
            ),
            shape=self.batch_size,
            device=self._device,
        )

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        """Reset environment to initial state.

        Args:
            tensordict: Optional tensordict with reset parameters

        Returns:
            TensorDict with initial observation
        """
        # Randomize initial positions if configured
        snake_pos = None
        prey_pos = None

        if self.config.randomize_initial_state:
            # Small random offset for snake start
            snake_pos = np.array([0.0, 0.0, self.config.physics.snake_radius])
            snake_pos[:2] += np.random.uniform(-0.1, 0.1, 2)

        if self.config.randomize_prey_position:
            # Random prey position in front of snake
            distance = np.random.uniform(*self.config.prey_position_range)
            angle = np.random.uniform(-np.pi / 4, np.pi / 4)
            prey_pos = np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                self.config.physics.prey_length / 2,
            ])

        # Reset simulation
        self.sim.reset(snake_pos, prey_pos)

        # Reset tracking
        self._step_count = 0
        self._episode_reward = 0.0

        # Get initial state
        self._current_state = self.sim.get_state()

        return self._make_tensordict()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step.

        Args:
            tensordict: TensorDict with action

        Returns:
            TensorDict with next observation, reward, done flags
        """
        # Extract action
        action = tensordict["action"].cpu().numpy()

        # Scale action to curvature range
        curvatures = action * self.config.action_scale * 5.0  # Scale to reasonable curvature range

        # Store previous state for reward computation
        prev_state = self._current_state

        # Apply action and step simulation
        self.sim.set_curvature_control(curvatures)

        for _ in range(self.config.action_repeat):
            self._current_state = self.sim.step()

        self._step_count += 1

        # Compute reward
        reward = self._compute_reward(prev_state, self._current_state, action)
        self._episode_reward += reward

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._step_count >= self.config.max_episode_steps
        done = terminated or truncated

        return self._make_tensordict(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            done=done,
        )

    def _make_tensordict(
        self,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        done: bool = False,
    ) -> TensorDictBase:
        """Create TensorDict from current state.

        Args:
            reward: Reward for this step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            done: Whether episode is done

        Returns:
            TensorDict with observations and metadata
        """
        state = self._current_state

        # Build flat observation
        obs = self.sim.get_observation(
            include_curvatures=self.config.include_curvatures,
            state_representation=self.config.state_representation,
        )

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self._device),
                "snake_positions": torch.tensor(
                    state["positions"], dtype=torch.float32, device=self._device
                ),
                "snake_velocities": torch.tensor(
                    state["velocities"], dtype=torch.float32, device=self._device
                ),
                "snake_curvatures": torch.tensor(
                    state["curvatures"], dtype=torch.float32, device=self._device
                ),
                "prey_position": torch.tensor(
                    state["prey_position"], dtype=torch.float32, device=self._device
                ),
                "prey_distance": torch.tensor(
                    [state["prey_distance"]], dtype=torch.float32, device=self._device
                ),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self._device),
                "done": torch.tensor([done], dtype=torch.bool, device=self._device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self._device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self._device),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _compute_reward(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        action: np.ndarray,
    ) -> float:
        """Compute reward for transition.

        Override in subclasses for task-specific rewards.

        Args:
            prev_state: State before action
            curr_state: State after action
            action: Action taken

        Returns:
            Reward value
        """
        # Base reward: negative for time passing (encourage efficiency)
        reward = -0.001

        # Energy penalty (discourage large actions)
        energy_penalty = 0.001 * np.sum(action**2)
        reward -= energy_penalty

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate.

        Override in subclasses for task-specific termination.

        Returns:
            Whether episode should terminate
        """
        # Base environment doesn't terminate early
        return False

    def _set_seed(self, seed: Optional[int]) -> None:
        """Set random seed.

        Args:
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    @property
    def state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return self._current_state

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count

    @property
    def episode_reward(self) -> float:
        """Get cumulative episode reward."""
        return self._episode_reward
