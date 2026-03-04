"""Potential-Based Reward Shaping (PBRS) for snake RL.

Implements reward shaping functions that preserve optimal policies
by using the difference in potential functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class PotentialFunction(ABC):
    """Abstract base class for potential functions."""

    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> float:
        """Compute potential for given state.

        Args:
            state: Dictionary containing simulation state

        Returns:
            Potential value (higher is better/closer to goal)
        """
        pass


class ApproachPotential(PotentialFunction):
    """Potential function for approaching prey.

    Higher potential when snake is closer to prey.
    """

    def __init__(
        self,
        max_distance: float = 2.0,
        distance_scale: float = 1.0,
        velocity_bonus_scale: float = 0.1,
    ):
        """Initialize approach potential.

        Args:
            max_distance: Maximum relevant distance (for normalization)
            distance_scale: Scale factor for distance component
            velocity_bonus_scale: Bonus for velocity towards prey
        """
        self.max_distance = max_distance
        self.distance_scale = distance_scale
        self.velocity_bonus_scale = velocity_bonus_scale

    def __call__(self, state: Dict[str, Any]) -> float:
        """Compute approach potential.

        Args:
            state: Must contain 'prey_distance', 'positions', 'velocities',
                   and 'prey_position'

        Returns:
            Potential value
        """
        # Distance component (negative distance, normalized)
        distance = state["prey_distance"]
        distance_potential = -self.distance_scale * (distance / self.max_distance)

        # Velocity component (bonus for moving towards prey)
        velocity_potential = 0.0
        if self.velocity_bonus_scale > 0:
            head_pos = state["positions"][0]
            head_vel = state["velocities"][0]
            prey_pos = state["prey_position"]

            # Direction to prey
            to_prey = prey_pos - head_pos
            to_prey_norm = np.linalg.norm(to_prey)

            if to_prey_norm > 1e-6:
                to_prey = to_prey / to_prey_norm

                # Velocity component towards prey
                vel_towards_prey = np.dot(head_vel, to_prey)
                velocity_potential = self.velocity_bonus_scale * vel_towards_prey

        return distance_potential + velocity_potential


class CoilPotential(PotentialFunction):
    """Potential function for coiling around prey.

    Higher potential when snake has more contact and wrap angle.
    """

    def __init__(
        self,
        contact_weight: float = 1.0,
        wrap_weight: float = 2.0,
        constriction_weight: float = 1.0,
        target_wraps: float = 2.0,
    ):
        """Initialize coil potential.

        Args:
            contact_weight: Weight for contact fraction component
            wrap_weight: Weight for wrap angle component
            constriction_weight: Weight for constriction (being close to prey)
            target_wraps: Target number of wraps for full potential
        """
        self.contact_weight = contact_weight
        self.wrap_weight = wrap_weight
        self.constriction_weight = constriction_weight
        self.target_wraps = target_wraps

    def __call__(self, state: Dict[str, Any]) -> float:
        """Compute coil potential.

        Args:
            state: Must contain 'contact_fraction', 'wrap_count'

        Returns:
            Potential value
        """
        # Contact component (0 to 1)
        contact_fraction = state.get("contact_fraction", 0.0)
        contact_potential = self.contact_weight * contact_fraction

        # Wrap component (normalized by target wraps)
        wrap_count = abs(state.get("wrap_count", 0.0))
        wrap_potential = self.wrap_weight * min(wrap_count / self.target_wraps, 1.0)

        # Constriction component (average distance of contact points to prey)
        constriction_potential = 0.0
        if self.constriction_weight > 0 and "positions" in state:
            contact_mask = state.get("contact_mask", np.zeros(len(state["positions"]), dtype=bool))
            if np.any(contact_mask):
                prey_pos = state["prey_position"]
                contact_positions = state["positions"][contact_mask]
                distances = np.linalg.norm(contact_positions - prey_pos, axis=1)
                avg_distance = np.mean(distances)
                # Inverse distance (closer is better)
                constriction_potential = self.constriction_weight / (1.0 + avg_distance)

        return contact_potential + wrap_potential + constriction_potential


class PotentialBasedRewardShaping:
    """Potential-Based Reward Shaping (PBRS).

    Computes shaped reward as: R' = R + gamma * Phi(s') - Phi(s)

    This preserves optimal policies while providing denser rewards.
    """

    def __init__(
        self,
        potential_fn: PotentialFunction,
        gamma: float = 0.99,
        scale: float = 1.0,
    ):
        """Initialize PBRS.

        Args:
            potential_fn: Potential function to use
            gamma: Discount factor
            scale: Scale factor for shaping reward
        """
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.scale = scale
        self._prev_potential: Optional[float] = None

    def reset(self) -> None:
        """Reset shaping state for new episode."""
        self._prev_potential = None

    def compute_shaping_reward(
        self,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        done: bool = False,
    ) -> float:
        """Compute shaping reward component.

        Args:
            state: Current state
            next_state: Next state after action
            done: Whether episode is done

        Returns:
            Shaping reward: gamma * Phi(s') - Phi(s)
        """
        current_potential = self.potential_fn(state)
        next_potential = self.potential_fn(next_state)

        if done:
            # Terminal state has zero potential
            shaping = -current_potential
        else:
            shaping = self.gamma * next_potential - current_potential

        return self.scale * shaping

    def __call__(
        self,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        base_reward: float,
        done: bool = False,
    ) -> float:
        """Compute total shaped reward.

        Args:
            state: Current state
            next_state: Next state
            base_reward: Original environment reward
            done: Whether episode is done

        Returns:
            Shaped reward: base_reward + shaping_reward
        """
        shaping = self.compute_shaping_reward(state, next_state, done)
        return base_reward + shaping


class CompositeRewardShaping:
    """Combine multiple reward shaping functions."""

    def __init__(self, gamma: float = 0.99):
        """Initialize composite shaping.

        Args:
            gamma: Discount factor for PBRS
        """
        self.gamma = gamma
        self.shapers: Dict[str, PotentialBasedRewardShaping] = {}

    def add_shaper(
        self,
        name: str,
        potential_fn: PotentialFunction,
        weight: float = 1.0,
    ) -> None:
        """Add a shaping component.

        Args:
            name: Name for this component
            potential_fn: Potential function
            weight: Weight for this component
        """
        self.shapers[name] = PotentialBasedRewardShaping(
            potential_fn, self.gamma, scale=weight
        )

    def reset(self) -> None:
        """Reset all shapers for new episode."""
        for shaper in self.shapers.values():
            shaper.reset()

    def compute_shaping_reward(
        self,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        done: bool = False,
    ) -> Dict[str, float]:
        """Compute shaping rewards from all components.

        Returns:
            Dictionary with individual and total shaping rewards
        """
        rewards = {}
        total = 0.0

        for name, shaper in self.shapers.items():
            reward = shaper.compute_shaping_reward(state, next_state, done)
            rewards[name] = reward
            total += reward

        rewards["total"] = total
        return rewards

    def __call__(
        self,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        base_reward: float,
        done: bool = False,
    ) -> float:
        """Compute total shaped reward.

        Args:
            state: Current state
            next_state: Next state
            base_reward: Original reward
            done: Whether episode is done

        Returns:
            Total shaped reward
        """
        shaping_rewards = self.compute_shaping_reward(state, next_state, done)
        return base_reward + shaping_rewards["total"]


# Convenience functions for creating standard shapers


def create_approach_shaper(
    gamma: float = 0.99,
    distance_scale: float = 1.0,
    velocity_bonus: float = 0.1,
) -> PotentialBasedRewardShaping:
    """Create reward shaper for approach task."""
    potential = ApproachPotential(
        distance_scale=distance_scale,
        velocity_bonus_scale=velocity_bonus,
    )
    return PotentialBasedRewardShaping(potential, gamma)


def create_coil_shaper(
    gamma: float = 0.99,
    contact_weight: float = 1.0,
    wrap_weight: float = 2.0,
    constriction_weight: float = 1.0,
) -> PotentialBasedRewardShaping:
    """Create reward shaper for coil task."""
    potential = CoilPotential(
        contact_weight=contact_weight,
        wrap_weight=wrap_weight,
        constriction_weight=constriction_weight,
    )
    return PotentialBasedRewardShaping(potential, gamma)


def create_full_task_shaper(gamma: float = 0.99) -> CompositeRewardShaping:
    """Create composite shaper for full snake predation task."""
    shaper = CompositeRewardShaping(gamma)
    shaper.add_shaper("approach", ApproachPotential(), weight=1.0)
    shaper.add_shaper("coil", CoilPotential(), weight=1.5)
    return shaper


def create_gait_shaper(
    demo_buffer: "DemonstrationBuffer",
    gamma: float = 0.99,
    sigma: float = 1.0,
    weight: float = 0.5,
) -> PotentialBasedRewardShaping:
    """Create PBRS with gait potential for demonstration-guided shaping.

    This creates a reward shaper that encourages the agent to match
    demonstrated locomotion patterns using a Gaussian potential in
    feature space.

    Args:
        demo_buffer: DemonstrationBuffer containing demonstration features
        gamma: Discount factor for PBRS
        sigma: Gaussian kernel width (controls sharpness of potential)
        weight: Weight/scale for the gait shaping reward

    Returns:
        PotentialBasedRewardShaping instance with GaitPotential

    Example:
        >>> from observations import CompositeFeatureExtractor, CurvatureModeExtractor
        >>> from behavioral_cloning import DemonstrationBuffer, SerpenoidGenerator
        >>> from rewards.shaping import create_gait_shaper
        >>>
        >>> # Create and populate buffer
        >>> extractor = CompositeFeatureExtractor([CurvatureModeExtractor()])
        >>> buffer = DemonstrationBuffer(extractor)
        >>> # ... add demonstrations ...
        >>>
        >>> # Create shaper
        >>> gait_shaper = create_gait_shaper(buffer, sigma=1.0, weight=0.5)
        >>> reward = gait_shaper.compute_shaping_reward(prev_state, curr_state)
    """
    from .gait_potential import GaitPotential

    potential = GaitPotential(demo_buffer, sigma=sigma, scale=1.0)
    return PotentialBasedRewardShaping(potential, gamma, scale=weight)


def create_curriculum_gait_shaper(
    demo_buffer: "DemonstrationBuffer",
    gamma: float = 0.99,
    sigma_init: float = 2.0,
    sigma_final: float = 0.5,
    weight: float = 0.5,
    schedule: str = "linear",
) -> PotentialBasedRewardShaping:
    """Create PBRS with curriculum gait potential.

    The sigma parameter anneals from sigma_init to sigma_final over training,
    providing curriculum learning where the agent first receives broad guidance
    and then is shaped toward precise demonstration matching.

    Args:
        demo_buffer: DemonstrationBuffer containing demonstration features
        gamma: Discount factor for PBRS
        sigma_init: Initial sigma (broader, more forgiving)
        sigma_final: Final sigma (tighter, more precise)
        weight: Weight/scale for the gait shaping reward
        schedule: Annealing schedule - "linear", "cosine", or "exponential"

    Returns:
        PotentialBasedRewardShaping instance with CurriculumGaitPotential

    Note:
        Call shaper.potential_fn.set_progress(progress) to update sigma
        during training, where progress is in [0, 1].
    """
    from .gait_potential import CurriculumGaitPotential

    potential = CurriculumGaitPotential(
        demo_buffer,
        sigma_init=sigma_init,
        sigma_final=sigma_final,
        scale=1.0,
        schedule=schedule,
    )
    return PotentialBasedRewardShaping(potential, gamma, scale=weight)
