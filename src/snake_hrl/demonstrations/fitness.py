"""Fitness evaluation functions for CPG locomotion trajectories.

This module provides functions for evaluating trajectory fitness based on
displacement metrics and filtering for successful locomotion patterns with
directional diversity.

The key insight is that different CPG parameters naturally produce movement
in different directions. By evaluating displacement vectors, we can:
1. Filter for efficient locomotion (displacement magnitude)
2. Ensure directional diversity (coverage of multiple movement directions)
3. Bootstrap RL training for approach tasks

Example:
    >>> from snake_hrl.demonstrations.fitness import (
    ...     evaluate_trajectory,
    ...     filter_successful_trajectories,
    ... )
    >>>
    >>> # Evaluate a single trajectory
    >>> fitness_info = evaluate_trajectory(trajectory)
    >>> print(f"Displacement: {fitness_info['displacement_magnitude']:.3f}m")
    >>>
    >>> # Filter trajectories by displacement and direction diversity
    >>> filtered_trajs, filtered_params, filtered_info = filter_successful_trajectories(
    ...     trajectories, parameters,
    ...     min_displacement=0.1,
    ...     ensure_direction_diversity=True,
    ... )
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_displacement_vector(trajectory: List[Dict[str, Any]]) -> np.ndarray:
    """Compute displacement vector from initial to final head position.

    Args:
        trajectory: List of state dictionaries with 'positions' key.
                   positions[0] is the head position.

    Returns:
        3D displacement vector (final_head - initial_head)
    """
    initial_head = np.array(trajectory[0]["positions"][0])
    final_head = np.array(trajectory[-1]["positions"][0])
    return final_head - initial_head


def compute_displacement_magnitude(trajectory: List[Dict[str, Any]]) -> float:
    """Compute displacement magnitude (efficiency metric).

    Args:
        trajectory: List of state dictionaries

    Returns:
        Euclidean distance from initial to final head position (meters)
    """
    return float(np.linalg.norm(compute_displacement_vector(trajectory)))


def compute_displacement_direction(trajectory: List[Dict[str, Any]]) -> np.ndarray:
    """Compute normalized displacement direction (2D, ignoring z).

    Args:
        trajectory: List of state dictionaries

    Returns:
        2D unit vector in the direction of displacement (xy plane)
    """
    vec = compute_displacement_vector(trajectory)[:2]  # xy only
    mag = np.linalg.norm(vec)
    if mag > 1e-6:
        return vec / mag
    else:
        return np.array([1.0, 0.0])  # Default direction if no movement


def evaluate_trajectory(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate trajectory fitness metrics.

    Computes comprehensive fitness metrics for a trajectory including
    displacement vector, magnitude, direction, and average speed.

    Args:
        trajectory: List of state dictionaries. Each state must have:
                   - 'positions': list of segment positions, [0] is head
                   - 'time': simulation timestamp

    Returns:
        Dictionary with fitness metrics:
        - displacement_vector: 3D vector (meters)
        - displacement_magnitude: scalar (meters)
        - displacement_direction: 2D unit vector (xy plane)
        - duration: trajectory duration (seconds)
        - avg_speed: average speed (meters/second)
    """
    displacement_vec = compute_displacement_vector(trajectory)
    magnitude = float(np.linalg.norm(displacement_vec))
    duration = trajectory[-1]["time"] - trajectory[0]["time"]

    # Compute 2D direction (xy plane)
    xy_mag = np.linalg.norm(displacement_vec[:2])
    if xy_mag > 1e-6:
        direction = displacement_vec[:2] / xy_mag
    else:
        direction = np.array([1.0, 0.0])

    return {
        "displacement_vector": displacement_vec,      # 3D vector
        "displacement_magnitude": magnitude,          # scalar (meters)
        "displacement_direction": direction,          # 2D unit vector
        "duration": duration,
        "avg_speed": magnitude / duration if duration > 0 else 0.0,
    }


def compute_direction_bin(direction: np.ndarray, num_bins: int = 8) -> int:
    """Compute direction bin index from direction vector.

    Bins directions into equal angular sectors:
    - 8 bins: E, NE, N, NW, W, SW, S, SE (counterclockwise from +x)

    Args:
        direction: 2D unit vector (or will be normalized)
        num_bins: Number of direction bins (default: 8)

    Returns:
        Bin index (0 to num_bins-1)
    """
    # Compute angle in radians (-pi to pi), counterclockwise from +x axis
    angle = np.arctan2(direction[1], direction[0])

    # Shift to (0 to 2*pi)
    angle_positive = (angle + 2 * np.pi) % (2 * np.pi)

    bin_size = 2 * np.pi / num_bins

    # Shift by half bin so that bin 0 is centered on +x axis (East)
    # E.g., for 8 bins: bin 0 = -22.5 to 22.5 degrees = East
    angle_shifted = (angle_positive + bin_size / 2) % (2 * np.pi)

    return int(angle_shifted / bin_size)


def get_direction_bin_name(bin_index: int, num_bins: int = 8) -> str:
    """Get human-readable name for a direction bin.

    Args:
        bin_index: Bin index (0 to num_bins-1)
        num_bins: Number of direction bins

    Returns:
        Direction name (e.g., "E", "NE", "N", etc.)
    """
    if num_bins == 8:
        names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        return names[bin_index % 8]
    else:
        return f"bin_{bin_index}"


def filter_successful_trajectories(
    trajectories: List[List[Dict[str, Any]]],
    parameters: List[Dict[str, float]],
    min_displacement: float = 0.1,
    ensure_direction_diversity: bool = True,
    num_direction_bins: int = 8,
    top_k_per_bin: int = 1,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, float]], List[Dict[str, Any]]]:
    """Filter trajectories by displacement threshold with direction diversity.

    This function:
    1. Evaluates all trajectories for fitness metrics
    2. Filters by minimum displacement threshold
    3. Optionally ensures coverage of multiple movement directions
       by keeping the best trajectory(ies) per direction bin

    Args:
        trajectories: List of trajectories (each is a list of state dicts)
        parameters: List of CPG parameters for each trajectory
        min_displacement: Minimum displacement to consider "successful" (meters)
        ensure_direction_diversity: If True, keep best trajectories per direction bin
        num_direction_bins: Number of direction bins (8 = N/NE/E/SE/S/SW/W/NW)
        top_k_per_bin: Number of best trajectories to keep per direction bin

    Returns:
        Tuple of (filtered_trajectories, filtered_parameters, fitness_info_list)
    """
    # 1. Evaluate all trajectories
    fitness_info_list = [evaluate_trajectory(t) for t in trajectories]

    # Add direction bin to each fitness info
    for info in fitness_info_list:
        info["direction_bin"] = compute_direction_bin(
            info["displacement_direction"], num_direction_bins
        )
        info["direction_bin_name"] = get_direction_bin_name(
            info["direction_bin"], num_direction_bins
        )

    # 2. Filter by minimum displacement
    passing_indices = [
        i for i, info in enumerate(fitness_info_list)
        if info["displacement_magnitude"] >= min_displacement
    ]

    if not passing_indices:
        # No trajectories pass threshold
        return [], [], []

    # 3. Optionally ensure direction diversity
    if ensure_direction_diversity:
        # Group by direction bin
        bins: Dict[int, List[int]] = {i: [] for i in range(num_direction_bins)}
        for idx in passing_indices:
            direction_bin = fitness_info_list[idx]["direction_bin"]
            bins[direction_bin].append(idx)

        # Keep top_k best per bin (by displacement magnitude)
        selected_indices = []
        for bin_idx in range(num_direction_bins):
            bin_indices = bins[bin_idx]
            if not bin_indices:
                continue

            # Sort by displacement magnitude (descending)
            bin_indices_sorted = sorted(
                bin_indices,
                key=lambda i: fitness_info_list[i]["displacement_magnitude"],
                reverse=True,
            )

            # Keep top k
            selected_indices.extend(bin_indices_sorted[:top_k_per_bin])

        # Sort selected indices by original order
        selected_indices = sorted(selected_indices)
    else:
        # Keep all passing trajectories
        selected_indices = passing_indices

    # 4. Build filtered results
    filtered_trajectories = [trajectories[i] for i in selected_indices]
    filtered_parameters = [parameters[i] for i in selected_indices]
    filtered_fitness_info = [fitness_info_list[i] for i in selected_indices]

    return filtered_trajectories, filtered_parameters, filtered_fitness_info


def compute_direction_coverage(
    fitness_info_list: List[Dict[str, Any]],
    num_bins: int = 8,
) -> Dict[str, Any]:
    """Compute statistics about direction coverage.

    Args:
        fitness_info_list: List of fitness info dictionaries
        num_bins: Number of direction bins

    Returns:
        Dictionary with:
        - bins_covered: number of bins with at least one trajectory
        - total_bins: total number of bins
        - coverage_ratio: bins_covered / total_bins
        - bin_counts: dict mapping bin name to count
        - best_per_bin: dict mapping bin name to best displacement in that bin
    """
    if not fitness_info_list:
        return {
            "bins_covered": 0,
            "total_bins": num_bins,
            "coverage_ratio": 0.0,
            "bin_counts": {},
            "best_per_bin": {},
        }

    bin_counts: Dict[str, int] = {}
    best_per_bin: Dict[str, float] = {}

    for info in fitness_info_list:
        bin_name = info.get("direction_bin_name", get_direction_bin_name(
            compute_direction_bin(info["displacement_direction"], num_bins),
            num_bins,
        ))

        # Count
        bin_counts[bin_name] = bin_counts.get(bin_name, 0) + 1

        # Track best
        mag = info["displacement_magnitude"]
        if bin_name not in best_per_bin or mag > best_per_bin[bin_name]:
            best_per_bin[bin_name] = mag

    return {
        "bins_covered": len(bin_counts),
        "total_bins": num_bins,
        "coverage_ratio": len(bin_counts) / num_bins,
        "bin_counts": bin_counts,
        "best_per_bin": best_per_bin,
    }


def get_best_parameters_per_direction(
    parameters: List[Dict[str, float]],
    fitness_info_list: List[Dict[str, Any]],
    num_bins: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """Get the best parameters for each direction bin.

    Args:
        parameters: List of CPG parameters for each trajectory
        fitness_info_list: List of fitness info for each trajectory
        num_bins: Number of direction bins

    Returns:
        Dictionary mapping bin name to:
        - parameters: best CPG parameters for that direction
        - displacement: displacement magnitude achieved
    """
    best_per_bin: Dict[str, Dict[str, Any]] = {}

    for params, info in zip(parameters, fitness_info_list):
        bin_name = info.get("direction_bin_name", get_direction_bin_name(
            info["direction_bin"], num_bins
        ))
        mag = info["displacement_magnitude"]

        if bin_name not in best_per_bin or mag > best_per_bin[bin_name]["displacement"]:
            best_per_bin[bin_name] = {
                "parameters": params,
                "displacement": mag,
            }

    return best_per_bin
