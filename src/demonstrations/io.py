"""I/O utilities for saving and loading demonstration data.

This module provides functions for serializing demonstration trajectories
and buffers to disk and loading them back.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from observations.extractors import FeatureExtractor
from demonstrations.buffer import DemonstrationBuffer


def save_demonstrations(
    trajectories: List[List[Dict[str, Any]]],
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    parameters: Optional[List[Dict[str, float]]] = None,
    fitness_info: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save demonstration trajectories to disk.

    Saves trajectories as a pickle file containing:
        - trajectories: List of trajectory state lists
        - metadata: Optional metadata dict (e.g., generation parameters)
        - parameters: Optional per-trajectory parameters
        - fitness_info: Optional fitness evaluation results per trajectory

    Args:
        trajectories: List of trajectories, each a list of state dicts
        path: Output file path (should end with .pkl)
        metadata: Optional metadata dictionary
        parameters: Optional list of parameter dicts (one per trajectory)
        fitness_info: Optional list of fitness info dicts (one per trajectory)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for serialization
    serializable_trajectories = []
    for traj in trajectories:
        serializable_traj = []
        for state in traj:
            serializable_state = {}
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    serializable_state[key] = value.tolist()
                else:
                    serializable_state[key] = value
            serializable_traj.append(serializable_state)
        serializable_trajectories.append(serializable_traj)

    # Convert fitness_info numpy arrays to lists
    serializable_fitness_info = None
    if fitness_info is not None:
        serializable_fitness_info = []
        for info in fitness_info:
            serializable_info = {}
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    serializable_info[key] = value.tolist()
                else:
                    serializable_info[key] = value
            serializable_fitness_info.append(serializable_info)

    data = {
        "trajectories": serializable_trajectories,
        "metadata": metadata or {},
        "parameters": parameters,
        "fitness_info": serializable_fitness_info,
        "num_trajectories": len(trajectories),
        "total_states": sum(len(t) for t in trajectories),
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_demonstrations(
    path: Union[str, Path],
    convert_to_numpy: bool = True,
) -> Dict[str, Any]:
    """Load demonstration trajectories from disk.

    Args:
        path: Path to pickle file
        convert_to_numpy: If True, convert lists back to numpy arrays

    Returns:
        Dictionary containing:
            - trajectories: List of trajectory state lists
            - metadata: Metadata dictionary
            - parameters: Per-trajectory parameters (if available)
            - fitness_info: Fitness evaluation results (if available)
            - num_trajectories: Number of trajectories
            - total_states: Total number of states
    """
    path = Path(path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    if convert_to_numpy:
        # Convert lists back to numpy arrays
        trajectories = []
        for traj in data["trajectories"]:
            numpy_traj = []
            for state in traj:
                numpy_state = {}
                for key, value in state.items():
                    if isinstance(value, list):
                        numpy_state[key] = np.array(value)
                    else:
                        numpy_state[key] = value
                numpy_traj.append(numpy_state)
            trajectories.append(numpy_traj)
        data["trajectories"] = trajectories

        # Convert fitness_info arrays back to numpy
        if data.get("fitness_info"):
            fitness_info = []
            for info in data["fitness_info"]:
                numpy_info = {}
                for key, value in info.items():
                    if isinstance(value, list):
                        numpy_info[key] = np.array(value)
                    else:
                        numpy_info[key] = value
                fitness_info.append(numpy_info)
            data["fitness_info"] = fitness_info

    return data


def save_buffer(
    buffer: DemonstrationBuffer,
    path: Union[str, Path],
    include_extractor_config: bool = True,
) -> None:
    """Save a demonstration buffer to disk.

    Args:
        buffer: DemonstrationBuffer to save
        path: Output file path
        include_extractor_config: Include extractor class info for reconstruction
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "features": [f.tolist() for f in buffer.features],
        "trajectory_ids": buffer.trajectory_ids,
        "timestamps": buffer.timestamps,
        "feature_dim": buffer.feature_dim,
        "num_samples": buffer.num_samples,
        "num_trajectories": buffer.num_trajectories,
    }

    if include_extractor_config:
        data["extractor_class"] = type(buffer.feature_extractor).__name__

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_buffer(
    path: Union[str, Path],
    feature_extractor: FeatureExtractor,
    build_index: bool = True,
) -> DemonstrationBuffer:
    """Load a demonstration buffer from disk.

    Args:
        path: Path to buffer file
        feature_extractor: Feature extractor to use (must match saved buffer)
        build_index: Whether to build KDTree index after loading

    Returns:
        Loaded DemonstrationBuffer
    """
    path = Path(path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    buffer = DemonstrationBuffer(feature_extractor)
    buffer.features = [np.array(f) for f in data["features"]]
    buffer.trajectory_ids = data["trajectory_ids"]
    buffer.timestamps = data["timestamps"]

    if build_index:
        buffer.build_index()

    return buffer


def export_to_json(
    trajectories: List[List[Dict[str, Any]]],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """Export demonstrations to JSON format (human-readable).

    Note: JSON files will be larger than pickle but are human-readable
    and can be loaded by other languages.

    Args:
        trajectories: List of trajectories
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists
    serializable = []
    for traj in trajectories:
        serializable_traj = []
        for state in traj:
            serializable_state = {}
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    serializable_state[key] = value.tolist()
                elif isinstance(value, (np.floating, np.integer)):
                    serializable_state[key] = float(value)
                else:
                    serializable_state[key] = value
            serializable_traj.append(serializable_state)
        serializable.append(serializable_traj)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=indent)


def load_from_json(path: Union[str, Path]) -> List[List[Dict[str, Any]]]:
    """Load demonstrations from JSON format.

    Args:
        path: Path to JSON file

    Returns:
        List of trajectories with numpy arrays
    """
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    # Convert lists to numpy arrays
    trajectories = []
    for traj in data:
        numpy_traj = []
        for state in traj:
            numpy_state = {}
            for key, value in state.items():
                if isinstance(value, list):
                    numpy_state[key] = np.array(value)
                else:
                    numpy_state[key] = value
            numpy_traj.append(numpy_state)
        trajectories.append(numpy_traj)

    return trajectories


def populate_buffer_from_trajectories(
    buffer: DemonstrationBuffer,
    trajectories: List[List[Dict[str, Any]]],
    build_index: bool = True,
) -> int:
    """Populate a demonstration buffer from trajectory data.

    Args:
        buffer: DemonstrationBuffer to populate
        trajectories: List of trajectories
        build_index: Whether to build KDTree index after population

    Returns:
        Total number of states added
    """
    total_added = 0

    for traj_id, trajectory in enumerate(trajectories):
        # Generate timestamps based on trajectory index
        timestamps = [float(i) for i in range(len(trajectory))]
        added = buffer.add_trajectory(
            trajectory,
            trajectory_id=traj_id,
            timestamps=timestamps,
        )
        total_added += added

    if build_index:
        buffer.build_index()

    return total_added
