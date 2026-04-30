"""Dataset classes for surrogate model training (DisMech backend).

Provides single-step data loading from .pt files saved by collect_data.py.

Classes:
    FlatStepDataset: Flat-format dataset for DisMech single-step transitions.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from aprx_model_dismech.state import (
    POS_X,
    POS_Y,
    VEL_X,
    VEL_Y,
    YAW,
    OMEGA_Z,
    encode_per_element_phase_batch,
)


def compute_density_weights(
    states: torch.Tensor,
    n_bins: int = 20,
    clip_max: float = 10.0,
) -> torch.Tensor:
    """Compute inverse-density sample weights from state summary features.

    Projects 124D states to 4 summary features (CoM_x, CoM_y, velocity
    magnitude, mean |angular velocity|), bins them into a joint histogram,
    and returns weights proportional to 1/bin_count so rare states are upweighted.

    Args:
        states: (N, 124) state tensor.
        n_bins: Number of histogram bins per feature dimension.
        clip_max: Maximum weight (prevents extreme outlier domination).

    Returns:
        (N,) weight tensor, normalized to mean=1.
    """
    # Extract summary features from 124D state
    com_x = states[:, POS_X].mean(dim=1)
    com_y = states[:, POS_Y].mean(dim=1)
    vel_mag = (states[:, VEL_X] ** 2 + states[:, VEL_Y] ** 2).mean(dim=1).sqrt()
    mean_omega = states[:, OMEGA_Z].abs().mean(dim=1)

    features = torch.stack([com_x, com_y, vel_mag, mean_omega], dim=1)  # (N, 4)

    # Compute joint bin index from 4 independent histograms
    n = features.shape[0]
    bin_indices = torch.zeros(n, dtype=torch.long)

    for i in range(features.shape[1]):
        col = features[:, i]
        lo, hi = col.min().item(), col.max().item()
        if hi - lo < 1e-8:
            continue
        bins = ((col - lo) / (hi - lo) * (n_bins - 1)).clamp(0, n_bins - 1).long()
        bin_indices = bin_indices * n_bins + bins

    # Count samples per joint bin
    unique_bins, inverse, counts = torch.unique(
        bin_indices, return_inverse=True, return_counts=True
    )

    # Weight = 1 / count -> rare bins get high weight
    weights = 1.0 / counts[inverse].float()

    # Normalize to mean=1, clip to prevent outliers
    weights = weights / weights.mean()
    weights = weights.clamp(max=clip_max)

    return weights


class FlatStepDataset(Dataset):
    """Dataset for DisMech flat-format .pt batches.

    Loads batch files produced by collect_data.py with flat_output=True.
    Each file contains flat (state, action, next_state) transitions -- one row
    per RL step.

    Format (per batch file):
        states:      (N, 124) float32 -- rod state before RL step
        next_states: (N, 124) float32 -- rod state after RL step
        actions:     (N, 5)   float32
        t_start:     (N,)     float32 -- serpenoid time at step start
        episode_ids: (N,)     int64
        step_ids:    (N,)     int64    (optional)
        forces: {              (optional)
            external_forces:  (N, 3, 21)
            internal_forces:  (N, 3, 21)
            external_torques: (N, 3, 20)
            internal_torques: (N, 3, 20)
        }

    __getitem__ returns:
        state:      (124,)
        action:     (5,)
        t_start:    scalar
        next_state: (124,)
        delta:      (124,)  = next_state - state
        forces:     dict with per-item (3,21)/(3,20) tensors, or None
    """

    _FORCE_KEYS = ("external_forces", "internal_forces", "external_torques", "internal_torques")

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        """Load and concatenate all batch_*.pt files from data_dir.

        Args:
            data_dir: Directory containing batch_*.pt files.
            split: "train" or "val".
            val_fraction: Fraction of episodes held out for validation.
                          Set to 0.0 to use all data for training.
            seed: Random seed for reproducible train/val split.
        """
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("batch_*.pt"))
        if not files:
            raise FileNotFoundError(f"No batch_*.pt files in {data_dir}")

        all_states: List[torch.Tensor] = []
        all_next_states: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_t_starts: List[torch.Tensor] = []
        all_episode_ids: List[torch.Tensor] = []
        force_accum: Dict[str, List[torch.Tensor]] = {k: [] for k in self._FORCE_KEYS}
        has_forces: Optional[bool] = None

        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=True)
            all_states.append(d["states"])
            all_next_states.append(d["next_states"])
            all_actions.append(d["actions"])
            all_t_starts.append(d["t_start"])
            all_episode_ids.append(d["episode_ids"])
            if has_forces is None:
                has_forces = "forces" in d
            if has_forces and "forces" in d:
                for k in self._FORCE_KEYS:
                    force_accum[k].append(d["forces"][k])

        self.states = torch.cat(all_states, dim=0)
        self.next_states = torch.cat(all_next_states, dim=0)
        self.actions = torch.cat(all_actions, dim=0)
        self.t_starts = torch.cat(all_t_starts, dim=0)
        self.episode_ids = torch.cat(all_episode_ids, dim=0)
        self.forces: Optional[Dict[str, torch.Tensor]] = (
            {k: torch.cat(force_accum[k], dim=0) for k in self._FORCE_KEYS}
            if has_forces and force_accum[self._FORCE_KEYS[0]]
            else None
        )

        # Train/val split by episode_id (not by transition index -- avoids leakage)
        rng_np = np.random.default_rng(seed)
        unique_eps = torch.unique(self.episode_ids).numpy()
        rng_np.shuffle(unique_eps)
        n_val = max(1, int(len(unique_eps) * val_fraction)) if val_fraction > 0.0 else 0
        val_set = set(unique_eps[:n_val].tolist())
        mask = torch.tensor([ep.item() not in val_set for ep in self.episode_ids])
        if split == "val":
            mask = ~mask

        self.states = self.states[mask]
        self.next_states = self.next_states[mask]
        self.actions = self.actions[mask]
        self.t_starts = self.t_starts[mask]
        self.episode_ids = self.episode_ids[mask]
        if self.forces is not None:
            self.forces = {k: v[mask] for k, v in self.forces.items()}

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return one transition.

        Returns dict with:
            state:      (124,)
            action:     (5,)
            t_start:    scalar tensor
            next_state: (124,)
            delta:      (124,) = next_state - state
            forces:     dict with per-item tensors, or None
        """
        state = self.states[idx]
        next_state = self.next_states[idx]
        return {
            "state":      state,
            "action":     self.actions[idx],
            "t_start":    self.t_starts[idx],
            "next_state": next_state,
            "delta":      next_state - state,
            "forces": (
                {k: v[idx] for k, v in self.forces.items()}
                if self.forces is not None else None
            ),
        }

    def get_sample_weights(self, n_bins: int = 20, clip_max: float = 10.0) -> torch.Tensor:
        """Compute inverse-frequency sample weights over the state distribution.

        Args:
            n_bins: Number of bins per feature dimension.
            clip_max: Maximum weight value.

        Returns:
            (N,) float32 weight tensor, normalized so the mean weight is <= clip_max.
        """
        data = self.states.numpy()
        counts = np.ones(len(data), dtype=np.float64)
        for dim_idx in range(min(n_bins, data.shape[1])):
            col = data[:, dim_idx]
            bins = np.linspace(col.min(), col.max() + 1e-8, n_bins + 1)
            bin_ids = np.clip(np.digitize(col, bins) - 1, 0, n_bins - 1)
            bin_counts = np.maximum(
                np.bincount(bin_ids, minlength=n_bins).astype(np.float64), 1.0
            )
            counts *= bin_counts[bin_ids]
        weights = 1.0 / counts
        weights = np.clip(weights / weights.mean(), 0.0, clip_max)
        return torch.from_numpy(weights.astype(np.float32))
