"""Dataset classes for surrogate model training.

Provides single-step and multi-step (trajectory) data loading from .pt or
.parquet files saved by collect_data.py.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from aprx_model_elastica.state import POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z


def compute_density_weights(
    states: torch.Tensor,
    n_bins: int = 20,
    clip_max: float = 10.0,
) -> torch.Tensor:
    """Compute inverse-density sample weights from state summary features.

    Projects 124D states to 4 summary features (CoM_x, CoM_y, velocity
    magnitude, mean |angular velocity|), bins them into a joint histogram,
    and returns weights ∝ 1/bin_count so rare states are upweighted.

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

    # Weight = 1 / count → rare bins get high weight
    weights = 1.0 / counts[inverse].float()

    # Normalize to mean=1, clip to prevent outliers
    weights = weights / weights.mean()
    weights = weights.clamp(max=clip_max)

    return weights


def _load_parquet_batch(path: Path) -> Dict[str, torch.Tensor]:
    """Load a .parquet batch file and return tensors matching .pt format."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {
        "states": torch.tensor(
            np.array(table["states"].to_pylist(), dtype=np.float32)
        ),
        "actions": torch.tensor(
            np.array(table["actions"].to_pylist(), dtype=np.float32)
        ),
        "serpenoid_times": torch.tensor(
            table["serpenoid_times"].to_numpy(zero_copy_only=False).astype(np.float32)
        ),
        "next_states": torch.tensor(
            np.array(table["next_states"].to_pylist(), dtype=np.float32)
        ),
        "episode_ids": torch.tensor(
            table["episode_ids"].to_numpy(zero_copy_only=False).astype(np.int64)
        ),
        "step_indices": torch.tensor(
            table["step_indices"].to_numpy(zero_copy_only=False).astype(np.int64)
        ),
    }


class SurrogateDataset(Dataset):
    """Single-step transition dataset for surrogate training.

    Each item is a (state, action, serpenoid_time, next_state, delta) tuple.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        """Load and concatenate all .pt batch files from data_dir.

        Args:
            data_dir: Directory containing batch_NNNN.pt files.
            split: "train" or "val".
            val_fraction: Fraction of episodes held out for validation.
            seed: Random seed for train/val split.
        """
        self.data_dir = Path(data_dir)

        # Load all batch files (support both .pt and .parquet)
        pt_files = sorted(self.data_dir.glob("batch_*.pt"))
        pq_files = sorted(self.data_dir.glob("batch_*.parquet"))
        if not pt_files and not pq_files:
            raise FileNotFoundError(
                f"No batch_*.pt or batch_*.parquet files found in {data_dir}"
            )

        all_states = []
        all_actions = []
        all_serp_times = []
        all_next_states = []
        all_episode_ids = []
        all_step_indices = []

        episode_offset = 0
        for bf in pt_files:
            data = torch.load(bf, map_location="cpu", weights_only=True)
            all_states.append(data["states"])
            all_actions.append(data["actions"])
            all_serp_times.append(data["serpenoid_times"])
            all_next_states.append(data["next_states"])
            all_episode_ids.append(data["episode_ids"] + episode_offset)
            all_step_indices.append(data["step_indices"])
            episode_offset = all_episode_ids[-1].max().item() + 1

        for bf in pq_files:
            data = _load_parquet_batch(bf)
            all_states.append(data["states"])
            all_actions.append(data["actions"])
            all_serp_times.append(data["serpenoid_times"])
            all_next_states.append(data["next_states"])
            all_episode_ids.append(data["episode_ids"] + episode_offset)
            all_step_indices.append(data["step_indices"])
            episode_offset = all_episode_ids[-1].max().item() + 1

        self.states = torch.cat(all_states, dim=0)
        self.actions = torch.cat(all_actions, dim=0)
        self.serpenoid_times = torch.cat(all_serp_times, dim=0)
        self.next_states = torch.cat(all_next_states, dim=0)
        self.episode_ids = torch.cat(all_episode_ids, dim=0)
        self.step_indices = torch.cat(all_step_indices, dim=0)

        # Compute deltas
        self.deltas = self.next_states - self.states

        # Train/val split by episode (not random transition — avoids leakage)
        unique_episodes = torch.unique(self.episode_ids).numpy()
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_episodes)
        n_val = max(1, int(len(unique_episodes) * val_fraction))
        if split == "val":
            keep_episodes = set(unique_episodes[:n_val].tolist())
        else:
            keep_episodes = set(unique_episodes[n_val:].tolist())

        mask = torch.tensor(
            [eid.item() in keep_episodes for eid in self.episode_ids],
            dtype=torch.bool,
        )
        self.states = self.states[mask]
        self.actions = self.actions[mask]
        self.serpenoid_times = self.serpenoid_times[mask]
        self.next_states = self.next_states[mask]
        self.deltas = self.deltas[mask]
        self.episode_ids = self.episode_ids[mask]
        self.step_indices = self.step_indices[mask]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "serpenoid_time": self.serpenoid_times[idx],
            "next_state": self.next_states[idx],
            "delta": self.deltas[idx],
        }

    def get_sample_weights(
        self,
        n_bins: int = 20,
        clip_max: float = 10.0,
    ) -> torch.Tensor:
        """Compute inverse-density sample weights for WeightedRandomSampler.

        Upweights rare states, downweights overrepresented regions.

        Args:
            n_bins: Histogram bins per feature dimension.
            clip_max: Maximum sample weight.

        Returns:
            (N,) weight tensor.
        """
        return compute_density_weights(self.states, n_bins=n_bins, clip_max=clip_max)


class TrajectoryDataset(Dataset):
    """Trajectory-window dataset for multi-step rollout loss.

    Each item is a contiguous window of `rollout_length` transitions from
    the same episode.
    """

    def __init__(
        self,
        data_dir: str,
        rollout_length: int = 8,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        # Load single-step dataset to get episode-grouped data
        base = SurrogateDataset(data_dir, split=split, val_fraction=val_fraction, seed=seed)

        self.states = base.states
        self.actions = base.actions
        self.serpenoid_times = base.serpenoid_times
        self.next_states = base.next_states
        self.deltas = base.deltas
        self.episode_ids = base.episode_ids
        self.step_indices = base.step_indices
        self.rollout_length = rollout_length

        # Build index: (episode_id, start_step) → list of contiguous indices
        self._build_trajectory_index()

    def _build_trajectory_index(self):
        """Build mapping from (episode, start_step) to dataset indices."""
        # Group transitions by episode
        episodes: Dict[int, List[Tuple[int, int]]] = {}
        for i in range(len(self.episode_ids)):
            eid = self.episode_ids[i].item()
            step = self.step_indices[i].item()
            if eid not in episodes:
                episodes[eid] = []
            episodes[eid].append((step, i))

        # Sort each episode by step index
        for eid in episodes:
            episodes[eid].sort(key=lambda x: x[0])

        # Find valid windows of rollout_length contiguous steps
        self.windows: List[List[int]] = []
        for eid, steps_and_indices in episodes.items():
            indices = [idx for _, idx in steps_and_indices]
            step_nums = [s for s, _ in steps_and_indices]

            for start in range(len(indices) - self.rollout_length):
                # Check that steps are contiguous
                window_steps = step_nums[start : start + self.rollout_length + 1]
                if window_steps[-1] - window_steps[0] == self.rollout_length:
                    self.windows.append(
                        indices[start : start + self.rollout_length + 1]
                    )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a trajectory window.

        Returns dict with:
            states:          (rollout_length + 1, 124)
            actions:         (rollout_length, 5)
            serpenoid_times: (rollout_length,)
        """
        indices = self.windows[idx]
        # states includes the final target state
        return {
            "states": self.states[indices],                          # (L+1, 124)
            "actions": self.actions[indices[:-1]],                   # (L, 5)
            "serpenoid_times": self.serpenoid_times[indices[:-1]],   # (L,)
        }


class HistoryDataset(TrajectoryDataset):
    """Single-step transition dataset with K prior steps as history context.

    Each item includes K prior (state, action) pairs before the current step,
    enabling history-conditioned surrogate models (HistorySurrogateModel).

    Window layout (history_k + 2 indices per window):
        indices[0 .. history_k-1]  — K prior transitions (history)
        indices[history_k]         — current step (input)
        indices[history_k + 1]     — next step (target for delta)

    Uses TrajectoryDataset's window-building logic with
    rollout_length = history_k + 1 to get windows of the required size.
    """

    def __init__(
        self,
        data_dir: str,
        history_k: int = 2,
        rollout_length: int = 1,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        """Create a HistoryDataset.

        Args:
            data_dir: Directory containing batch_*.pt files.
            history_k: Number of prior steps to include as history context.
            rollout_length: Ignored — present for API compatibility. Window
                length is always history_k + 1 (K prior + current + next).
            split: "train" or "val".
            val_fraction: Fraction of episodes held out for validation.
            seed: Random seed for train/val split.
        """
        # We need windows of length history_k + 2:
        #   K prior steps + current step + next step (for delta)
        # TrajectoryDataset with rollout_length=history_k+1 produces
        # windows of exactly that size (rollout_length + 1 indices).
        super().__init__(
            data_dir,
            rollout_length=history_k + 1,
            split=split,
            val_fraction=val_fraction,
            seed=seed,
        )
        self.history_k = history_k

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single transition with history context.

        Returns dict with:
            history_states:  (K, 124) — K prior states (oldest first)
            history_actions: (K, 5)   — K prior actions (oldest first)
            state:           (124,)   — current state
            action:          (5,)     — current action
            serpenoid_time:  ()       — current serpenoid time scalar
            delta:           (124,)   — next_state - current_state (target)
        """
        indices = self.windows[idx]  # length = history_k + 2

        # Prior K steps: indices[0..history_k-1]
        history_states = self.states[indices[:self.history_k]]     # (K, 124)
        history_actions = self.actions[indices[:self.history_k]]   # (K, 5)

        # Current step: indices[history_k]
        current_idx = indices[self.history_k]
        current_state = self.states[current_idx]
        current_action = self.actions[current_idx]
        current_time = self.serpenoid_times[current_idx]

        # Target: indices[history_k + 1]
        next_idx = indices[self.history_k + 1]
        next_state = self.states[next_idx]
        delta = next_state - current_state

        return dict(
            history_states=history_states,
            history_actions=history_actions,
            state=current_state,
            action=current_action,
            serpenoid_time=current_time,
            delta=delta,
        )
