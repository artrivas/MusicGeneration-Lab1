from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PianoRollWindowDataset(Dataset):
    """Random fixed-length windows from sequence-aware piano-roll cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        context_len: int = 512,
        split: str = "train",
        val_fraction: float = 0.1,
        steps_per_epoch: int = 1000,
        seed: int = 1234,
        sequence_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.context_len = int(context_len)
        self.window_len = self.context_len + 1
        self.split = split
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 100_000))

        self.rolls_flat = np.load(self.cache_dir / "rolls_flat.npy", mmap_mode="r")
        self.offsets = np.load(self.cache_dir / "offsets.npy", mmap_mode="r").astype(np.int64, copy=False)
        self.ids = np.load(self.cache_dir / "ids.npy", mmap_mode="r")

        info_path = self.cache_dir / "dataset_info.json"
        self.info = {}
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as f:
                self.info = json.load(f)

        lengths = np.diff(self.offsets)
        valid = np.flatnonzero(lengths >= self.window_len)
        if sequence_ids is not None:
            wanted = set(np.asarray(sequence_ids).tolist())
            valid = np.asarray([i for i in valid if self.ids[i].item() in wanted], dtype=np.int64)
        else:
            valid = self._split_indices(valid, val_fraction, seed)

        if len(valid) == 0:
            raise ValueError(
                f"No {split} sequences are at least context_len+1={self.window_len} steps long. "
                "Lower --context_len or check the dataset."
            )

        self.seq_indices = valid.astype(np.int64)
        self.seq_lengths = lengths[self.seq_indices].astype(np.int64)
        self.weights = self.seq_lengths - self.window_len + 1
        self.weights = self.weights.astype(np.float64) / float(self.weights.sum())

    def _split_indices(self, valid: np.ndarray, val_fraction: float, seed: int) -> np.ndarray:
        if self.split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val', or 'all'")
        if self.split == "all":
            return valid
        rng = np.random.default_rng(seed)
        perm = valid.copy()
        rng.shuffle(perm)
        n_val = max(1, int(round(len(perm) * val_fraction))) if len(perm) > 1 else 0
        val_set = set(perm[:n_val].tolist())
        if self.split == "val":
            return np.asarray([i for i in valid if i in val_set], dtype=np.int64)
        return np.asarray([i for i in valid if i not in val_set], dtype=np.int64)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, index: int) -> torch.Tensor:
        del index
        choice = int(self.rng.choice(len(self.seq_indices), p=self.weights))
        seq_idx = int(self.seq_indices[choice])
        seq_len = int(self.seq_lengths[choice])
        start_in_seq = int(self.rng.integers(0, seq_len - self.window_len + 1))
        start = int(self.offsets[seq_idx]) + start_in_seq
        end = start + self.window_len
        window = np.asarray(self.rolls_flat[start:end], dtype=np.float32)
        return torch.from_numpy(window)


def estimate_note_density(cache_dir: str | Path, max_frames: int = 250_000, seed: int = 1234) -> float:
    rolls = np.load(Path(cache_dir) / "rolls_flat.npy", mmap_mode="r")
    n = len(rolls)
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    take = min(int(max_frames), n)
    idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
    sample = np.asarray(rolls[idx], dtype=np.float32)
    return float(sample.mean())


def estimate_pos_weight(cache_dir: str | Path, max_frames: int = 250_000, seed: int = 1234) -> float:
    density = estimate_note_density(cache_dir, max_frames=max_frames, seed=seed)
    if density <= 0:
        return 1.0
    return float(min(100.0, max(1.0, (1.0 - density) / density)))
