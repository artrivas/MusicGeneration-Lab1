from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset):
    """Random fixed-length token windows with sequence-level train/val split."""

    def __init__(
        self,
        cache_dir: str | Path,
        context_len: int = 1024,
        split: str = "train",
        val_fraction: float = 0.1,
        steps_per_epoch: int = 1000,
        seed: int = 1234,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.context_len = int(context_len)
        self.window_len = self.context_len + 1
        self.split = split
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 100_000))
        self.tokens_flat = np.load(self.cache_dir / "tokens_flat.npy", mmap_mode="r")
        self.token_offsets = np.load(self.cache_dir / "token_offsets.npy", mmap_mode="r").astype(np.int64, copy=False)

        info_path = self.cache_dir / "dataset_info.json"
        self.info = {}
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as f:
                self.info = json.load(f)

        lengths = np.diff(self.token_offsets)
        valid = np.flatnonzero(lengths >= self.window_len)
        valid = self._split_indices(valid, val_fraction, seed)
        if len(valid) == 0:
            raise ValueError(
                f"No {split} token sequences are at least context_len+1={self.window_len} tokens. "
                "Lower --context_len or rebuild the token cache."
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        choice = int(self.rng.choice(len(self.seq_indices), p=self.weights))
        seq_idx = int(self.seq_indices[choice])
        seq_len = int(self.seq_lengths[choice])
        start_in_seq = int(self.rng.integers(0, seq_len - self.window_len + 1))
        start = int(self.token_offsets[seq_idx]) + start_in_seq
        window = np.asarray(self.tokens_flat[start : start + self.window_len], dtype=np.int64)
        input_ids = torch.from_numpy(window[:-1].copy())
        target_ids = torch.from_numpy(window[1:].copy())
        return input_ids, target_ids
