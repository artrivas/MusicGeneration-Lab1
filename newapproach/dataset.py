from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray


def make_sequence_split(num_sequences: int, val_fraction: float = 0.1, seed: int = 1234) -> SequenceSplit:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_sequences)
    rng.shuffle(indices)
    val_count = max(1, int(round(num_sequences * val_fraction))) if num_sequences > 1 else 0
    return SequenceSplit(train_indices=np.sort(indices[val_count:]), val_indices=np.sort(indices[:val_count]))


class TokenWindowDataset(Dataset):
    def __init__(
        self,
        tokens_flat_path: str,
        token_offsets_path: str,
        context_len: int,
        sequence_indices: np.ndarray | list[int] | None = None,
        samples_per_epoch: int | None = None,
        seed: int = 1234,
    ) -> None:
        self.tokens_flat = np.load(tokens_flat_path, mmap_mode="r")
        self.token_offsets = np.load(token_offsets_path, mmap_mode="r")
        self.context_len = int(context_len)
        all_indices = np.arange(len(self.token_offsets) - 1)
        if sequence_indices is None:
            sequence_indices = all_indices
        self.sequence_indices = np.asarray(sequence_indices, dtype=np.int64)
        self.eligible = []
        for idx in self.sequence_indices:
            start = int(self.token_offsets[idx])
            end = int(self.token_offsets[idx + 1])
            if end - start >= self.context_len + 1:
                self.eligible.append(int(idx))
        self.eligible = np.asarray(self.eligible, dtype=np.int64)
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch is not None else max(1, len(self.eligible) * 64)
        self.rng = np.random.default_rng(seed)
        if len(self.eligible) == 0:
            raise ValueError("No sequences are long enough for context_len + 1.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx = int(self.rng.choice(self.eligible))
        seq_start = int(self.token_offsets[seq_idx])
        seq_end = int(self.token_offsets[seq_idx + 1])
        max_start = seq_end - seq_start - self.context_len - 1
        local_start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        start = seq_start + local_start
        x = np.asarray(self.tokens_flat[start : start + self.context_len], dtype=np.int64)
        y = np.asarray(self.tokens_flat[start + 1 : start + self.context_len + 1], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
