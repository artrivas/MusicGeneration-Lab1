from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vocab import BOS, EOS, PAD, UNK_CHORD, ChordVocab, load_rolls_flat_npz, scalar, sequence_view


@dataclass
class EventSequence:
    delta: np.ndarray
    chord: np.ndarray
    notes: np.ndarray
    cum: np.ndarray
    card: np.ndarray
    total_steps: int


def roll_to_events(
    roll: np.ndarray,
    vocab: ChordVocab,
    max_delta: int = 1200,
    add_bos: bool = True,
    add_eos: bool = True,
) -> EventSequence:
    roll = np.asarray(roll, dtype=np.uint8)
    active_ts = np.flatnonzero(np.sum(roll > 0, axis=1) > 0)

    deltas: list[int] = []
    chords: list[int] = []
    notes: list[np.ndarray] = []
    cums: list[int] = []
    cards: list[int] = []

    if add_bos:
        deltas.append(0)
        chords.append(BOS)
        notes.append(np.zeros(vocab.num_notes, dtype=np.uint8))
        cums.append(0)
        cards.append(0)

    prev_t = -1
    for t in active_ts:
        frame = roll[int(t), : vocab.num_notes].astype(np.uint8, copy=False)
        delta = int(t) - prev_t
        deltas.append(min(delta, max_delta))
        chords.append(vocab.encode_frame(frame))
        notes.append(frame.copy())
        cums.append(int(t))
        cards.append(int(np.sum(frame > 0)))
        prev_t = int(t)

    if add_eos:
        eos_step = int(len(roll))
        delta = eos_step - prev_t if prev_t >= 0 else eos_step + 1
        deltas.append(min(delta, max_delta))
        chords.append(EOS)
        notes.append(np.zeros(vocab.num_notes, dtype=np.uint8))
        cums.append(eos_step)
        cards.append(0)

    if not deltas:
        deltas = [0]
        chords = [PAD]
        notes = [np.zeros(vocab.num_notes, dtype=np.uint8)]
        cums = [0]
        cards = [0]

    return EventSequence(
        delta=np.asarray(deltas, dtype=np.int64),
        chord=np.asarray(chords, dtype=np.int64),
        notes=np.stack(notes).astype(np.float32),
        cum=np.asarray(cums, dtype=np.int64),
        card=np.asarray(cards, dtype=np.int64),
        total_steps=int(len(roll)),
    )


class EventChunkDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        vocab: ChordVocab,
        sequence_indices: list[int],
        context_len: int = 1024,
        max_delta: int = 1200,
        random_crop: bool = True,
        samples_per_sequence: int = 1,
    ) -> None:
        self.rolls_flat, self.offsets, self.data = load_rolls_flat_npz(npz_path)
        self.vocab = vocab
        self.sequence_indices = list(sequence_indices)
        self.context_len = int(context_len)
        self.max_delta = int(max_delta)
        self.random_crop = bool(random_crop)
        self.samples_per_sequence = max(1, int(samples_per_sequence))
        self.events = [
            roll_to_events(sequence_view(self.rolls_flat, self.offsets, i), vocab, max_delta=max_delta)
            for i in self.sequence_indices
        ]

    def __len__(self) -> int:
        if self.random_crop:
            return len(self.events) * self.samples_per_sequence
        return len(self.events)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.random_crop and self.events:
            index = index % len(self.events)
        ev = self.events[index]
        need = self.context_len + 1
        n = len(ev.delta)
        if n > need:
            if self.random_crop:
                start = int(np.random.randint(0, n - need + 1))
            else:
                start = max(0, (n - need) // 2)
            sl = slice(start, start + need)
            delta = ev.delta[sl]
            chord = ev.chord[sl]
            notes = ev.notes[sl]
            cum = ev.cum[sl]
            card = ev.card[sl]
        else:
            pad = need - n
            delta = np.pad(ev.delta, (0, pad), constant_values=0)
            chord = np.pad(ev.chord, (0, pad), constant_values=PAD)
            notes = np.pad(ev.notes, ((0, pad), (0, 0)), constant_values=0)
            cum = np.pad(ev.cum, (0, pad), constant_values=0)
            card = np.pad(ev.card, (0, pad), constant_values=0)

        return {
            "delta_in": torch.as_tensor(delta[:-1], dtype=torch.long),
            "chord_in": torch.as_tensor(chord[:-1], dtype=torch.long),
            "notes_in": torch.as_tensor(notes[:-1], dtype=torch.float32),
            "cum_in": torch.as_tensor(cum[:-1], dtype=torch.long),
            "card_in": torch.as_tensor(card[:-1], dtype=torch.long),
            "delta_target": torch.as_tensor(delta[1:], dtype=torch.long),
            "chord_target": torch.as_tensor(chord[1:], dtype=torch.long),
            "notes_target": torch.as_tensor(notes[1:], dtype=torch.float32),
            "card_target": torch.as_tensor(card[1:], dtype=torch.long),
            "target_mask": torch.as_tensor(chord[1:] != PAD, dtype=torch.bool),
        }


def split_sequence_indices(num_sequences: int, val_frac: float = 0.1, seed: int = 42) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_sequences)
    rng.shuffle(indices)
    n_val = max(1, int(round(num_sequences * val_frac))) if num_sequences > 1 else 0
    val = sorted(indices[:n_val].astype(int).tolist())
    train = sorted(indices[n_val:].astype(int).tolist())
    return train, val


def dataset_metadata(npz_path: str | Path) -> dict[str, Any]:
    _, offsets, data = load_rolls_flat_npz(npz_path)
    return {
        "num_sequences": len(offsets) - 1,
        "step_sec": float(scalar(data, "step_sec", 0.05)),
        "note_min": int(scalar(data, "note_min", 21)),
        "note_max": int(scalar(data, "note_max", 108)),
        "num_positions": int(scalar(data, "num_positions", 88)),
        "representation": str(scalar(data, "representation", "onset")),
    }
