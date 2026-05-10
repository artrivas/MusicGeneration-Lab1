from __future__ import annotations

import argparse
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PAD = 0
BOS = 1
EOS = 2
UNK_CHORD = 3
SPECIAL_TOKENS = ("PAD", "BOS", "EOS", "UNK_CHORD")


def scalar(data: Any, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.shape == ():
        value = arr.item()
    elif arr.size:
        value = arr.ravel()[0]
        if isinstance(value, np.generic):
            value = value.item()
    else:
        value = default
    return value


def load_rolls_flat_npz(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, Any]:
    data = np.load(str(npz_path), allow_pickle=True)
    if "rolls_flat" not in data.files or "offsets" not in data.files:
        raise KeyError(f"{npz_path} must contain rolls_flat and offsets")
    rolls_flat = np.asarray(data["rolls_flat"], dtype=np.uint8)
    offsets = np.asarray(data["offsets"], dtype=np.int64)
    return rolls_flat, offsets, data


def sequence_view(rolls_flat: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    start = int(offsets[index])
    end = int(offsets[index + 1])
    return np.asarray(rolls_flat[start:end], dtype=np.uint8)


def pack_chord(frame: np.ndarray) -> bytes:
    return np.packbits(np.asarray(frame, dtype=np.uint8), bitorder="big").tobytes()


def unpack_chord(key: bytes, num_notes: int = 88) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8), bitorder="big")
    return bits[:num_notes].astype(np.uint8, copy=False)


@dataclass
class ChordVocab:
    key_to_id: dict[bytes, int]
    id_to_key: list[bytes | None]
    num_notes: int = 88

    @property
    def size(self) -> int:
        return len(self.id_to_key)

    def encode_frame(self, frame: np.ndarray) -> int:
        return self.key_to_id.get(pack_chord(frame), UNK_CHORD)

    def decode_id(self, chord_id: int) -> np.ndarray | None:
        if chord_id < 0 or chord_id >= len(self.id_to_key):
            return None
        key = self.id_to_key[chord_id]
        if key is None:
            return None
        return unpack_chord(key, self.num_notes)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"key_to_id": self.key_to_id, "id_to_key": self.id_to_key, "num_notes": self.num_notes},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ChordVocab":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        return cls(
            key_to_id=payload["key_to_id"],
            id_to_key=payload["id_to_key"],
            num_notes=int(payload.get("num_notes", 88)),
        )


def build_chord_vocab(train_npz: str | Path, top_k: int = 50_000, num_notes: int = 88) -> ChordVocab:
    rolls_flat, offsets, _ = load_rolls_flat_npz(train_npz)
    counts: Counter[bytes] = Counter()
    for i in range(len(offsets) - 1):
        roll = sequence_view(rolls_flat, offsets, i)
        active_rows = roll[np.sum(roll > 0, axis=1) > 0]
        for frame in active_rows:
            counts[pack_chord(frame[:num_notes])] += 1

    id_to_key: list[bytes | None] = [None, None, None, None]
    key_to_id: dict[bytes, int] = {}
    for key, _ in counts.most_common(top_k):
        key_to_id[key] = len(id_to_key)
        id_to_key.append(key)
    return ChordVocab(key_to_id=key_to_id, id_to_key=id_to_key, num_notes=num_notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a top-K non-empty chord vocabulary.")
    parser.add_argument("--train", type=Path, default=Path("../train.npz"))
    parser.add_argument("--out", type=Path, default=Path("vocab.pkl"))
    parser.add_argument("--top-k", type=int, default=50_000)
    args = parser.parse_args()

    vocab = build_chord_vocab(args.train, top_k=args.top_k)
    vocab.save(args.out)
    print(f"saved {vocab.size} chord tokens to {args.out}")


if __name__ == "__main__":
    main()
