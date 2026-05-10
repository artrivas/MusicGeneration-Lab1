from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _scalar(data: Any, key: str, default: Any) -> Any:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    if value.size:
        return value.ravel()[0].item() if hasattr(value.ravel()[0], "item") else value.ravel()[0]
    return default


def load_npz_sequences(npz_path: str | Path) -> tuple[list[np.ndarray], Any]:
    data = np.load(str(npz_path), allow_pickle=True)
    if "rolls_flat" in data.files and "offsets" in data.files:
        rolls_flat = np.asarray(data["rolls_flat"], dtype=np.uint8)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        rolls = [rolls_flat[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(offsets) - 1)]
        return rolls, data
    if "sequences" in data.files:
        seqs = data["sequences"]
        return [np.asarray(seq, dtype=np.uint8) for seq in seqs], data
    raise KeyError("Expected .npz with rolls_flat+offsets or sequences.")


def get_sequence(data: Any, index: int) -> np.ndarray:
    if "rolls_flat" in data.files and "offsets" in data.files:
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        return np.asarray(data["rolls_flat"][int(offsets[index]) : int(offsets[index + 1])], dtype=np.uint8)
    if "sequences" in data.files:
        return np.asarray(data["sequences"][index], dtype=np.uint8)
    raise KeyError("Expected .npz with rolls_flat+offsets or sequences.")


def get_dataset_defaults(data: Any) -> dict[str, Any]:
    return {
        "step_sec": float(_scalar(data, "step_sec", 0.05)),
        "note_min": int(_scalar(data, "note_min", 21)),
        "note_max": int(_scalar(data, "note_max", 108)),
        "num_positions": int(_scalar(data, "num_positions", 88)),
        "representation": str(_scalar(data, "representation", "onset")),
    }


def save_rolls_flat_npz(
    out_path: str | Path,
    rolls: Iterable[np.ndarray],
    ids: Iterable[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    roll_list = [np.asarray(r, dtype=np.uint8) for r in rolls]
    if roll_list:
        num_notes = int(roll_list[0].shape[1])
        rolls_flat = np.concatenate(roll_list, axis=0).astype(np.uint8, copy=False)
    else:
        num_notes = 88
        rolls_flat = np.zeros((0, num_notes), dtype=np.uint8)
    offsets = np.zeros(len(roll_list) + 1, dtype=np.int64)
    total = 0
    for i, roll in enumerate(roll_list):
        total += int(len(roll))
        offsets[i + 1] = total
    if ids is None:
        ids_arr = np.asarray([f"seq_{i:04d}" for i in range(len(roll_list))])
    else:
        ids_arr = np.asarray(list(ids))
    payload: dict[str, Any] = {"rolls_flat": rolls_flat, "offsets": offsets, "ids": ids_arr}
    for key, value in (metadata or {}).items():
        if value is not None:
            payload[key] = value
    if "num_positions" not in payload:
        payload["num_positions"] = num_notes
    np.savez_compressed(out_path, **payload)
    return out_path


def event_density(roll: np.ndarray) -> float:
    roll = np.asarray(roll)
    if len(roll) == 0:
        return 0.0
    return float(np.count_nonzero(np.sum(roll > 0, axis=1)) / len(roll))


def avg_notes_per_event(roll: np.ndarray) -> float:
    roll = np.asarray(roll)
    if len(roll) == 0:
        return 0.0
    counts = np.sum(roll > 0, axis=1)
    event_counts = counts[counts > 0]
    return float(np.mean(event_counts)) if len(event_counts) else 0.0


def longest_silent_run(roll: np.ndarray) -> int:
    roll = np.asarray(roll)
    best = cur = 0
    for active in np.sum(roll > 0, axis=1) > 0:
        if active:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return int(best)


def top_patterns(roll: np.ndarray, top_k: int = 10) -> list[tuple[str, int]]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for frame in np.asarray(roll):
        active = np.flatnonzero(frame > 0)
        if len(active):
            counter["-".join(str(int(i)) for i in active)] += 1
    return counter.most_common(top_k)
