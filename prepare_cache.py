from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.ravel()[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def prepare_cache(train_npz: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rolls_path = cache_dir / "rolls_flat.npy"
    offsets_path = cache_dir / "offsets.npy"
    ids_path = cache_dir / "ids.npy"
    info_path = cache_dir / "dataset_info.json"

    if rolls_path.exists() and offsets_path.exists() and ids_path.exists() and info_path.exists():
        return info_path

    if not train_npz.exists():
        raise FileNotFoundError(f"Training NPZ does not exist: {train_npz}")

    try:
        data = np.load(train_npz, allow_pickle=True)
        if "rolls_flat" not in data.files or "offsets" not in data.files:
            raise KeyError("Expected train NPZ to contain rolls_flat and offsets.")

        np.save(rolls_path, data["rolls_flat"])
        np.save(offsets_path, data["offsets"].astype(np.int64, copy=False))
        if "ids" in data.files:
            ids = data["ids"]
        else:
            ids = np.arange(len(data["offsets"]) - 1, dtype=np.int64)
        np.save(ids_path, ids)

        offsets = np.asarray(data["offsets"], dtype=np.int64)
        lengths = np.diff(offsets)
        info = {
            "source_npz": str(train_npz),
            "rolls_flat_shape": list(data["rolls_flat"].shape),
            "rolls_flat_dtype": str(data["rolls_flat"].dtype),
            "offsets_shape": list(offsets.shape),
            "ids_shape": list(ids.shape),
            "num_sequences": int(len(lengths)),
            "total_steps": int(offsets[-1]) if len(offsets) else 0,
            "min_length": int(lengths.min()) if len(lengths) else 0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
            "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
            "step_sec": _scalar(data, "step_sec", 0.05),
            "note_min": _scalar(data, "note_min", 21),
            "note_max": _scalar(data, "note_max", 108),
            "num_positions": _scalar(data, "num_positions", int(data["rolls_flat"].shape[1])),
            "representation": str(_scalar(data, "representation", "onset")),
        }
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return info_path
    except MemoryError as exc:
        raise MemoryError(
            "Not enough RAM to decompress train.npz into the cache. Close other programs, "
            "use a machine with more memory, or ask the dataset provider for uncompressed .npy files."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert train.npz into a memory-mapped .npy cache.")
    parser.add_argument("--train_npz", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    args = parser.parse_args()
    info_path = prepare_cache(args.train_npz, args.cache_dir)
    print(f"cache ready: {info_path}")


if __name__ == "__main__":
    main()
