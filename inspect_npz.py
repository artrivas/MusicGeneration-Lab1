from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.ravel()[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a piano-roll NPZ dataset.")
    parser.add_argument("--npz", required=True, type=Path, help="Path to train/eval .npz file")
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"NPZ file does not exist: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)
    print(f"path: {args.npz}")
    print("keys:")
    for key in data.files:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    if "rolls_flat" in data.files:
        rolls = data["rolls_flat"]
        print(f"rolls_flat: shape={rolls.shape}, dtype={rolls.dtype}")
    elif "sequences" in data.files:
        seqs = data["sequences"]
        print(f"sequences: shape={seqs.shape}, dtype={seqs.dtype}")

    if "offsets" in data.files:
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        lengths = np.diff(offsets)
        print(f"offsets: shape={offsets.shape}, dtype={offsets.dtype}")
        print(f"num_sequences: {len(lengths)}")
        if len(offsets):
            print(f"offset_start: {int(offsets[0])}")
            print(f"offset_end: {int(offsets[-1])}")
        if len(lengths):
            print(
                "sequence_lengths: "
                f"min={int(lengths.min())}, max={int(lengths.max())}, "
                f"mean={float(lengths.mean()):.2f}, median={float(np.median(lengths)):.2f}"
            )
            print(f"first_lengths: {lengths[:10].astype(int).tolist()}")
    elif "sequences" in data.files:
        seqs = data["sequences"]
        print(f"num_sequences: {len(seqs)}")
        try:
            lengths = np.asarray([len(s) for s in seqs], dtype=np.int64)
            print(
                "sequence_lengths: "
                f"min={int(lengths.min())}, max={int(lengths.max())}, "
                f"mean={float(lengths.mean()):.2f}, median={float(np.median(lengths)):.2f}"
            )
        except Exception:
            pass

    if "ids" in data.files:
        ids = data["ids"]
        print(f"ids: shape={ids.shape}, dtype={ids.dtype}, first={ids[:10].tolist()}")

    print(f"step_sec: {scalar(data, 'step_sec', 'missing')}")
    print(f"note_min: {scalar(data, 'note_min', 'missing')}")
    print(f"note_max: {scalar(data, 'note_max', 'missing')}")
    print(f"num_positions: {scalar(data, 'num_positions', 'missing')}")
    print(f"representation: {scalar(data, 'representation', 'missing')}")
    if "is_prefix" in data.files:
        print(f"is_prefix: {scalar(data, 'is_prefix')}")
    if "prefix_steps" in data.files:
        print(f"prefix_steps: {np.asarray(data['prefix_steps']).tolist()}")


if __name__ == "__main__":
    main()
