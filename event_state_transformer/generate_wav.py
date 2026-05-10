from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_play import save_wav, synthesize_musicbox_roll  # noqa: E402
from vocab import load_rolls_flat_npz, scalar, sequence_view  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export generated NPZ sequences to WAV.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("wav"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    rolls_flat, offsets, data = load_rolls_flat_npz(args.npz)
    step_sec = float(scalar(data, "step_sec", 0.05))
    note_min = int(scalar(data, "note_min", 21))
    representation = str(scalar(data, "representation", "onset"))
    ids = np.asarray(data["ids"]) if "ids" in data.files else np.asarray([f"seq_{i:04d}" for i in range(len(offsets) - 1)])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = len(offsets) - 1 if args.limit is None else min(args.limit, len(offsets) - 1)
    for i in range(n):
        roll = sequence_view(rolls_flat, offsets, i)
        audio, sr = synthesize_musicbox_roll(
            roll,
            step_sec=step_sec,
            note_min=note_min,
            sample_rate=args.sample_rate,
            representation=representation,
            max_events=args.max_events,
        )
        safe_id = str(ids[i]).replace("/", "_")
        path = args.out_dir / f"{i:04d}_{safe_id}.wav"
        save_wav(path, audio, sr)
        print(path)


if __name__ == "__main__":
    main()
