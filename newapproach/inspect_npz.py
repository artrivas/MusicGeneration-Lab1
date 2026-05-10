from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils_npz import avg_notes_per_event, event_density, get_dataset_defaults, load_npz_sequences, longest_silent_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect symbolic piano-roll .npz files.")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    rolls, data = load_npz_sequences(args.npz)
    print(f"path: {args.npz}")
    print(f"keys: {data.files}")
    print(f"defaults: {get_dataset_defaults(data)}")
    print(f"num_sequences: {len(rolls)}")
    for i, roll in enumerate(rolls[: args.limit]):
        print(
            f"{i}: shape={roll.shape} notes={int(np.sum(roll > 0))} "
            f"density={event_density(roll):.4f} avg_notes/event={avg_notes_per_event(roll):.3f} "
            f"longest_silence={longest_silent_run(roll)}"
        )


if __name__ == "__main__":
    main()
