from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from vocab import load_rolls_flat_npz, pack_chord, scalar, sequence_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect rolls_flat/offsets dataset and event-state statistics.")
    parser.add_argument("--npz", type=Path, default=Path("../train.npz"))
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    rolls_flat, offsets, data = load_rolls_flat_npz(args.npz)
    lengths = np.diff(offsets)
    active = np.sum(rolls_flat > 0, axis=1) > 0
    note_counts = np.sum(rolls_flat > 0, axis=1)
    event_note_counts = note_counts[active]

    deltas: list[int] = []
    chord_counts: Counter[bytes] = Counter()
    seq_events: list[int] = []
    for i in range(len(offsets) - 1):
        roll = sequence_view(rolls_flat, offsets, i)
        ts = np.flatnonzero(np.sum(roll > 0, axis=1) > 0)
        seq_events.append(len(ts))
        prev = -1
        for t in ts:
            deltas.append(int(t) - prev)
            chord_counts[pack_chord(roll[int(t)])] += 1
            prev = int(t)

    print(f"path: {args.npz}")
    print(f"keys: {data.files}")
    print(f"rolls_flat: shape={rolls_flat.shape} dtype={rolls_flat.dtype}")
    print(f"num_sequences: {len(lengths)}")
    print(f"sequence_frames: min={int(lengths.min())} max={int(lengths.max())} mean={float(lengths.mean()):.2f} median={float(np.median(lengths)):.2f}")
    print(f"non_empty_frames: {int(active.sum())} / {len(active)} ({float(active.mean()):.4f})")
    print(f"unique_non_empty_chords: {len(chord_counts)}")
    if seq_events:
        arr = np.asarray(seq_events)
        print(f"events_per_sequence: min={int(arr.min())} max={int(arr.max())} mean={float(arr.mean()):.2f} median={float(np.median(arr)):.2f}")
    if deltas:
        arr = np.asarray(deltas)
        print(f"delta_t: min={int(arr.min())} max={int(arr.max())} mean={float(arr.mean()):.2f} p99={float(np.percentile(arr, 99)):.2f}")
        print(f"deltas_over_1200: {int(np.sum(arr > 1200))}")
    if len(event_note_counts):
        print(f"notes_per_event: min={int(event_note_counts.min())} max={int(event_note_counts.max())} mean={float(event_note_counts.mean()):.2f}")
    print(f"step_sec: {scalar(data, 'step_sec', 'missing')}")
    print(f"note_min: {scalar(data, 'note_min', 'missing')}")
    print(f"note_max: {scalar(data, 'note_max', 'missing')}")
    print(f"num_positions: {scalar(data, 'num_positions', 'missing')}")
    print(f"representation: {scalar(data, 'representation', 'missing')}")
    print("top_chord_frequencies:")
    for rank, (_, count) in enumerate(chord_counts.most_common(args.top), start=1):
        print(f"  {rank}: {count}")


if __name__ == "__main__":
    main()
