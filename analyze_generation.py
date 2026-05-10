from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from tokenizer import PianoEventTokenizer


def get_sequence(data: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    offsets = np.asarray(data["offsets"], dtype=np.int64)
    return np.asarray(data["rolls_flat"][offsets[index] : offsets[index + 1]], dtype=np.uint8)


def chord_key(frame: np.ndarray) -> str:
    active = np.flatnonzero(frame > 0)
    if len(active) == 0:
        return "silence"
    return ",".join(str(int(x)) for x in active)


def longest_silent_run(roll: np.ndarray) -> int:
    best = 0
    cur = 0
    for is_silent in (roll.sum(axis=1) == 0):
        if is_silent:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def event_density(roll: np.ndarray) -> float:
    if len(roll) == 0:
        return 0.0
    return float((roll.sum(axis=1) > 0).sum() / len(roll))


def unique_chords(roll: np.ndarray) -> int:
    event_frames = roll[roll.sum(axis=1) > 0]
    return len({chord_key(frame) for frame in event_frames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated piano-roll NPZ for silence or noisy density.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--prefix_npz", type=Path, default=None)
    parser.add_argument("--full_npz", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--training_event_density", type=float, default=None)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    prefix_data = np.load(args.prefix_npz, allow_pickle=True) if args.prefix_npz is not None else None
    full_data = np.load(args.full_npz, allow_pickle=True) if args.full_npz is not None else None
    training_density = args.training_event_density
    if args.tokenizer is not None:
        training_density = PianoEventTokenizer.load(args.tokenizer).event_density

    offsets = np.asarray(data["offsets"], dtype=np.int64)
    n_seq = len(offsets) - 1
    for i in range(n_seq):
        roll = get_sequence(data, i)
        if prefix_data is not None:
            prefix_steps = len(get_sequence(prefix_data, i))
        elif "prefix_steps" in data.files:
            prefix_arr = np.asarray(data["prefix_steps"]).ravel()
            prefix_steps = int(prefix_arr[i] if len(prefix_arr) > 1 else prefix_arr[0])
        else:
            prefix_steps = 0
        generated = roll[prefix_steps:]
        prefix = roll[:prefix_steps]
        generated_note_count = int(generated.sum())
        event_frames = generated[generated.sum(axis=1) > 0]
        event_density = float(len(event_frames) / max(1, len(generated)))
        patterns = Counter(chord_key(frame) for frame in event_frames)

        print(f"sequence {i}")
        print(f"  total_steps: {len(roll)}")
        print(f"  prefix_steps: {prefix_steps}")
        print(f"  generated_steps: {len(generated)}")
        print(f"  prefix_note_count: {int(prefix.sum())}")
        print(f"  generated_note_count: {generated_note_count}")
        print(f"  generated_event_density: {event_density:.6f}")
        print(f"  unique_generated_chord_patterns: {len(patterns)}")
        print(f"  longest_silent_run: {longest_silent_run(generated)}")
        print("  top_10_generated_chord_patterns:")
        for key, count in patterns.most_common(10):
            print(f"    {key}: {count}")
        if generated_note_count == 0:
            print("  WARNING: generated continuation is completely silent.")
        if training_density is not None and event_density > training_density * 4.0:
            print(
                "  WARNING: generated event density is much higher than training average "
                f"({event_density:.6f} vs {training_density:.6f})."
            )
        elif training_density is None and event_density > 0.2:
            print("  WARNING: generated event density is high for sparse onset data.")
        if full_data is not None:
            full_roll = get_sequence(full_data, i)
            real = full_roll[prefix_steps : prefix_steps + len(generated)]
            print("  full_file_comparison:")
            print(f"    real_event_density: {event_density(real):.6f}")
            print(f"    generated_event_density: {event_density(generated):.6f}")
            print(f"    real_longest_silent_run: {longest_silent_run(real)}")
            print(f"    generated_longest_silent_run: {longest_silent_run(generated)}")
            print(f"    real_unique_chords: {unique_chords(real)}")
            print(f"    generated_unique_chords: {unique_chords(generated)}")


if __name__ == "__main__":
    main()
