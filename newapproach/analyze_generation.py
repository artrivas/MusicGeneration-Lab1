from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from utils_npz import avg_notes_per_event, event_density, get_dataset_defaults, load_npz_sequences, longest_silent_run


def event_pattern_list(roll: np.ndarray, note_min: int = 0) -> list[str]:
    patterns: list[str] = []
    for frame in np.asarray(roll):
        active = np.flatnonzero(frame > 0)
        if len(active):
            patterns.append("-".join(str(int(note_min + i)) for i in active))
    return patterns


def intervals_for_pattern(pattern: str) -> list[int]:
    if not pattern:
        return []
    notes = [int(x) for x in pattern.split("-") if x != ""]
    intervals: list[int] = []
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            intervals.append(abs(notes[j] - notes[i]) % 12)
    return intervals


def average_run_length(patterns: list[str]) -> float:
    if not patterns:
        return 0.0
    runs: list[int] = []
    cur = 1
    for prev, now in zip(patterns, patterns[1:]):
        if now == prev:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def repetition_metrics(roll: np.ndarray, note_min: int = 0) -> dict:
    patterns = event_pattern_list(roll, note_min=note_min)
    pattern_counts = Counter(patterns)
    event_count = max(1, len(patterns))
    note_counts: Counter[int] = Counter()
    interval_counts: Counter[int] = Counter()
    for pattern in patterns:
        notes = [int(x) for x in pattern.split("-") if x != ""]
        note_counts.update(notes)
        interval_counts.update(intervals_for_pattern(pattern))
    note_total = max(1, sum(note_counts.values()))
    interval_total = max(1, sum(interval_counts.values()))
    max_pattern_count = max(pattern_counts.values(), default=0)
    max_note_count = max(note_counts.values(), default=0)
    max_interval_count = max(interval_counts.values(), default=0)
    repeated_pattern_events = sum(count for count in pattern_counts.values() if count > 1)
    return {
        "top_10_generated_patterns": pattern_counts.most_common(10),
        "max_pattern_count": int(max_pattern_count),
        "max_pattern_share": float(max_pattern_count / event_count),
        "repeated_pattern_ratio": float(repeated_pattern_events / event_count),
        "top_10_note_pitches": note_counts.most_common(10),
        "max_note_pitch_count": int(max_note_count),
        "max_note_pitch_share": float(max_note_count / note_total),
        "top_10_intervals": interval_counts.most_common(10),
        "max_interval_share": float(max_interval_count / interval_total),
        "average_run_length_of_same_pattern": average_run_length(patterns),
    }


def repetition_warnings(metrics: dict) -> list[str]:
    warnings: list[str] = []
    if metrics["max_pattern_share"] > 0.15:
        warnings.append("max_pattern_share > 0.15: one event pattern dominates locally.")
    if metrics["max_note_pitch_share"] > 0.20:
        warnings.append("max_note_pitch_share > 0.20: one note pitch is overused.")
    if metrics["max_interval_share"] > 0.35:
        warnings.append("max_interval_share > 0.35: one interval class dominates.")
    if metrics["average_run_length_of_same_pattern"] > 3.0:
        warnings.append("average_run_length_of_same_pattern is high: adjacent event loops are likely audible.")
    return warnings


def summarize(label: str, roll: np.ndarray) -> dict:
    return {
        f"{label}_note_count": int(np.sum(roll > 0)),
        f"{label}_event_density": event_density(roll),
        f"{label}_average_notes_per_event": avg_notes_per_event(roll),
        f"{label}_unique_patterns": int(len({tuple(np.flatnonzero(f > 0).tolist()) for f in roll if np.any(f > 0)})),
        f"{label}_longest_silent_run": longest_silent_run(roll),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated piano-roll continuations.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--prefix_npz", type=Path, required=True)
    parser.add_argument("--full_npz", type=Path)
    args = parser.parse_args()

    gen_rolls, gen_data = load_npz_sequences(args.npz)
    prefix_rolls, _ = load_npz_sequences(args.prefix_npz)
    full_rolls = load_npz_sequences(args.full_npz)[0] if args.full_npz else None
    note_min = int(get_dataset_defaults(gen_data).get("note_min", 0))

    for i, gen in enumerate(gen_rolls):
        prefix = prefix_rolls[i]
        prefix_steps = len(prefix)
        continuation = gen[prefix_steps:]
        print(f"sequence {i}")
        print(f"  total_steps: {len(gen)}")
        print(f"  prefix_steps: {prefix_steps}")
        print(f"  generated_steps: {len(continuation)}")
        base = summarize("prefix", prefix)
        cont = summarize("generated", continuation)
        for key, value in {**base, **cont}.items():
            print(f"  {key}: {value}")
        metrics = repetition_metrics(continuation, note_min=note_min)
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        warnings = repetition_warnings(metrics)
        if warnings:
            print("  warnings:")
            for warning in warnings:
                print(f"    - {warning}")
        if full_rolls is not None:
            real_cont = full_rolls[i][prefix_steps : len(gen)]
            real = summarize("real", real_cont)
            print("  comparison_to_real:")
            for key in [
                "real_event_density",
                "generated_event_density",
                "real_average_notes_per_event",
                "generated_average_notes_per_event",
                "real_longest_silent_run",
                "generated_longest_silent_run",
                "real_unique_patterns",
                "generated_unique_patterns",
            ]:
                value = real.get(key, cont.get(key))
                print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
