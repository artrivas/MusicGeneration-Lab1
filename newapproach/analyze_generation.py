from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils_npz import avg_notes_per_event, event_density, load_npz_sequences, longest_silent_run, top_patterns


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

    gen_rolls, _ = load_npz_sequences(args.npz)
    prefix_rolls, _ = load_npz_sequences(args.prefix_npz)
    full_rolls = load_npz_sequences(args.full_npz)[0] if args.full_npz else None

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
        print(f"  top_10_generated_patterns: {top_patterns(continuation, 10)}")
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
