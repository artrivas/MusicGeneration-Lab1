from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tokenizer_mirexlike import MirexLikePianoTokenizer
from utils_npz import avg_notes_per_event, event_density, get_dataset_defaults, load_npz_sequences, longest_silent_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode piano-roll .npz data into compact event tokens.")
    parser.add_argument("--train_npz", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--max_shift", type=int, default=64)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rolls, data = load_npz_sequences(args.train_npz)
    defaults = get_dataset_defaults(data)
    tokenizer = MirexLikePianoTokenizer(
        max_shift=args.max_shift,
        num_notes=int(defaults["num_positions"]),
        note_min=int(defaults["note_min"]),
        note_max=int(defaults["note_max"]),
        step_sec=float(defaults["step_sec"]),
    )

    token_chunks: list[np.ndarray] = []
    token_offsets = np.zeros(len(rolls) + 1, dtype=np.int64)
    total_tokens = 0
    total_frames = 0
    note_count = time_count = chord_count = 0
    densities: list[float] = []
    avg_notes: list[float] = []
    silent_runs: list[int] = []

    for i, roll in enumerate(rolls):
        tokens = np.asarray(tokenizer.encode_roll(roll, add_bos_eos=True), dtype=np.int32)
        token_chunks.append(tokens)
        total_tokens += int(len(tokens))
        token_offsets[i + 1] = total_tokens
        total_frames += int(len(roll))
        note_count += sum(1 for t in tokens if tokenizer.is_note_on(int(t)))
        time_count += sum(1 for t in tokens if tokenizer.is_time_shift(int(t)))
        chord_count += sum(1 for t in tokens if tokenizer.is_chord_end(int(t)))
        densities.append(event_density(roll))
        avg_notes.append(avg_notes_per_event(roll))
        silent_runs.append(longest_silent_run(roll))
        if (i + 1) % 50 == 0 or i + 1 == len(rolls):
            print(f"encoded {i + 1}/{len(rolls)} sequences, total_tokens={total_tokens}")

    tokens_flat = np.concatenate(token_chunks).astype(np.int32, copy=False) if token_chunks else np.zeros(0, dtype=np.int32)
    ids = np.asarray(data["ids"]) if "ids" in data.files else np.asarray([f"seq_{i:04d}" for i in range(len(rolls))])

    tokenizer.save(args.cache_dir / "tokenizer.json")
    np.save(args.cache_dir / "tokens_flat.npy", tokens_flat)
    np.save(args.cache_dir / "token_offsets.npy", token_offsets)
    np.save(args.cache_dir / "ids.npy", ids)

    total_non_special = max(1, total_tokens)
    info = {
        "num_sequences": len(rolls),
        "total_frames": total_frames,
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.vocab_size,
        "avg_tokens_per_sequence": float(total_tokens / max(1, len(rolls))),
        "avg_frames_per_sequence": float(total_frames / max(1, len(rolls))),
        "train_event_density": float(np.mean(densities)) if densities else 0.0,
        "train_avg_notes_per_event": float(np.mean(avg_notes)) if avg_notes else 0.0,
        "train_longest_silent_run_mean": float(np.mean(silent_runs)) if silent_runs else 0.0,
        "train_longest_silent_run_p95": float(np.percentile(silent_runs, 95)) if silent_runs else 0.0,
        "note_token_ratio": float(note_count / total_non_special),
        "time_shift_token_ratio": float(time_count / total_non_special),
        "chord_end_ratio": float(chord_count / total_non_special),
        "step_sec": defaults["step_sec"],
        "note_min": defaults["note_min"],
        "note_max": defaults["note_max"],
        "num_positions": defaults["num_positions"],
        "representation": defaults["representation"],
        "max_shift": args.max_shift,
    }
    with (args.cache_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"saved cache to {args.cache_dir} vocab_size={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
