from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tokenizer import PianoEventTokenizer


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.ravel()[0]
    return value.item() if isinstance(value, np.generic) else value


def prepare_token_cache(
    train_npz: Path,
    cache_dir: Path,
    max_shift: int = 64,
    min_chord_freq: int = 1,
    max_chord_vocab: int = 20000,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache_dir / "tokenizer.json"
    tokens_path = cache_dir / "tokens_flat.npy"
    offsets_path = cache_dir / "token_offsets.npy"
    ids_path = cache_dir / "ids.npy"
    info_path = cache_dir / "dataset_info.json"
    if tokenizer_path.exists() and tokens_path.exists() and offsets_path.exists() and info_path.exists():
        print(f"token cache already exists: {cache_dir}")
        return

    data = np.load(train_npz, allow_pickle=True)
    if "rolls_flat" not in data.files or "offsets" not in data.files:
        raise KeyError("Expected train NPZ to contain rolls_flat and offsets.")

    tokenizer = PianoEventTokenizer(max_shift=max_shift)
    tokenizer.fit_from_npz(
        train_npz=train_npz,
        cache_dir=cache_dir,
        max_shift=max_shift,
        min_chord_freq=min_chord_freq,
        max_chord_vocab=max_chord_vocab,
    )
    tokenizer.save(tokenizer_path)

    rolls_flat = data["rolls_flat"]
    offsets = np.asarray(data["offsets"], dtype=np.int64)
    ids = np.asarray(data["ids"]) if "ids" in data.files else np.arange(len(offsets) - 1, dtype=np.int64)
    token_offsets = [0]
    token_sequences: list[np.ndarray] = []
    time_shift_count = 0
    chord_count = 0
    unk_count = 0

    try:
        for seq_idx in range(len(offsets) - 1):
            start = int(offsets[seq_idx])
            end = int(offsets[seq_idx + 1])
            tokens = tokenizer.encode_roll(rolls_flat[start:end], add_bos=True, add_eos=True)
            arr = np.asarray(tokens, dtype=np.int32)
            token_sequences.append(arr)
            token_offsets.append(token_offsets[-1] + len(arr))
            time_shift_count += sum(1 for x in tokens if tokenizer.is_time_shift(x))
            chord_count += sum(1 for x in tokens if tokenizer.is_chord(x))
            unk_count += sum(1 for x in tokens if x == tokenizer.UNK_CHORD)
    except MemoryError as exc:
        raise MemoryError(
            "Not enough RAM while building token cache. Try a larger machine or reduce max_chord_vocab."
        ) from exc

    tokens_flat = np.concatenate(token_sequences).astype(np.int32, copy=False)
    np.save(tokens_path, tokens_flat)
    np.save(offsets_path, np.asarray(token_offsets, dtype=np.int64))
    np.save(ids_path, ids)

    info = {
        "source_npz": str(train_npz),
        "num_sequences": int(len(offsets) - 1),
        "total_frames": int(offsets[-1]) if len(offsets) else 0,
        "total_tokens": int(len(tokens_flat)),
        "vocab_size": int(tokenizer.vocab_size),
        "max_shift": int(max_shift),
        "min_chord_freq": int(min_chord_freq),
        "max_chord_vocab": int(max_chord_vocab),
        "event_density": float(tokenizer.event_density),
        "time_shift_token_count": int(time_shift_count),
        "chord_token_count": int(chord_count),
        "unk_chord_token_count": int(unk_count),
        "step_sec": scalar(data, "step_sec", 0.05),
        "note_min": scalar(data, "note_min", 21),
        "note_max": scalar(data, "note_max", 108),
        "num_positions": scalar(data, "num_positions", int(rolls_flat.shape[1])),
        "representation": str(scalar(data, "representation", "onset")),
    }
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"saved token cache: {cache_dir}")
    print(f"vocab_size={tokenizer.vocab_size} total_tokens={len(tokens_flat)} event_density={tokenizer.event_density:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build token vocabulary and tokenized training cache.")
    parser.add_argument("--train_npz", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--max_shift", type=int, default=64)
    parser.add_argument("--min_chord_freq", type=int, default=1)
    parser.add_argument("--max_chord_vocab", type=int, default=20000)
    args = parser.parse_args()
    prepare_token_cache(
        train_npz=args.train_npz,
        cache_dir=args.cache_dir,
        max_shift=args.max_shift,
        min_chord_freq=args.min_chord_freq,
        max_chord_vocab=args.max_chord_vocab,
    )


if __name__ == "__main__":
    main()
