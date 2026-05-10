from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np


class PianoEventTokenizer:
    """Tokenizes sparse onset piano rolls into TIME_SHIFT and observed CHORD tokens."""

    PAD = 0
    BOS = 1
    EOS = 2
    UNK_CHORD = 3

    def __init__(
        self,
        max_shift: int = 64,
        num_notes: int = 88,
        note_min: int = 21,
        step_sec: float = 0.05,
    ) -> None:
        self.max_shift = int(max_shift)
        self.num_notes = int(num_notes)
        self.note_min = int(note_min)
        self.step_sec = float(step_sec)
        self.special_tokens = ["PAD", "BOS", "EOS", "UNK_CHORD"]
        self.time_shift_start = len(self.special_tokens)
        self.chord_start = self.time_shift_start + self.max_shift
        self.chord_to_id: dict[str, int] = {}
        self.id_to_chord: dict[int, list[int]] = {}
        self.chord_counts: dict[str, int] = {}
        self.event_density = 0.0
        self.stats: dict[str, float] = {}

    @property
    def vocab_size(self) -> int:
        return self.chord_start + len(self.chord_to_id)

    def time_shift_id(self, shift: int) -> int:
        if shift < 1 or shift > self.max_shift:
            raise ValueError(f"TIME_SHIFT must be in [1, {self.max_shift}], got {shift}")
        return self.time_shift_start + shift - 1

    def is_time_shift(self, token_id: int) -> bool:
        return self.time_shift_start <= int(token_id) < self.chord_start

    def time_shift_value(self, token_id: int) -> int:
        return int(token_id) - self.time_shift_start + 1

    def get_time_shift_value(self, token_id: int) -> int:
        return self.time_shift_value(token_id)

    def is_chord(self, token_id: int) -> bool:
        return int(token_id) >= self.chord_start

    def chord_token_ids(self) -> list[int]:
        return list(range(self.chord_start, self.vocab_size))

    def time_shift_token_ids(self) -> list[int]:
        return list(range(self.time_shift_start, self.chord_start))

    def decode_chord_token(self, token_id: int) -> np.ndarray:
        frame = np.zeros(self.num_notes, dtype=np.uint8)
        if self.is_chord(token_id):
            for note_idx in self.id_to_chord.get(int(token_id), []):
                if 0 <= note_idx < self.num_notes:
                    frame[note_idx] = 1
        return frame

    def chord_key_from_indices(self, indices: np.ndarray | list[int]) -> str:
        return ",".join(str(int(i)) for i in indices)

    def chord_key_from_frame(self, frame: np.ndarray) -> str:
        return self.chord_key_from_indices(np.flatnonzero(frame > 0))

    def _emit_shift(self, tokens: list[int], shift: int) -> None:
        while shift > 0:
            n = min(shift, self.max_shift)
            tokens.append(self.time_shift_id(n))
            shift -= n

    def fit_from_npz(
        self,
        train_npz: str | Path,
        cache_dir: str | Path | None = None,
        max_shift: int | None = None,
        min_chord_freq: int = 1,
        max_chord_vocab: int = 20000,
    ) -> "PianoEventTokenizer":
        if max_shift is not None and int(max_shift) != self.max_shift:
            self.__init__(max_shift=int(max_shift), num_notes=self.num_notes, note_min=self.note_min, step_sec=self.step_sec)

        data = np.load(train_npz, allow_pickle=True)
        if "rolls_flat" not in data.files or "offsets" not in data.files:
            raise KeyError("Expected train NPZ with rolls_flat and offsets.")
        rolls_flat = data["rolls_flat"]
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        self.num_notes = int(rolls_flat.shape[1])
        self.note_min = int(np.asarray(data["note_min"]).ravel()[0]) if "note_min" in data.files else 21
        self.step_sec = float(np.asarray(data["step_sec"]).ravel()[0]) if "step_sec" in data.files else 0.05

        counter: Counter[str] = Counter()
        total_steps = 0
        event_steps = 0
        for seq_idx in range(len(offsets) - 1):
            start = int(offsets[seq_idx])
            end = int(offsets[seq_idx + 1])
            seq = rolls_flat[start:end]
            total_steps += len(seq)
            nonzero = np.flatnonzero(np.asarray(seq).sum(axis=1) > 0)
            event_steps += int(len(nonzero))
            for t in nonzero:
                counter[self.chord_key_from_frame(seq[int(t)])] += 1

        for note_idx in range(self.num_notes):
            counter.setdefault(str(note_idx), 1)

        selected = [
            (key, count)
            for key, count in counter.most_common()
            if count >= int(min_chord_freq)
        ][: int(max_chord_vocab)]
        selected_keys = {key for key, _ in selected}
        for note_idx in range(self.num_notes):
            selected_keys.add(str(note_idx))

        ordered = sorted(selected_keys, key=lambda k: (-counter.get(k, 1), k))
        self.chord_to_id = {key: self.chord_start + i for i, key in enumerate(ordered)}
        self.id_to_chord = {
            token_id: [int(x) for x in key.split(",") if x != ""]
            for key, token_id in self.chord_to_id.items()
        }
        self.chord_counts = {key: int(counter.get(key, 0)) for key in ordered}
        self.event_density = float(event_steps / total_steps) if total_steps else 0.0

        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return self

    def encode_roll(self, roll: np.ndarray, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        roll = np.asarray(roll)
        if roll.ndim != 2 or roll.shape[1] != self.num_notes:
            raise ValueError(f"Expected roll shape [T, {self.num_notes}], got {roll.shape}")
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.BOS)
        silent = 0
        for frame in roll:
            if np.any(frame > 0):
                if silent:
                    self._emit_shift(tokens, silent)
                    silent = 0
                key = self.chord_key_from_frame(frame)
                if key in self.chord_to_id:
                    tokens.append(self.chord_to_id[key])
                else:
                    active = np.flatnonzero(frame > 0)
                    if len(active) == 1 and str(int(active[0])) in self.chord_to_id:
                        tokens.append(self.chord_to_id[str(int(active[0]))])
                    else:
                        tokens.append(self.UNK_CHORD)
            else:
                silent += 1
        if silent:
            self._emit_shift(tokens, silent)
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode_tokens(self, tokens: list[int], target_steps: Optional[int] = None) -> np.ndarray:
        frames: list[np.ndarray] = []
        pos = 0
        for token in tokens:
            token = int(token)
            if token in {self.PAD, self.BOS}:
                continue
            if token == self.EOS and target_steps is None:
                break
            if token == self.EOS:
                continue
            if self.is_time_shift(token):
                shift = self.time_shift_value(token)
                for _ in range(shift):
                    if target_steps is not None and pos >= target_steps:
                        break
                    frames.append(np.zeros(self.num_notes, dtype=np.uint8))
                    pos += 1
            elif self.is_chord(token):
                if target_steps is not None and pos >= target_steps:
                    break
                frame = np.zeros(self.num_notes, dtype=np.uint8)
                for note_idx in self.id_to_chord.get(token, []):
                    if 0 <= note_idx < self.num_notes:
                        frame[note_idx] = 1
                frames.append(frame)
                pos += 1
            elif token == self.UNK_CHORD:
                if target_steps is not None and pos >= target_steps:
                    break
                frames.append(np.zeros(self.num_notes, dtype=np.uint8))
                pos += 1

        if target_steps is not None:
            while len(frames) < target_steps:
                frames.append(np.zeros(self.num_notes, dtype=np.uint8))
            frames = frames[:target_steps]
        if not frames:
            return np.zeros((0 if target_steps is None else target_steps, self.num_notes), dtype=np.uint8)
        return np.stack(frames).astype(np.uint8, copy=False)

    def to_dict(self) -> dict:
        return {
            "max_shift": self.max_shift,
            "num_notes": self.num_notes,
            "note_min": self.note_min,
            "step_sec": self.step_sec,
            "special_tokens": self.special_tokens,
            "time_shift_start": self.time_shift_start,
            "chord_start": self.chord_start,
            "chord_to_id": self.chord_to_id,
            "chord_counts": self.chord_counts,
            "event_density": self.event_density,
            "stats": self.stats,
            "vocab_size": self.vocab_size,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PianoEventTokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls(
            max_shift=int(obj["max_shift"]),
            num_notes=int(obj.get("num_notes", 88)),
            note_min=int(obj.get("note_min", 21)),
            step_sec=float(obj.get("step_sec", 0.05)),
        )
        tok.special_tokens = list(obj.get("special_tokens", tok.special_tokens))
        tok.time_shift_start = int(obj.get("time_shift_start", len(tok.special_tokens)))
        tok.chord_start = int(obj.get("chord_start", tok.time_shift_start + tok.max_shift))
        tok.chord_to_id = {str(k): int(v) for k, v in obj["chord_to_id"].items()}
        tok.id_to_chord = {
            int(v): [int(x) for x in str(k).split(",") if x != ""]
            for k, v in tok.chord_to_id.items()
        }
        tok.chord_counts = {str(k): int(v) for k, v in obj.get("chord_counts", {}).items()}
        tok.event_density = float(obj.get("event_density", 0.0))
        tok.stats = {str(k): float(v) for k, v in obj.get("stats", {}).items()}
        return tok


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and save a piano event tokenizer.")
    parser.add_argument("--train_npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max_shift", type=int, default=64)
    parser.add_argument("--min_chord_freq", type=int, default=1)
    parser.add_argument("--max_chord_vocab", type=int, default=20000)
    args = parser.parse_args()
    tok = PianoEventTokenizer(max_shift=args.max_shift)
    tok.fit_from_npz(args.train_npz, min_chord_freq=args.min_chord_freq, max_chord_vocab=args.max_chord_vocab)
    tok.save(args.out)
    print(f"saved tokenizer: {args.out} vocab_size={tok.vocab_size} event_density={tok.event_density:.6f}")


if __name__ == "__main__":
    main()
