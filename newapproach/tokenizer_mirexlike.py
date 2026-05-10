from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class MirexLikePianoTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    CHORD_END = 3

    def __init__(
        self,
        max_shift: int = 64,
        num_notes: int = 88,
        note_min: int = 21,
        note_max: int = 108,
        step_sec: float = 0.05,
    ) -> None:
        self.max_shift = int(max_shift)
        self.num_notes = int(num_notes)
        self.note_min = int(note_min)
        self.note_max = int(note_max)
        self.step_sec = float(step_sec)
        self.special_tokens = ["PAD", "BOS", "EOS", "CHORD_END"]
        self.time_shift_start = len(self.special_tokens)
        self.note_on_start = self.time_shift_start + self.max_shift
        self.vocab_size = self.note_on_start + self.num_notes

    def time_shift_id(self, shift: int) -> int:
        shift = int(shift)
        if shift < 1 or shift > self.max_shift:
            raise ValueError(f"TIME_SHIFT must be in [1, {self.max_shift}], got {shift}")
        return self.time_shift_start + shift - 1

    def note_on_id(self, note_index: int) -> int:
        note_index = int(note_index)
        if note_index < 0 or note_index >= self.num_notes:
            raise ValueError(f"note index must be in [0, {self.num_notes}), got {note_index}")
        return self.note_on_start + note_index

    def _emit_shift(self, tokens: list[int], shift: int) -> None:
        while shift > 0:
            n = min(int(shift), self.max_shift)
            tokens.append(self.time_shift_id(n))
            shift -= n

    def encode_roll(self, roll: np.ndarray, add_bos_eos: bool = True) -> list[int]:
        roll = np.asarray(roll)
        if roll.ndim != 2 or roll.shape[1] != self.num_notes:
            raise ValueError(f"Expected roll shape [T, {self.num_notes}], got {roll.shape}")
        tokens: list[int] = []
        if add_bos_eos:
            tokens.append(self.BOS)
        silent = 0
        for frame in roll:
            active = np.flatnonzero(frame > 0)
            if len(active) == 0:
                silent += 1
                continue
            if silent:
                self._emit_shift(tokens, silent)
                silent = 0
            for note_idx in active:
                if 0 <= int(note_idx) < self.num_notes:
                    tokens.append(self.note_on_id(int(note_idx)))
            tokens.append(self.CHORD_END)
        if silent:
            self._emit_shift(tokens, silent)
        if add_bos_eos:
            tokens.append(self.EOS)
        return tokens

    def decode_tokens(self, tokens: list[int] | np.ndarray, target_steps: Optional[int] = None) -> np.ndarray:
        frames: list[np.ndarray] = []
        pos = 0
        open_notes: set[int] = set()
        tokens_since_chord_start = 0
        max_unclosed_notes = 32

        def append_silence(n: int) -> None:
            nonlocal pos
            for _ in range(int(n)):
                if target_steps is not None and pos >= target_steps:
                    return
                frames.append(np.zeros(self.num_notes, dtype=np.uint8))
                pos += 1

        def flush_chord(advance_empty: bool = False) -> None:
            nonlocal pos, open_notes, tokens_since_chord_start
            if target_steps is not None and pos >= target_steps:
                open_notes = set()
                tokens_since_chord_start = 0
                return
            if open_notes or advance_empty:
                frame = np.zeros(self.num_notes, dtype=np.uint8)
                for note_idx in open_notes:
                    if 0 <= note_idx < self.num_notes:
                        frame[note_idx] = 1
                frames.append(frame)
                pos += 1
            open_notes = set()
            tokens_since_chord_start = 0

        for raw in tokens:
            token = int(raw)
            if token in {self.PAD, self.BOS}:
                continue
            if token == self.EOS:
                if target_steps is None:
                    break
                continue
            if self.is_time_shift(token):
                if open_notes:
                    flush_chord()
                append_silence(self.get_time_shift_value(token))
            elif self.is_note_on(token):
                note_idx = self.get_note_pitch(token) - self.note_min
                if 0 <= note_idx < self.num_notes:
                    open_notes.add(note_idx)
                    tokens_since_chord_start += 1
                if tokens_since_chord_start >= max_unclosed_notes:
                    flush_chord()
            elif self.is_chord_end(token):
                flush_chord(advance_empty=False)
            if target_steps is not None and pos >= target_steps:
                break
        if open_notes and (target_steps is None or pos < target_steps):
            flush_chord()
        if target_steps is not None:
            append_silence(max(0, int(target_steps) - len(frames)))
            frames = frames[: int(target_steps)]
        if not frames:
            return np.zeros((0 if target_steps is None else int(target_steps), self.num_notes), dtype=np.uint8)
        return np.stack(frames).astype(np.uint8, copy=False)

    def token_to_string(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id == self.PAD:
            return "PAD"
        if token_id == self.BOS:
            return "BOS"
        if token_id == self.EOS:
            return "EOS"
        if token_id == self.CHORD_END:
            return "CHORD_END"
        if self.is_time_shift(token_id):
            return f"TIME_SHIFT_{self.get_time_shift_value(token_id)}"
        if self.is_note_on(token_id):
            return f"NOTE_ON_{self.get_note_pitch(token_id)}"
        return f"UNK_{token_id}"

    def string_to_token(self, token_str: str) -> int:
        s = str(token_str).strip().upper()
        if s in {"PAD", "BOS", "EOS", "CHORD_END"}:
            return getattr(self, s)
        if s.startswith("TIME_SHIFT_"):
            return self.time_shift_id(int(s.split("_")[-1]))
        if s.startswith("NOTE_ON_"):
            midi = int(s.split("_")[-1])
            return self.note_on_id(midi - self.note_min)
        raise KeyError(f"Unknown token string: {token_str}")

    def is_time_shift(self, token_id: int) -> bool:
        return self.time_shift_start <= int(token_id) < self.note_on_start

    def is_note_on(self, token_id: int) -> bool:
        return self.note_on_start <= int(token_id) < self.vocab_size

    def is_chord_end(self, token_id: int) -> bool:
        return int(token_id) == self.CHORD_END

    def get_time_shift_value(self, token_id: int) -> int:
        return int(token_id) - self.time_shift_start + 1

    def get_note_pitch(self, token_id: int) -> int:
        return self.note_min + int(token_id) - self.note_on_start

    def time_shift_token_ids(self) -> list[int]:
        return list(range(self.time_shift_start, self.note_on_start))

    def note_on_token_ids(self) -> list[int]:
        return list(range(self.note_on_start, self.vocab_size))

    def to_dict(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "max_shift": self.max_shift,
            "num_notes": self.num_notes,
            "note_min": self.note_min,
            "note_max": self.note_max,
            "step_sec": self.step_sec,
            "special_tokens": self.special_tokens,
            "time_shift_start": self.time_shift_start,
            "note_on_start": self.note_on_start,
            "vocab_size": self.vocab_size,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MirexLikePianoTokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(
            max_shift=int(obj.get("max_shift", 64)),
            num_notes=int(obj.get("num_notes", 88)),
            note_min=int(obj.get("note_min", 21)),
            note_max=int(obj.get("note_max", 108)),
            step_sec=float(obj.get("step_sec", 0.05)),
        )
