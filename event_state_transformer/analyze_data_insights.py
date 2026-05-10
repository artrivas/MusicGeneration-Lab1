from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


CANDIDATE_TOP_K = [128, 256, 512, 1000, 2000, 5000, 10000, 20000, 50000]
DELTA_THRESHOLDS = [64, 128, 256, 512, 1200]
SHORT_SECONDS = [5, 10, 20, 30, 60]


def scalar(data: Any, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.shape == ():
        value = arr.item()
    elif arr.size:
        value = arr.ravel()[0]
        if isinstance(value, np.generic):
            value = value.item()
    else:
        value = default
    return value


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return as_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "n/a"
        return f"{float(value):,.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def pct(part: float, total: float) -> float:
    return 100.0 * float(part) / float(total) if total else 0.0


def describe_array(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values)
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p99_5": None,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p99_5": float(np.percentile(values, 99.5)),
    }


def describe_for_markdown(stats: dict[str, Any], fields: list[str]) -> str:
    return " | ".join(fmt_num(stats.get(field)) for field in fields)


def load_npz(path: Path) -> Any:
    return np.load(str(path), allow_pickle=True)


def npz_overview(path: Path, data: Any) -> dict[str, Any]:
    keys = {}
    for key in data.files:
        arr = np.asarray(data[key])
        item = None
        if arr.shape == ():
            try:
                item = arr.item()
            except Exception:
                item = None
        keys[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "scalar_value": as_jsonable(item) if item is not None else None,
        }
    return {"path": str(path), "keys": keys}


def require_rolls_flat(data: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    if "rolls_flat" not in data.files or "offsets" not in data.files:
        raise KeyError(f"{name} must contain rolls_flat and offsets")
    rolls_flat = np.asarray(data["rolls_flat"], dtype=np.uint8)
    offsets = np.asarray(data["offsets"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 1:
        raise ValueError(f"{name}: offsets must be a non-empty 1D array")
    if len(offsets) > 1 and (offsets[0] != 0 or np.any(np.diff(offsets) < 0)):
        raise ValueError(f"{name}: offsets must start at 0 and be monotonically nondecreasing")
    if len(offsets) and int(offsets[-1]) > len(rolls_flat):
        raise ValueError(f"{name}: offsets[-1] exceeds rolls_flat length")
    return rolls_flat, offsets


def sequence_view(rolls_flat: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    return rolls_flat[int(offsets[index]) : int(offsets[index + 1])]


def sequence_lengths(offsets: np.ndarray) -> np.ndarray:
    return np.diff(offsets).astype(np.int64, copy=False)


def pack_chord(frame: np.ndarray) -> bytes:
    return np.packbits(np.asarray(frame, dtype=np.uint8), bitorder="big").tobytes()


def chord_label(key: bytes, note_min: int, num_notes: int) -> str:
    bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8), bitorder="big")[:num_notes]
    idx = np.flatnonzero(bits > 0)
    pitches = [int(note_min + i) for i in idx]
    return "[" + ",".join(str(p) for p in pitches) + "]"


def split_sequence_indices(num_sequences: int, val_frac: float = 0.1, seed: int = 42) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_sequences)
    rng.shuffle(indices)
    n_val = max(1, int(round(num_sequences * val_frac))) if num_sequences > 1 else 0
    val = sorted(indices[:n_val].astype(int).tolist())
    train = sorted(indices[n_val:].astype(int).tolist())
    return train, val


def active_mask_for_rolls(rolls_flat: np.ndarray, num_notes: int) -> np.ndarray:
    return np.sum(rolls_flat[:, :num_notes] > 0, axis=1) > 0


def per_sequence_activity(
    rolls_flat: np.ndarray, offsets: np.ndarray, num_notes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = sequence_lengths(offsets)
    active_counts = np.zeros(len(lengths), dtype=np.int64)
    deltas: list[np.ndarray] = []
    note_counts_all: list[np.ndarray] = []
    note_frequency = np.zeros(num_notes, dtype=np.int64)

    for i in range(len(lengths)):
        roll = sequence_view(rolls_flat, offsets, i)[:, :num_notes]
        note_counts = np.sum(roll > 0, axis=1).astype(np.int16, copy=False)
        active_ts = np.flatnonzero(note_counts > 0)
        active_counts[i] = int(active_ts.size)
        if active_ts.size:
            note_counts_all.append(note_counts[active_ts])
            note_frequency += np.sum(roll[active_ts] > 0, axis=0, dtype=np.int64)
            prev = np.concatenate((np.array([-1], dtype=np.int64), active_ts[:-1]))
            deltas.append((active_ts.astype(np.int64) - prev).astype(np.int64, copy=False))

    all_deltas = np.concatenate(deltas) if deltas else np.array([], dtype=np.int64)
    active_note_counts = np.concatenate(note_counts_all) if note_counts_all else np.array([], dtype=np.int16)
    return active_counts, all_deltas, active_note_counts, note_frequency


def build_chord_counter(rolls_flat: np.ndarray, offsets: np.ndarray, num_notes: int) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for i in range(len(offsets) - 1):
        roll = sequence_view(rolls_flat, offsets, i)[:, :num_notes]
        active_rows = roll[np.sum(roll > 0, axis=1) > 0]
        for frame in active_rows:
            counts[pack_chord(frame)] += 1
    return counts


def chord_oov_rate(rolls_flat: np.ndarray, offsets: np.ndarray, top_keys: set[bytes], num_notes: int) -> dict[str, Any]:
    total = 0
    oov = 0
    for i in range(len(offsets) - 1):
        roll = sequence_view(rolls_flat, offsets, i)[:, :num_notes]
        active_rows = roll[np.sum(roll > 0, axis=1) > 0]
        total += int(len(active_rows))
        for frame in active_rows:
            if pack_chord(frame) not in top_keys:
                oov += 1
    return {"events": total, "oov_events": oov, "oov_pct": pct(oov, total), "coverage_pct": 100.0 - pct(oov, total)}


def coverage_table(counts: Counter[bytes], top_ks: list[int]) -> dict[str, Any]:
    total = sum(counts.values())
    ordered = counts.most_common()
    cumulative = np.cumsum([c for _, c in ordered], dtype=np.int64) if ordered else np.array([], dtype=np.int64)
    rows = []
    for k in top_ks:
        effective = min(k, len(ordered))
        covered = int(cumulative[effective - 1]) if effective else 0
        rows.append(
            {
                "k": int(k),
                "effective_vocab_size": int(effective),
                "covered_events": covered,
                "coverage_pct": pct(covered, total),
                "oov_pct": 100.0 - pct(covered, total),
            }
        )
    return {"total_events": int(total), "unique_chords": int(len(counts)), "rows": rows}


def entropy_from_counts(counts: Counter[bytes]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {"bits": 0.0, "nats": 0.0}
    probs = np.asarray(list(counts.values()), dtype=np.float64) / float(total)
    nats = float(-np.sum(probs * np.log(probs)))
    return {"bits": float(nats / math.log(2.0)), "nats": nats}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_hist(path: Path, values: np.ndarray, title: str, xlabel: str, bins: int = 80, clip_p99: bool = False) -> bool:
    if plt is None or values.size == 0:
        return False
    plot_values = np.asarray(values)
    if clip_p99 and plot_values.size:
        upper = np.percentile(plot_values, 99)
        plot_values = plot_values[plot_values <= upper]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(plot_values, bins=bins, color="#3b82f6", edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def save_line(path: Path, x: list[float] | np.ndarray, y: list[float] | np.ndarray, title: str, xlabel: str, ylabel: str) -> bool:
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o", color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def save_bar(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> bool:
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, y, color="#0f766e")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def basic_roll_stats(data: Any, rolls_flat: np.ndarray, offsets: np.ndarray, name: str) -> dict[str, Any]:
    num_notes = int(scalar(data, "num_positions", rolls_flat.shape[1] if rolls_flat.ndim == 2 else 88) or 88)
    num_notes = min(num_notes, rolls_flat.shape[1])
    step_sec = float(scalar(data, "step_sec", 0.05) or 0.05)
    lengths = sequence_lengths(offsets)
    active_counts, deltas, active_note_counts, note_frequency = per_sequence_activity(rolls_flat, offsets, num_notes)
    total_frames = int(np.sum(lengths))
    total_active = int(np.sum(active_counts))
    total_empty = total_frames - total_active
    return {
        "name": name,
        "num_sequences": int(len(lengths)),
        "step_sec": step_sec,
        "note_min": int(scalar(data, "note_min", 21) or 21),
        "note_max": int(scalar(data, "note_max", 108) or 108),
        "num_positions": num_notes,
        "representation": str(scalar(data, "representation", "unknown")),
        "total_frames": total_frames,
        "total_active_frames": total_active,
        "total_empty_frames": int(total_empty),
        "empty_frame_pct": pct(total_empty, total_frames),
        "active_frame_ratio": float(total_active / total_frames) if total_frames else 0.0,
        "avg_notes_per_frame": float(np.sum(note_frequency) / total_frames) if total_frames else 0.0,
        "avg_notes_per_active_frame": float(np.mean(active_note_counts)) if active_note_counts.size else 0.0,
        "lengths_frames": lengths,
        "active_counts": active_counts,
        "deltas": deltas,
        "active_note_counts": active_note_counts,
        "note_frequency": note_frequency,
    }


def split_diagnostics(
    rolls_flat: np.ndarray,
    offsets: np.ndarray,
    num_notes: int,
    chord_counts_by_split: dict[str, Counter[bytes]],
) -> dict[str, Any]:
    lengths = sequence_lengths(offsets)
    train_idx, val_idx = split_sequence_indices(len(lengths), val_frac=0.1, seed=42)

    def subset(indices: list[int]) -> dict[str, Any]:
        idx = np.asarray(indices, dtype=np.int64)
        active_counts = []
        deltas = []
        note_total = 0
        frame_total = 0
        for i in idx:
            roll = sequence_view(rolls_flat, offsets, int(i))[:, :num_notes]
            note_counts = np.sum(roll > 0, axis=1)
            active_ts = np.flatnonzero(note_counts > 0)
            active_counts.append(int(active_ts.size))
            frame_total += int(len(roll))
            note_total += int(np.sum(note_counts[active_ts])) if active_ts.size else 0
            if active_ts.size:
                prev = np.concatenate((np.array([-1], dtype=np.int64), active_ts[:-1]))
                deltas.append(active_ts.astype(np.int64) - prev)
        active_arr = np.asarray(active_counts, dtype=np.int64)
        delta_arr = np.concatenate(deltas) if deltas else np.array([], dtype=np.int64)
        active_total = int(np.sum(active_arr))
        return {
            "num_sequences": int(len(indices)),
            "lengths_frames": describe_array(lengths[idx] if len(idx) else np.array([], dtype=np.int64)),
            "active_frame_ratio": float(active_total / frame_total) if frame_total else 0.0,
            "events_per_sequence": describe_array(active_arr),
            "delta": describe_array(delta_arr),
            "delta_p95": float(np.percentile(delta_arr, 95)) if delta_arr.size else None,
            "delta_p99": float(np.percentile(delta_arr, 99)) if delta_arr.size else None,
            "unique_chords": int(len(chord_counts_by_split["train" if indices is train_idx else "val"])),
            "avg_notes_per_active_frame": float(note_total / active_total) if active_total else 0.0,
        }

    train = subset(train_idx)
    val = subset(val_idx)
    warnings = []
    if train["lengths_frames"]["mean"] and val["lengths_frames"]["mean"]:
        ratio = float(val["lengths_frames"]["mean"]) / float(train["lengths_frames"]["mean"])
        if ratio > 1.25 or ratio < 0.75:
            warnings.append(f"Validation mean sequence length differs from train by {ratio:.2f}x.")
    if train["active_frame_ratio"]:
        ratio = val["active_frame_ratio"] / train["active_frame_ratio"]
        if ratio > 1.25 or ratio < 0.75:
            warnings.append(f"Validation active-frame ratio differs from train by {ratio:.2f}x.")
    if train["events_per_sequence"]["mean"] and val["events_per_sequence"]["mean"]:
        ratio = float(val["events_per_sequence"]["mean"]) / float(train["events_per_sequence"]["mean"])
        if ratio > 1.25 or ratio < 0.75:
            warnings.append(f"Validation events-per-sequence differs from train by {ratio:.2f}x.")
    return {"train_indices": train_idx, "val_indices": val_idx, "train": train, "val": val, "warnings": warnings}


def counter_for_indices(rolls_flat: np.ndarray, offsets: np.ndarray, indices: list[int], num_notes: int) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for i in indices:
        roll = sequence_view(rolls_flat, offsets, int(i))[:, :num_notes]
        active_rows = roll[np.sum(roll > 0, axis=1) > 0]
        for frame in active_rows:
            counts[pack_chord(frame)] += 1
    return counts


def analyze_eval_sets(
    prefix_rolls: np.ndarray,
    prefix_offsets: np.ndarray,
    full_rolls: np.ndarray,
    full_offsets: np.ndarray,
    train_top_keys: dict[int, set[bytes]],
    num_notes: int,
    step_sec: float,
    train_active_ratio: float,
) -> dict[str, Any]:
    prefix_lengths = sequence_lengths(prefix_offsets)
    full_lengths = sequence_lengths(full_offsets)
    n = min(len(prefix_lengths), len(full_lengths))
    continuations = full_lengths[:n] - prefix_lengths[:n]
    ratios = np.divide(prefix_lengths[:n], full_lengths[:n], out=np.zeros(n, dtype=float), where=full_lengths[:n] != 0)
    mismatches = []
    for i in range(n):
        prefix = sequence_view(prefix_rolls, prefix_offsets, i)[:, :num_notes]
        full_start = sequence_view(full_rolls, full_offsets, i)[: len(prefix), :num_notes]
        if prefix.shape != full_start.shape or not np.array_equal(prefix, full_start):
            mismatches.append(int(i))
            if len(mismatches) >= 20:
                break

    def density_parts() -> dict[str, Any]:
        prefix_active = 0
        prefix_frames = 0
        cont_active = 0
        cont_frames = 0
        full_active = 0
        full_frames = 0
        for i in range(n):
            pref = sequence_view(prefix_rolls, prefix_offsets, i)[:, :num_notes]
            full = sequence_view(full_rolls, full_offsets, i)[:, :num_notes]
            cont = full[len(pref) :]
            prefix_active += int(np.sum(np.sum(pref > 0, axis=1) > 0))
            prefix_frames += int(len(pref))
            full_active += int(np.sum(np.sum(full > 0, axis=1) > 0))
            full_frames += int(len(full))
            cont_active += int(np.sum(np.sum(cont > 0, axis=1) > 0))
            cont_frames += int(len(cont))
        return {
            "prefix_active_frame_ratio": float(prefix_active / prefix_frames) if prefix_frames else 0.0,
            "full_active_frame_ratio": float(full_active / full_frames) if full_frames else 0.0,
            "continuation_active_frame_ratio": float(cont_active / cont_frames) if cont_frames else 0.0,
            "train_active_frame_ratio": train_active_ratio,
        }

    oov = {
        str(k): {
            "prefix": chord_oov_rate(prefix_rolls, prefix_offsets, keys, num_notes),
            "full": chord_oov_rate(full_rolls, full_offsets, keys, num_notes),
        }
        for k, keys in train_top_keys.items()
    }
    return {
        "same_sequence_count": len(prefix_lengths) == len(full_lengths),
        "prefix_num_sequences": int(len(prefix_lengths)),
        "full_num_sequences": int(len(full_lengths)),
        "compared_sequences": int(n),
        "prefix_lengths_frames": describe_array(prefix_lengths),
        "full_lengths_frames": describe_array(full_lengths),
        "continuation_lengths_frames": describe_array(continuations),
        "continuation_lengths_seconds": describe_array(continuations * step_sec),
        "prefix_full_ratio": describe_array(ratios),
        "prefix_matches_full_start": len(mismatches) == 0 and len(prefix_lengths) == len(full_lengths),
        "prefix_mismatch_count_first_20": int(len(mismatches)),
        "prefix_mismatch_examples": mismatches,
        "density": density_parts(),
        "oov_by_top_k": oov,
    }


def analyze_metadata(path: Path, train_sequence_count: int) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "warning": f"{path} does not exist"}
    result: dict[str, Any] = {"available": True, "path": str(path), "pandas_available": pd is not None}
    if pd is not None:
        df = pd.read_csv(path)
        result["row_count"] = int(len(df))
        result["columns"] = list(map(str, df.columns))
        result["missing_values"] = {str(k): int(v) for k, v in df.isna().sum().to_dict().items()}
        low_card = {}
        for col in df.columns:
            nunique = int(df[col].nunique(dropna=True))
            if nunique <= 20:
                low_card[str(col)] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).head(20).to_dict().items()}
        result["low_cardinality_value_counts"] = low_card
        summaries = {}
        for col in df.columns:
            low = str(col).lower()
            if any(token in low for token in ["duration", "length", "step", "num_steps"]):
                try:
                    desc = df[col].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
                    summaries[str(col)] = {str(k): as_jsonable(v) for k, v in desc.to_dict().items()}
                except Exception:
                    pass
            if any(token in low for token in ["composer", "genre", "split", "id"]):
                summaries[str(col)] = {
                    "unique": int(df[col].nunique(dropna=True)),
                    "top_values": {str(k): int(v) for k, v in df[col].value_counts(dropna=False).head(20).to_dict().items()},
                }
        result["selected_column_summaries"] = summaries
    else:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        result["row_count"] = len(rows)
        result["columns"] = list(reader.fieldnames or [])
        result["missing_values"] = {
            col: sum(1 for row in rows if row.get(col, "") == "") for col in result["columns"]
        }
        result["low_cardinality_value_counts"] = {}
    result["matches_train_sequence_count"] = result.get("row_count") == train_sequence_count
    if not result["matches_train_sequence_count"]:
        result["warning"] = f"metadata row count {result.get('row_count')} does not match train sequence count {train_sequence_count}"
    return result


def recommendation_text(stats: dict[str, Any]) -> list[str]:
    chord_rows = stats["train"]["chord_coverage"]["rows"]
    def cov(k: int) -> float:
        return next((r["coverage_pct"] for r in chord_rows if r["k"] == k), 0.0)

    delta_p99 = stats["train"]["delta_stats"]["p99"]
    delta_p995 = stats["train"]["delta_stats"]["p99_5"]
    compression = stats["train"]["event_state"]["compression_ratio_active_over_total"]
    empty_pct = stats["train"]["basic"]["empty_frame_pct"]
    top_50000_effective = next((r["effective_vocab_size"] for r in chord_rows if r["k"] == 50000), 0)
    unique = stats["train"]["chord_vocab"]["unique_non_empty_chord_states"]

    lines = []
    if cov(10000) >= 99.0:
        lines.append(f"1. top_k=50000 is likely larger than necessary: top_k=10000 already covers {cov(10000):.2f}% of train chord events.")
    elif cov(20000) >= 99.0:
        lines.append(f"1. top_k=50000 may be excessive: top_k=20000 covers {cov(20000):.2f}% of train chord events.")
    else:
        lines.append(f"1. top_k=50000 is more defensible because top_k=20000 covers only {cov(20000):.2f}%, but the effective size is {top_50000_effective:,} of {unique:,} unique chords.")

    recommended_delta = int(math.ceil((delta_p995 or delta_p99 or 0) / 10.0) * 10) if (delta_p995 or delta_p99) else 1200
    lines.append(f"2. A reasonable max_delta is near the p99.5 delta, about {recommended_delta:,} frames; compare this to the current 1,200 cap.")

    if compression < 0.10:
        lines.append(f"3. The event-state representation is highly compressed: only {compression * 100:.2f}% of frames are active and {empty_pct:.2f}% are empty.")
    else:
        lines.append(f"3. The event-state representation is moderately compressed: {compression * 100:.2f}% of frames remain as events.")

    entropy_bits = stats["train"]["chord_vocab"]["empirical_entropy_bits"]
    if entropy_bits > 10:
        lines.append("4. Validation loss is probably dominated by chord_ce because the empirical chord entropy is high; delta_ce can also matter if many deltas hit or approach the cap.")
    else:
        lines.append("4. Validation loss is less likely to be dominated only by chord_ce; inspect delta_ce and note_bce because chord entropy is moderate.")

    split_warnings = stats["train_val_split"]["warnings"]
    if split_warnings:
        lines.append("5. The train/val split may be mismatched: " + " ".join(split_warnings))
    else:
        lines.append("5. The train/val split does not show a large first-order mismatch in length, active ratio, or event count.")

    eval_density = stats["eval"]["density"]
    full_ratio = eval_density["full_active_frame_ratio"]
    train_ratio = eval_density["train_active_frame_ratio"]
    if train_ratio and (full_ratio / train_ratio > 1.25 or full_ratio / train_ratio < 0.75):
        lines.append(f"6. eval_set_01_full differs from train density by {full_ratio / train_ratio:.2f}x, so evaluation may not be train-like.")
    else:
        lines.append("6. eval_set_01_prefix/full look broadly similar to train by active-frame ratio; chord OOV rates below give the stricter check.")

    lines.append("7. Loss weights should be adjusted only after checking component losses; if chord_ce dominates, smaller top_k or chord smoothing is preferable before simply downweighting it.")
    lines.append("8. Increasing samples_per_sequence or using more random crops is useful if sequences contain many events and validation uses center crops that miss train-time variety.")
    lines.append("9. A frame-level Transformer/GRU baseline is worth considering if chord-token OOV is high or chord_ce remains dominant, because multi-label note prediction can generalize across unseen chord combinations.")
    return lines


def write_report(path: Path, stats: dict[str, Any], plots: list[str], console_lines: list[str]) -> None:
    train = stats["train"]
    lines = [
        "# Event-State Transformer Data Analysis",
        "",
        "## Console summary",
        "",
        "```text",
        *console_lines,
        "```",
        "",
        "## NPZ keys and arrays",
        "",
    ]
    for name in ["train_npz", "prefix_npz", "full_npz"]:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| key | shape | dtype | scalar |")
        lines.append("|---|---:|---|---|")
        for key, info in stats["npz_overview"][name]["keys"].items():
            lines.append(f"| `{key}` | `{info['shape']}` | `{info['dtype']}` | `{info['scalar_value']}` |")
        lines.append("")

    basic = train["basic"]
    lines += [
        "## A. Basic dataset information",
        "",
        f"- `rolls_flat`: shape `{basic['rolls_flat_shape']}`, dtype `{basic['rolls_flat_dtype']}`",
        f"- `offsets`: shape `{basic['offsets_shape']}`, dtype `{basic['offsets_dtype']}`",
        f"- sequences: {basic['num_sequences']:,}",
        f"- step_sec: {basic['step_sec']}",
        f"- note range: {basic['note_min']} to {basic['note_max']}",
        f"- num_positions: {basic['num_positions']}",
        f"- representation: {basic['representation']}",
        f"- total frames: {basic['total_frames']:,}",
        f"- active frames: {basic['total_active_frames']:,}",
        f"- empty frames: {basic['total_empty_frames']:,} ({basic['empty_frame_pct']:.2f}%)",
        f"- average notes per frame: {basic['avg_notes_per_frame']:.4f}",
        f"- average notes per active frame: {basic['avg_notes_per_active_frame']:.4f}",
        "",
        "## B. Sequence length statistics",
        "",
        "| units | count | min | max | mean | median | p25 | p75 | p90 | p95 | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| frames | {describe_for_markdown(train['length_stats_frames'], ['count','min','max','mean','median','p25','p75','p90','p95','p99'])} |",
        f"| seconds | {describe_for_markdown(train['length_stats_seconds'], ['count','min','max','mean','median','p25','p75','p90','p95','p99'])} |",
        "",
        "| shorter than | sequence count | percentage |",
        "|---:|---:|---:|",
    ]
    for key, value in train["short_sequence_counts"].items():
        lines.append(f"| {key} | {value['count']:,} | {value['pct']:.2f}% |")

    lines += [
        "",
        "## C. Event-state statistics",
        "",
        f"- compression ratio active frames / total frames: {train['event_state']['compression_ratio_active_over_total']:.6f}",
        "",
        "| metric | count | min | max | mean | median | p90 | p95 | p99 | p99.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| active events / sequence | {describe_for_markdown(train['events_per_sequence_stats'], ['count','min','max','mean','median','p90','p95','p99','p99_5'])} |",
        f"| delta | {describe_for_markdown(train['delta_stats'], ['count','min','max','mean','median','p90','p95','p99','p99_5'])} |",
        "",
        "| delta threshold | count | percentage |",
        "|---:|---:|---:|",
    ]
    for threshold, row in train["delta_thresholds"].items():
        lines.append(f"| > {threshold} | {row['count']:,} | {row['pct']:.2f}% |")
    lines += [
        "",
        f"- recommended max_delta from p99/p99.5: {train['event_state']['recommended_max_delta']:,}",
        "",
        "## D. Chord vocabulary analysis",
        "",
        f"- unique non-empty chord states: {train['chord_vocab']['unique_non_empty_chord_states']:,}",
        f"- total non-empty chord events: {train['chord_vocab']['total_non_empty_chord_events']:,}",
        f"- empirical chord entropy: {train['chord_vocab']['empirical_entropy_bits']:.3f} bits / {train['chord_vocab']['empirical_entropy_nats']:.3f} nats",
        "",
        "| K | coverage | OOV | effective vocab size |",
        "|---:|---:|---:|---:|",
    ]
    for row in train["chord_coverage"]["rows"]:
        lines.append(f"| {row['k']:,} | {row['coverage_pct']:.2f}% | {row['oov_pct']:.2f}% | {row['effective_vocab_size']:,} |")
    lines += ["", "Top 20 chord frequencies:", "", "| rank | MIDI pitches | count | percentage |", "|---:|---|---:|---:|"]
    for row in train["chord_vocab"]["top_20"]:
        lines.append(f"| {row['rank']} | `{row['pitches']}` | {row['count']:,} | {row['pct']:.2f}% |")

    lines += [
        "",
        "## E. Note-level statistics",
        "",
        f"- notes per active frame: {describe_for_markdown(train['notes_per_active_frame_stats'], ['min','max','mean','median','p90','p95','p99'])}",
        "",
        "Most common notes:",
        "",
        "| rank | pitch_index | midi_pitch | count |",
        "|---:|---:|---:|---:|",
    ]
    for row in train["note_stats"]["most_common_notes"]:
        lines.append(f"| {row['rank']} | {row['pitch_index']} | {row['midi_pitch']} | {row['count']:,} |")
    lines += ["", "Least common notes:", "", "| rank | pitch_index | midi_pitch | count |", "|---:|---:|---:|---:|"]
    for row in train["note_stats"]["least_common_notes"]:
        lines.append(f"| {row['rank']} | {row['pitch_index']} | {row['midi_pitch']} | {row['count']:,} |")
    lines += ["", "Active-frame chord cardinality:", "", "| notes in frame | percentage | count |", "|---:|---:|---:|"]
    for row in train["note_stats"]["active_frame_cardinality"]:
        lines.append(f"| {row['label']} | {row['pct']:.2f}% | {row['count']:,} |")

    split = stats["train_val_split"]
    lines += [
        "",
        "## F. Train/validation split diagnostic",
        "",
        f"- train sequences: {split['train']['num_sequences']:,}",
        f"- validation sequences: {split['val']['num_sequences']:,}",
        f"- train active-frame ratio: {split['train']['active_frame_ratio']:.6f}",
        f"- validation active-frame ratio: {split['val']['active_frame_ratio']:.6f}",
        f"- train unique chords: {split['train']['unique_chords']:,}",
        f"- validation unique chords: {split['val']['unique_chords']:,}",
        f"- train delta p95/p99: {fmt_num(split['train']['delta_p95'])} / {fmt_num(split['train']['delta_p99'])}",
        f"- validation delta p95/p99: {fmt_num(split['val']['delta_p95'])} / {fmt_num(split['val']['delta_p99'])}",
    ]
    if split["warnings"]:
        lines += ["", "Warnings:", *[f"- {warning}" for warning in split["warnings"]]]

    ev = stats["eval"]
    lines += [
        "",
        "## G. eval_set_01_prefix/full analysis",
        "",
        f"- prefix sequences: {ev['prefix_num_sequences']:,}",
        f"- full sequences: {ev['full_num_sequences']:,}",
        f"- same number of sequences: {ev['same_sequence_count']}",
        f"- prefix equals beginning of full: {ev['prefix_matches_full_start']}",
        f"- prefix mismatch examples: {ev['prefix_mismatch_examples']}",
        f"- continuation length frames: {describe_for_markdown(ev['continuation_lengths_frames'], ['min','max','mean','median','p90','p95','p99'])}",
        f"- continuation length seconds: {describe_for_markdown(ev['continuation_lengths_seconds'], ['min','max','mean','median','p90','p95','p99'])}",
        f"- prefix/full ratio: {describe_for_markdown(ev['prefix_full_ratio'], ['min','max','mean','median','p90','p95','p99'])}",
        f"- train active-frame ratio: {ev['density']['train_active_frame_ratio']:.6f}",
        f"- prefix active-frame ratio: {ev['density']['prefix_active_frame_ratio']:.6f}",
        f"- full active-frame ratio: {ev['density']['full_active_frame_ratio']:.6f}",
        f"- continuation active-frame ratio: {ev['density']['continuation_active_frame_ratio']:.6f}",
        "",
        "Eval chord OOV under train vocabularies:",
        "",
        "| K | prefix OOV | full OOV | prefix coverage | full coverage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in ["5000", "10000", "20000", "50000"]:
        row = ev["oov_by_top_k"][k]
        lines.append(
            f"| {int(k):,} | {row['prefix']['oov_pct']:.2f}% | {row['full']['oov_pct']:.2f}% | "
            f"{row['prefix']['coverage_pct']:.2f}% | {row['full']['coverage_pct']:.2f}% |"
        )

    meta = stats["metadata"]
    lines += ["", "## H. metadata.csv analysis", ""]
    if meta.get("available"):
        lines += [
            f"- pandas available: {meta['pandas_available']}",
            f"- rows: {meta['row_count']:,}",
            f"- columns: {', '.join('`' + c + '`' for c in meta['columns'])}",
            f"- matches train sequence count: {meta['matches_train_sequence_count']}",
            f"- missing values: `{meta['missing_values']}`",
        ]
        if meta.get("warning"):
            lines.append(f"- warning: {meta['warning']}")
        if meta.get("selected_column_summaries"):
            lines += ["", "Selected column summaries:", ""]
            for col, summary in meta["selected_column_summaries"].items():
                lines.append(f"- `{col}`: `{summary}`")
    else:
        lines.append(f"- unavailable: {meta.get('warning')}")

    lines += [
        "",
        "## Plots",
        "",
    ]
    if plots:
        lines.extend(f"- `{plot}`" for plot in plots)
    else:
        lines.append("- No plots generated because matplotlib is unavailable or inputs were empty.")

    lines += [
        "",
        "## Actionable recommendations for event_state_transformer",
        "",
        *recommendation_text(stats),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze event-state Transformer data and produce diagnostics.")
    parser.add_argument("--train", type=Path, default=Path("../train.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("../metadata.csv"))
    parser.add_argument("--prefix", type=Path, default=Path("../eval_set_01_prefix.npz"))
    parser.add_argument("--full", type=Path, default=Path("../eval_set_01_full.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs"))
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    train_data = load_npz(args.train)
    prefix_data = load_npz(args.prefix)
    full_data = load_npz(args.full)
    train_rolls, train_offsets = require_rolls_flat(train_data, "train")
    prefix_rolls, prefix_offsets = require_rolls_flat(prefix_data, "prefix")
    full_rolls, full_offsets = require_rolls_flat(full_data, "full")

    train_roll_stats = basic_roll_stats(train_data, train_rolls, train_offsets, "train")
    num_notes = train_roll_stats["num_positions"]
    step_sec = train_roll_stats["step_sec"]
    note_min = train_roll_stats["note_min"]

    chord_counts = build_chord_counter(train_rolls, train_offsets, num_notes)
    chord_cov = coverage_table(chord_counts, CANDIDATE_TOP_K)
    entropy = entropy_from_counts(chord_counts)
    ordered_keys = [key for key, _ in chord_counts.most_common()]
    top_key_sets = {k: set(ordered_keys[: min(k, len(ordered_keys))]) for k in CANDIDATE_TOP_K}

    lengths = train_roll_stats["lengths_frames"]
    active_counts = train_roll_stats["active_counts"]
    deltas = train_roll_stats["deltas"]
    active_note_counts = train_roll_stats["active_note_counts"]
    note_frequency = train_roll_stats["note_frequency"]

    train_idx, val_idx = split_sequence_indices(len(lengths), val_frac=0.1, seed=42)
    chord_counts_train = counter_for_indices(train_rolls, train_offsets, train_idx, num_notes)
    chord_counts_val = counter_for_indices(train_rolls, train_offsets, val_idx, num_notes)
    split_stats = split_diagnostics(
        train_rolls,
        train_offsets,
        num_notes,
        {"train": chord_counts_train, "val": chord_counts_val},
    )

    eval_stats = analyze_eval_sets(
        prefix_rolls,
        prefix_offsets,
        full_rolls,
        full_offsets,
        {k: top_key_sets[k] for k in [5000, 10000, 20000, 50000]},
        num_notes,
        step_sec,
        train_roll_stats["active_frame_ratio"],
    )

    card_rows = []
    for n in range(1, 10):
        count = int(np.sum(active_note_counts == n))
        card_rows.append({"label": str(n), "count": count, "pct": pct(count, len(active_note_counts))})
    count_10 = int(np.sum(active_note_counts >= 10))
    card_rows.append({"label": "10+", "count": count_10, "pct": pct(count_10, len(active_note_counts))})

    note_order_desc = np.argsort(-note_frequency)
    note_order_asc = np.argsort(note_frequency)
    top_20 = []
    total_events = sum(chord_counts.values())
    for rank, (key, count) in enumerate(chord_counts.most_common(20), start=1):
        top_20.append(
            {
                "rank": rank,
                "pitches": chord_label(key, note_min, num_notes),
                "count": int(count),
                "pct": pct(count, total_events),
            }
        )

    delta_thresholds = {
        str(threshold): {"count": int(np.sum(deltas > threshold)), "pct": pct(np.sum(deltas > threshold), len(deltas))}
        for threshold in DELTA_THRESHOLDS
    }
    delta_p99 = float(np.percentile(deltas, 99)) if deltas.size else 0.0
    delta_p995 = float(np.percentile(deltas, 99.5)) if deltas.size else delta_p99
    recommended_max_delta = int(math.ceil(max(delta_p99, delta_p995) / 10.0) * 10) if deltas.size else 1200

    stats: dict[str, Any] = {
        "npz_overview": {
            "train_npz": npz_overview(args.train, train_data),
            "prefix_npz": npz_overview(args.prefix, prefix_data),
            "full_npz": npz_overview(args.full, full_data),
        },
        "train": {
            "basic": {
                "rolls_flat_shape": list(train_rolls.shape),
                "rolls_flat_dtype": str(train_rolls.dtype),
                "offsets_shape": list(train_offsets.shape),
                "offsets_dtype": str(train_offsets.dtype),
                **{k: v for k, v in train_roll_stats.items() if not isinstance(v, np.ndarray)},
            },
            "length_stats_frames": describe_array(lengths),
            "length_stats_seconds": describe_array(lengths * step_sec),
            "short_sequence_counts": {
                f"{seconds}s": {
                    "count": int(np.sum(lengths * step_sec < seconds)),
                    "pct": pct(np.sum(lengths * step_sec < seconds), len(lengths)),
                }
                for seconds in SHORT_SECONDS
            },
            "event_state": {
                "compression_ratio_active_over_total": train_roll_stats["active_frame_ratio"],
                "recommended_max_delta": recommended_max_delta,
            },
            "events_per_sequence_stats": describe_array(active_counts),
            "delta_stats": describe_array(deltas),
            "delta_thresholds": delta_thresholds,
            "chord_vocab": {
                "unique_non_empty_chord_states": int(len(chord_counts)),
                "total_non_empty_chord_events": int(total_events),
                "empirical_entropy_bits": entropy["bits"],
                "empirical_entropy_nats": entropy["nats"],
                "top_20": top_20,
            },
            "chord_coverage": chord_cov,
            "notes_per_active_frame_stats": describe_array(active_note_counts),
            "note_stats": {
                "note_frequency": note_frequency.tolist(),
                "most_common_notes": [
                    {
                        "rank": rank,
                        "pitch_index": int(idx),
                        "midi_pitch": int(note_min + idx),
                        "count": int(note_frequency[idx]),
                    }
                    for rank, idx in enumerate(note_order_desc[:20], start=1)
                ],
                "least_common_notes": [
                    {
                        "rank": rank,
                        "pitch_index": int(idx),
                        "midi_pitch": int(note_min + idx),
                        "count": int(note_frequency[idx]),
                    }
                    for rank, idx in enumerate(note_order_asc[:20], start=1)
                ],
                "active_frame_cardinality": card_rows,
            },
        },
        "train_val_split": split_stats,
        "eval": eval_stats,
        "metadata": analyze_metadata(args.metadata, len(lengths)),
    }

    plots = []
    if save_hist(args.out_dir / "sequence_lengths_seconds.png", lengths * step_sec, "Sequence lengths", "Seconds"):
        plots.append("sequence_lengths_seconds.png")
    if save_hist(args.out_dir / "delta_hist_clipped_p99.png", deltas, "Delta distribution clipped to p99", "Frames", clip_p99=True):
        plots.append("delta_hist_clipped_p99.png")
    if save_line(
        args.out_dir / "topk_coverage.png",
        [row["k"] for row in chord_cov["rows"]],
        [row["coverage_pct"] for row in chord_cov["rows"]],
        "Top-K chord coverage",
        "K",
        "Coverage (%)",
    ):
        plots.append("topk_coverage.png")
    if save_bar(
        args.out_dir / "note_frequency.png",
        np.arange(num_notes),
        note_frequency,
        "Note frequency by pitch index",
        "Pitch index",
        "Active-frame count",
    ):
        plots.append("note_frequency.png")
    if save_hist(args.out_dir / "notes_per_active_frame.png", active_note_counts, "Notes per active frame", "Notes", bins=40):
        plots.append("notes_per_active_frame.png")

    console_lines = [
        f"Train sequences: {len(lengths):,}; frames: {train_roll_stats['total_frames']:,}; active-frame ratio: {train_roll_stats['active_frame_ratio']:.4f}",
        f"Unique non-empty chords: {len(chord_counts):,}; total chord events: {total_events:,}; entropy: {entropy['bits']:.2f} bits",
        f"Top-K coverage: K=5k {next(r['coverage_pct'] for r in chord_cov['rows'] if r['k'] == 5000):.2f}%, "
        f"K=10k {next(r['coverage_pct'] for r in chord_cov['rows'] if r['k'] == 10000):.2f}%, "
        f"K=20k {next(r['coverage_pct'] for r in chord_cov['rows'] if r['k'] == 20000):.2f}%, "
        f"K=50k {next(r['coverage_pct'] for r in chord_cov['rows'] if r['k'] == 50000):.2f}%",
        f"Delta p95/p99/p99.5: {np.percentile(deltas, 95):.1f}/{delta_p99:.1f}/{delta_p995:.1f}; recommended max_delta: {recommended_max_delta}",
        f"Eval prefix/full sequences: {eval_stats['prefix_num_sequences']}/{eval_stats['full_num_sequences']}; prefix matches full start: {eval_stats['prefix_matches_full_start']}",
        f"Metadata rows: {stats['metadata'].get('row_count', 'n/a')}; matches train sequences: {stats['metadata'].get('matches_train_sequence_count', False)}",
    ]
    for warning in split_stats["warnings"]:
        console_lines.append(f"WARNING: {warning}")
    if stats["metadata"].get("warning"):
        console_lines.append(f"WARNING: {stats['metadata']['warning']}")
    if plt is None:
        console_lines.append("WARNING: matplotlib is not available; plots were skipped.")
    if pd is None:
        console_lines.append("WARNING: pandas is not available; metadata analysis used csv fallback.")

    write_report(args.out_dir / "analysis_report.md", stats, plots, console_lines)
    (args.out_dir / "analysis_stats.json").write_text(json.dumps(as_jsonable(stats), indent=2), encoding="utf-8")

    print("\n".join(console_lines))
    print(f"Saved report: {args.out_dir / 'analysis_report.md'}")
    print(f"Saved stats: {args.out_dir / 'analysis_stats.json'}")
    if plots:
        print("Saved plots: " + ", ".join(plots))


if __name__ == "__main__":
    main()
