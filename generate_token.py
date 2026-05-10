from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from token_model import CausalTokenTransformer
from tokenizer import PianoEventTokenizer


MODEL_TYPE = "NestedMusicTransformerInspired_TokenChordTransformer"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.ravel()[0]
    return value.item() if isinstance(value, np.generic) else value


def get_sequence(data: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    offsets = np.asarray(data["offsets"], dtype=np.int64)
    return np.asarray(data["rolls_flat"][offsets[index] : offsets[index + 1]], dtype=np.uint8)


def longest_silent_run(roll: np.ndarray) -> int:
    best = 0
    cur = 0
    for silent in (roll.sum(axis=1) == 0):
        if silent:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def trailing_silent_run(roll: np.ndarray) -> int:
    cur = 0
    for frame in reversed(roll):
        if int(frame.sum()) == 0:
            cur += 1
        else:
            break
    return cur


def roll_stats(roll: np.ndarray) -> dict[str, float]:
    steps = int(len(roll))
    event_mask = roll.sum(axis=1) > 0 if steps else np.asarray([], dtype=bool)
    events = int(event_mask.sum())
    note_count = int(roll.sum())
    return {
        "prefix_steps": steps,
        "prefix_note_count": note_count,
        "prefix_event_frames": events,
        "prefix_event_density": float(events / max(1, steps)),
        "prefix_avg_notes_per_event": float(note_count / max(1, events)),
        "prefix_longest_silent_run": float(longest_silent_run(roll)),
    }


def chord_key(frame: np.ndarray) -> str:
    active = np.flatnonzero(frame > 0)
    return ",".join(str(int(x)) for x in active) if len(active) else "silence"


def resolve_target_lengths(prefix_data, args) -> list[int]:
    prefix_offsets = np.asarray(prefix_data["offsets"], dtype=np.int64)
    prefix_lengths = np.diff(prefix_offsets).astype(int).tolist()
    if args.target_full_npz is not None and args.match_full_lengths:
        full = np.load(args.target_full_npz, allow_pickle=True)
        full_offsets = np.asarray(full["offsets"], dtype=np.int64)
        lengths = np.diff(full_offsets).astype(int).tolist()
        if len(lengths) != len(prefix_lengths):
            raise ValueError("target_full_npz sequence count does not match prefix_npz.")
        return lengths
    if args.target_total_steps is not None:
        return [int(args.target_total_steps) for _ in prefix_lengths]
    return [int(n + args.continuation_steps) for n in prefix_lengths]


def build_model(ckpt, tokenizer: PianoEventTokenizer, args, device):
    cfg = ckpt.get("config", {})
    model_context_len = int(cfg.get("context_len", args.context_len or 1024))
    model = CausalTokenTransformer(
        vocab_size=int(cfg.get("vocab_size", tokenizer.vocab_size)),
        context_len=model_context_len,
        d_model=int(cfg.get("d_model", 384)),
        n_layers=int(cfg.get("n_layers", 8)),
        n_heads=int(cfg.get("n_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        pad_id=tokenizer.PAD,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, min(int(args.context_len or model_context_len), model_context_len)


def load_training_stats(tokenizer: PianoEventTokenizer, tokenizer_path: Path) -> dict[str, float]:
    stats = dict(getattr(tokenizer, "stats", {}) or {})
    info_path = tokenizer_path.parent / "dataset_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        for key in [
            "train_event_density",
            "train_avg_notes_per_event",
            "train_longest_silent_run_mean",
            "train_longest_silent_run_p95",
            "train_chord_ratio",
            "train_timeshift_ratio",
        ]:
            if key in info:
                stats[key] = float(info[key])
    stats.setdefault("train_event_density", float(getattr(tokenizer, "event_density", 0.0) or 0.18))
    stats.setdefault("train_avg_notes_per_event", 1.3)
    stats.setdefault("train_longest_silent_run_mean", 24.0)
    stats.setdefault("train_longest_silent_run_p95", 48.0)
    stats.setdefault("train_chord_ratio", 0.5)
    stats.setdefault("train_timeshift_ratio", 0.5)
    return stats


def target_density_for_sequence(prefix_stats: dict[str, float], train_stats: dict[str, float], args) -> float:
    if args.target_density_auto:
        density = 0.60 * train_stats["train_event_density"] + 0.40 * prefix_stats["prefix_event_density"]
    else:
        density = train_stats["train_event_density"]
    return float(np.clip(density, args.min_target_density, args.max_target_density))


def filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered[filtered < threshold] = -float("inf")
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_indices[remove]] = -float("inf")
    return filtered


def penalize_repeated_chords(
    logits: torch.Tensor,
    tokenizer: PianoEventTokenizer,
    recent_chords: deque[int],
    args,
) -> int:
    activations = 0
    counts = Counter(recent_chords)
    for token_id, count in counts.items():
        if not tokenizer.is_chord(token_id):
            continue
        if count >= args.max_same_chord_repeats:
            logits[token_id] = -float("inf")
            activations += 1
        elif count > 0 and args.chord_repetition_penalty > 1.0:
            logits[token_id] -= float(np.log(args.chord_repetition_penalty)) * count
            activations += 1
    return activations


def mask_long_time_shifts(
    logits: torch.Tensor,
    tokenizer: PianoEventTokenizer,
    current_silent_run: int,
    args,
) -> None:
    if not args.mask_long_time_shifts:
        return
    remaining = args.max_silent_frames - current_silent_run
    valid_any = False
    for token_id in tokenizer.time_shift_token_ids():
        shift = tokenizer.get_time_shift_value(token_id)
        if remaining <= 0 or shift > max(1, remaining):
            logits[token_id] = -float("inf")
        elif torch.isfinite(logits[token_id]):
            valid_any = True
    if not valid_any:
        for token_id in tokenizer.time_shift_token_ids():
            logits[token_id] = -float("inf")


def adjusted_logits(
    logits: torch.Tensor,
    tokenizer: PianoEventTokenizer,
    args,
    target_density: float,
    generated_frames: int,
    generated_events: int,
    current_silent_run: int,
    recent_chords: deque[int],
) -> tuple[torch.Tensor, dict[str, int]]:
    out = logits.float().clone()
    out[tokenizer.PAD] = -float("inf")
    out[tokenizer.BOS] = -float("inf")
    out[tokenizer.UNK_CHORD] = -float("inf")
    if not args.allow_eos:
        out[tokenizer.EOS] = -float("inf")

    chord_ids = tokenizer.chord_token_ids()
    shift_ids = tokenizer.time_shift_token_ids()
    counters = {"density": 0, "silence": 0, "repetition": 0}

    if args.density_control and generated_frames >= 8:
        current_density = generated_events / max(1, generated_frames)
        diff = target_density - current_density
        scale = min(1.0, abs(diff) / max(target_density, 1e-6))
        amount = args.density_strength * scale
        if diff > 0:
            out[chord_ids] += amount
            out[shift_ids] -= 0.75 * amount
            counters["density"] = 1
        elif diff < 0:
            out[shift_ids] += 0.5 * amount
            out[chord_ids] -= 0.75 * amount
            counters["density"] = 1

    if current_silent_run >= args.max_silent_frames:
        out[shift_ids] -= args.silence_penalty
        out[chord_ids] += args.chord_boost_after_silence
        counters["silence"] = 1

    counters["repetition"] += penalize_repeated_chords(out, tokenizer, recent_chords, args)
    mask_long_time_shifts(out, tokenizer, current_silent_run, args)
    return out, counters


def sample_token(logits: torch.Tensor, args) -> int:
    logits = logits / max(args.temperature, 1e-6)
    filtered = filter_top_k_top_p(logits, args.top_k, args.top_p)
    probs = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        finite = torch.isfinite(logits)
        probs = finite.float() / max(1, int(finite.sum().item()))
    return int(torch.multinomial(probs, num_samples=1).item())


def token_frame_delta(tokenizer: PianoEventTokenizer, token_id: int) -> tuple[int, bool]:
    if tokenizer.is_time_shift(token_id):
        return tokenizer.get_time_shift_value(token_id), False
    if tokenizer.is_chord(token_id):
        return 1, True
    return 0, False


@torch.no_grad()
def generate_one(
    model,
    tokenizer: PianoEventTokenizer,
    prefix_roll: np.ndarray,
    target_steps: int,
    context_len: int,
    train_stats: dict[str, float],
    args,
    device,
) -> tuple[np.ndarray, int, dict]:
    prefix_len = len(prefix_roll)
    if target_steps < prefix_len:
        raise ValueError(f"target_steps={target_steps} is shorter than prefix length {prefix_len}.")

    prefix_stats = roll_stats(prefix_roll)
    target_density = target_density_for_sequence(prefix_stats, train_stats, args)
    tokens = tokenizer.encode_roll(prefix_roll, add_bos=True, add_eos=False)
    generated_tokens: list[int] = []
    recent_chords: deque[int] = deque(maxlen=args.recent_chord_window)
    generated_frames = 0
    generated_events = 0
    current_silent_run = trailing_silent_run(prefix_roll)
    control_counts = Counter()
    max_new_tokens = max(args.max_new_tokens, (target_steps - prefix_len) * 4 + 128)

    for _ in range(max_new_tokens):
        if prefix_len + generated_frames >= target_steps:
            break
        input_ids = torch.tensor(tokens[-context_len:], dtype=torch.long, device=device).unsqueeze(0)
        base_logits = model(input_ids)[0, -1]
        logits, counters = adjusted_logits(
            base_logits,
            tokenizer,
            args,
            target_density,
            generated_frames,
            generated_events,
            current_silent_run,
            recent_chords,
        )
        control_counts.update(counters)
        next_id = sample_token(logits, args)
        tokens.append(next_id)
        generated_tokens.append(next_id)

        frame_delta, is_event = token_frame_delta(tokenizer, next_id)
        generated_frames += frame_delta
        if is_event:
            generated_events += 1
            current_silent_run = 0
            recent_chords.append(next_id)
        elif tokenizer.is_time_shift(next_id):
            current_silent_run += frame_delta

    decoded = tokenizer.decode_tokens(tokens, target_steps=target_steps)
    if len(decoded) < target_steps:
        padded = np.zeros((target_steps, tokenizer.num_notes), dtype=np.uint8)
        padded[: len(decoded)] = decoded
        decoded = padded
    decoded = decoded[:target_steps].astype(np.uint8, copy=False)
    decoded[:prefix_len] = prefix_roll.astype(np.uint8, copy=False)

    generated = decoded[prefix_len:]
    if generated.sum() == 0 and len(generated) > 0:
        note = min(max(60 - tokenizer.note_min, 0), tokenizer.num_notes - 1)
        step = max(8, min(args.max_silent_frames, int(round(1.0 / tokenizer.step_sec))))
        generated[::step, note] = 1
        decoded[prefix_len:] = generated

    generated_event_mask = generated.sum(axis=1) > 0 if len(generated) else np.asarray([], dtype=bool)
    generated_patterns = Counter(chord_key(frame) for frame in generated[generated_event_mask])
    debug = {
        **prefix_stats,
        "target_density": target_density,
        "target_total_steps": target_steps,
        "generated_steps": target_steps - prefix_len,
        "final_generated_event_density": float(generated_event_mask.sum() / max(1, len(generated))),
        "generated_note_count": int(generated.sum()),
        "unique_generated_chords": len(generated_patterns),
        "longest_silent_run": longest_silent_run(generated),
        "density_control_activations": int(control_counts["density"]),
        "silence_control_activations": int(control_counts["silence"]),
        "chord_repetition_control_activations": int(control_counts["repetition"]),
        "top_patterns": generated_patterns.most_common(10),
    }
    return decoded, target_steps - prefix_len, debug


def print_debug(index: int, debug: dict) -> None:
    print(f"generation stats sequence {index}")
    for key in [
        "prefix_steps",
        "prefix_note_count",
        "prefix_event_frames",
        "prefix_avg_notes_per_event",
        "prefix_longest_silent_run",
        "target_total_steps",
        "generated_steps",
        "prefix_event_density",
        "target_density",
        "final_generated_event_density",
        "generated_note_count",
        "unique_generated_chords",
        "longest_silent_run",
        "density_control_activations",
        "silence_control_activations",
        "chord_repetition_control_activations",
    ]:
        value = debug[key]
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    print("  top_10_generated_chord_patterns:")
    for key, count in debug["top_patterns"]:
        print(f"    {key}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate piano-roll continuations with controlled token sampling.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prefix_npz", type=Path, required=True)
    parser.add_argument("--out_npz", type=Path, required=True)
    parser.add_argument("--continuation_steps", type=int, default=2048)
    parser.add_argument("--target_total_steps", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top_k", type=int, default=48)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--density_control", action="store_true")
    parser.add_argument("--target_density_auto", action="store_true")
    parser.add_argument("--min_target_density", type=float, default=0.08)
    parser.add_argument("--max_target_density", type=float, default=0.32)
    parser.add_argument("--density_strength", type=float, default=1.5)
    parser.add_argument("--max_silent_frames", type=int, default=48)
    parser.add_argument("--silence_penalty", type=float, default=3.0)
    parser.add_argument("--chord_boost_after_silence", type=float, default=1.5)
    parser.add_argument("--mask_long_time_shifts", action="store_true")
    parser.add_argument("--chord_repetition_penalty", type=float, default=1.15)
    parser.add_argument("--recent_chord_window", type=int, default=32)
    parser.add_argument("--max_same_chord_repeats", type=int, default=8)
    parser.add_argument("--allow_eos", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=12000)
    parser.add_argument("--debug_generation_stats", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target_full_npz", type=Path, default=None)
    parser.add_argument("--match_full_lengths", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    tokenizer = PianoEventTokenizer.load(args.tokenizer)
    train_stats = load_training_stats(tokenizer, args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model, context_len = build_model(ckpt, tokenizer, args, device)

    prefix_data = np.load(args.prefix_npz, allow_pickle=True)
    if "rolls_flat" not in prefix_data.files or "offsets" not in prefix_data.files:
        raise KeyError("prefix_npz must contain rolls_flat and offsets.")
    target_lengths = resolve_target_lengths(prefix_data, args)
    n_seq = len(prefix_data["offsets"]) - 1

    rolls: list[np.ndarray] = []
    offsets = [0]
    generated_steps = []
    for i in range(n_seq):
        prefix = get_sequence(prefix_data, i)
        roll, gen_steps, debug = generate_one(
            model, tokenizer, prefix, int(target_lengths[i]), context_len, train_stats, args, device
        )
        if not np.array_equal(roll[: len(prefix)], prefix):
            raise RuntimeError("Internal error: generated roll changed the prefix.")
        rolls.append(roll)
        offsets.append(offsets[-1] + len(roll))
        generated_steps.append(gen_steps)
        print(
            f"sequence {i}: prefix={len(prefix)} total={len(roll)} generated={gen_steps} "
            f"generated_notes={debug['generated_note_count']} event_density={debug['final_generated_event_density']:.4f}"
        )
        if args.debug_generation_stats:
            print_debug(i, debug)

    out_rolls = np.concatenate(rolls, axis=0).astype(np.uint8, copy=False)
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "rolls_flat": out_rolls,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "ids": np.asarray(prefix_data["ids"]) if "ids" in prefix_data.files else np.arange(n_seq),
        "step_sec": np.asarray(scalar(prefix_data, "step_sec", tokenizer.step_sec)),
        "note_min": np.asarray(scalar(prefix_data, "note_min", tokenizer.note_min)),
        "note_max": np.asarray(scalar(prefix_data, "note_max", tokenizer.note_min + tokenizer.num_notes - 1)),
        "num_positions": np.asarray(tokenizer.num_notes),
        "representation": np.asarray("onset"),
        "is_prefix": np.asarray(False),
        "prefix_steps": (
            np.asarray(prefix_data["prefix_steps"])
            if "prefix_steps" in prefix_data.files
            else np.diff(np.asarray(prefix_data["offsets"], dtype=np.int64))
        ),
        "generated_steps": np.asarray(generated_steps, dtype=np.int64),
        "model_type": np.asarray(MODEL_TYPE),
    }
    np.savez_compressed(args.out_npz, **save)
    print(f"saved: {args.out_npz}")


if __name__ == "__main__":
    main()
