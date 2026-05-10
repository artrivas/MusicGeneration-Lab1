from __future__ import annotations

import argparse
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
    generation_context_len = min(int(args.context_len or model_context_len), model_context_len)
    return model, generation_context_len


def apply_repetition_penalty(logits: torch.Tensor, recent_tokens: list[int], penalty: float) -> None:
    if penalty <= 1.0:
        return
    for token_id in set(recent_tokens):
        if 0 <= token_id < logits.numel():
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty


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


def sample_token(
    logits: torch.Tensor,
    tokenizer: PianoEventTokenizer,
    args,
    recent_tokens: list[int],
    generated_frames: int,
    generated_events: int,
    consecutive_shift_frames: int,
    consecutive_chords: int,
) -> int:
    logits = logits.float().clone()
    logits[tokenizer.PAD] = -float("inf")
    logits[tokenizer.BOS] = -float("inf")
    logits[tokenizer.EOS] = -float("inf")
    logits[tokenizer.UNK_CHORD] = -float("inf")

    apply_repetition_penalty(logits, recent_tokens, args.repetition_penalty)

    chord_slice = slice(tokenizer.chord_start, tokenizer.vocab_size)
    shift_slice = slice(tokenizer.time_shift_start, tokenizer.chord_start)
    target_density = max(float(tokenizer.event_density), 1e-5)
    current_density = generated_events / max(1, generated_frames)
    if generated_frames >= 16 and current_density < target_density * 0.35:
        logits[chord_slice] += args.density_boost
    if generated_frames >= 16 and current_density > target_density * 2.5:
        logits[shift_slice] += args.density_boost
        logits[chord_slice] -= args.density_boost
    if consecutive_shift_frames >= args.max_silent_frames:
        logits[chord_slice] += args.degeneracy_boost
    if consecutive_chords >= args.max_consecutive_chords:
        logits[shift_slice] += args.degeneracy_boost
        logits[chord_slice] -= args.degeneracy_boost

    logits = logits / max(args.temperature, 1e-6)
    filtered = filter_top_k_top_p(logits, args.top_k, args.top_p)
    probs = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def token_frame_delta(tokenizer: PianoEventTokenizer, token_id: int) -> tuple[int, int]:
    if tokenizer.is_time_shift(token_id):
        return tokenizer.time_shift_value(token_id), 0
    if tokenizer.is_chord(token_id) or token_id == tokenizer.UNK_CHORD:
        return 1, 1 if tokenizer.is_chord(token_id) else 0
    return 0, 0


@torch.no_grad()
def generate_one(
    model,
    tokenizer: PianoEventTokenizer,
    prefix_roll: np.ndarray,
    target_steps: int,
    context_len: int,
    args,
    device,
) -> tuple[np.ndarray, int]:
    prefix_len = len(prefix_roll)
    if target_steps < prefix_len:
        raise ValueError(f"target_steps={target_steps} is shorter than prefix length {prefix_len}.")

    tokens = tokenizer.encode_roll(prefix_roll, add_bos=True, add_eos=False)
    generated_frames = 0
    generated_events = 0
    consecutive_shift_frames = 0
    consecutive_chords = 0
    max_new_tokens = max(args.max_new_tokens, (target_steps - prefix_len) * 4 + 128)

    for _ in range(max_new_tokens):
        if prefix_len + generated_frames >= target_steps:
            break
        ctx = tokens[-context_len:]
        input_ids = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_ids)[0, -1]
        next_id = sample_token(
            logits=logits,
            tokenizer=tokenizer,
            args=args,
            recent_tokens=tokens[-args.repetition_window :],
            generated_frames=generated_frames,
            generated_events=generated_events,
            consecutive_shift_frames=consecutive_shift_frames,
            consecutive_chords=consecutive_chords,
        )
        tokens.append(next_id)
        frame_delta, event_delta = token_frame_delta(tokenizer, next_id)
        generated_frames += frame_delta
        generated_events += event_delta
        if tokenizer.is_time_shift(next_id):
            consecutive_shift_frames += frame_delta
            consecutive_chords = 0
        elif tokenizer.is_chord(next_id):
            consecutive_shift_frames = 0
            consecutive_chords += 1
        else:
            consecutive_chords = 0

    decoded = tokenizer.decode_tokens(tokens, target_steps=target_steps)
    if len(decoded) < target_steps:
        padded = np.zeros((target_steps, tokenizer.num_notes), dtype=np.uint8)
        padded[: len(decoded)] = decoded
        decoded = padded
    decoded = decoded[:target_steps].astype(np.uint8, copy=False)
    decoded[:prefix_len] = prefix_roll.astype(np.uint8, copy=False)

    generated = decoded[prefix_len:]
    if generated.sum() == 0 and len(generated) > 0:
        # Last-resort anti-silence fallback: place a safe single-note onset on a sparse grid.
        note = min(max(60 - tokenizer.note_min, 0), tokenizer.num_notes - 1)
        step = max(8, int(round(1.0 / tokenizer.step_sec)))
        generated[::step, note] = 1
        decoded[prefix_len:] = generated
    return decoded, target_steps - prefix_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate piano-roll continuations with token chord Transformer.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prefix_npz", type=Path, required=True)
    parser.add_argument("--out_npz", type=Path, required=True)
    parser.add_argument("--continuation_steps", type=int, default=2048)
    parser.add_argument("--target_total_steps", type=int, default=None)
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--repetition_window", type=int, default=64)
    parser.add_argument("--max_silent_frames", type=int, default=160)
    parser.add_argument("--max_consecutive_chords", type=int, default=8)
    parser.add_argument("--density_boost", type=float, default=1.0)
    parser.add_argument("--degeneracy_boost", type=float, default=2.0)
    parser.add_argument("--max_new_tokens", type=int, default=12000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target_full_npz", type=Path, default=None)
    parser.add_argument("--match_full_lengths", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    tokenizer = PianoEventTokenizer.load(args.tokenizer)
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
        roll, gen_steps = generate_one(model, tokenizer, prefix, int(target_lengths[i]), context_len, args, device)
        if not np.array_equal(roll[: len(prefix)], prefix):
            raise RuntimeError("Internal error: generated roll changed the prefix.")
        rolls.append(roll)
        offsets.append(offsets[-1] + len(roll))
        generated_steps.append(gen_steps)
        gen_notes = int(roll[len(prefix) :].sum())
        print(f"sequence {i}: prefix={len(prefix)} total={len(roll)} generated={gen_steps} generated_notes={gen_notes}")

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
