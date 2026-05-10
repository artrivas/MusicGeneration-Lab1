from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_transformer import CausalTransformerLM
from tokenizer_mirexlike import MirexLikePianoTokenizer
from utils_npz import avg_notes_per_event, event_density, get_dataset_defaults, load_npz_sequences, longest_silent_run, save_rolls_flat_npz


def build_model_from_checkpoint(ckpt: dict, args: argparse.Namespace, vocab_size: int) -> torch.nn.Module:
    cfg = dict(ckpt.get("config", {}))
    model_type = args.model_type or cfg.get("model_type", "transformer")
    if model_type == "rwkv":
        from model_rwkv import RWKVLanguageModel

        return RWKVLanguageModel()
    return CausalTransformerLM(
        vocab_size=vocab_size,
        context_len=args.context_len or int(cfg.get("context_len", 1024)),
        d_model=int(cfg.get("d_model", 384)),
        n_layers=int(cfg.get("n_layers", 8)),
        n_heads=int(cfg.get("n_heads", 8)),
        dropout=float(cfg.get("dropout", 0.0)),
    )


def apply_repetition_penalty(logits: torch.Tensor, recent: deque[int], penalty: float) -> None:
    if penalty <= 1.0:
        return
    for token in set(recent):
        logits[token] = logits[token] / penalty if logits[token] > 0 else logits[token] * penalty


def block_repeated_ngram(logits: torch.Tensor, tokens: list[int], n: int) -> None:
    if n <= 1 or len(tokens) < n - 1:
        return
    prefix = tuple(tokens[-(n - 1) :])
    banned = set()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i : i + n - 1]) == prefix:
            banned.add(tokens[i + n - 1])
    for token in banned:
        logits[int(token)] = -float("inf")


def top_k_top_p_sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    logits = logits / max(1e-6, float(temperature))
    if top_k > 0 and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < threshold, torch.full_like(logits, -float("inf")), logits)
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits[remove] = -float("inf")
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(0, sorted_idx, sorted_logits)
        logits = filtered
    probs = F.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
        probs = torch.ones_like(logits) / logits.numel()
    return int(torch.multinomial(probs, num_samples=1).item())


def target_length_for_sequence(args: argparse.Namespace, prefix_len: int, full_rolls: list[np.ndarray] | None, index: int) -> int:
    if args.match_full_lengths:
        if full_rolls is None:
            raise ValueError("--match_full_lengths requires --target_full_npz")
        return int(len(full_rolls[index]))
    if args.target_total_steps is not None:
        return int(args.target_total_steps)
    return int(prefix_len + args.continuation_steps)


def generated_stats(roll: np.ndarray) -> dict[str, float | int]:
    return {
        "event_density": event_density(roll),
        "note_count": int(np.sum(roll > 0)),
        "unique_patterns": int(len({tuple(np.flatnonzero(f > 0).tolist()) for f in roll if np.any(f > 0)})),
        "longest_silent_run": longest_silent_run(roll),
        "avg_notes_per_event": avg_notes_per_event(roll),
    }


@torch.no_grad()
def generate_one(
    model: torch.nn.Module,
    tokenizer: MirexLikePianoTokenizer,
    prefix_roll: np.ndarray,
    target_total_steps: int,
    args: argparse.Namespace,
    train_density: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float | int]]:
    prefix_tokens = tokenizer.encode_roll(prefix_roll, add_bos_eos=True)
    if prefix_tokens and prefix_tokens[-1] == tokenizer.EOS:
        prefix_tokens = prefix_tokens[:-1]
    tokens = list(prefix_tokens)
    generated_tokens: list[int] = []
    recent = deque(maxlen=128)
    prefix_density = event_density(prefix_roll)
    if args.target_density_auto:
        target_density = 0.60 * float(train_density) + 0.40 * float(prefix_density)
    else:
        target_density = float(args.target_density)
    target_density = float(np.clip(target_density, args.min_target_density, args.max_target_density))

    gen_frames = 0
    gen_event_frames = 0
    generated_note_count = 0
    current_silent_run = 0
    open_chord_notes: set[int] = set()
    open_chord_tokens = 0
    density_acts = silence_acts = chord_acts = 0
    context_len = int(args.context_len)

    time_ids = tokenizer.time_shift_token_ids()
    note_ids = tokenizer.note_on_token_ids()

    while len(prefix_roll) + gen_frames < target_total_steps and len(generated_tokens) < args.max_new_tokens:
        x = torch.tensor(tokens[-context_len:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1].float()
        logits[tokenizer.PAD] = -float("inf")
        logits[tokenizer.BOS] = -float("inf")
        if len(prefix_roll) + gen_frames < target_total_steps - 2:
            logits[tokenizer.EOS] -= 2.0

        apply_repetition_penalty(logits, recent, args.repetition_penalty)
        block_repeated_ngram(logits, tokens, args.no_repeat_ngram_size)

        if args.note_repetition_penalty > 1.0:
            for note_idx in open_chord_notes:
                token_id = tokenizer.note_on_id(note_idx)
                logits[token_id] = logits[token_id] / args.note_repetition_penalty if logits[token_id] > 0 else logits[token_id] * args.note_repetition_penalty

        current_density = gen_event_frames / max(1, gen_frames)
        if args.density_control and gen_frames >= 8:
            if current_density < target_density - args.density_margin:
                logits[time_ids] -= args.density_strength
                logits[note_ids] += 0.5 * args.density_strength
                density_acts += 1
            elif current_density > target_density + args.density_margin:
                logits[time_ids] += args.density_strength
                logits[note_ids] -= 0.5 * args.density_strength
                density_acts += 1

        if current_silent_run > args.max_silent_frames:
            logits[time_ids] -= args.silence_penalty
            logits[note_ids] += args.note_boost_after_silence
            silence_acts += 1

        if open_chord_notes:
            if open_chord_tokens >= args.max_open_chord_tokens or len(open_chord_notes) >= args.max_notes_per_chord:
                logits[tokenizer.CHORD_END] += 5.0
                logits[note_ids] -= 2.0
            else:
                logits[tokenizer.CHORD_END] += 0.3
            chord_acts += 1
        else:
            logits[tokenizer.CHORD_END] -= 3.0

        token = top_k_top_p_sample(logits, args.temperature, args.top_k, args.top_p)
        tokens.append(token)
        generated_tokens.append(token)
        recent.append(token)

        if tokenizer.is_time_shift(token):
            if open_chord_notes:
                gen_frames += 1
                gen_event_frames += 1
                generated_note_count += len(open_chord_notes)
                open_chord_notes.clear()
                open_chord_tokens = 0
                current_silent_run = 0
            shift = tokenizer.get_time_shift_value(token)
            remaining = target_total_steps - len(prefix_roll) - gen_frames
            used = max(0, min(shift, remaining))
            gen_frames += used
            current_silent_run += used
        elif tokenizer.is_note_on(token):
            open_chord_notes.add(tokenizer.get_note_pitch(token) - tokenizer.note_min)
            open_chord_tokens += 1
        elif tokenizer.is_chord_end(token):
            if open_chord_notes:
                gen_frames += 1
                gen_event_frames += 1
                generated_note_count += len(open_chord_notes)
                current_silent_run = 0
            open_chord_notes.clear()
            open_chord_tokens = 0
        elif token == tokenizer.EOS:
            tokens.pop()
            generated_tokens.pop()
            recent.pop()
            logits[tokenizer.EOS] = -float("inf")
            continue

    decoded = tokenizer.decode_tokens(tokens, target_steps=target_total_steps)
    if len(decoded) < target_total_steps:
        pad = np.zeros((target_total_steps - len(decoded), tokenizer.num_notes), dtype=np.uint8)
        decoded = np.concatenate([decoded, pad], axis=0)
    final_roll = decoded[:target_total_steps].astype(np.uint8, copy=True)
    final_roll[: len(prefix_roll)] = prefix_roll.astype(np.uint8, copy=False)
    continuation = final_roll[len(prefix_roll) :]
    stats = generated_stats(continuation)
    stats.update(
        {
            "prefix_steps": int(len(prefix_roll)),
            "target_total_steps": int(target_total_steps),
            "generated_steps": int(len(continuation)),
            "prefix_event_density": float(prefix_density),
            "target_density": float(target_density),
            "final_generated_event_density": float(stats["event_density"]),
            "generated_note_count": int(stats["note_count"]),
            "unique_generated_note_patterns": int(stats["unique_patterns"]),
            "number_of_density_control_activations": int(density_acts),
            "number_of_silence_control_activations": int(silence_acts),
            "number_of_chord_end_control_activations": int(chord_acts),
        }
    )
    return final_roll, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate symbolic piano continuation from prefix .npz.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prefix_npz", type=Path, required=True)
    parser.add_argument("--out_npz", type=Path, required=True)
    parser.add_argument("--continuation_steps", type=int, default=2048)
    parser.add_argument("--target_total_steps", type=int)
    parser.add_argument("--target_full_npz", type=Path)
    parser.add_argument("--match_full_lengths", action="store_true")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--model_type", choices=["transformer", "rwkv"])
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top_k", type=int, default=48)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--density_control", action="store_true")
    parser.add_argument("--target_density_auto", action="store_true")
    parser.add_argument("--target_density", type=float, default=0.18)
    parser.add_argument("--min_target_density", type=float, default=0.08)
    parser.add_argument("--max_target_density", type=float, default=0.32)
    parser.add_argument("--density_margin", type=float, default=0.03)
    parser.add_argument("--density_strength", type=float, default=0.5)
    parser.add_argument("--max_silent_frames", type=int, default=80)
    parser.add_argument("--silence_penalty", type=float, default=1.5)
    parser.add_argument("--note_boost_after_silence", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--note_repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--max_notes_per_chord", type=int, default=8)
    parser.add_argument("--max_open_chord_tokens", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=20000)
    parser.add_argument("--debug_generation_stats", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = MirexLikePianoTokenizer.load(args.tokenizer)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(ckpt, args, tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prefix_rolls, prefix_data = load_npz_sequences(args.prefix_npz)
    full_rolls = load_npz_sequences(args.target_full_npz)[0] if args.target_full_npz else None
    defaults = get_dataset_defaults(prefix_data)
    dataset_info_path = args.tokenizer.parent / "dataset_info.json"
    train_density = 0.18
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as f:
            train_density = float(json.load(f).get("train_event_density", train_density))
    ids = np.asarray(prefix_data["ids"]) if "ids" in prefix_data.files else np.asarray([f"seq_{i:04d}" for i in range(len(prefix_rolls))])

    out_rolls = []
    for i, prefix_roll in enumerate(prefix_rolls):
        target_steps = target_length_for_sequence(args, len(prefix_roll), full_rolls, i)
        target_steps = max(target_steps, len(prefix_roll))
        final_roll, stats = generate_one(model, tokenizer, prefix_roll, target_steps, args, train_density, device)
        out_rolls.append(final_roll)
        if args.debug_generation_stats:
            print(f"sequence {i}")
            for key in [
                "prefix_steps",
                "target_total_steps",
                "generated_steps",
                "prefix_event_density",
                "target_density",
                "final_generated_event_density",
                "generated_note_count",
                "unique_generated_note_patterns",
                "longest_silent_run",
                "avg_notes_per_event",
                "number_of_density_control_activations",
                "number_of_silence_control_activations",
                "number_of_chord_end_control_activations",
            ]:
                print(f"  {key}: {stats[key]}")

    metadata = dict(defaults)
    metadata.update({"is_generated": True, "source_prefix_npz": str(args.prefix_npz)})
    out = save_rolls_flat_npz(args.out_npz, out_rolls, ids=ids, metadata=metadata)
    print(f"saved generated npz: {out}")


if __name__ == "__main__":
    main()
