from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dataset import roll_to_events
from model import EventStateMusicTransformer
from vocab import BOS, EOS, PAD, UNK_CHORD, ChordVocab, load_rolls_flat_npz, scalar, sequence_view


def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float | None = None) -> int:
    logits = logits.float() / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = sorted_probs[keep]
        probs = filtered / filtered.sum().clamp_min(1e-12)
    return int(torch.multinomial(probs, 1).item())


def sample_notes_with_cardinality(
    note_logits: torch.Tensor,
    k: int,
    temperature: float = 0.9,
    top_p: float | None = None,
) -> np.ndarray:
    k = max(1, min(int(k), int(note_logits.numel())))
    logits = note_logits.float() / max(temperature, 1e-6)
    probs = torch.sigmoid(logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs / sorted_probs.sum().clamp_min(1e-12), dim=-1)
        keep = cdf <= top_p
        keep[0] = True
        allowed = torch.zeros_like(probs, dtype=torch.bool)
        allowed[sorted_idx[keep]] = True
        probs = probs.masked_fill(~allowed, 0.0)
    if int((probs > 0).sum().item()) < k:
        idx = torch.topk(probs, k=k).indices
    else:
        idx = torch.multinomial(probs / probs.sum().clamp_min(1e-12), k, replacement=False)
    sampled = torch.zeros_like(probs, dtype=torch.uint8)
    sampled[idx] = 1
    return sampled.cpu().numpy().astype(np.uint8)


def sample_cardinality(card_logits: torch.Tensor, temperature: float, min_notes: int, max_notes: int) -> int:
    logits = card_logits.float().clone()
    logits[: max(1, min_notes)] = -float("inf")
    logits[max_notes + 1 :] = -float("inf")
    if max_notes >= 5:
        logits[5 : max_notes + 1] -= 0.75
    if max_notes >= 8:
        logits[8:] = -float("inf")
    k = sample_logits(logits, temperature=temperature)
    return max(min_notes, min(int(k), max_notes))


def build_model_from_checkpoint(checkpoint: Path, device: torch.device) -> tuple[EventStateMusicTransformer, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = EventStateMusicTransformer(
        vocab_size=int(cfg["vocab_size"]),
        max_delta=int(cfg["max_delta"]),
        num_notes=int(cfg.get("num_notes", 88)),
        max_seq_len=int(cfg["context_len"]),
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        d_ff=int(cfg["d_ff"]),
        dropout=float(cfg["dropout"]),
        use_rope=bool(cfg.get("use_rope", False)),
    ).to(device)
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "warning: loaded checkpoint with non-strict state dict; "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, cfg


@torch.no_grad()
def generate_one(
    model: EventStateMusicTransformer,
    vocab: ChordVocab,
    prefix_roll: np.ndarray,
    total_steps: int,
    device: torch.device,
    max_delta: int,
    context_len: int,
    delta_temp: float,
    chord_temp: float,
    chord_top_p: float,
    note_temp: float,
    card_temp: float,
    min_notes: int,
    max_notes: int,
    repeat_penalty: float,
    max_same_chord_run: int,
    prefer_note_head_prob: float,
) -> np.ndarray:
    out = np.zeros((total_steps, vocab.num_notes), dtype=np.uint8)
    copy_len = min(len(prefix_roll), total_steps)
    out[:copy_len] = prefix_roll[:copy_len, : vocab.num_notes]

    ev = roll_to_events(prefix_roll, vocab, max_delta=max_delta, add_bos=True, add_eos=False)
    deltas = ev.delta.astype(np.int64).tolist()
    chords = ev.chord.astype(np.int64).tolist()
    notes = ev.notes.astype(np.float32).tolist()
    cums = ev.cum.astype(np.int64).tolist()
    cards = ev.card.astype(np.int64).tolist()

    cumulative_step = copy_len - 1
    if len(ev.cum) > 1:
        cumulative_step = max(cumulative_step, int(ev.cum[-1]))

    while cumulative_step < total_steps - 1:
        sl = slice(max(0, len(chords) - context_len), len(chords))
        batch = {
            "delta": torch.tensor([deltas[sl]], dtype=torch.long, device=device),
            "chord": torch.tensor([chords[sl]], dtype=torch.long, device=device),
            "notes": torch.tensor([notes[sl]], dtype=torch.float32, device=device),
            "cum": torch.tensor([cums[sl]], dtype=torch.long, device=device),
            "card": torch.tensor([cards[sl]], dtype=torch.long, device=device),
        }
        outputs = model(batch["delta"], batch["chord"], batch["notes"], batch["cum"], batch["card"])
        delta_logits = outputs["delta_logits"][0, -1].clone()
        chord_logits = outputs["chord_logits"][0, -1].clone()
        note_logits = outputs["note_logits"][0, -1]
        card_logits = outputs["card_logits"][0, -1]

        delta_logits[0] = -float("inf")
        delta_logits[max_delta + 1 :] = -float("inf")
        sampled_delta = sample_logits(delta_logits, temperature=delta_temp)
        sampled_delta = max(1, min(int(sampled_delta), max_delta))

        chord_logits[PAD] = -float("inf")
        chord_logits[BOS] = -float("inf")
        chord_logits[EOS] = -float("inf")
        recent_counts = Counter(chords[-64:])
        penalty = max(1.0, repeat_penalty)
        if penalty > 1.0:
            for chord_id, count in recent_counts.items():
                if 0 <= chord_id < chord_logits.numel():
                    chord_logits[chord_id] -= float(count) * float(np.log(penalty))
        if max_same_chord_run > 0 and chords:
            run_id = chords[-1]
            run_len = 0
            for chord_id in reversed(chords):
                if chord_id != run_id:
                    break
                run_len += 1
            if run_len >= max_same_chord_run and 0 <= run_id < chord_logits.numel():
                chord_logits[run_id] = -float("inf")
        sampled_chord = sample_logits(chord_logits, temperature=chord_temp, top_p=chord_top_p)
        if sampled_chord == EOS:
            sampled_chord = UNK_CHORD
        sampled_card = sample_cardinality(card_logits, card_temp, min_notes, max_notes)

        cumulative_step += sampled_delta
        if cumulative_step >= total_steps:
            break

        decoded = vocab.decode_id(sampled_chord)
        use_note_head = decoded is None or random.random() < prefer_note_head_prob
        if decoded is not None:
            decoded = decoded.astype(np.uint8, copy=False)
            decoded_card = int(decoded.sum())
            if decoded_card <= 0 or decoded_card > max_notes:
                use_note_head = True
        if use_note_head:
            decoded = sample_notes_with_cardinality(note_logits, sampled_card, temperature=note_temp)
        decoded = decoded.astype(np.uint8, copy=False)
        out[cumulative_step, : vocab.num_notes] = decoded

        deltas.append(sampled_delta)
        chords.append(sampled_chord)
        notes.append(decoded.astype(np.float32).tolist())
        cums.append(cumulative_step)
        cards.append(int(decoded.sum()))

    out[:copy_len] = prefix_roll[:copy_len, : vocab.num_notes]
    return out


def save_generated_npz(out_path: Path, rolls: list[np.ndarray], source_data: Any, ids: np.ndarray) -> None:
    rolls_flat = np.concatenate(rolls, axis=0).astype(np.uint8, copy=False) if rolls else np.zeros((0, 88), dtype=np.uint8)
    offsets = np.zeros(len(rolls) + 1, dtype=np.int64)
    total = 0
    for i, roll in enumerate(rolls):
        total += len(roll)
        offsets[i + 1] = total
    payload = {
        "rolls_flat": rolls_flat,
        "offsets": offsets,
        "ids": ids,
        "step_sec": scalar(source_data, "step_sec", 0.05),
        "note_min": scalar(source_data, "note_min", 21),
        "note_max": scalar(source_data, "note_max", 108),
        "num_positions": scalar(source_data, "num_positions", rolls_flat.shape[1] if rolls_flat.size else 88),
        "representation": scalar(source_data, "representation", "onset"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate continuations with an Event-State Music Transformer.")
    parser.add_argument("--data-dir", type=Path, default=Path(".."))
    parser.add_argument("--prefix-npz", type=Path, default=None)
    parser.add_argument("--full-npz", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/event_state/eval_set_01_generated_event_state.npz"))
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--delta-temp", type=float, default=0.8)
    parser.add_argument("--chord-temp", type=float, default=0.95)
    parser.add_argument("--chord-top-p", type=float, default=0.92)
    parser.add_argument("--note-temp", type=float, default=0.9)
    parser.add_argument("--card-temp", type=float, default=0.9)
    parser.add_argument("--max-notes", type=int, default=7)
    parser.add_argument("--min-notes", type=int, default=1)
    parser.add_argument("--repeat-penalty", type=float, default=1.15)
    parser.add_argument("--max-same-chord-run", type=int, default=3)
    parser.add_argument("--prefer-note-head-prob", type=float, default=0.25)
    args = parser.parse_args()
    args.min_notes = max(1, int(args.min_notes))
    args.max_notes = max(args.min_notes, int(args.max_notes))

    prefix_npz = args.prefix_npz or (args.data_dir / "eval_set_01_prefix.npz")
    full_npz = args.full_npz or (args.data_dir / "eval_set_01_full.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model_from_checkpoint(args.checkpoint, device)
    vocab = ChordVocab.load(args.vocab or Path(cfg["vocab_path"]))
    max_delta = int(cfg["max_delta"])
    context_len = int(cfg["context_len"])

    prefix_flat, prefix_offsets, prefix_data = load_rolls_flat_npz(prefix_npz)
    ids = np.asarray(prefix_data["ids"]) if "ids" in prefix_data.files else np.asarray([f"seq_{i:04d}" for i in range(len(prefix_offsets) - 1)])

    if args.total_steps is not None:
        total_steps = int(args.total_steps)
        target_lengths = [total_steps] * (len(prefix_offsets) - 1)
    elif full_npz.exists():
        _, full_offsets, _ = load_rolls_flat_npz(full_npz)
        target_lengths = [int(full_offsets[i + 1] - full_offsets[i]) for i in range(len(prefix_offsets) - 1)]
    else:
        prefix_lengths = np.diff(prefix_offsets)
        target_lengths = [int(x) * 4 for x in prefix_lengths]

    rolls: list[np.ndarray] = []
    for i in range(len(prefix_offsets) - 1):
        prefix = sequence_view(prefix_flat, prefix_offsets, i)
        total = max(int(target_lengths[i]), len(prefix))
        roll = generate_one(
            model,
            vocab,
            prefix,
            total,
            device,
            max_delta,
            context_len,
            args.delta_temp,
            args.chord_temp,
            args.chord_top_p,
            args.note_temp,
            args.card_temp,
            args.min_notes,
            args.max_notes,
            args.repeat_penalty,
            args.max_same_chord_run,
            args.prefer_note_head_prob,
        )
        rolls.append(roll)
        print(f"generated {i + 1}/{len(prefix_offsets) - 1}: prefix={len(prefix)} total={len(roll)} events={int((roll.sum(axis=1) > 0).sum())}")

    save_generated_npz(args.out, rolls, prefix_data, ids)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
