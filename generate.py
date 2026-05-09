from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import NestedPianoRollTransformer


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
    if isinstance(value, np.generic):
        value = value.item()
    return value


def get_sequence(data: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    offsets = np.asarray(data["offsets"], dtype=np.int64)
    return np.asarray(data["rolls_flat"][offsets[index] : offsets[index + 1]], dtype=np.uint8)


def target_lengths(prefix_data, args) -> list[int]:
    prefix_offsets = np.asarray(prefix_data["offsets"], dtype=np.int64)
    prefix_lengths = np.diff(prefix_offsets).astype(int).tolist()
    if args.target_full_npz and args.match_full_lengths:
        full = np.load(args.target_full_npz, allow_pickle=True)
        full_offsets = np.asarray(full["offsets"], dtype=np.int64)
        return np.diff(full_offsets).astype(int).tolist()
    return [int(n + args.continuation_steps) for n in prefix_lengths]


def build_model_from_checkpoint(ckpt, args, device):
    cfg = ckpt.get("config", {})
    model_context_len = int(cfg.get("context_len", args.context_len or 512))
    model = NestedPianoRollTransformer(
        num_notes=int(cfg.get("num_notes", 88)),
        d_model=int(cfg.get("d_model", 256)),
        n_layers=int(cfg.get("n_layers", 6)),
        n_heads=int(cfg.get("n_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        context_len=model_context_len,
        head_type=str(cfg.get("head_type", "mlp")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    generation_context_len = int(args.context_len or model_context_len)
    generation_context_len = min(generation_context_len, model_context_len)
    return model, cfg, generation_context_len


def sample_frame(
    probs: np.ndarray,
    rng: np.random.Generator,
    threshold: float,
    top_k_notes: int,
    min_notes: int,
    max_notes: int,
) -> np.ndarray:
    frame = (rng.random(probs.shape) < probs).astype(np.uint8)
    frame[probs < threshold] = 0
    active = np.flatnonzero(frame)
    cap = max_notes if max_notes > 0 else len(probs)
    if top_k_notes > 0:
        cap = min(cap, top_k_notes)
    if len(active) > cap:
        keep = active[np.argsort(probs[active])[-cap:]]
        frame[:] = 0
        frame[keep] = 1
    if min_notes > 0 and int(frame.sum()) < min_notes:
        keep = np.argsort(probs)[-min_notes:]
        frame[keep] = 1
    return frame


@torch.no_grad()
def generate_one(model, prefix, total_len, generation_context_len, args, device, rng) -> tuple[np.ndarray, int]:
    num_notes = prefix.shape[1]
    out = np.zeros((total_len, num_notes), dtype=np.uint8)
    prefix_len = min(len(prefix), total_len)
    out[:prefix_len] = prefix[:prefix_len]
    repeated = 0
    last_generated = None

    for t in range(prefix_len, total_len):
        start = max(0, t - generation_context_len)
        ctx = out[start:t].astype(np.float32)
        if len(ctx) == 0:
            ctx = np.zeros((1, num_notes), dtype=np.float32)
        x = torch.from_numpy(ctx).unsqueeze(0).to(device)
        logits = model(x)[0, -1].float().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits / max(args.temperature, 1e-6)))
        if args.target_density > 0:
            current_density = float(probs.mean())
            if current_density > args.target_density * 4.0:
                probs *= (args.target_density * 4.0) / max(current_density, 1e-8)
        frame = sample_frame(
            probs,
            rng,
            args.note_threshold,
            args.top_k_notes,
            args.min_notes_per_event,
            args.max_notes_per_event,
        )
        if last_generated is not None and np.array_equal(frame, last_generated):
            repeated += 1
        else:
            repeated = 0
        if repeated >= args.repetition_patience and frame.sum() > 0:
            drop = np.flatnonzero(frame)
            frame[rng.choice(drop)] = 0
            repeated = 0
        out[t] = frame
        last_generated = frame.copy()
    return out, max(0, total_len - prefix_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate continuations from any prefix NPZ.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix_npz", type=Path, required=True)
    parser.add_argument("--out_npz", type=Path, required=True)
    parser.add_argument("--continuation_steps", type=int, default=2048)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--note_threshold", type=float, default=0.35)
    parser.add_argument("--top_k_notes", type=int, default=8)
    parser.add_argument("--min_notes_per_event", type=int, default=0)
    parser.add_argument("--max_notes_per_event", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target_full_npz", type=Path, default=None)
    parser.add_argument("--match_full_lengths", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target_density", type=float, default=-1.0)
    parser.add_argument("--repetition_patience", type=int, default=8)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model, cfg, generation_context_len = build_model_from_checkpoint(ckpt, args, device)
    if args.target_density < 0:
        args.target_density = float(cfg.get("note_density", 0.0))

    prefix_data = np.load(args.prefix_npz, allow_pickle=True)
    if "rolls_flat" not in prefix_data.files or "offsets" not in prefix_data.files:
        raise KeyError("Prefix NPZ must contain rolls_flat and offsets.")
    lengths = target_lengths(prefix_data, args)
    n_seq = len(prefix_data["offsets"]) - 1
    if len(lengths) != n_seq:
        raise ValueError("Target full NPZ sequence count does not match prefix NPZ sequence count.")

    rolls = []
    offsets = [0]
    generated_steps = []
    for i in range(n_seq):
        prefix = get_sequence(prefix_data, i)
        total_len = int(lengths[i])
        if total_len < len(prefix):
            raise ValueError(f"Target length {total_len} is shorter than prefix length {len(prefix)} for index {i}.")
        roll, gen = generate_one(model, prefix, total_len, generation_context_len, args, device, rng)
        if not np.array_equal(roll[: len(prefix)], prefix):
            raise RuntimeError("Internal error: generated output changed the prefix.")
        rolls.append(roll)
        offsets.append(offsets[-1] + len(roll))
        generated_steps.append(gen)
        print(f"generated sequence {i}: prefix={len(prefix)} total={len(roll)} generated={gen}")

    out_rolls = np.concatenate(rolls, axis=0).astype(np.uint8, copy=False)
    out_offsets = np.asarray(offsets, dtype=np.int64)
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "rolls_flat": out_rolls,
        "offsets": out_offsets,
        "ids": np.asarray(prefix_data["ids"]) if "ids" in prefix_data.files else np.arange(n_seq),
        "step_sec": np.asarray(scalar(prefix_data, "step_sec", 0.05)),
        "note_min": np.asarray(scalar(prefix_data, "note_min", 21)),
        "note_max": np.asarray(scalar(prefix_data, "note_max", 108)),
        "num_positions": np.asarray(scalar(prefix_data, "num_positions", out_rolls.shape[1])),
        "representation": np.asarray(scalar(prefix_data, "representation", "onset")),
        "is_prefix": np.asarray(False),
        "generated_steps": np.asarray(generated_steps, dtype=np.int64),
    }
    if "prefix_steps" in prefix_data.files:
        save["prefix_steps"] = np.asarray(prefix_data["prefix_steps"])
    else:
        save["prefix_steps"] = np.diff(np.asarray(prefix_data["offsets"], dtype=np.int64))
    np.savez_compressed(args.out_npz, **save)
    print(f"saved: {args.out_npz}")


if __name__ == "__main__":
    main()
