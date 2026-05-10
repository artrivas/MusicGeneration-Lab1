from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import EventChunkDataset, split_sequence_indices
from model import EventStateMusicTransformer, event_state_loss
from vocab import ChordVocab, build_chord_vocab, load_rolls_flat_npz


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def serializable_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


@torch.no_grad()
def evaluate(model: EventStateMusicTransformer, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(batch["delta_in"], batch["chord_in"], batch["notes_in"], batch["cum_in"], batch["card_in"])
            _, metrics = event_state_loss(outputs, batch)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Event-State Music Transformer.")
    parser.add_argument("--data-dir", type=Path, default=Path(".."))
    parser.add_argument("--train-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/event_state"))
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50_000)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--max-delta", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_npz = args.train_npz or (args.data_dir / "train.npz")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.vocab or (args.out_dir / "vocab.pkl")

    if vocab_path.exists():
        vocab = ChordVocab.load(vocab_path)
        print(f"loaded vocab: {vocab.size} tokens")
    else:
        print("building chord vocabulary...")
        vocab = build_chord_vocab(train_npz, top_k=args.top_k)
        vocab.save(vocab_path)
        print(f"saved vocab: {vocab.size} tokens -> {vocab_path}")

    _, offsets, _ = load_rolls_flat_npz(train_npz)
    train_idx, val_idx = split_sequence_indices(len(offsets) - 1, val_frac=args.val_frac, seed=args.seed)
    print(f"split sequences: train={len(train_idx)} val={len(val_idx)}")

    train_ds = EventChunkDataset(train_npz, vocab, train_idx, args.context_len, args.max_delta, random_crop=True)
    val_ds = EventChunkDataset(train_npz, vocab, val_idx, args.context_len, args.max_delta, random_crop=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventStateMusicTransformer(
        vocab_size=vocab.size,
        max_delta=args.max_delta,
        num_notes=vocab.num_notes,
        max_seq_len=args.context_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, args.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    config = serializable_config(args)
    config.update({"vocab_size": vocab.size, "num_notes": vocab.num_notes, "vocab_path": str(vocab_path)})
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(batch["delta_in"], batch["chord_in"], batch["notes_in"], batch["cum_in"], batch["card_in"])
                loss, _ = event_state_loss(outputs, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach().cpu())
            global_step += 1
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} train_loss={running / step:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch} train_loss={running / max(1, len(train_loader)):.4f} "
            f"val_loss={val_metrics['loss']:.4f} delta={val_metrics['delta_ce']:.4f} "
            f"chord={val_metrics['chord_ce']:.4f} note={val_metrics['note_bce']:.4f}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, args.out_dir / "best.pt")
            print(f"saved best checkpoint: {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
