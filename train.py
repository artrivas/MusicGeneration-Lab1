from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import PianoRollWindowDataset, estimate_note_density, estimate_pos_weight
from model import NestedPianoRollTransformer
from prepare_cache import prepare_cache


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path, model, optimizer, scaler, epoch, step, best_val, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
            "config": config,
        },
        path,
    )


def run_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled, grad_clip) -> float:
    model.train()
    total = 0.0
    for step, window in enumerate(loader, start=1):
        window = window.to(device, non_blocking=True)
        x = window[:, :-1]
        y = window[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.detach().cpu())
        if step % 50 == 0:
            print(f"step {step}/{len(loader)} loss={total / step:.5f}")
    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device, amp_enabled) -> float:
    model.eval()
    total = 0.0
    for window in loader:
        window = window.to(device, non_blocking=True)
        x = window[:, :-1]
        y = window[:, 1:]
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss = criterion(model(x), y)
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a causal nested piano-roll Transformer.")
    parser.add_argument("--train_npz", type=Path, default=Path("train.npz"))
    parser.add_argument("--cache_dir", type=Path, default=Path("cache"))
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--val_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_type", choices=["mlp", "nested"], default="mlp")
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pos_weight", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    prepare_cache(args.train_npz, args.cache_dir)
    device = pick_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    density = estimate_note_density(args.cache_dir, seed=args.seed)
    if args.pos_weight == "auto":
        pos_weight_value = estimate_pos_weight(args.cache_dir, seed=args.seed)
    else:
        pos_weight_value = float(args.pos_weight)

    config = vars(args).copy()
    config.update(
        {
            "train_npz": str(args.train_npz),
            "cache_dir": str(args.cache_dir),
            "out_dir": str(args.out_dir),
            "resume": str(args.resume) if args.resume else None,
            "num_notes": 88,
            "note_density": density,
            "pos_weight_value": pos_weight_value,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_ds = PianoRollWindowDataset(
        args.cache_dir,
        context_len=args.context_len,
        split="train",
        steps_per_epoch=args.steps_per_epoch * args.batch_size,
        seed=args.seed,
    )
    val_ds = PianoRollWindowDataset(
        args.cache_dir,
        context_len=args.context_len,
        split="val",
        steps_per_epoch=args.val_steps * args.batch_size,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2, pin_memory=device.type == "cuda")

    model = NestedPianoRollTransformer(
        num_notes=88,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        context_len=args.context_len,
        head_type=args.head_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 1
    best_val = float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state") and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(f"device={device} amp={amp_enabled} density={density:.6f} pos_weight={pos_weight_value:.3f}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device, scaler, amp_enabled, args.grad_clip
        )
        val_loss = validate(model, val_loader, criterion, device, amp_enabled)
        print(f"epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, scaler, epoch, 0, best_val, config)
            print(f"saved best checkpoint: {args.out_dir / 'best.pt'}")
        save_checkpoint(args.out_dir / "latest.pt", model, optimizer, scaler, epoch, 0, best_val, config)


if __name__ == "__main__":
    main()
