from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from token_dataset import TokenWindowDataset
from token_model import CausalTokenTransformer
from tokenizer import PianoEventTokenizer


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path, model, optimizer, scaler, epoch, best_val, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val": best_val,
            "config": config,
        },
        path,
    )


def run_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled, grad_clip) -> float:
    model.train()
    total = 0.0
    for step, (input_ids, target_ids) in enumerate(loader, start=1):
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
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
def validate(model, loader, criterion, tokenizer, device, amp_enabled) -> tuple[float, float]:
    model.eval()
    total = 0.0
    chord_predictions = 0
    shift_predictions = 0
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
        total += float(loss.detach().cpu())
        preds = logits.argmax(dim=-1).detach().cpu().reshape(-1).tolist()
        chord_predictions += sum(1 for x in preds if tokenizer.is_chord(x))
        shift_predictions += sum(1 for x in preds if tokenizer.is_time_shift(x))
    denom = max(1, chord_predictions + shift_predictions)
    return total / max(1, len(loader)), chord_predictions / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Train token-level chord/time-shift Transformer.")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints_token"))
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--val_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer_path = args.cache_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer cache: {tokenizer_path}. Run prepare_token_cache.py first.")
    tokenizer = PianoEventTokenizer.load(tokenizer_path)

    device = pick_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    train_ds = TokenWindowDataset(
        args.cache_dir,
        context_len=args.context_len,
        split="train",
        steps_per_epoch=args.steps_per_epoch * args.batch_size,
        seed=args.seed,
    )
    val_ds = TokenWindowDataset(
        args.cache_dir,
        context_len=args.context_len,
        split="val",
        steps_per_epoch=args.val_steps * args.batch_size,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2, pin_memory=device.type == "cuda")

    model = CausalTokenTransformer(
        vocab_size=tokenizer.vocab_size,
        context_len=args.context_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        pad_id=tokenizer.PAD,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    config = vars(args).copy()
    config.update(
        {
            "cache_dir": str(args.cache_dir),
            "out_dir": str(args.out_dir),
            "resume": str(args.resume) if args.resume else None,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.PAD,
            "model_type": "NestedMusicTransformerInspired_TokenChordTransformer",
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

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

    print(f"device={device} amp={amp_enabled} vocab_size={tokenizer.vocab_size}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device, scaler, amp_enabled, args.grad_clip
        )
        val_loss, val_chord_ratio = validate(model, val_loader, criterion, tokenizer, device, amp_enabled)
        print(
            f"epoch {epoch}: train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_argmax_chord_ratio={val_chord_ratio:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, scaler, epoch, best_val, config)
            print(f"saved best checkpoint: {args.out_dir / 'best.pt'}")
        save_checkpoint(args.out_dir / "latest.pt", model, optimizer, scaler, epoch, best_val, config)


if __name__ == "__main__":
    main()
