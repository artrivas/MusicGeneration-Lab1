from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TokenWindowDataset, make_sequence_split
from model_transformer import CausalTransformerLM
from tokenizer_mirexlike import MirexLikePianoTokenizer


def build_model(args: argparse.Namespace, vocab_size: int) -> nn.Module:
    if args.model_type == "transformer":
        return CausalTransformerLM(
            vocab_size=vocab_size,
            context_len=args.context_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )
    if args.model_type == "rwkv":
        from model_rwkv import RWKVLanguageModel

        return RWKVLanguageModel()
    raise ValueError(f"unknown model_type {args.model_type}")


def ratios(tokens: torch.Tensor, tok: MirexLikePianoTokenizer) -> dict[str, float]:
    flat = tokens.detach().view(-1).cpu().numpy()
    denom = max(1, len(flat))
    return {
        "note": float(sum(tok.is_note_on(int(t)) for t in flat) / denom),
        "time": float(sum(tok.is_time_shift(int(t)) for t in flat) / denom),
        "chord": float(sum(tok.is_chord_end(int(t)) for t in flat) / denom),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, tok: MirexLikePianoTokenizer, max_batches: int = 50) -> dict[str, float]:
    model.eval()
    losses = []
    top1_total = top5_total = count_total = 0
    pred_chunks = []
    target_chunks = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(float(loss.item()))
        pred = torch.argmax(logits, dim=-1)
        top1_total += int((pred == y).sum().item())
        top5 = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
        top5_total += int((top5 == y.unsqueeze(-1)).any(dim=-1).sum().item())
        count_total += int(y.numel())
        pred_chunks.append(pred.cpu())
        target_chunks.append(y.cpu())
    if not losses:
        return {}
    pred_all = torch.cat([p.reshape(-1) for p in pred_chunks])
    target_all = torch.cat([t.reshape(-1) for t in target_chunks])
    pr = ratios(pred_all, tok)
    tr = ratios(target_all, tok)
    loss_mean = float(np.mean(losses))
    return {
        "val_loss": loss_mean,
        "val_perplexity": float(math.exp(min(20.0, loss_mean))),
        "val_top1_accuracy": float(top1_total / max(1, count_total)),
        "val_top5_accuracy": float(top5_total / max(1, count_total)),
        "val_note_token_ratio_pred": pr["note"],
        "val_time_shift_ratio_pred": pr["time"],
        "val_chord_end_ratio_pred": pr["chord"],
        "val_note_token_ratio_target": tr["note"],
        "val_time_shift_ratio_target": tr["time"],
        "val_chord_end_ratio_target": tr["chord"],
    }


def lr_scale(step: int, total_steps: int, warmup_steps: int, scheduler: str) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    if scheduler != "cosine":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a next-token symbolic piano continuation model.")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_type", choices=["transformer", "rwkv"], default="transformer")
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = MirexLikePianoTokenizer.load(args.cache_dir / "tokenizer.json")
    offsets = np.load(args.cache_dir / "token_offsets.npy", mmap_mode="r")
    split = make_sequence_split(len(offsets) - 1, args.val_fraction, args.seed)
    samples = args.steps_per_epoch * args.batch_size
    train_ds = TokenWindowDataset(str(args.cache_dir / "tokens_flat.npy"), str(args.cache_dir / "token_offsets.npy"), args.context_len, split.train_indices, samples, args.seed)
    val_ds = TokenWindowDataset(str(args.cache_dir / "tokens_flat.npy"), str(args.cache_dir / "token_offsets.npy"), args.context_len, split.val_indices if len(split.val_indices) else split.train_indices, min(samples, 256 * args.batch_size), args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, tok.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tok.PAD, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    total_steps = args.epochs * args.steps_per_epoch

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"vocab_size": tok.vocab_size, "device": str(device)})
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics_path = args.out_dir / "metrics.csv"
    best_val = float("inf")
    global_step = 0
    fields = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for step, (x, y) in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            global_step += 1
            scale = lr_scale(global_step, total_steps, args.warmup_steps, args.scheduler)
            for group in opt.param_groups:
                group["lr"] = args.lr * scale
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "train_perplexity": float(math.exp(min(20.0, train_loss))),
        }
        row.update(evaluate(model, val_loader, criterion, device, tok))
        if fields is None:
            fields = list(row.keys())
            with metrics_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)
        ckpt = {"model_state": model.state_dict(), "config": config, "tokenizer": tok.to_dict(), "metrics": row}
        torch.save(ckpt, args.out_dir / "latest.pt")
        if row.get("val_loss", float("inf")) < best_val:
            best_val = row["val_loss"]
            torch.save(ckpt, args.out_dir / "best.pt")
        print(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
