from __future__ import annotations

import argparse
import csv
import json
import math
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


def make_criterion(pad_id: int, label_smoothing: float):
    try:
        return nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing:
            print("warning: this PyTorch version does not support label_smoothing; using 0.0")
        return nn.CrossEntropyLoss(ignore_index=pad_id)


def make_scheduler(optimizer, scheduler_name: str, warmup_steps: int, total_steps: int):
    if scheduler_name == "none":
        return None
    if scheduler_name != "cosine":
        raise ValueError("--scheduler must be 'none' or 'cosine'")

    def lr_lambda(step: int) -> float:
        step = max(1, step)
        if warmup_steps > 0 and step <= warmup_steps:
            return step / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, amp_enabled, grad_clip) -> tuple[float, float]:
    model.train()
    total = 0.0
    grad_total = 0.0
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
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach().cpu())
        grad_total += float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm)
        if step % 50 == 0:
            avg_loss = total / step
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step}/{len(loader)} loss={avg_loss:.5f} "
                f"ppl={math.exp(min(20.0, avg_loss)):.2f} lr={lr:.6g} grad_norm={grad_total / step:.3f}"
            )
    return total / max(1, len(loader)), grad_total / max(1, len(loader))


@torch.no_grad()
def _ratio(mask_count: int, denom: int) -> float:
    return float(mask_count / max(1, denom))


@torch.no_grad()
def validate(model, loader, criterion, tokenizer, device, amp_enabled) -> dict[str, float]:
    model.eval()
    total = 0.0
    total_targets = 0
    chord_targets = 0
    shift_targets = 0
    chord_predictions = 0
    shift_predictions = 0
    top1_correct = 0
    top5_correct = 0
    chord_top5_correct = 0
    shift_top5_correct = 0
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
        total += float(loss.detach().cpu())
        flat_targets = target_ids.reshape(-1)
        valid = flat_targets.ne(tokenizer.PAD)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        preds = flat_logits.argmax(dim=-1)
        top5 = torch.topk(flat_logits, k=min(5, flat_logits.shape[-1]), dim=-1).indices

        valid_targets = flat_targets[valid]
        valid_preds = preds[valid]
        valid_top5 = top5[valid]
        total_targets += int(valid_targets.numel())
        top1_correct += int(valid_preds.eq(valid_targets).sum().item())
        top5_correct += int((valid_top5 == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item())

        target_list = valid_targets.detach().cpu().tolist()
        pred_list = valid_preds.detach().cpu().tolist()
        chord_mask = torch.tensor([tokenizer.is_chord(int(x)) for x in target_list], device=device, dtype=torch.bool)
        shift_mask = torch.tensor([tokenizer.is_time_shift(int(x)) for x in target_list], device=device, dtype=torch.bool)
        chord_targets += int(chord_mask.sum().item())
        shift_targets += int(shift_mask.sum().item())
        chord_predictions += sum(1 for x in pred_list if tokenizer.is_chord(x))
        shift_predictions += sum(1 for x in pred_list if tokenizer.is_time_shift(x))
        if chord_mask.any():
            chord_top5_correct += int((valid_top5[chord_mask] == valid_targets[chord_mask].unsqueeze(-1)).any(dim=-1).sum().item())
        if shift_mask.any():
            shift_top5_correct += int((valid_top5[shift_mask] == valid_targets[shift_mask].unsqueeze(-1)).any(dim=-1).sum().item())
    loss = total / max(1, len(loader))
    return {
        "val_loss": loss,
        "val_perplexity": math.exp(min(20.0, loss)),
        "val_chord_target_ratio": _ratio(chord_targets, total_targets),
        "val_timeshift_target_ratio": _ratio(shift_targets, total_targets),
        "val_argmax_chord_ratio": _ratio(chord_predictions, total_targets),
        "val_argmax_timeshift_ratio": _ratio(shift_predictions, total_targets),
        "val_top1_accuracy": _ratio(top1_correct, total_targets),
        "val_top5_accuracy": _ratio(top5_correct, total_targets),
        "val_chord_top5_accuracy": _ratio(chord_top5_correct, chord_targets),
        "val_timeshift_top5_accuracy": _ratio(shift_top5_correct, shift_targets),
    }


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
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--transpose_augmentation", action="store_true")
    parser.add_argument("--transpose_min", type=int, default=-5)
    parser.add_argument("--transpose_max", type=int, default=6)
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
        tokenizer=tokenizer,
        transpose_augmentation=args.transpose_augmentation,
        transpose_min=args.transpose_min,
        transpose_max=args.transpose_max,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_train_steps = max(1, args.epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, args.scheduler, args.warmup_steps, total_train_steps)
    criterion = make_criterion(tokenizer.PAD, args.label_smoothing)
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
    epochs_without_improvement = 0
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
    metrics_path = args.out_dir / "metrics.csv"
    write_header = not metrics_path.exists() or start_epoch == 1
    metrics_fields = [
        "epoch", "train_loss", "train_perplexity", "grad_norm", "lr",
        "val_loss", "val_perplexity", "val_chord_target_ratio", "val_timeshift_target_ratio",
        "val_argmax_chord_ratio", "val_argmax_timeshift_ratio", "val_top1_accuracy",
        "val_top5_accuracy", "val_chord_top5_accuracy", "val_timeshift_top5_accuracy",
    ]
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, grad_norm = run_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, amp_enabled, args.grad_clip
        )
        val_metrics = validate(model, val_loader, criterion, tokenizer, device, amp_enabled)
        val_loss = val_metrics["val_loss"]
        print(
            f"epoch {epoch}: train_loss={train_loss:.5f} train_ppl={math.exp(min(20.0, train_loss)):.2f} "
            f"val_loss={val_loss:.5f} val_ppl={val_metrics['val_perplexity']:.2f} "
            f"val_top1={val_metrics['val_top1_accuracy']:.4f} val_top5={val_metrics['val_top5_accuracy']:.4f} "
            f"target_chord={val_metrics['val_chord_target_ratio']:.4f} "
            f"argmax_chord={val_metrics['val_argmax_chord_ratio']:.4f}"
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_perplexity": math.exp(min(20.0, train_loss)),
            "grad_norm": grad_norm,
            "lr": optimizer.param_groups[0]["lr"],
            **val_metrics,
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, scaler, epoch, best_val, config)
            print(f"saved best checkpoint: {args.out_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1
        save_checkpoint(args.out_dir / "latest.pt", model, optimizer, scaler, epoch, best_val, config)
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early stopping after {epochs_without_improvement} epochs without validation improvement")
            break


if __name__ == "__main__":
    main()
