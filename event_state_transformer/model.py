from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import PAD


class EventStateMusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_delta: int = 1200,
        num_notes: int = 88,
        max_seq_len: int = 1024,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_delta = max_delta
        self.num_notes = num_notes
        self.max_seq_len = max_seq_len
        self.use_rope = bool(use_rope)

        self.chord_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.delta_emb = nn.Embedding(max_delta + 1, d_model)
        self.card_emb = nn.Embedding(num_notes + 1, d_model)
        self.mod4_emb = nn.Embedding(4, d_model)
        self.mod8_emb = nn.Embedding(8, d_model)
        self.mod16_emb = nn.Embedding(16, d_model)
        self.mod32_emb = nn.Embedding(32, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.note_proj = nn.Linear(num_notes, d_model)
        self.in_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Linear(d_model, max_delta + 1)
        self.chord_head = nn.Linear(d_model, vocab_size)
        self.note_head = nn.Linear(d_model, num_notes)
        self.card_head = nn.Linear(d_model, num_notes + 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        delta: torch.Tensor,
        chord: torch.Tensor,
        notes: torch.Tensor,
        cum: torch.Tensor,
        card: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = chord.shape
        if seq_len > self.max_seq_len:
            delta = delta[:, -self.max_seq_len :]
            chord = chord[:, -self.max_seq_len :]
            notes = notes[:, -self.max_seq_len :]
            cum = cum[:, -self.max_seq_len :]
            card = card[:, -self.max_seq_len :]
            seq_len = self.max_seq_len

        pos = torch.arange(seq_len, device=chord.device).unsqueeze(0).expand(bsz, seq_len)
        x = (
            self.chord_emb(chord)
            + self.delta_emb(delta.clamp(0, self.max_delta))
            + self.card_emb(card.clamp(0, self.num_notes))
            + self.note_proj(notes)
            + self.mod4_emb(cum.remainder(4))
            + self.mod8_emb(cum.remainder(8))
            + self.mod16_emb(cum.remainder(16))
            + self.mod32_emb(cum.remainder(32))
            + self.pos_emb(pos)
        )
        x = self.dropout(self.in_norm(x))
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=chord.device, dtype=torch.bool), diagonal=1)
        padding_mask = chord.eq(PAD)
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.out_norm(x)
        return {
            "delta_logits": self.delta_head(x),
            "chord_logits": self.chord_head(x),
            "note_logits": self.note_head(x),
            "card_logits": self.card_head(x),
        }


def event_state_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    delta_weight: float = 1.0,
    chord_weight: float = 0.5,
    note_weight: float = 1.0,
    card_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["target_mask"]
    denom = mask.sum().clamp_min(1).float()

    delta_loss_all = F.cross_entropy(
        outputs["delta_logits"].transpose(1, 2),
        batch["delta_target"].clamp_min(0),
        reduction="none",
    )
    delta_loss = (delta_loss_all * mask.float()).sum() / denom

    chord_loss = F.cross_entropy(
        outputs["chord_logits"].transpose(1, 2),
        batch["chord_target"],
        ignore_index=PAD,
    )

    note_loss_all = F.binary_cross_entropy_with_logits(
        outputs["note_logits"],
        batch["notes_target"],
        reduction="none",
    ).mean(dim=-1)
    note_loss = (note_loss_all * mask.float()).sum() / denom

    card_loss_all = F.cross_entropy(
        outputs["card_logits"].transpose(1, 2),
        batch["card_target"].clamp(0, 88),
        reduction="none",
    )
    card_loss = (card_loss_all * mask.float()).sum() / denom

    loss = (
        delta_weight * delta_loss
        + chord_weight * chord_loss
        + note_weight * note_loss
        + card_weight * card_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "delta_ce": float(delta_loss.detach().cpu()),
        "chord_ce": float(chord_loss.detach().cpu()),
        "note_bce": float(note_loss.detach().cpu()),
        "card_ce": float(card_loss.detach().cpu()),
    }
    return loss, metrics
