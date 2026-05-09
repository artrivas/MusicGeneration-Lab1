from __future__ import annotations

import torch
from torch import nn


class NestedNoteHead(nn.Module):
    def __init__(self, d_model: int, num_notes: int, dropout: float) -> None:
        super().__init__()
        self.pitch_embed = nn.Embedding(num_notes, d_model)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, l, d = h.shape
        pitch = self.pitch_embed.weight.view(1, 1, -1, d)
        h = h.view(b, l, 1, d) + pitch
        return self.out(self.proj(h)).squeeze(-1)


class NestedPianoRollTransformer(nn.Module):
    def __init__(
        self,
        num_notes: int = 88,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        context_len: int = 512,
        head_type: str = "mlp",
    ) -> None:
        super().__init__()
        if head_type not in {"mlp", "nested"}:
            raise ValueError("head_type must be 'mlp' or 'nested'")
        self.num_notes = num_notes
        self.d_model = d_model
        self.context_len = context_len
        self.head_type = head_type

        self.input_proj = nn.Linear(num_notes, d_model)
        self.pos_embed = nn.Embedding(context_len, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_notes),
            )
        else:
            self.head = NestedNoteHead(d_model=d_model, num_notes=num_notes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.num_notes:
            raise ValueError(f"Expected x with shape [B, L, {self.num_notes}], got {tuple(x.shape)}")
        b, l, _ = x.shape
        if l > self.context_len:
            raise ValueError(f"Input length {l} exceeds configured context_len {self.context_len}")
        pos = torch.arange(l, device=x.device)
        h = self.input_proj(x) + self.pos_embed(pos).view(1, l, self.d_model)
        h = self.dropout(h)
        mask = torch.triu(torch.ones(l, l, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.transformer(h, mask=mask)
        return self.head(self.norm(h))
