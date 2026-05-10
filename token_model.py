from __future__ import annotations

import torch
from torch import nn


class CausalTokenTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_len = int(context_len)
        self.pad_id = int(pad_id)
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
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
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids [B,L], got {tuple(input_ids.shape)}")
        b, l = input_ids.shape
        if l > self.context_len:
            raise ValueError(f"Input length {l} exceeds context_len {self.context_len}")
        pos = torch.arange(l, device=input_ids.device)
        h = self.token_embed(input_ids) + self.pos_embed(pos).view(1, l, -1)
        h = self.dropout(h)
        causal_mask = torch.triu(torch.ones(l, l, device=input_ids.device, dtype=torch.bool), diagonal=1)
        padding_mask = input_ids.eq(self.pad_id)
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output(self.norm(h))
