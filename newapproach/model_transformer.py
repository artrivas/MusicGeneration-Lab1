from __future__ import annotations

import torch
import torch.nn as nn


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_len = int(context_len)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.context_len:
            raise ValueError(f"seq_len {seq_len} exceeds context_len {self.context_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)
