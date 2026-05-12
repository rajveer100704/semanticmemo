"""Small pair classifier used to judge prompt equivalence."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

_torch: Any = torch


def build_pair_features(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Build the standard Sentence-BERT pair representation."""

    if emb_a.shape != emb_b.shape:
        msg = f"Embedding tensors must have the same shape, got {emb_a.shape} and {emb_b.shape}"
        raise ValueError(msg)
    return _torch.cat([emb_a, emb_b, _torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1)


class PairClassifier(nn.Module):
    """CPU-friendly MLP over two fixed embedding vectors."""

    def __init__(
        self,
        *,
        embed_dim: int = 384,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            msg = "embed_dim must be positive"
            raise ValueError(msg)
        self.embed_dim = embed_dim
        self.network = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        if emb_a.shape[-1] != self.embed_dim:
            msg = f"Expected embedding dimension {self.embed_dim}, got {emb_a.shape[-1]}"
            raise ValueError(msg)
        pair_features = build_pair_features(emb_a, emb_b)
        logits = self.network(pair_features).squeeze(-1)
        return _torch.sigmoid(logits)
