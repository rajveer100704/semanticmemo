from __future__ import annotations

from typing import Any

import torch

from equivcache.classifier import PairClassifier, build_pair_features

_torch: Any = torch


def test_build_pair_features_uses_matching_representation() -> None:
    emb_a = _torch.tensor([[1.0, 2.0]])
    emb_b = _torch.tensor([[3.0, 1.0]])

    features = build_pair_features(emb_a, emb_b)

    assert features.tolist() == [[1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 3.0, 2.0]]


def test_pair_classifier_outputs_probability_per_pair() -> None:
    model = PairClassifier(embed_dim=2, dropout=0.0)
    emb_a = _torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    emb_b = _torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    probabilities = model(emb_a, emb_b)

    assert probabilities.shape == (2,)
    assert _torch.all(probabilities >= 0)
    assert _torch.all(probabilities <= 1)
