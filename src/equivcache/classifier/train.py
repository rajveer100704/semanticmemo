"""Training loop for the prompt equivalence classifier."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from equivcache.classifier.data import EmbeddedPairDataset, PairRecord
from equivcache.classifier.evaluate import EvaluationMetrics, evaluate_model
from equivcache.classifier.model import PairClassifier
from equivcache.types import EmbeddingProvider

_torch: Any = torch


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters for the small classifier head."""

    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    threshold: float = 0.85
    device: str = "cpu"
    seed: int = 7
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned after a classifier training run."""

    checkpoint_path: Path
    train_examples: int
    validation_examples: int
    final_train_loss: float
    validation_metrics: EvaluationMetrics | None


def train_classifier(
    *,
    train_records: Sequence[PairRecord],
    embedding_provider: EmbeddingProvider,
    output_path: Path | str,
    config: TrainingConfig | None = None,
    validation_records: Sequence[PairRecord] | None = None,
) -> TrainingResult:
    """Train and checkpoint a PairClassifier on labeled prompt pairs."""

    active_config = config or TrainingConfig()
    if not train_records:
        msg = "train_classifier requires at least one training record"
        raise ValueError(msg)
    _validate_config(active_config)
    _torch.manual_seed(active_config.seed)

    target_device = _torch.device(active_config.device)
    dataset = EmbeddedPairDataset(train_records, embedding_provider)
    loader = DataLoader(dataset, batch_size=active_config.batch_size, shuffle=True)
    model = PairClassifier(embed_dim=embedding_provider.dim, dropout=active_config.dropout).to(
        target_device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=active_config.learning_rate)
    positive_weight, negative_weight = _class_weights(train_records)
    final_loss = 0.0

    for _ in range(active_config.epochs):
        model.train()
        batch_losses: list[float] = []
        for emb_a, emb_b, labels in loader:
            emb_a = emb_a.to(target_device)
            emb_b = emb_b.to(target_device)
            labels = labels.to(target_device)
            optimizer.zero_grad()
            probabilities = model(emb_a, emb_b)
            weights = _torch.where(
                labels > 0.5,
                _torch.full_like(labels, positive_weight),
                _torch.full_like(labels, negative_weight),
            )
            loss = F.binary_cross_entropy(probabilities, labels, weight=weights)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        final_loss = sum(batch_losses) / len(batch_losses)

    validation_metrics = None
    if validation_records:
        validation_metrics = evaluate_model(
            model=model,
            records=validation_records,
            embedding_provider=embedding_provider,
            threshold=active_config.threshold,
            batch_size=active_config.batch_size,
            device=active_config.device,
        )

    checkpoint_path = Path(output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _torch.save(
        {
            "model_type": "PairClassifier",
            "model_state_dict": model.state_dict(),
            "embed_dim": embedding_provider.dim,
            "threshold": active_config.threshold,
            "training_config": asdict(active_config),
            "final_train_loss": final_loss,
            "validation_metrics": validation_metrics.to_dict() if validation_metrics else None,
        },
        checkpoint_path,
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        train_examples=len(train_records),
        validation_examples=len(validation_records or []),
        final_train_loss=final_loss,
        validation_metrics=validation_metrics,
    )


def _validate_config(config: TrainingConfig) -> None:
    if config.epochs <= 0:
        msg = "epochs must be positive"
        raise ValueError(msg)
    if config.batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if config.learning_rate <= 0:
        msg = "learning_rate must be positive"
        raise ValueError(msg)
    if not 0 <= config.threshold <= 1:
        msg = "threshold must be between 0 and 1"
        raise ValueError(msg)


def _class_weights(records: Sequence[PairRecord]) -> tuple[float, float]:
    positive_count = sum(record.label for record in records)
    negative_count = len(records) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 1.0, 1.0
    total = len(records)
    positive_weight = total / (2 * positive_count)
    negative_weight = total / (2 * negative_count)
    return positive_weight, negative_weight
