from __future__ import annotations

from pathlib import Path

import numpy as np

from equivcache.classifier import ClassifierService, TrainingConfig, train_classifier
from equivcache.classifier.data import PairRecord
from equivcache.types import FloatVector


class Provider:
    dim = 4

    def embed(self, text: str) -> FloatVector:
        vectors = {
            "approve refund": np.array([1, 0, 0, 0], dtype=np.float32),
            "accept refund": np.array([1, 0, 0, 0], dtype=np.float32),
            "deny refund": np.array([0, 1, 0, 0], dtype=np.float32),
            "reject refund": np.array([0, 1, 0, 0], dtype=np.float32),
            "reset password": np.array([0, 0, 1, 0], dtype=np.float32),
            "delete account": np.array([0, 0, 0, 1], dtype=np.float32),
        }
        return vectors[text]


def test_train_classifier_saves_loadable_checkpoint(tmp_path: Path) -> None:
    provider = Provider()
    train_records = [
        PairRecord(prompt_a="approve refund", prompt_b="accept refund", label=1),
        PairRecord(prompt_a="deny refund", prompt_b="reject refund", label=1),
        PairRecord(prompt_a="approve refund", prompt_b="deny refund", label=0),
        PairRecord(prompt_a="reset password", prompt_b="delete account", label=0),
    ]
    validation_records = [
        PairRecord(prompt_a="approve refund", prompt_b="accept refund", label=1),
        PairRecord(prompt_a="approve refund", prompt_b="deny refund", label=0),
    ]
    checkpoint = tmp_path / "classifier.pt"

    result = train_classifier(
        train_records=train_records,
        validation_records=validation_records,
        embedding_provider=provider,
        output_path=checkpoint,
        config=TrainingConfig(epochs=2, batch_size=2, dropout=0.0),
    )
    service = ClassifierService(checkpoint)
    score = service.predict(
        provider.embed("approve refund"),
        provider.embed("accept refund"),
    )

    assert result.checkpoint_path == checkpoint
    assert result.train_examples == 4
    assert result.validation_examples == 2
    assert result.validation_metrics is not None
    assert checkpoint.exists()
    assert 0 <= score <= 1
