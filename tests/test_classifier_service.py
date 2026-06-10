from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from semanticmemo.classifier.model import PairClassifier
from semanticmemo.classifier.service import ClassifierService
from semanticmemo.resources import bundled_classifier_path

_DIM = 384


def _service() -> ClassifierService:
    return ClassifierService(bundled_classifier_path())


def test_loads_bundled_checkpoint() -> None:
    service = _service()
    assert isinstance(service.model, PairClassifier)
    assert 0.0 <= service.threshold <= 1.0


def test_threshold_override() -> None:
    service = ClassifierService(bundled_classifier_path(), threshold=0.5)
    assert service.threshold == 0.5


def test_predict_matches_predict_batch() -> None:
    service = _service()
    a = np.ones(_DIM, dtype=np.float32)
    b = np.arange(_DIM, dtype=np.float32)
    assert service.predict(a, b) == service.predict_batch([(a, b)])[0]


def test_predict_batch_empty_returns_empty_list() -> None:
    assert _service().predict_batch([]) == []


def test_is_equivalent_consistent_with_threshold() -> None:
    service = _service()
    a = np.ones(_DIM, dtype=np.float32)
    b = np.arange(_DIM, dtype=np.float32)
    expected = service.predict(a, b) >= service.threshold
    assert service.is_equivalent(a, b) is expected


def test_rejects_unsupported_checkpoint(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"model_type": "NotPairClassifier"}, bad)
    with pytest.raises(ValueError, match="Unsupported"):
        ClassifierService(bad)


def test_rejects_checkpoint_without_state_dict(tmp_path: Path) -> None:
    bad = tmp_path / "no-state.pt"
    torch.save({"model_type": "PairClassifier", "embed_dim": _DIM}, bad)
    with pytest.raises(ValueError, match="model_state_dict"):
        ClassifierService(bad)


def test_reload_swaps_checkpoint() -> None:
    service = ClassifierService(bundled_classifier_path(), threshold=0.5)
    service.reload(bundled_classifier_path())
    assert service.model_path == bundled_classifier_path()
    assert isinstance(service.model, PairClassifier)
