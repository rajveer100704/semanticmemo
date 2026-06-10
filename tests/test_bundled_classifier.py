from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from semanticmemo import CacheConfig, ClassifierConfig, SemanticMemo
from semanticmemo.embedding import HashEmbeddingProvider
from semanticmemo.exceptions import SemanticMemoError
from semanticmemo.resources import bundled_classifier_path

_BUNDLED_DIM = 384


def test_bundled_classifier_checkpoint_is_packaged() -> None:
    path = bundled_classifier_path()
    assert path.is_file()
    assert path.suffix == ".pt"


def test_bundled_config_defaults_to_precision_first_threshold() -> None:
    config = ClassifierConfig.bundled()
    assert config.model_path is not None
    assert config.model_path.is_file()
    assert config.threshold == 0.95
    assert config.device == "cpu"


def test_bundled_config_accepts_overrides() -> None:
    config = ClassifierConfig.bundled(device="cpu", threshold=0.8)
    assert config.threshold == 0.8


def test_bundled_path_reports_missing_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from semanticmemo import resources

    monkeypatch.setattr(resources, "BUNDLED_CLASSIFIER_NAME", "does-not-exist.pt")
    with pytest.raises(SemanticMemoError, match="missing"):
        resources.bundled_classifier_path()


async def test_bundled_classifier_gates_cache_lookups(tmp_path: Path) -> None:
    cache = SemanticMemo(
        domain="customer-support",
        config=CacheConfig(
            db_path=tmp_path / "cache.db",
            embedding_dim=_BUNDLED_DIM,
            estimated_llm_cost_usd=Decimal("0.002"),
        ),
        embedding_provider=HashEmbeddingProvider(dim=_BUNDLED_DIM),
        classifier=ClassifierConfig.bundled(),
        use_faiss=False,
    )
    try:

        async def call_llm(prompt: str) -> str:
            return f"response::{prompt}"

        first = await cache.get_or_call(prompt="Approve the refund", llm_function=call_llm)
        assert first.was_cache_hit is False

        # An identical prompt yields a cosine candidate, so the learned
        # classifier evaluates it and a classifier score is recorded.
        second = await cache.get_or_call(prompt="Approve the refund", llm_function=call_llm)
        assert second.classifier_score is not None
        assert 0.0 <= second.classifier_score <= 1.0
    finally:
        cache.close()


def test_shipped_model_card_meets_acceptance_gate() -> None:
    report_path = bundled_classifier_path().with_suffix(".report.json")
    if not report_path.is_file():
        pytest.skip("model card report is not present in this installation")
    report = json.loads(report_path.read_text())
    assert report["gold_vs_cosine"]["gate_passed"] is True
