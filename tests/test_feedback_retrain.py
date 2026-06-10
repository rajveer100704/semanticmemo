from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from semanticmemo.classifier import TrainingConfig
from semanticmemo.feedback import RetrainConfig, retrain_from_feedback
from semanticmemo.store import SQLiteCacheStore
from semanticmemo.types import FloatVector


class Provider:
    dim = 4

    def embed(self, text: str) -> FloatVector:
        if "approve" in text or "accept" in text:
            return np.array([1, 0, 0, 0], dtype=np.float32)
        if "deny" in text or "reject" in text:
            return np.array([0, 1, 0, 0], dtype=np.float32)
        if "reset" in text:
            return np.array([0, 0, 1, 0], dtype=np.float32)
        return np.array([0, 0, 0, 1], dtype=np.float32)


def test_retrain_from_feedback_promotes_when_gates_pass(tmp_path: Path) -> None:
    store = _store_with_feedback(tmp_path, label=0)
    validation_path = _write_validation_pairs(tmp_path)
    checkpoint_path = tmp_path / "classifier-candidate.pt"
    promoted_path = tmp_path / "classifier-active.pt"

    result = retrain_from_feedback(
        store=store,
        embedding_provider=Provider(),
        config=RetrainConfig(
            output_path=checkpoint_path,
            validation_data_path=validation_path,
            promote_to=promoted_path,
            min_precision=0.0,
            training=TrainingConfig(epochs=1, batch_size=1, dropout=0.0),
        ),
    )

    report = json.loads(result.report_path.read_text())
    assert result.feedback_examples == 1
    assert result.seed_examples == 0
    assert result.train_examples == 1
    assert result.gates_passed is True
    assert result.promoted_path == promoted_path
    assert checkpoint_path.exists()
    assert promoted_path.exists()
    assert report["gates_passed"] is True
    assert report["feedback_examples"] == 1
    store.close()


def test_retrain_merges_seed_data_and_filters_domain(tmp_path: Path) -> None:
    store = _store_with_feedback(tmp_path, label=0, domain="customer-support")
    validation_path = _write_validation_pairs(tmp_path, domain="customer-support")
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_a": "approve refund",
                        "prompt_b": "accept refund",
                        "label": 1,
                        "domain": "customer-support",
                        "split": "train",
                    }
                ),
                json.dumps(
                    {
                        "prompt_a": "legal hold",
                        "prompt_b": "delete legal hold",
                        "label": 0,
                        "domain": "legal",
                        "split": "train",
                    }
                ),
            ]
        )
        + "\n"
    )

    result = retrain_from_feedback(
        store=store,
        embedding_provider=Provider(),
        config=RetrainConfig(
            output_path=tmp_path / "classifier.pt",
            validation_data_path=validation_path,
            seed_data_path=seed_path,
            domain="customer-support",
            min_precision=0.0,
            training=TrainingConfig(epochs=1, batch_size=1, dropout=0.0),
        ),
    )

    assert result.feedback_examples == 1
    assert result.seed_examples == 1
    assert result.train_examples == 2
    store.close()


def test_retrain_does_not_promote_when_gates_fail(tmp_path: Path) -> None:
    store = _store_with_feedback(tmp_path, label=0)
    validation_path = _write_validation_pairs(tmp_path)
    checkpoint_path = tmp_path / "classifier-candidate.pt"
    promoted_path = tmp_path / "classifier-active.pt"

    result = retrain_from_feedback(
        store=store,
        embedding_provider=Provider(),
        config=RetrainConfig(
            output_path=checkpoint_path,
            validation_data_path=validation_path,
            promote_to=promoted_path,
            min_precision=1.0,
            min_recall=1.0,
            training=TrainingConfig(epochs=1, batch_size=1, dropout=0.0),
        ),
    )

    report = json.loads(result.report_path.read_text())
    assert result.gates_passed is False
    assert result.promoted_path is None
    assert checkpoint_path.exists()
    assert not promoted_path.exists()
    assert report["gates_passed"] is False
    store.close()


def test_retrain_requires_feedback_or_seed_data(tmp_path: Path) -> None:
    store = SQLiteCacheStore(tmp_path / "empty.db")
    validation_path = _write_validation_pairs(tmp_path)

    with pytest.raises(ValueError, match="at least one feedback or seed"):
        retrain_from_feedback(
            store=store,
            embedding_provider=Provider(),
            config=RetrainConfig(
                output_path=tmp_path / "classifier.pt",
                validation_data_path=validation_path,
                training=TrainingConfig(epochs=1, batch_size=1),
            ),
        )
    store.close()


def test_retrain_cli_smoke_with_hash_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from semanticmemo.cli import main

    db_path = tmp_path / "cache.db"
    store = _store_with_feedback(tmp_path, db_path=db_path, label=0)
    store.close()
    validation_path = _write_validation_pairs(tmp_path)
    checkpoint_path = tmp_path / "cli-candidate.pt"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "SemanticMemo",
            "--db-path",
            str(db_path),
            "retrain",
            "--out",
            str(checkpoint_path),
            "--validation-data",
            str(validation_path),
            "--embedding-provider",
            "hash",
            "--embedding-dim",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--min-precision",
            "0.0",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["feedback_examples"] == 1
    assert checkpoint_path.exists()
    assert Path(f"{checkpoint_path}.report.json").exists()


def _store_with_feedback(
    tmp_path: Path,
    *,
    label: int,
    domain: str = "customer-support",
    db_path: Path | None = None,
) -> SQLiteCacheStore:
    store = SQLiteCacheStore(db_path or tmp_path / "cache.db")
    entry_id = store.add(
        prompt="approve refund",
        embedding=np.array([1, 0, 0, 0], dtype=np.float32),
        response="approved",
        model="test-model",
    )
    query_id = uuid4()
    store.record_lookup(
        query_id=query_id,
        domain=domain,
        prompt="deny refund",
        embedding=np.array([0, 1, 0, 0], dtype=np.float32),
        cache_entry_id=entry_id,
        similarity_score=0.97,
        classifier_score=0.9,
    )
    store.record_feedback(query_id=query_id, label=label, reason="operator feedback")
    return store


def _write_validation_pairs(
    tmp_path: Path,
    *,
    domain: str = "customer-support",
) -> Path:
    path = tmp_path / "validation.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_a": "approve refund",
                        "prompt_b": "accept refund",
                        "label": 1,
                        "domain": domain,
                        "split": "validation",
                    }
                ),
                json.dumps(
                    {
                        "prompt_a": "approve refund",
                        "prompt_b": "deny refund",
                        "label": 0,
                        "domain": domain,
                        "split": "validation",
                    }
                ),
            ]
        )
        + "\n"
    )
    return path
