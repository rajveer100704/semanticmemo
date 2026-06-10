"""Manual feedback-to-checkpoint retraining workflow."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from semanticmemo._logging import get_logger
from semanticmemo.classifier import (
    EvaluationMetrics,
    TrainingConfig,
    load_pair_records,
    train_classifier,
)
from semanticmemo.classifier.data import PairRecord
from semanticmemo.store import SQLiteCacheStore
from semanticmemo.types import EmbeddingProvider

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrainConfig:
    """Configuration for one manual feedback-driven retraining run."""

    output_path: Path
    validation_data_path: Path
    seed_data_path: Path | None = None
    report_path: Path | None = None
    promote_to: Path | None = None
    domain: str | None = None
    seed_split: str = "train"
    validation_split: str = "validation"
    threshold: float = 0.85
    min_precision: float = 0.95
    min_recall: float = 0.0
    training: TrainingConfig | None = None


@dataclass(frozen=True)
class RetrainResult:
    """Auditable result from one feedback-driven retraining run."""

    checkpoint_path: Path
    report_path: Path
    feedback_examples: int
    seed_examples: int
    train_examples: int
    validation_examples: int
    validation_metrics: EvaluationMetrics
    precision_gate_passed: bool
    recall_gate_passed: bool
    gates_passed: bool
    promoted_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "report_path": str(self.report_path),
            "feedback_examples": self.feedback_examples,
            "seed_examples": self.seed_examples,
            "train_examples": self.train_examples,
            "validation_examples": self.validation_examples,
            "validation_metrics": self.validation_metrics.to_dict(),
            "precision_gate_passed": self.precision_gate_passed,
            "recall_gate_passed": self.recall_gate_passed,
            "gates_passed": self.gates_passed,
            "promoted_path": str(self.promoted_path) if self.promoted_path else None,
        }


def retrain_from_feedback(
    *,
    store: SQLiteCacheStore,
    embedding_provider: EmbeddingProvider,
    config: RetrainConfig,
) -> RetrainResult:
    """Train a candidate classifier from feedback and optional seed data."""

    _validate_config(config)
    feedback_records = _load_feedback_records(store=store, domain=config.domain)
    seed_records = _load_seed_records(config)
    train_records = [*feedback_records, *seed_records]
    if not train_records:
        msg = "retrain requires at least one feedback or seed training record"
        raise ValueError(msg)
    logger.info(
        "retrain started: %d feedback + %d seed training records",
        len(feedback_records),
        len(seed_records),
    )

    validation_records = load_pair_records(
        config.validation_data_path,
        split=config.validation_split,
        domain=config.domain,
    )
    if not validation_records:
        msg = "retrain requires at least one validation record"
        raise ValueError(msg)

    training_config = _training_config(config)
    training_result = train_classifier(
        train_records=train_records,
        validation_records=validation_records,
        embedding_provider=embedding_provider,
        output_path=config.output_path,
        config=training_config,
    )
    if training_result.validation_metrics is None:
        msg = "retrain expected validation metrics but training returned none"
        raise RuntimeError(msg)

    metrics = training_result.validation_metrics
    precision_gate_passed = metrics.precision >= config.min_precision
    recall_gate_passed = metrics.recall >= config.min_recall
    gates_passed = precision_gate_passed and recall_gate_passed
    promoted_path = None
    if config.promote_to is not None and gates_passed:
        promoted_path = config.promote_to
        promoted_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(training_result.checkpoint_path, promoted_path)
    elif config.promote_to is not None and not gates_passed:
        logger.warning(
            "retrain candidate not promoted: gates failed (precision=%.4f recall=%.4f)",
            metrics.precision,
            metrics.recall,
        )
    logger.info(
        "retrain finished: precision=%.4f recall=%.4f gates_passed=%s promoted=%s",
        metrics.precision,
        metrics.recall,
        gates_passed,
        promoted_path is not None,
    )

    report_path = config.report_path or Path(f"{config.output_path}.report.json")
    result = RetrainResult(
        checkpoint_path=training_result.checkpoint_path,
        report_path=report_path,
        feedback_examples=len(feedback_records),
        seed_examples=len(seed_records),
        train_examples=training_result.train_examples,
        validation_examples=training_result.validation_examples,
        validation_metrics=metrics,
        precision_gate_passed=precision_gate_passed,
        recall_gate_passed=recall_gate_passed,
        gates_passed=gates_passed,
        promoted_path=promoted_path,
    )
    _write_report(
        path=report_path,
        result=result,
        config=config,
        training_config=training_config,
    )
    return result


def _load_feedback_records(
    *,
    store: SQLiteCacheStore,
    domain: str | None,
) -> list[PairRecord]:
    with TemporaryDirectory(prefix="SemanticMemo-feedback-") as directory:
        feedback_path = Path(directory) / "feedback_pairs.jsonl"
        store.export_feedback_pairs(feedback_path, split="train")
        return load_pair_records(feedback_path, split="train", domain=domain)


def _load_seed_records(config: RetrainConfig) -> list[PairRecord]:
    if config.seed_data_path is None:
        return []
    return load_pair_records(
        config.seed_data_path,
        split=config.seed_split,
        domain=config.domain,
    )


def _training_config(config: RetrainConfig) -> TrainingConfig:
    base = config.training or TrainingConfig()
    return TrainingConfig(
        epochs=base.epochs,
        batch_size=base.batch_size,
        learning_rate=base.learning_rate,
        threshold=config.threshold,
        device=base.device,
        seed=base.seed,
        dropout=base.dropout,
    )


def _write_report(
    *,
    path: Path,
    result: RetrainResult,
    config: RetrainConfig,
    training_config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint_path": str(result.checkpoint_path),
        "promoted_path": str(result.promoted_path) if result.promoted_path else None,
        "domain": config.domain,
        "threshold": config.threshold,
        "min_precision": config.min_precision,
        "min_recall": config.min_recall,
        "feedback_examples": result.feedback_examples,
        "seed_examples": result.seed_examples,
        "train_examples": result.train_examples,
        "validation_examples": result.validation_examples,
        "validation_metrics": result.validation_metrics.to_dict(),
        "precision_gate_passed": result.precision_gate_passed,
        "recall_gate_passed": result.recall_gate_passed,
        "gates_passed": result.gates_passed,
        "seed_data_path": str(config.seed_data_path) if config.seed_data_path else None,
        "validation_data_path": str(config.validation_data_path),
        "seed_split": config.seed_split,
        "validation_split": config.validation_split,
        "training_config": asdict(training_config),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_config(config: RetrainConfig) -> None:
    if not 0 <= config.threshold <= 1:
        msg = "threshold must be between 0 and 1"
        raise ValueError(msg)
    if not 0 <= config.min_precision <= 1:
        msg = "min_precision must be between 0 and 1"
        raise ValueError(msg)
    if not 0 <= config.min_recall <= 1:
        msg = "min_recall must be between 0 and 1"
        raise ValueError(msg)
