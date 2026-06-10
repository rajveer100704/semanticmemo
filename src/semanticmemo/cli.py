"""Small CLI for local smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from semanticmemo import CacheConfig
from semanticmemo.store import SQLiteCacheStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="semanticmemo")
    parser.add_argument("--db-path", default=str(CacheConfig().db_path))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stats", help="Show persisted cache entry counts.")
    _add_train_classifier_parser(subparsers)
    _add_eval_classifier_parser(subparsers)
    _add_export_feedback_parser(subparsers)
    _add_export_active_learning_parser(subparsers)
    _add_retrain_parser(subparsers)
    args = parser.parse_args()

    if args.command == "stats":
        store = SQLiteCacheStore(args.db_path)
        print(f"entries={store.count()} total_hits={store.total_hit_count()}")
        store.close()
    elif args.command == "train-classifier":
        _train_classifier(args)
    elif args.command == "eval-classifier":
        _eval_classifier(args)
    elif args.command == "export-feedback":
        _export_feedback(args)
    elif args.command == "export-active-learning":
        _export_active_learning(args)
    elif args.command == "retrain":
        _retrain(args)


def _add_train_classifier_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser(
        "train-classifier",
        help="Train a PairClassifier checkpoint from JSONL prompt pairs.",
    )
    parser.add_argument("--data", required=True, help="Path to JSONL pair dataset.")
    parser.add_argument("--out", required=True, help="Checkpoint output path.")
    parser.add_argument("--domain", default=None, help="Optional domain filter.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    _add_embedding_args(parser)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--device", default="cpu")


def _add_eval_classifier_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser(
        "eval-classifier",
        help="Evaluate a PairClassifier checkpoint from JSONL prompt pairs.",
    )
    parser.add_argument("--data", required=True, help="Path to JSONL pair dataset.")
    parser.add_argument("--model", required=True, help="Classifier checkpoint path.")
    parser.add_argument("--domain", default=None, help="Optional domain filter.")
    parser.add_argument("--split", default="test")
    _add_embedding_args(parser)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cpu")


def _add_export_feedback_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser(
        "export-feedback",
        help="Export feedback events as classifier training JSONL pairs.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--split", default="train", help="Split value written to each record.")


def _add_retrain_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser(
        "retrain",
        help="Train and optionally promote a classifier from durable feedback.",
    )
    parser.add_argument("--out", required=True, help="Candidate checkpoint output path.")
    parser.add_argument(
        "--validation-data",
        required=True,
        help="JSONL validation dataset used for promotion gates.",
    )
    parser.add_argument("--seed-data", default=None, help="Optional JSONL seed training dataset.")
    parser.add_argument("--report", default=None, help="Report output path.")
    parser.add_argument("--promote-to", default=None, help="Copy checkpoint here if gates pass.")
    parser.add_argument("--domain", default=None, help="Optional domain filter.")
    parser.add_argument("--seed-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    _add_embedding_args(parser)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence-transformer", "hash"],
        default="sentence-transformer",
        help="Use real SentenceTransformers embeddings or deterministic hash embeddings.",
    )
    parser.add_argument(
        "--embedding-model",
        default=CacheConfig().embedding_model,
        help="SentenceTransformers model name.",
    )
    parser.add_argument("--embedding-dim", type=int, default=CacheConfig().embedding_dim)


def _train_classifier(args: argparse.Namespace) -> None:
    from semanticmemo.classifier import TrainingConfig, load_pair_records, train_classifier

    train_records = load_pair_records(args.data, split=args.train_split, domain=args.domain)
    validation_records = load_pair_records(
        args.data,
        split=args.validation_split,
        domain=args.domain,
    )
    provider = _embedding_provider(args)
    result = train_classifier(
        train_records=train_records,
        validation_records=validation_records,
        embedding_provider=provider,
        output_path=Path(args.out),
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            threshold=args.threshold,
            device=args.device,
        ),
    )
    payload: dict[str, Any] = {
        "checkpoint_path": str(result.checkpoint_path),
        "train_examples": result.train_examples,
        "validation_examples": result.validation_examples,
        "final_train_loss": result.final_train_loss,
        "validation_metrics": (
            result.validation_metrics.to_dict() if result.validation_metrics is not None else None
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _eval_classifier(args: argparse.Namespace) -> None:
    from semanticmemo.classifier import ClassifierService, evaluate_model, load_pair_records

    service = ClassifierService(
        args.model,
        device=args.device,
        threshold=args.threshold,
    )
    records = load_pair_records(args.data, split=args.split, domain=args.domain)
    provider = _embedding_provider(args)
    metrics = evaluate_model(
        model=service.model,
        records=records,
        embedding_provider=provider,
        threshold=service.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))


def _export_feedback(args: argparse.Namespace) -> None:
    store = SQLiteCacheStore(args.db_path)
    count = store.export_feedback_pairs(args.out, split=args.split)
    store.close()
    print(f"exported={count} path={args.out}")


def _retrain(args: argparse.Namespace) -> None:
    from semanticmemo.classifier import TrainingConfig
    from semanticmemo.feedback import RetrainConfig, retrain_from_feedback

    store = SQLiteCacheStore(args.db_path)
    try:
        result = retrain_from_feedback(
            store=store,
            embedding_provider=_embedding_provider(args),
            config=RetrainConfig(
                output_path=Path(args.out),
                validation_data_path=Path(args.validation_data),
                seed_data_path=Path(args.seed_data) if args.seed_data else None,
                report_path=Path(args.report) if args.report else None,
                promote_to=Path(args.promote_to) if args.promote_to else None,
                domain=args.domain,
                seed_split=args.seed_split,
                validation_split=args.validation_split,
                threshold=args.threshold,
                min_precision=args.min_precision,
                min_recall=args.min_recall,
                training=TrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    threshold=args.threshold,
                    device=args.device,
                ),
            ),
        )
    finally:
        store.close()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _embedding_provider(args: argparse.Namespace):
    if args.embedding_provider == "hash":
        from semanticmemo.embedding import HashEmbeddingProvider

        return HashEmbeddingProvider(dim=args.embedding_dim)

    from semanticmemo.embedding import SentenceTransformerEmbeddingProvider

    return SentenceTransformerEmbeddingProvider(
        model_name=args.embedding_model,
        dim=args.embedding_dim,
    )


def _add_export_active_learning_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser(
        "export-active-learning",
        help="Export recorded active learning pairs as classifier training JSONL pairs.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--split", default="train", help="Split value written to each record.")


def _export_active_learning(args: argparse.Namespace) -> None:
    store = SQLiteCacheStore(args.db_path)
    count = store.export_active_learning_pairs(args.out, split=args.split)
    store.close()
    print(f"exported={count} path={args.out}")

