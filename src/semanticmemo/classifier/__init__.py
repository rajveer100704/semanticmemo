"""Classifier training and inference utilities."""

from semanticmemo.classifier.cross_encoder_service import CrossEncoderService
from semanticmemo.classifier.data import EmbeddedPairDataset, PairRecord, load_pair_records
from semanticmemo.classifier.evaluate import (
    EvaluationMetrics,
    compute_binary_metrics,
    evaluate_model,
)
from semanticmemo.classifier.model import PairClassifier, build_pair_features
from semanticmemo.classifier.service import ClassifierService
from semanticmemo.classifier.train import TrainingConfig, TrainingResult, train_classifier

__all__ = [
    "ClassifierService",
    "CrossEncoderService",
    "EmbeddedPairDataset",
    "EvaluationMetrics",
    "PairClassifier",
    "PairRecord",
    "TrainingConfig",
    "TrainingResult",
    "build_pair_features",
    "compute_binary_metrics",
    "evaluate_model",
    "load_pair_records",
    "train_classifier",
]
