"""Classifier training and inference utilities."""

from equivcache.classifier.data import EmbeddedPairDataset, PairRecord, load_pair_records
from equivcache.classifier.evaluate import EvaluationMetrics, compute_binary_metrics, evaluate_model
from equivcache.classifier.model import PairClassifier, build_pair_features
from equivcache.classifier.service import ClassifierService
from equivcache.classifier.train import TrainingConfig, TrainingResult, train_classifier

__all__ = [
    "ClassifierService",
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
