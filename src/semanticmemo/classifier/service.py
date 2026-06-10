"""Inference wrapper for trained equivalence classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from semanticmemo._logging import get_logger
from semanticmemo.classifier.model import PairClassifier
from semanticmemo.embedding.service import normalize
from semanticmemo.types import FloatVector

_torch: Any = torch

logger = get_logger(__name__)


# Global cache for loaded model checkpoints to avoid reloading overhead
_MODEL_CACHE: dict[tuple[Path, torch.device], tuple[PairClassifier, float]] = {}


class ClassifierService:
    """Load a PairClassifier checkpoint and run CPU-friendly inference."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str = "cpu",
        threshold: float | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = _torch.device(device)
        self.model, checkpoint_threshold = self._load_model_cached(self.model_path)
        self.threshold = threshold if threshold is not None else checkpoint_threshold

    def predict(self, emb_a: FloatVector, emb_b: FloatVector) -> float:
        return self.predict_batch([(emb_a, emb_b)])[0]

    def predict_batch(self, pairs: Sequence[tuple[FloatVector, FloatVector]]) -> list[float]:
        if not pairs:
            return []
        embeddings_a = _torch.tensor(
            np.stack([normalize(np.asarray(pair[0], dtype=np.float32)) for pair in pairs]),
            dtype=_torch.float32,
            device=self.device,
        )
        embeddings_b = _torch.tensor(
            np.stack([normalize(np.asarray(pair[1], dtype=np.float32)) for pair in pairs]),
            dtype=_torch.float32,
            device=self.device,
        )
        self.model.eval()
        with _torch.no_grad():
            probabilities = self.model(embeddings_a, embeddings_b)
        return [float(value) for value in probabilities.detach().cpu().tolist()]

    def is_equivalent(self, emb_a: FloatVector, emb_b: FloatVector) -> bool:
        return self.predict(emb_a, emb_b) >= self.threshold

    def reload(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        self.model, checkpoint_threshold = self._load_model_cached(
            self.model_path, force_reload=True
        )
        self.threshold = checkpoint_threshold

    def _load_model_cached(
        self, model_path: Path, *, force_reload: bool = False
    ) -> tuple[PairClassifier, float]:
        cache_key = (model_path, self.device)
        if not force_reload and cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        model, threshold = self._load_model(model_path)
        _MODEL_CACHE[cache_key] = (model, threshold)
        return model, threshold

    def _load_model(self, model_path: Path) -> tuple[PairClassifier, float]:
        checkpoint: dict[str, Any] = _torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        if checkpoint.get("model_type") != "PairClassifier":
            msg = f"Unsupported classifier checkpoint at {model_path}"
            raise ValueError(msg)
        embed_dim = int(checkpoint.get("embed_dim", 384))
        model = PairClassifier(embed_dim=embed_dim)
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            msg = f"Classifier checkpoint at {model_path} has no model_state_dict"
            raise ValueError(msg)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        threshold = float(checkpoint.get("threshold", 0.85))
        logger.debug(
            "loaded classifier checkpoint: path=%s embed_dim=%d threshold=%.4f",
            model_path,
            embed_dim,
            threshold,
        )
        return model, threshold
