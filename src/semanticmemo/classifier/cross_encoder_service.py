"""Inference wrapper for second-stage Cross-Encoder models."""

from __future__ import annotations

import math
from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from semanticmemo._logging import get_logger

logger = get_logger(__name__)

# Global cache for instantiated CrossEncoder models to avoid reloading overhead
_MODEL_CACHE: dict[tuple[str, str], CrossEncoder] = {}


class CrossEncoderService:
    """Wrapper around sentence-transformers CrossEncoder for high-precision validation."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "cpu",
        threshold: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            cache_key = (self.model_name, self.device)
            if cache_key in _MODEL_CACHE:
                self._model = _MODEL_CACHE[cache_key]
            else:
                logger.info("Loading CrossEncoder: %s on %s", self.model_name, self.device)
                self._model = CrossEncoder(self.model_name, device=self.device)
                _MODEL_CACHE[cache_key] = self._model
        return self._model

    def predict(self, prompt_a: str, prompt_b: str) -> float:
        """Score a single pair of prompts."""
        return self.predict_batch([(prompt_a, prompt_b)])[0]

    def predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Score a list of prompt pairs.

        Returns probability scores mapped to [0, 1] using sigmoid.
        """
        if not pairs:
            return []

        # Convert sequence to list of lists/tuples as expected by sentence-transformers
        formatted_pairs = [list(pair) for pair in pairs]

        # Run prediction
        raw_scores = self.model.predict(formatted_pairs, show_progress_bar=False)

        # Handle single float vs list output
        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]

        # Apply sigmoid because ms-marco output is a logit, but we need [0, 1] probability
        # Also handle potential NaN or infinity safely
        probs = []
        for score in raw_scores:
            s = float(score)
            # sigmoid(x) = 1 / (1 + exp(-x))
            try:
                prob = 1.0 / (1.0 + math.exp(-s))
            except OverflowError:
                prob = 0.0 if s < 0 else 1.0
            probs.append(prob)

        return probs

    def is_equivalent(self, prompt_a: str, prompt_b: str) -> bool:
        """Check if two prompts are equivalent based on the threshold."""
        threshold = 0.90 if self.threshold is None else self.threshold
        return self.predict(prompt_a, prompt_b) >= threshold
