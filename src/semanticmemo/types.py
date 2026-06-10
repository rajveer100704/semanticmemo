"""Shared typing protocols."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatVector: TypeAlias = NDArray[np.float32]


class EmbeddingProvider(Protocol):
    """Protocol implemented by prompt embedding providers."""

    dim: int

    def embed(self, text: str) -> FloatVector:
        """Return one embedding vector for text."""
        ...


class EquivalenceClassifier(Protocol):
    """Protocol implemented by prompt-pair equivalence classifiers."""

    threshold: float

    def predict_batch(self, pairs: Sequence[tuple[FloatVector, FloatVector]]) -> list[float]:
        """Return equivalence probabilities for embedding pairs."""
        ...


LLMFunction: TypeAlias = Callable[[str], str | Awaitable[str]]
