"""Shared typing protocols."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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


LLMFunction: TypeAlias = Callable[[str], str | Awaitable[str]]
