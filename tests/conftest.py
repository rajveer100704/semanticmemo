from __future__ import annotations

import os
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

# faiss-cpu and torch each bundle their own OpenMP runtime; on macOS loading
# both in one process trips "OMP: Error #15". This test-only workaround lets
# the suite exercise the faiss-backed index path. It is a no-op on Linux CI,
# and must be set before smartmemo (and therefore torch/faiss) is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from smartmemo import CacheConfig, SmartMemo  # noqa: E402
from smartmemo.types import FloatVector  # noqa: E402


class ToyEmbeddingProvider:
    dim = 4

    def embed(self, text: str) -> FloatVector:
        match text:
            case "alpha":
                return np.array([1, 0, 0, 0], dtype=np.float32)
            case "alpha duplicate":
                return np.array([1, 0, 0, 0], dtype=np.float32)
            case "near alpha":
                return np.array([0.8, 0.2, 0, 0], dtype=np.float32)
            case "beta":
                return np.array([0, 1, 0, 0], dtype=np.float32)
            case _:
                return np.array([0, 0, 1, 0], dtype=np.float32)


@pytest.fixture
def cache_config(tmp_path: Path) -> CacheConfig:
    return CacheConfig(
        db_path=tmp_path / "smartmemo.db",
        embedding_dim=ToyEmbeddingProvider.dim,
        cosine_threshold=0.95,
        candidate_k=3,
        max_entries=10,
        estimated_llm_cost_usd=Decimal("0.003"),
    )


@pytest.fixture
def cache(cache_config: CacheConfig) -> Generator[SmartMemo]:
    instance = SmartMemo(
        domain="test",
        config=cache_config,
        embedding_provider=ToyEmbeddingProvider(),
        use_faiss=False,
    )
    yield instance
    instance.close()
