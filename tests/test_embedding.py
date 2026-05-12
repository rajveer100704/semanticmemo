from __future__ import annotations

from uuid import uuid4

import numpy as np

from equivcache.embedding import EmbeddingService, InMemoryVectorIndex
from equivcache.types import FloatVector


class Provider:
    dim = 3

    def embed(self, text: str) -> FloatVector:
        if text == "x":
            return np.array([2, 0, 0], dtype=np.float32)
        return np.array([0, 1, 0], dtype=np.float32)


def test_embedding_service_normalizes_and_searches() -> None:
    service = EmbeddingService(Provider(), InMemoryVectorIndex(dim=3))
    x_id = uuid4()
    y_id = uuid4()

    service.add(x_id, np.array([2, 0, 0], dtype=np.float32))
    service.add(y_id, np.array([0, 1, 0], dtype=np.float32))

    candidates = service.search(service.embed("x"), k=2)

    assert candidates[0].entry_id == x_id
    assert candidates[0].score == 1.0
    assert candidates[1].entry_id == y_id
