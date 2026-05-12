from __future__ import annotations

import equivcache


def test_public_api_exports() -> None:
    assert equivcache.EquivCache is not None
    assert equivcache.CacheConfig is not None
    assert equivcache.CacheResult is not None
    assert equivcache.CacheStats is not None
    assert equivcache.CacheEntry is not None
    assert equivcache.EvictionPolicy is not None
