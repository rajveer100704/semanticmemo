"""EquivCache public API."""

from equivcache.cache import EquivCache
from equivcache.models import (
    CacheConfig,
    CacheEntry,
    CacheResult,
    CacheStats,
    EvictionPolicy,
)

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "CacheResult",
    "CacheStats",
    "EquivCache",
    "EvictionPolicy",
]
