"""Project-specific exceptions."""


class EquivCacheError(Exception):
    """Base exception for EquivCache errors."""


class MissingDependencyError(EquivCacheError):
    """Raised when an optional dependency is required but not installed."""


class CacheStoreError(EquivCacheError):
    """Raised for persistence-layer failures."""
