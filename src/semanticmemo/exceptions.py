"""Project-specific exceptions."""


class SemanticMemoError(Exception):
    """Base exception for SemanticMemo errors."""


class MissingDependencyError(SemanticMemoError):
    """Raised when an optional dependency is required but not installed."""


class CacheStoreError(SemanticMemoError):
    """Raised for persistence-layer failures."""


class LLMCallError(SemanticMemoError):
    """Raised when the user-supplied LLM function fails after all retry attempts.

    Only raised when retries are enabled via ``CacheConfig.retry``. The final
    underlying exception is chained as ``__cause__``. Without retries enabled,
    the original exception propagates unchanged instead.
    """
