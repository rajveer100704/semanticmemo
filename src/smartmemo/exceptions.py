"""Project-specific exceptions."""


class SmartMemoError(Exception):
    """Base exception for SmartMemo errors."""


class MissingDependencyError(SmartMemoError):
    """Raised when an optional dependency is required but not installed."""


class CacheStoreError(SmartMemoError):
    """Raised for persistence-layer failures."""


class LLMCallError(SmartMemoError):
    """Raised when the user-supplied LLM function fails after all retry attempts.

    Only raised when retries are enabled via ``CacheConfig.retry``. The final
    underlying exception is chained as ``__cause__``. Without retries enabled,
    the original exception propagates unchanged instead.
    """
