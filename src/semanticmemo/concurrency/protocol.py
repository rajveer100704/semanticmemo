"""Single-flight concurrency protocol for request coalescing."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class SingleFlightManager(Protocol):
    """Protocol for coalescing concurrent duplicate operations under a single flight key."""

    async def execute(
        self,
        flight_key: str,
        operation: Callable[[], Awaitable[T]],
        *,
        result_loader: Callable[[], Awaitable[T | None]],
        result_writer: Callable[[T], Awaitable[None]] | None = None,
    ) -> T:
        """Execute operation for leader or wait and load cached result for followers.

        Args:
            flight_key: Unique identity encoding pipeline, scope, tenant, resource,
                version, and prompt hash.
            operation: Asynchronous operation producing the fresh result (e.g. LLM execution).
            result_loader: Callback returning the persisted result from the authoritative
                scoped cache store.
            result_writer: Optional callback to persist the fresh result to the authoritative
                scoped cache (invoked only if caller has write authorization).

        Returns:
            The resolved result (from fresh execution or shared cache).
        """
        ...
