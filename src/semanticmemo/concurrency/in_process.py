"""In-process single-flight manager using asyncio.Future coalescing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class InProcessSingleFlightManager:
    """In-process single-flight manager using asyncio.Future.

    Coalesces concurrent identical operations within a single Python process.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self._flight_generations: dict[str, int] = {}

    async def execute(
        self,
        flight_key: str,
        operation: Callable[[], Awaitable[T]],
        *,
        result_loader: Callable[[], Awaitable[T | None]],
        result_writer: Callable[[T], Awaitable[None]] | None = None,
    ) -> T:
        loop = asyncio.get_running_loop()

        # Follower path: an identical flight is already in flight in this process
        if flight_key in self._inflight:
            fut = self._inflight[flight_key]
            try:
                result = await asyncio.shield(fut)
                return result
            except Exception:
                # If leader failed, fallback to checking store or direct execution
                cached = await result_loader()
                if cached is not None:
                    return cached
                return await operation()

        # Leader path: register future and execute
        leader_fut: asyncio.Future[T] = loop.create_future()
        self._inflight[flight_key] = leader_fut
        gen = self._flight_generations.get(flight_key, 0) + 1
        self._flight_generations[flight_key] = gen

        try:
            result = await operation()
            if result_writer is not None:
                import inspect

                sig = inspect.signature(result_writer)
                if len(sig.parameters) >= 2 or any(
                    p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
                ):
                    await result_writer(result, gen)  # type: ignore[call-arg]
                else:
                    await result_writer(result)
            if not leader_fut.done():
                leader_fut.set_result(result)
            return result
        except BaseException as exc:
            if not leader_fut.done():
                leader_fut.set_exception(exc)
            # Retrieve exception to avoid unretrieved future loop warnings
            _ = leader_fut.exception()
            raise
        finally:
            self._inflight.pop(flight_key, None)
