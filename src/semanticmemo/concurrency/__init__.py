"""Single-flight concurrency managers for distributed and local request coalescing."""

from semanticmemo.concurrency.in_process import InProcessSingleFlightManager
from semanticmemo.concurrency.protocol import SingleFlightManager
from semanticmemo.concurrency.redis_flight import (
    REDIS_AVAILABLE,
    RedisConfig,
    RedisFlightMetrics,
    RedisSingleFlightManager,
    build_flight_key,
)

__all__ = [
    "InProcessSingleFlightManager",
    "REDIS_AVAILABLE",
    "RedisConfig",
    "RedisFlightMetrics",
    "RedisSingleFlightManager",
    "SingleFlightManager",
    "build_flight_key",
]
