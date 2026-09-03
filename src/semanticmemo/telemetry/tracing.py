"""Distributed tracing facade with OpenTelemetry support and zero-overhead fallback."""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
import time
from collections.abc import Generator
from types import TracebackType
from typing import Any

from semanticmemo._logging import get_logger

logger = get_logger(__name__)

# Check if debug trace logging is explicitly requested via environment
_TRACE_DEBUG = os.environ.get("SEMANTICMEMO_TRACE_DEBUG", "false").lower() in ("true", "1")


def hash_identifier(val: str | None) -> str:
    """Generate SHA-256 hash for sensitive identifiers to prevent PII / data leaks."""
    if not val:
        return "none"
    return hashlib.sha256(val.encode("utf-8")).hexdigest()[:16]


def sanitize_attributes(
    attrs: dict[str, Any],
    *,
    include_raw_identifiers: bool = False,
) -> dict[str, Any]:
    """Sanitize span attributes ensuring raw prompts and sensitive tokens are never traced.

    By default, hashes tenant, resource, and user identifiers to protect privacy.
    """
    sanitized: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        # Strictly reject raw prompts and responses
        sensitive_keys = (
            "prompt",
            "query_prompt",
            "response",
            "cached_response",
            "token",
            "jwt",
            "api_key",
        )
        if k in sensitive_keys:
            continue

        if k in ("tenant_id", "user_id", "resource_id") and not include_raw_identifiers:
            sanitized[f"semanticmemo.{k.replace('_id', '_hash')}"] = hash_identifier(str(v))
        elif k.startswith("semanticmemo.") or k.startswith("service.") or k.startswith("error."):
            sanitized[k] = v
        else:
            sanitized[f"semanticmemo.{k}"] = v
    return sanitized


class Span:
    """Span protocol / wrapper interface."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def set_status_ok(self) -> None:
        pass

    def set_status_error(self, description: str | None = None) -> None:
        pass

    def end(self) -> None:
        pass


class NoOpSpan(Span):
    """Minimal-overhead no-op span."""

    def __enter__(self) -> Span:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


class DebugLogSpan(Span):
    """Diagnostic span that logs execution details only when explicitly requested."""

    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.start_time = time.perf_counter()

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)

    def record_exception(self, exception: BaseException) -> None:
        logger.error("[TRACE:ERROR] %s: %s", self.name, exception)

    def end(self) -> None:
        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        logger.debug("[TRACE:END] %s (%0.3fms) attrs=%s", self.name, duration_ms, self.attributes)

    def __enter__(self) -> Span:
        logger.debug("[TRACE:START] %s attrs=%s", self.name, self.attributes)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is not None:
            self.record_exception(exc_val)
        self.end()


class OTelSpanWrapper(Span):
    """Wrapper around active opentelemetry.trace.Span."""

    def __init__(self, otel_span: Any) -> None:
        self._span = otel_span

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            if self._span and hasattr(self._span, "set_attribute"):
                self._span.set_attribute(key, value)
        except Exception:
            pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        try:
            if self._span and hasattr(self._span, "set_attributes"):
                self._span.set_attributes(attributes)
        except Exception:
            pass

    def record_exception(self, exception: BaseException) -> None:
        try:
            if self._span and hasattr(self._span, "record_exception"):
                self._span.record_exception(exception)
        except Exception:
            pass

    def set_status_ok(self) -> None:
        try:
            from opentelemetry.trace import (  # pyright: ignore[reportMissingImports]
                Status,
                StatusCode,
            )

            if self._span:
                self._span.set_status(Status(status_code=StatusCode.OK))
        except Exception:
            pass

    def set_status_error(self, description: str | None = None) -> None:
        try:
            from opentelemetry.trace import (  # pyright: ignore[reportMissingImports]
                Status,
                StatusCode,
            )

            if self._span:
                self._span.set_status(Status(status_code=StatusCode.ERROR, description=description))
        except Exception:
            pass

    def end(self) -> None:
        try:
            if self._span and hasattr(self._span, "end"):
                self._span.end()
        except Exception:
            pass


class SemanticMemoTracer:
    """Facade for OpenTelemetry tracing with automatic W3C trace context propagation."""

    def __init__(
        self,
        service_name: str = "semanticmemo",
        *,
        include_raw_identifiers: bool = False,
        debug_mode: bool | None = None,
        sample_rate: float | None = None,
        enable_otel: bool = True,
    ) -> None:
        self.service_name = service_name
        self.include_raw_identifiers = include_raw_identifiers
        self.debug_mode = _TRACE_DEBUG if debug_mode is None else debug_mode
        self.enable_otel = enable_otel

        env_sample_rate = os.getenv("SEMANTICMEMO_TRACE_SAMPLE_RATE")
        if sample_rate is not None:
            self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        elif env_sample_rate is not None:
            try:
                self.sample_rate = max(0.0, min(1.0, float(env_sample_rate)))
            except ValueError:
                self.sample_rate = 1.0
        else:
            self.sample_rate = 1.0

        self._otel_tracer: Any = None
        if self.enable_otel:
            self._init_otel()

    def _init_otel(self) -> None:
        try:
            from opentelemetry import trace  # pyright: ignore[reportMissingImports]

            self._otel_tracer = trace.get_tracer(self.service_name)
        except ImportError:
            self._otel_tracer = None

    @property
    def is_otel_available(self) -> bool:
        return self._otel_tracer is not None

    def should_sample(self, is_error: bool = False) -> bool:
        """Evaluate probabilistic trace sampling with 100% retention on errors."""
        if is_error or self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() < self.sample_rate

    @contextlib.contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        traceparent: str | None = None,
        force_sample: bool = False,
    ) -> Generator[Span, None, None]:
        """Start a new span, optionally continuing from a W3C traceparent string."""
        if not force_sample and not self.should_sample():
            yield NoOpSpan()
            return

        sanitized = sanitize_attributes(
            attributes or {},
            include_raw_identifiers=self.include_raw_identifiers,
        )
        sanitized["service.name"] = self.service_name

        if self._otel_tracer is not None:
            # Active OpenTelemetry Tracer
            context = None
            if traceparent:
                try:
                    from opentelemetry.trace.propagation.tracecontext import (  # pyright: ignore[reportMissingImports]
                        TraceContextTextMapPropagator,
                    )

                    carrier = {"traceparent": traceparent}
                    context = TraceContextTextMapPropagator().extract(carrier=carrier)
                except Exception as exc:
                    logger.debug("Trace context propagation error: %s", exc)

            otel_cm = None
            try:
                otel_cm = self._otel_tracer.start_as_current_span(
                    name,
                    context=context,
                    attributes=sanitized,
                )
            except Exception as exc:
                logger.debug("OTel start_as_current_span failed: %s", exc)

            if otel_cm is not None:
                with otel_cm as otel_span:
                    yield OTelSpanWrapper(otel_span)
                return

        # Fallback: DebugLogSpan if explicitly enabled, otherwise minimal-overhead NoOpSpan
        if self.debug_mode:
            span = DebugLogSpan(name, sanitized)
            with span as s:
                yield s
        else:
            yield NoOpSpan()

    def inject_trace_context(self, carrier: dict[str, str]) -> None:
        """Inject current W3C trace context (traceparent) into a dictionary carrier."""
        if self._otel_tracer is not None:
            try:
                from opentelemetry.trace.propagation.tracecontext import (  # pyright: ignore[reportMissingImports]
                    TraceContextTextMapPropagator,
                )

                TraceContextTextMapPropagator().inject(carrier=carrier)
            except Exception:
                pass

    def extract_trace_context(self, carrier: dict[str, str]) -> str | None:
        """Extract W3C traceparent from carrier dictionary."""
        return carrier.get("traceparent") or carrier.get("TRACEPARENT")


# Global default tracer instance
_default_tracer = SemanticMemoTracer()


def get_tracer() -> SemanticMemoTracer:
    return _default_tracer
