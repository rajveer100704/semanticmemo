"""Telemetry module: Metrics, Prometheus exposition, and OpenTelemetry tracing."""

from __future__ import annotations

from semanticmemo.telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    SemanticMemoMetrics,
    StageTimer,
)
from semanticmemo.telemetry.tracing import (
    NoOpSpan,
    SemanticMemoTracer,
    Span,
    get_tracer,
    hash_identifier,
    sanitize_attributes,
)

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "NoOpSpan",
    "SemanticMemoMetrics",
    "SemanticMemoTracer",
    "Span",
    "StageTimer",
    "get_tracer",
    "hash_identifier",
    "sanitize_attributes",
]
