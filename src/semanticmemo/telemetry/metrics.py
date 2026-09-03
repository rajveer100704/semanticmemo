"""Low-cardinality Prometheus-compatible metrics registry and exposition engine."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# Standard duration histogram buckets in seconds (0.5ms to 10s)
DEFAULT_DURATION_BUCKETS: tuple[float, ...] = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

# Candidate space reduction ratio buckets (0.0 to 1.0)
DEFAULT_RATIO_BUCKETS: tuple[float, ...] = (
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
    0.999,
    1.0,
)

# Allowed low-cardinality label keys to prevent cardinality explosion
ALLOWED_METRIC_LABELS: frozenset[str] = frozenset(
    {
        "scope",
        "decision",
        "domain",
        "backend",
        "stage",
        "role",
        "event_type",
        "status",
        "action",
        "type",
        "model",
    }
)


def _sanitize_labels(labels: dict[str, str]) -> dict[str, str]:
    """Ensure only strictly bounded low-cardinality labels are exported."""
    return {k: str(v) for k, v in sorted(labels.items()) if k in ALLOWED_METRIC_LABELS}


def _labels_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


@dataclass
class MetricSample:
    name: str
    labels: dict[str, str]
    value: float
    timestamp: float | None = None


class Counter:
    """Thread-safe monotonic counter."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if amount < 0:
            raise ValueError("Counters can only be incremented by non-negative amounts.")
        sanitized = _sanitize_labels(labels or {})
        key = _labels_key(sanitized)
        with self._lock:
            self._values[key] += amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        sanitized = _sanitize_labels(labels or {})
        key = _labels_key(sanitized)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[MetricSample]:
        with self._lock:
            return [MetricSample(self.name, dict(k), v) for k, v in sorted(self._values.items())]


class Gauge:
    """Thread-safe gauge representing a value that can go up and down."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        sanitized = _sanitize_labels(labels or {})
        key = _labels_key(sanitized)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        sanitized = _sanitize_labels(labels or {})
        key = _labels_key(sanitized)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        sanitized = _sanitize_labels(labels or {})
        key = _labels_key(sanitized)
        with self._lock:
            self._values[key] -= amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        sanitized = _sanitize_labels(labels or {})
        key = _labels_key(sanitized)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[MetricSample]:
        with self._lock:
            return [MetricSample(self.name, dict(k), v) for k, v in sorted(self._values.items())]


@dataclass
class HistogramData:
    buckets: tuple[float, ...]
    counts: list[int]
    total_count: int = 0
    total_sum: float = 0.0


class Histogram:
    """Thread-safe cumulative histogram with configurable buckets in seconds."""

    def __init__(
        self,
        name: str,
        description: str,
        buckets: tuple[float, ...] = DEFAULT_DURATION_BUCKETS,
    ) -> None:
        self.name = name
        self.description = description
        self.buckets = tuple(sorted(buckets))
        self._data: dict[tuple[tuple[str, str], ...], HistogramData] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        val = float(value)
        sanitized = _sanitize_labels(labels or {})
        key = _labels_key(sanitized)
        with self._lock:
            if key not in self._data:
                self._data[key] = HistogramData(
                    buckets=self.buckets,
                    counts=[0] * len(self.buckets),
                )
            hdata = self._data[key]
            hdata.total_count += 1
            hdata.total_sum += val
            for i, b in enumerate(hdata.buckets):
                if val <= b:
                    hdata.counts[i] += 1

    def collect(self) -> list[MetricSample]:
        samples: list[MetricSample] = []
        with self._lock:
            for key, hdata in sorted(self._data.items()):
                base_labels = dict(key)
                for b, count in zip(hdata.buckets, hdata.counts, strict=False):
                    b_str = "+Inf" if math.isinf(b) else str(b)
                    lbls = {**base_labels, "le": b_str}
                    samples.append(MetricSample(f"{self.name}_bucket", lbls, float(count)))
                lbls_inf = {**base_labels, "le": "+Inf"}
                samples.append(
                    MetricSample(f"{self.name}_bucket", lbls_inf, float(hdata.total_count))
                )
                samples.append(
                    MetricSample(f"{self.name}_sum", base_labels, float(hdata.total_sum))
                )
                samples.append(
                    MetricSample(f"{self.name}_count", base_labels, float(hdata.total_count))
                )
        return samples


class SemanticMemoMetrics:
    """Central metrics registry for SemanticMemo telemetry."""

    _instance: SemanticMemoMetrics | None = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        # Request Counters (Low Cardinality: scope, decision, domain, backend)
        self.cache_requests_total = Counter(
            "semanticmemo_cache_requests_total",
            "Total cache lookup requests partitioned by decision and scope",
        )
        # Latency Histograms (in Seconds)
        self.request_duration_seconds = Histogram(
            "semanticmemo_request_duration_seconds",
            "Latency of cache request lifecycle stages in seconds",
            buckets=DEFAULT_DURATION_BUCKETS,
        )
        # Search Space Reduction Ratio (0.0 - 1.0)
        self.candidate_space_reduction_ratio = Histogram(
            "semanticmemo_candidate_space_reduction_ratio",
            "Ratio of search-space reduction achieved by scoped pre-retrieval",
            buckets=DEFAULT_RATIO_BUCKETS,
        )
        # Single Flight Metrics
        self.single_flight_total = Counter(
            "semanticmemo_single_flight_total",
            "Total single-flight lock acquisitions partitioned by role (leader/follower)",
        )
        self.single_flight_duration_seconds = Histogram(
            "semanticmemo_single_flight_duration_seconds",
            "Duration of single-flight coordination in seconds",
            buckets=DEFAULT_DURATION_BUCKETS,
        )
        # Financial / Token Savings
        self.tokens_saved_total = Counter(
            "semanticmemo_tokens_saved_total",
            "Estimated total LLM tokens saved via verified semantic cache hits",
        )
        self.cost_saved_usd_total = Counter(
            "semanticmemo_cost_saved_usd_total",
            "Estimated total LLM API cost saved in USD",
        )
        # Outbox & Event Telemetry
        self.outbox_events_total = Counter(
            "semanticmemo_outbox_events_total",
            "Total outbox event lifecycle transitions by event_type and status",
        )
        # Reconciliation Metrics
        self.reconciliation_total = Counter(
            "semanticmemo_reconciliation_total",
            "Total vector index reconciliation actions performed",
        )
        # Active Entries Gauge
        self.cache_entries_active = Gauge(
            "semanticmemo_cache_entries_active",
            "Total active entries tracked in cache persistence",
        )

        self._all_metrics: list[Counter | Gauge | Histogram] = [
            self.cache_requests_total,
            self.request_duration_seconds,
            self.candidate_space_reduction_ratio,
            self.single_flight_total,
            self.single_flight_duration_seconds,
            self.tokens_saved_total,
            self.cost_saved_usd_total,
            self.outbox_events_total,
            self.reconciliation_total,
            self.cache_entries_active,
        ]

    @classmethod
    def default(cls) -> SemanticMemoMetrics:
        """Get or create the global singleton metrics instance."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for clean unit tests)."""
        with cls._singleton_lock:
            cls._instance = None

    def export_openmetrics_text(self) -> str:
        """Export all registered metrics in standard Prometheus/OpenMetrics text format."""
        lines: list[str] = []
        for metric in self._all_metrics:
            lines.append(f"# HELP {metric.name} {metric.description}")
            if isinstance(metric, Counter):
                metric_type = "counter"
            elif isinstance(metric, Gauge):
                metric_type = "gauge"
            else:
                metric_type = "histogram"
            lines.append(f"# TYPE {metric.name} {metric_type}")
            for sample in metric.collect():
                if sample.labels:
                    lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(sample.labels.items()))
                    lines.append(f"{sample.name}{{{lbl_str}}} {sample.value}")
                else:
                    lines.append(f"{sample.name} {sample.value}")
        lines.append("# EOF\n")
        return "\n".join(lines)


class StageTimer:
    """Context manager for measuring and recording stage durations in seconds."""

    def __init__(
        self,
        histogram: Histogram,
        stage: str,
        extra_labels: dict[str, str] | None = None,
    ) -> None:
        self.histogram = histogram
        self.stage = stage
        self.extra_labels = extra_labels or {}
        self.start_time = 0.0
        self.duration_seconds = 0.0

    def __enter__(self) -> StageTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.duration_seconds = time.perf_counter() - self.start_time
        labels = {"stage": self.stage, **self.extra_labels}
        self.histogram.observe(self.duration_seconds, labels=labels)
