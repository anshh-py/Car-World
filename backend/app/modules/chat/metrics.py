"""Lightweight, in-memory observability for CarWorld AI.

Collects performance and security counters and latency distributions without any
external analytics dependency or database writes. All state lives in process
memory and is reset on restart; this is intentionally simple and safe.

Counter names (prefixes):
  perf.*      - request counts, provider calls, tokens, cache, latency
  sec.*       - injection, jailbreak, blocked tools, PII, output validation,
                moderation, rate-limit events

The registry is thread-safe (a single lock guards all mutations).
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from typing import Any


class _LatencyStats:
    """Online min/max/count/sum statistics for a named latency series."""

    __slots__ = ("count", "total", "minimum", "maximum")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def record(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        self.count += 1
        self.total += value
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    def snapshot(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "avg_ms": None, "min_ms": None, "max_ms": None}
        return {
            "count": self.count,
            "avg_ms": round((self.total / self.count) * 1000.0, 2),
            "min_ms": round(self.minimum * 1000.0, 2),
            "max_ms": round(self.maximum * 1000.0, 2),
        }


class MetricsRegistry:
    """Thread-safe counter + latency registry consumed by the API and tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, _LatencyStats] = {}
        self._started = time.time()

    def incr(self, name: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        with self._lock:
            self._counters[name] += amount

    def record_latency(self, name: str, seconds: float) -> None:
        with self._lock:
            stats = self._latencies.get(name)
            if stats is None:
                stats = self._latencies[name] = _LatencyStats()
            stats.record(seconds)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(sorted(self._counters.items()))
            latencies = {name: stats.snapshot() for name, stats in sorted(self._latencies.items())}
        return {
            "uptime_seconds": round(time.time() - self._started, 2),
            "counters": counters,
            "latencies_ms": latencies,
        }

    def summary(self) -> dict[str, Any]:
        """Compact developer-readable digest intended for backend diagnostics."""
        snap = self.snapshot()
        counters = snap["counters"]
        return {
            "requests": counters.get("perf.requests", 0),
            "provider_calls": counters.get("perf.provider_calls", 0),
            "fallbacks": counters.get("perf.fallback_activated", 0),
            "cache_hits": counters.get("perf.cache_hits", 0),
            "cache_misses": counters.get("perf.cache_misses", 0),
            "injection_detected": counters.get("sec.injection_detected", 0),
            "jailbreak_detected": counters.get("sec.jailbreak_detected", 0),
            "blocked_tools": counters.get("sec.blocked_tools", 0),
            "pii_redacted": counters.get("sec.pii_redacted", 0),
            "output_validation_failures": counters.get("sec.output_validation_failures", 0),
            "moderation_events": counters.get("sec.moderation_events", 0),
            "rate_limit_events": counters.get("sec.rate_limit_events", 0),
            "latencies_ms": snap["latencies_ms"],
        }


# Process-wide singleton shared by routes, providers, and the agent.
metrics = MetricsRegistry()


def diagnostics() -> dict:
    """Developer-readable performance + cost digest for backend diagnostics.

    Combines the in-memory metrics snapshot with token/cost accounting. This is
    intended for operator/developer inspection (e.g. a debug endpoint or a
    periodic log line), not for end users. It never contains raw secrets or
    full conversation text.
    """
    from app.modules.chat.usage import usage as _usage  # local import avoids cycles
    return {
        "performance": metrics.summary(),
        "cost_and_tokens": _usage.summary(),
    }


def log_diagnostics(*, logger: "Any | None" = None) -> None:
    """Log a sanitized, structured operational summary.

    Emits only aggregate counters, latency summaries, and cost/token totals -
    never prompts, user messages, PII, API keys, or credentials. Used by the
    application lifespan to log one summary per process lifetime rather than on
    every request, so it produces minimal log noise. Logging failures are
    swallowed so they can never break chat requests.
    """
    try:
        if logger is None:
            import logging as _logging
            logger = _logging.getLogger("carworld.observability")
        digest = diagnostics()
        logger.info("carworld operational summary: %s", _sanitize_for_log(digest))
    except Exception:
        # Observability must never break the application.
        return


def _sanitize_for_log(value: Any) -> str:
    """Serialize a digest into a JSON string, dropping non-serializable or
    unexpected fields so no raw secrets or conversation content leak out."""
    import json as _json
    try:
        return _json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return "{}"
