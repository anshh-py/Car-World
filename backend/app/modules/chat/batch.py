"""Reusable batch processing for non-interactive CarWorld AI workloads.

This is strictly for *offline* / bulk work (evaluation datasets, prompt
reviews, token and cost analysis). It is intentionally NOT used in the normal
interactive chat path — a user request never waits on a batch job.

Design goals:
  * bounded input size and bounded concurrency (never fan out unboundedly);
  * rate-limit awareness: a configurable minimum delay is enforced between
    provider calls so bursts do not trip provider rate limits;
  * retries with exponential backoff for transient failures only (permanent
    errors are not retried);
  * partial-failure handling: a failed item is recorded and the batch
    continues, and the outcome is reported as structured data;
  * no unnecessary memory use: results are collected as the batch runs rather
    than pre-materializing every intermediate stage.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

# Error types that are considered transient (worth retrying). A process function
# may also raise ``RetryableError`` explicitly.
_TRANSIENT = (TimeoutError, ConnectionError, BrokenPipeError, OSError)


class RetryableError(Exception):
    """Raise inside a process function to force a retry for a transient fault."""


@dataclass
class BatchItemResult:
    """Outcome for a single input item."""
    index: int
    item: Any
    success: bool
    result: Any = None
    error: Optional[str] = None
    attempts: int = 1
    retried: bool = False


@dataclass
class BatchResult:
    """Structured aggregate outcome of a batch run."""
    total: int
    succeeded: int = 0
    failed: int = 0
    items: list[BatchItemResult] = field(default_factory=list)

    def structured(self) -> dict[str, Any]:
        """Compact summary suitable for log/JSON output."""
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "failed_indices": [r.index for r in self.items if not r.success],
            "error_messages": sorted(
                {r.error for r in self.items if not r.success and r.error}
            ),
        }


class BatchProcessor:
    """Run ``process_fn`` over ``items`` with bounded concurrency and retries.

    ``process_fn(item) -> result`` is expected to raise on failure. Retryable
    failures (retriable error types or ``RetryableError``) are retried up to
    ``retries`` times with exponential backoff. Permanent failures are recorded
    and the batch continues.
    """

    def __init__(
        self,
        *,
        max_workers: int = 4,
        retries: int = 2,
        base_backoff_seconds: float = 1.0,
        min_interval_seconds: float = 0.0,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self._max_workers = int(max_workers)
        self._retries = int(retries)
        self._base_backoff = float(base_backoff_seconds)
        self._min_interval = float(min_interval_seconds)
        self._last_call = 0.0
        self._throttle_lock = threading.Lock()

    def _throttle(self) -> None:
        """Enforce a minimum gap between provider calls (rate-limit awareness)."""
        if self._min_interval <= 0:
            return
        with self._throttle_lock:
            now = time.monotonic()
            delay = self._last_call + self._min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last_call = time.monotonic()

    def _is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, RetryableError):
            return True
        return isinstance(exc, _TRANSIENT)

    def _run_one(self, index: int, item: Any) -> BatchItemResult:
        attempts = 0
        last_error: Optional[str] = None
        retried = False
        while True:
            attempts += 1
            self._throttle()
            try:
                result = self._process_fn(item)
                return BatchItemResult(
                    index=index, item=item, success=True, result=result,
                    attempts=attempts, retried=retried,
                )
            except Exception as exc:  # noqa: BLE001 - captured per item
                last_error = f"{type(exc).__name__}: {exc}"
                if attempts > self._retries or not self._is_retryable(exc):
                    return BatchItemResult(
                        index=index, item=item, success=False, error=last_error,
                        attempts=attempts, retried=retried,
                    )
                retried = True
                backoff = self._base_backoff * (2 ** (attempts - 1))
                time.sleep(backoff)

    def run(
        self,
        items: Iterable[Any],
        process_fn: Callable[[Any], Any],
    ) -> BatchResult:
        """Process ``items`` and return a structured ``BatchResult``.

        ``process_fn`` must be thread-safe or operating on independent items;
        concurrency is controlled by ``max_workers``.
        """
        items_list = list(items)
        result = BatchResult(total=len(items_list))
        if not items_list:
            return result
        self._process_fn = process_fn

        if self._max_workers == 1 or len(items_list) == 1:
            for i, item in enumerate(items_list):
                item_result = self._run_one(i, item)
                result.items.append(item_result)
                if item_result.success:
                    result.succeeded += 1
                else:
                    result.failed += 1
            return result

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._run_one, i, item): i
                for i, item in enumerate(items_list)
            }
            outcome: dict[int, BatchItemResult] = {}
            for future in as_completed(futures):
                idx = futures[future]
                outcome[idx] = future.result()
        order = [outcome[i] for i in range(len(items_list))]
        for item_result in order:
            result.items.append(item_result)
            if item_result.success:
                result.succeeded += 1
            else:
                result.failed += 1
        return result


# Bounded default used by offline tooling (not the interactive chat path).
DEFAULT_BATCH_MAX_WORKERS = 4
DEFAULT_BATCH_RETRIES = 2
