"""Token usage accounting and optional cost estimation for CarWorld AI.

Tracks provider-reported token usage per provider/model and computes *estimated*
cost only when pricing is configured. Estimates are clearly distinct from actual
billing: they are computed from published per-1M-token list prices supplied via
environment variables, never from provider invoices.

Price environment variables (optional; cost estimation is disabled when unset):
  GROQ_INPUT_PRICE_PER_1M / GROQ_OUTPUT_PRICE_PER_1M
  QWEN3_INPUT_PRICE_PER_1M / QWEN3_OUTPUT_PRICE_PER_1M

Providers that do not report usage (e.g. the Qwen3 Gradio Space) are recorded
with token counts of 0 and are flagged as ``usage_unavailable``.
"""

from __future__ import annotations

import threading
from typing import Any, Optional


_DEFAULT_PRICES: dict[str, dict[str, Optional[float]]] = {
    "groq": {
        "input_per_1m": None,
        "output_per_1m": None,
    },
    "qwen3_fallback": {
        "input_per_1m": None,
        "output_per_1m": None,
    },
}


class UsageTracker:
    """Thread-safe accumulated usage and cost estimates."""

    def __init__(self, prices: Optional[dict[str, dict[str, Optional[float]]]] = None) -> None:
        self._lock = threading.Lock()
        self._prices = prices if prices is not None else _DEFAULT_PRICES
        self._usage: dict[str, dict[str, int]] = {}
        self._requests: dict[str, int] = {}

    def set_prices(self, provider: str, input_per_1m: Optional[float], output_per_1m: Optional[float]) -> None:
        with self._lock:
            self._prices.setdefault(provider, {})
            self._prices[provider]["input_per_1m"] = input_per_1m
            self._prices[provider]["output_per_1m"] = output_per_1m

    def record(
        self,
        provider: str,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        with self._lock:
            entry = self._usage.setdefault(provider, {"input_tokens": 0, "output_tokens": 0})
            self._requests[provider] = self._requests.get(provider, 0) + 1
            entry["input_tokens"] += int(input_tokens or 0)
            entry["output_tokens"] += int(output_tokens or 0)
            entry["usage_unavailable"] = entry.get("usage_unavailable", False) or (
                input_tokens is None and output_tokens is None
            )

    def cost_estimate(self) -> dict[str, Any]:
        """Per-provider estimated cost (USD) and totals; only when priced."""
        with self._lock:
            providers: dict[str, Any] = {}
            total_cost = 0.0
            priced_any = False
            for provider, entry in sorted(self._usage.items()):
                prices = self._prices.get(provider, {})
                input_price = prices.get("input_per_1m")
                output_price = prices.get("output_per_1m")
                has_price = isinstance(input_price, (int, float)) or isinstance(output_price, (int, float))
                cost = None
                if has_price:
                    in_cost = (entry["input_tokens"] / 1_000_000.0) * (input_price or 0.0)
                    out_cost = (entry["output_tokens"] / 1_000_000.0) * (output_price or 0.0)
                    cost = round(in_cost + out_cost, 6)
                    total_cost += cost
                    priced_any = True
                providers[provider] = {
                    "requests": self._requests.get(provider, 0),
                    "input_tokens": entry["input_tokens"],
                    "output_tokens": entry["output_tokens"],
                    "total_tokens": entry["input_tokens"] + entry["output_tokens"],
                    "usage_unavailable": entry.get("usage_unavailable", False),
                    "estimated_cost_usd": cost,
                    "priced": has_price,
                }
        return {
            "providers": providers,
            "estimated_total_cost_usd": round(total_cost, 6) if priced_any else None,
            "cost_estimation_enabled": priced_any,
            "disclaimer": "Estimated from configured list prices; not actual billing.",
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total_in = sum(e["input_tokens"] for e in self._usage.values())
            total_out = sum(e["output_tokens"] for e in self._usage.values())
            total_requests = sum(self._requests.values())
        return {
            "total_requests": total_requests,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "cost": self.cost_estimate(),
        }

    def reset(self) -> None:
        with self._lock:
            self._usage.clear()
            self._requests.clear()


# Process-wide singleton consumed by providers, routes, and tests.
usage = UsageTracker()
