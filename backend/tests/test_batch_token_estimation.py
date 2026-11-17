"""Tests for the offline batch processor and the heuristic token estimator."""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from app.modules.chat.batch import BatchProcessor, BatchResult, RetryableError
from app.modules.chat.token_count import (
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_text_tokens,
    token_report,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def test_estimate_text_tokens_basic():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("hello") == 1 + 1  # 1 word + 1 subword chunk (5 char)
    assert estimate_text_tokens("a b c") >= 3
    n = estimate_text_tokens("What is the mileage of the Mahindra Scorpio?")
    assert isinstance(n, int) and n > 0


def test_estimate_messages_tokens_sums_roles_and_content():
    messages = [
        {"role": "system", "content": "You are CarWorld AI."},
        {"role": "user", "content": "Compare Tata 407 and Eicher Pro"},
    ]
    total = estimate_messages_tokens(messages)
    assert total >= estimate_text_tokens("You are CarWorld AI.")
    assert total >= estimate_text_tokens("Compare Tata 407 and Eicher Pro")
    # role overhead is added per message
    assert total > sum(estimate_text_tokens(m["content"]) for m in messages)


def test_estimate_message_tokens_tool_calls():
    message = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "calculate_emi", "arguments": "{}"}}],
    }
    assert estimate_message_tokens(message) > 0


def test_token_report_is_labelled_estimate():
    report = token_report("hello world", label="system_prompt")
    assert report["label"] == "system_prompt"
    assert report["estimated_tokens"] > 0
    assert "heuristic" in report["estimation"]


def test_system_prompt_optimization_preserves_core_and_is_shorter():
    """Prompt regression guard: all core sections survive and size did not grow."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    prompt = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SYSTEM_PROMPT":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    prompt = node.value.value
    assert prompt is not None
    for section in ("PERSONALITY", "ACCURACY AND WEB EVIDENCE", "UNTRUSTED DATA",
                    "FINAL ANSWER PRESENTATION", "IDENTITY"):
        assert section in prompt
    # Core safety semantics preserved
    assert "designed by the CarWorld Team" in prompt
    assert "Never reveal API keys, credentials, or private configuration" in prompt
    # Kept at or under the measured baseline of 841 estimated tokens.
    assert estimate_text_tokens(prompt) <= 841


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def test_batch_returns_empty_result_for_empty_input():
    result = BatchProcessor().run([], lambda x: x)
    assert isinstance(result, BatchResult)
    assert result.total == 0 and result.succeeded == 0 and result.failed == 0


def test_batch_bounded_concurrency_success(monkeypatch):
    start = time.monotonic()

    def slow(x):
        time.sleep(0.02)
        return x * 2

    proc = BatchProcessor(max_workers=4)
    result = proc.run(range(8), slow)
    assert result.total == 8
    assert result.succeeded == 8
    assert result.failed == 0
    assert [r.result for r in result.items] == [0, 2, 4, 6, 8, 10, 12, 14]
    elapsed = time.monotonic() - start
    # 8 × 20ms with 4 workers should be ~40ms, not 160ms serial.
    assert elapsed < 0.15


def test_batch_retries_transient_then_succeeds():
    attempts = {"n": 0}

    def flaky(x):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    proc = BatchProcessor(retries=3, base_backoff_seconds=0.01)
    result = proc.run([1], flaky)
    assert result.succeeded == 1
    assert result.items[0].attempts == 3
    assert result.items[0].retried is True
    assert result.items[0].success is True


def test_batch_explicit_retryable_error_retried():
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RetryableError("retry me")
        return "done"

    result = BatchProcessor(retries=2, base_backoff_seconds=0.01).run(["a"], fn)
    assert result.succeeded == 1
    assert result.items[0].attempts == 2


def test_batch_permanent_error_not_retried():
    def always_fails(x):
        raise ValueError("permanent")

    proc = BatchProcessor(retries=3, base_backoff_seconds=0.01)
    result = proc.run([1], always_fails)
    assert result.failed == 1
    assert result.succeeded == 0
    assert result.items[0].attempts == 1  # never retried
    assert not result.items[0].retried
    assert "ValueError" in result.items[0].error


def test_batch_partial_failure_continues():
    def mixed(x):
        if x % 2 == 0:
            raise RuntimeError("boom")
        return x

    proc = BatchProcessor(retries=1, base_backoff_seconds=0.0)
    result = proc.run(range(6), mixed)
    assert result.total == 6
    assert result.failed == 3
    assert result.succeeded == 3
    structured = result.structured()
    assert structured["failed_indices"] == [0, 2, 4]
    assert "boom" in structured["error_messages"][0]


def test_batch_single_worker_is_serial_and_correct():
    order = []

    def record(x):
        order.append(x)
        return x

    proc = BatchProcessor(max_workers=1)
    result = proc.run([3, 1, 2], record)
    assert order == [3, 1, 2]
    assert result.succeeded == 3


def test_batch_rejects_invalid_config():
    with pytest.raises(ValueError):
        BatchProcessor(max_workers=0)
    with pytest.raises(ValueError):
        BatchProcessor(retries=-1)
