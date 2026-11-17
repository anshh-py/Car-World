"""Deterministic token estimation for CarWorld AI.

Provider usage is the source of truth where it is reported (see
``app.modules.chat.usage``). This module is only for *offline* measurement of
prompt/schema length where no provider is available (e.g. prompt optimization
audits and benchmarking). It uses a cheap, deterministic, word/subword
heuristic and every number it returns is **labelled as an estimate** — it must
never be conflated with provider-reported tokens.

Heuristic: each whitespace-delimited word contributes 1 token, plus one extra
token for every full 4 characters in the word (subword chunks). This
approximates a ~4-char/token English tokenizer closely enough for before/after
comparisons.
"""

from __future__ import annotations

import re
from typing import Iterable

_SPLIT = re.compile(r"\s+")


def estimate_text_tokens(text: str) -> int:
    """Estimate the number of tokens in a single text string."""
    if not text:
        return 0
    total = 0
    for word in _SPLIT.split(text.strip()):
        if not word:
            continue
        total += 1 + (len(word) // 4)
    return total


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for one chat message dict (role + content + tool bits)."""
    total = 0
    role = (message or {}).get("role")
    if role:
        total += 2  # role marker + delimiter overhead (estimated)
    content = (message or {}).get("content")
    if isinstance(content, str):
        total += estimate_text_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    total += estimate_text_tokens(part["text"])
            elif isinstance(part, str):
                total += estimate_text_tokens(part)
    tool_calls = (message or {}).get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict):
                fn = call.get("function") or {}
                if isinstance(fn.get("name"), str):
                    total += estimate_text_tokens(fn["name"])
                if isinstance(fn.get("arguments"), str):
                    total += estimate_text_tokens(fn["arguments"])
    return total


def estimate_messages_tokens(messages: Iterable[dict]) -> int:
    """Estimate total tokens for a list of chat messages."""
    return sum(estimate_message_tokens(m) for m in messages)


def token_report(text: str, *, label: str = "text") -> dict:
    """Compact, developer-readable token measurement for a single string."""
    return {
        "label": label,
        "chars": len(text),
        "words": len(_SPLIT.split(text.strip())) if text.strip() else 0,
        "estimated_tokens": estimate_text_tokens(text),
        "estimation": "heuristic (approx 4 chars/token); not provider-reported",
    }
