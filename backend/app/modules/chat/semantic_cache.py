"""A small, privacy-safe semantic cache for automotive questions.

Returns cached answers only when a new question is sufficiently similar to a
previously answered one AND the request is safe to cache. Safety is enforced by
``should_cache`` (no conversation history, no PII, no instruction-like content,
guest-only), so cached responses are never shared across users or contexts and
never contain user-specific or time-sensitive information.

Similarity uses a deterministic lexical fingerprint (token-set Jaccard) over
normalized text. This requires no embeddings API and no external service, which
keeps the feature free and dependency-free. Because it is lexical, the threshold
is conservative: we prefer a miss over a wrong hit.

Cache metrics (hit/miss/eviction/size) are recorded and exposed via ``stats``.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Optional

from carworld_security import (
    contains_instruction_like_text,
    redact_sensitive_text,
)

_TOKEN_SPLIT = re.compile(r"[^\w]+")
_WHITESPACE = re.compile(r"\s+")

DEFAULT_MAX_ENTRIES = 512
DEFAULT_TTL_SECONDS = 300.0
DEFAULT_SIMILARITY_THRESHOLD = 0.9


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and drop punctuation for fingerprinting."""
    lowered = str(text).strip().lower()
    lowered = _WHITESPACE.sub(" ", lowered)
    return _TOKEN_SPLIT.sub(" ", lowered)


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_normalize(text).split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets (0.0 .. 1.0)."""
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class SemanticCache:
    """TTL-bounded, size-bounded, thread-safe response cache.

    ``_entries`` maps a fingerprint (canonical token set) to a list of
    ``(token_set, answer, expiry)`` tuples. Lookup scans for the most similar
    stored entry that also matches the threshold and has not expired.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        enabled: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._max_entries = int(max_entries)
        self._ttl_seconds = float(ttl_seconds)
        self._threshold = float(similarity_threshold)
        self._enabled = bool(enabled)
        self._entries: dict[str, list[tuple[frozenset[str], str, float]]] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._stores = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(
        self,
        *,
        enabled: bool,
        max_entries: int,
        ttl_seconds: float,
        similarity_threshold: float,
    ) -> None:
        """Update configuration at runtime (e.g. from settings on startup)."""
        with self._lock:
            self._enabled = bool(enabled)
            self._max_entries = int(max_entries)
            self._ttl_seconds = float(ttl_seconds)
            self._threshold = float(similarity_threshold)

    def _purge_locked(self, now: float) -> None:
        expired_keys = []
        for key, bucket in self._entries.items():
            alive = [(tokens, answer, expiry) for tokens, answer, expiry in bucket if expiry > now]
            if alive:
                self._entries[key] = alive
            else:
                expired_keys.append(key)
        for key in expired_keys:
            del self._entries[key]

    def _prune_locked(self) -> None:
        total = sum(len(bucket) for bucket in self._entries.values())
        while total > self._max_entries:
            # Drop one entry from the largest bucket (FIFO within a bucket).
            biggest_key = max(self._entries, key=lambda k: len(self._entries[k]))
            bucket = self._entries[biggest_key]
            bucket.pop(0)
            if not bucket:
                del self._entries[biggest_key]
            self._evictions += 1
            total -= 1

    def lookup(self, question: str) -> Optional[str]:
        """Return a cached answer if a sufficiently similar entry exists."""
        if not self._enabled:
            return None
        tokens = _token_set(question)
        if not tokens:
            return None
        now = time.time()
        best: Optional[tuple[float, str]] = None
        with self._lock:
            self._purge_locked(now)
            # Only consider the bucket whose fingerprint matches the query exactly
            # (canonical token set), then keep the most similar expired-free match.
            key = frozenset(tokens)
            bucket = self._entries.get(key, [])
            for stored_tokens, answer, expiry in bucket:
                if expiry <= now:
                    continue
                sim = jaccard(tokens, stored_tokens)
                if sim >= self._threshold and (best is None or sim > best[0]):
                    best = (sim, answer)
        if best is not None:
            self._hits += 1
            return best[1]
        self._misses += 1
        return None

    def store(self, question: str, answer: str) -> None:
        """Record an answer for a normalized question (only if enabled)."""
        if not self._enabled:
            return
        answer = str(answer).strip()
        tokens = _token_set(question)
        if not tokens or not answer:
            return
        now = time.time()
        with self._lock:
            self._purge_locked(now)
            key = frozenset(tokens)
            bucket = self._entries.setdefault(key, [])
            bucket.append((tokens, answer, now + self._ttl_seconds))
            self._prune_locked()
            self._stores += 1

    def invalidate(self, question: str) -> int:
        """Remove all entries whose token set matches the question. Returns removed count."""
        tokens = _token_set(question)
        with self._lock:
            key = frozenset(tokens)
            bucket = self._entries.pop(key, [])
            return len(bucket)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._purge_locked(time.time())
            return {
                "enabled": self._enabled,
                "hits": self._hits,
                "misses": self._misses,
                "stores": self._stores,
                "evictions": self._evictions,
                "size": sum(len(bucket) for bucket in self._entries.values()),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl_seconds,
                "similarity_threshold": self._threshold,
            }


def should_cache(message: str, history: Optional[list[dict]] = None) -> bool:
    """Whether a request is safe to cache.

    A request is cacheable only when ALL of:
      * there is no conversation history (the answer does not depend on prior turns);
      * the message contains no sensitive personal data (redaction is a no-op);
      * the message is not instruction-like (no injection attempt);
      * the message is non-empty and of reasonable size.
    Guest-only / per-user differentiation is enforced by the caller (the cache is
    only consulted for unauthenticated requests).
    """
    if history:
        return False
    if not isinstance(message, str) or not message.strip():
        return False
    if len(message) > 4000:
        return False
    if redact_sensitive_text(message) != message:
        return False
    if contains_instruction_like_text(message):
        return False
    return True


# Process-wide shared cache instance; disabled by default until configured.
semantic_cache = SemanticCache(enabled=False)
