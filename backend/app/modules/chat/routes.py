from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request, status
from groq import BadRequestError, RateLimitError
from starlette.concurrency import run_in_threadpool

from app.core.config import PROJECT_ROOT, settings
from app.modules.auth.dependencies import get_optional_current_user
from app.modules.auth.schemas import AuthenticatedUser
from app.modules.chat.agent import get_agent
from app.modules.chat.schemas import ChatRequest, ChatResponse

project_root_string = str(PROJECT_ROOT)
if project_root_string not in sys.path:
    sys.path.insert(0, project_root_string)
from carworld_security import (
    redact_sensitive_text,
    validate_model_output,
    contains_abusive_content,
)
from app.modules.chat.metrics import metrics
from app.modules.chat.semantic_cache import semantic_cache, should_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
_request_times: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = asyncio.Lock()
CHAT_RATE_LIMITS = (
    (settings.chat_rate_limit_per_minute, 60),
    (settings.chat_rate_limit_per_day, 86400),
)
MAX_RATE_WINDOW_SECONDS = max(window for _limit, window in CHAT_RATE_LIMITS)


async def enforce_rate_limit(
    request: Request,
    user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> None:
    """Limit guests by IP and signed-in callers by verified account ID."""
    host = request.client.host if request.client else "unknown"
    client_key = f"user:{user.id}" if user else f"ip:{host}"
    now = time.monotonic()
    async with _rate_limit_lock:
        if len(_request_times) > 4096:
            expired_hosts = [key for key, entries in _request_times.items()
                             if not entries or now - entries[-1] >= MAX_RATE_WINDOW_SECONDS]
            for key in expired_hosts:
                _request_times.pop(key, None)
        active_windows = []
        for limit, window_seconds in CHAT_RATE_LIMITS:
            requests = _request_times[f"{client_key}:{window_seconds}"]
            while requests and now - requests[0] >= window_seconds:
                requests.popleft()
            active_windows.append((requests, limit, window_seconds))
            if len(requests) >= limit:
                retry_after = max(1, math.ceil(window_seconds - (now - requests[0])))
                metrics.incr("sec.rate_limit_events")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Chat limit reached. Please retry in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )
        for requests, _limit, _window_seconds in active_windows:
            requests.append(now)


def _safe_provider_error_fields(error: BadRequestError) -> tuple[str, str, str, str, str]:
    """Extract provider diagnostics without logging request headers or tool arguments."""
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error") if isinstance(body.get("error"), dict) else {}
    failed = details.get("failed_generation") if isinstance(details, dict) else None
    if not isinstance(failed, dict):
        failed = {}

    def safe_text(value: object, limit: int = 240) -> str:
        if not isinstance(value, str):
            return ""
        return redact_sensitive_text(value)[:limit]

    return (
        safe_text(details.get("type")),
        safe_text(details.get("code")),
        safe_text(details.get("param")),
        safe_text(details.get("message")),
        safe_text(failed.get("reason")),
    )


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    _rate_limit: None = Depends(enforce_rate_limit),
    _user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ChatResponse:
    message = body.message.strip()
    metrics.incr("perf.requests")
    _request_start = time.perf_counter()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    # Lightweight deterministic moderation gate (one layer, not the only defense).
    if contains_abusive_content(message):
        metrics.incr("sec.moderation_events")
        raise HTTPException(
            status_code=422,
            detail="That request is not supported by CarWorld AI.",
        )

    history_records = [entry.model_dump() for entry in body.history]
    conversation = list(history_records)
    conversation.append({"role": "user", "content": message})

    # Opt-in semantic cache (default off). Only consulted for guests without
    # history so cached answers are never shared across users or contexts.
    if semantic_cache.enabled and _user is None and should_cache(message, history_records):
        _lookup_start = time.perf_counter()
        cached = semantic_cache.lookup(message)
        metrics.record_latency("latency.cache_lookup", time.perf_counter() - _lookup_start)
        if cached is not None:
            metrics.incr("perf.cache_hits")
            validated_cache = validate_model_output(cached)
            metrics.record_latency("latency.total", time.perf_counter() - _request_start)
            return ChatResponse(reply=validated_cache)
        metrics.incr("perf.cache_misses")

    try:
        _agent_start = time.perf_counter()
        agent = get_agent()
        metrics.record_latency("latency.agent_init", time.perf_counter() - _agent_start)
        replies = await run_in_threadpool(lambda: list(agent.run(conversation)))
        reply = validate_model_output(replies[-1] if replies else None)
        metrics.record_latency("latency.total", time.perf_counter() - _request_start)
        if reply == "Sorry, I couldn't provide a safe response to that request. Please try again.":
            metrics.incr("sec.output_validation_failures")
        # Store in the cache only when enabled, safe, guest-only, and valid.
        if semantic_cache.enabled and _user is None and should_cache(message, history_records):
            semantic_cache.store(message, reply)
        return ChatResponse(reply=reply)
    except BadRequestError as error:
        provider_type, provider_code, provider_param, message, generation_reason = (
            _safe_provider_error_fields(error)
        )
        logger.warning(
            "Groq rejected chat request status=%s type=%s code=%s param=%s message=%s generation_reason=%s",
            error.status_code,
            provider_type,
            provider_code,
            provider_param,
            message,
            generation_reason,
        )
        raise HTTPException(
            status_code=502,
            detail="CarWorld AI could not process that request. Please rephrase it and try again.",
        ) from error
    except RateLimitError as error:
        logger.warning("Provider rate limit exceeded: %s", error)
        raise HTTPException(
            status_code=503,
            detail="CarWorld AI is temporarily unavailable due to rate limits. Please try again later.",
        ) from error
    except RuntimeError as error:
        if "GROQ_API_KEY" in str(error):
            logger.error("CarWorld AI is not configured: missing model credentials")
            raise HTTPException(status_code=503, detail="CarWorld AI is not configured.") from error
        if "no fallback provider configured" in str(error):
            logger.error("Groq rate limit exceeded and Qwen3 fallback not configured")
            raise HTTPException(
                status_code=503,
                detail="CarWorld AI is temporarily unavailable. Fallback provider not configured.",
            ) from error
        logger.error("CarWorld AI request failed (%s).", type(error).__name__)
        raise HTTPException(status_code=502, detail="CarWorld AI could not complete the request.") from error
    except Exception as error:
        logger.error("CarWorld AI request failed (%s).", type(error).__name__)
        raise HTTPException(status_code=502, detail="CarWorld AI could not complete the request.") from error
