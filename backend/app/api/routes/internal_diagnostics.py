from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Header, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.modules.chat.providers import create_qwen3_provider

logger = logging.getLogger(__name__)

DIAGNOSTIC_HEADER = "X-Qwen3-Diagnostic-Key"
DIAGNOSTIC_PROMPT = (
    "Explain the difference between petrol and diesel engines in three concise points. "
    "Give the practical differences."
)
RESPONSE_PREVIEW_LIMIT = 500

router = APIRouter(tags=["internal"])


@router.post("/internal/diagnostics/qwen3", include_in_schema=False)
async def qwen3_diagnostic(
    diagnostic_key: str | None = Header(default=None, alias=DIAGNOSTIC_HEADER),
) -> dict[str, str]:
    """Diagnose connectivity to the Qwen3 fallback inference Space.

    This endpoint is temporary and intentionally isolated from the public chat
    route. It never calls Groq or the ProviderRouter and never exposes the
    configured secret or Hugging Face credentials.
    """
    secret = settings.qwen3_diagnostic_secret
    if not secret:
        raise HTTPException(status_code=404, detail="Diagnostic endpoint is not configured.")

    if not diagnostic_key or not hmac.compare_digest(secret.encode("utf-8"), diagnostic_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Unauthorized.")

    provider = create_qwen3_provider()
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Qwen3 fallback is not configured (QWEN3_SPACE_ID / QWEN3_ENDPOINT_URL).",
        )

    messages = [{"role": "user", "content": DIAGNOSTIC_PROMPT}]
    try:
        response = await run_in_threadpool(
            lambda: provider.chat_completion(messages=messages, tools=None, temperature=0.2)
        )
    except Exception as exc:  # noqa: BLE001 - surface a safe generic failure
        logger.error("Qwen3 diagnostic inference failed (%s).", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Qwen3 diagnostic inference failed.") from exc

    if response.provider_name != "qwen3_fallback":
        logger.error("Qwen3 diagnostic returned unexpected provider=%s", response.provider_name)
        raise HTTPException(status_code=502, detail="Qwen3 diagnostic returned an unexpected provider.")

    if not isinstance(response.content, str) or not response.content.strip():
        logger.error("Qwen3 diagnostic returned an empty response.")
        raise HTTPException(status_code=502, detail="Qwen3 diagnostic returned an empty response.")

    preview = response.content.strip()[:RESPONSE_PREVIEW_LIMIT]
    return {"status": "ok", "provider": response.provider_name, "response_preview": preview}
