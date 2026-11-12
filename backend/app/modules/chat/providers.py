"""Provider abstraction for Groq primary and Qwen3 fallback inference."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
from groq import APIStatusError, Groq, RateLimitError

try:  # Gradio client is optional at import time; a missing install is reported per-request.
    import gradio_client
except ImportError:  # pragma: no cover - environment dependent
    gradio_client = None  # type: ignore[assignment]

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    """Standardized response from any provider."""
    content: str
    tool_calls: list[dict] | None = None
    provider_name: str = ""


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier for logging."""

    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider supports tool calling."""

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
    ) -> ProviderResponse:
        """Execute a chat completion request."""


class GroqProvider(Provider):
    """Groq API provider for openai/gpt-oss-120b."""

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self._client = Groq(api_key=api_key, timeout=60.0, max_retries=2)
        self._model = model

    @property
    def name(self) -> str:
        return "groq"

    def supports_tools(self) -> bool:
        return True

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        # When no tools are supplied for the request, omit tool_choice entirely.
        # Groq's openai/gpt-oss-120b can ignore tool_choice="none" and still emit a
        # tool call, which the API rejects with HTTP 400 "Tool choice is none, but
        # model called a tool". Omitting the argument lets a no-tool request be
        # answered as plain text without a contradictory tool_choice.

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]

        return ProviderResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            provider_name=self.name,
        )


class Qwen3Provider(Provider):
    """Fine-tuned Qwen3-4B provider via Hugging Face Gradio Space or local endpoint.

    This provider implements the fallback path. When QWEN3_SPACE_ID is configured,
    requests are sent through the gradio_client to the private Hugging Face Space,
    whose chat_completion function returns both base_model_response and
    finetuned_model_response; only the finetuned response is exposed.

    When QWEN3_SPACE_ID is unset and QWEN3_ENDPOINT_URL is set, the provider
    falls back to an OpenAI-compatible HTTP endpoint (/v1/chat/completions).

    Tool calling support depends on the deployed model and endpoint capabilities.
    If the endpoint does not support tool calling, tools will be ignored and only
    text completion will be used.
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        space_id: str | None = None,
        hf_token: str | None = None,
        model_id: str = "qwen3-4b-finetuned",
        timeout: float = 60.0,
        supports_tool_calling: bool = False,
    ):
        self._http_endpoint = endpoint_url.rstrip("/") if endpoint_url else None
        self._space_id = space_id
        self._hf_token = hf_token
        self._model_id = model_id
        self._timeout = timeout
        self._supports_tool_calling = supports_tool_calling
        self._gradio_client = None
        self._client_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "qwen3_fallback"

    def supports_tools(self) -> bool:
        return self._supports_tool_calling

    def _get_gradio_client(self):
        """Lazy, thread-safe initialization of the Gradio Client.

        The client is created once and reused for subsequent requests. It is not
        created at import time or during app startup — only when the provider
        receives its first request, and only if QWEN3_SPACE_ID is configured.

        QWEN3_SPACE_ID takes precedence: when a Space ID is configured the client
        is always built, even if QWEN3_ENDPOINT_URL is also set. The HTTP endpoint
        is only used when no Space ID is configured.
        """
        if self._space_id is None:
            return None

        if gradio_client is None:
            raise RuntimeError(
                "The Gradio client is not installed. "
                "Install it with: pip install gradio_client"
            )

        with self._client_lock:
            if self._gradio_client is None:
                try:
                    self._gradio_client = gradio_client.Client(
                        src=self._space_id,
                        token=self._hf_token,
                        httpx_kwargs={"timeout": self._timeout},
                        verbose=False,
                    )
                except ImportError as e:
                    raise RuntimeError(
                        "The Gradio client is not installed. "
                        "Install it with: pip install gradio_client"
                    ) from e
        return self._gradio_client

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
    ) -> ProviderResponse:
        """Execute a chat completion request via Gradio Space or HTTP endpoint."""
        gradio_client = self._get_gradio_client()

        if gradio_client is not None:
            return self._call_gradio(messages, tools, temperature, tool_choice)

        if self._http_endpoint is None:
            raise RuntimeError(
                "Qwen3 provider configured without a valid endpoint or Gradio Space ID"
            )

        return self._call_http(messages, tools, temperature, tool_choice)

    def _call_gradio(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        tool_choice: str | dict = "auto",
    ) -> ProviderResponse:
        """Send the request through the Gradio Space's chat_completion function.

        The Space's Gradio input is a gr.Code component that expects JSON text.
        We serialize the messages (and optional tool definitions) into a single
        JSON payload string and call client.predict(payload_text, api_name="/chat_completion").
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }

        if tools and self._supports_tool_calling:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        payload_text = json.dumps(payload)

        # Resolve the client outside the try block: a missing dependency raises a
        # RuntimeError that names the package and must not be folded into the
        # generic "temporarily unavailable" message.
        client = self._get_gradio_client()

        try:
            result = client.predict(payload_text, api_name="/chat_completion")
        except ImportError as e:
            raise RuntimeError(
                "The Gradio client is not installed. Install it with: pip install gradio_client"
            ) from e
        except Exception as e:
            # Do not propagate raw Gradio exception messages (they may contain HF tokens).
            # Classify failures without mislabeling them as Groq 429s.
            self._log_gradio_error(type(e).__name__, str(e))
            raise RuntimeError(
                "CarWorld AI fallback model is temporarily unavailable. Please try again later."
            ) from e

        return self._parse_gradio_response(result)

    def _parse_gradio_response(self, result: Any) -> ProviderResponse:
        """Extract finetuned_model_response from the Gradio prediction result.

        The Space returns a prediction result object containing
        base_model_response and finetuned_model_response. Only the fine-tuned
        response is exposed; it must be a non-empty string.

        If finetuned_model_response is missing or invalid, we do NOT silently
        substitute the base-model output: we raise a RuntimeError so the caller
        can surface a safe fallback.
        """
        try:
            if isinstance(result, dict):
                fine = result.get("finetuned_model_response")
                base = result.get("base_model_response")
            else:
                fine = getattr(result, "finetuned_model_response", None)
                base = getattr(result, "base_model_response", None)

            if isinstance(fine, str) and fine.strip():
                return ProviderResponse(
                    content=fine.strip(),
                    tool_calls=None,  # QWEN3_SUPPORTS_TOOLS is currently false
                    provider_name=self.name,
                )

            if base is not None and isinstance(base, str) and base.strip():
                # Diagnostic: base-model output is present but we must never expose it.
                logger.warning(
                    "Qwen3 Gradio Space returned a base_model_response but no "
                    "finetuned_model_response; not exposing diagnostic output."
                )
            else:
                logger.warning(
                    "Qwen3 Gradio Space returned an unexpected prediction result: %s",
                    type(result).__name__,
                )

        except Exception as e:
            self._log_gradio_error("parse", str(e))

        raise RuntimeError(
            "CarWorld AI fallback model returned an unexpected response. Please try again later."
        )

    def _log_gradio_error(self, stage: str, message: str) -> None:
        """Log a Gradio failure without leaking HF tokens or request payloads."""
        safe_message = self._redact(message)
        logger.error("Qwen3 Gradio provider error [%s]: %s", stage, safe_message)

    @staticmethod
    def _redact(message: str) -> str:
        """Remove HF token-like values from error strings."""
        redacted = message
        if isinstance(message, str):
            # Bearer headers first so the credential that follows is masked as a
            # bearer token rather than by the generic API-key pattern.
            redacted = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer [REDACTED]", redacted)
            # Hugging Face access tokens use the "hf_" prefix (underscore).
            redacted = re.sub(
                r"\bhf_[A-Za-z0-9]{10,}\b", "[HUGGINGFACE_TOKEN_REDACTED]", redacted
            )
            redacted = re.sub(
                r"\bhf-[A-Za-z0-9]{20,}\b", "[HUGGINGFACE_TOKEN_REDACTED]", redacted
            )
            # GitHub / Groq / Tavily style keys.
            redacted = re.sub(
                r"\b(?:ghp_|gsk_|tvly_|sk-)[A-Za-z0-9]{8,}\b",
                "[API_KEY_REDACTED]",
                redacted,
            )
        return redacted[:1024]

    def _call_http(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        tool_choice: str | dict = "auto",
    ) -> ProviderResponse:
        """Send an OpenAI-compatible request to a local /v1/chat/completions endpoint."""
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
            "stream": False,
        }

        if tools and self._supports_tool_calling:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        try:
            with httpx.Client(timeout=self._timeout) as session:
                response = session.post(
                    f"{self._http_endpoint}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            tool_calls = None
            if self._supports_tool_calling and message.get("tool_calls"):
                tool_calls = message["tool_calls"]

            return ProviderResponse(
                content=message.get("content", "") or "",
                tool_calls=tool_calls,
                provider_name=self.name,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(
                    "Qwen3 rate limit exceeded",
                    response=e.response,
                    body=e.response.json() if e.response.content else {},
                ) from e
            raise
        except Exception as e:
            logger.error("Qwen3 HTTP provider error: %s", type(e).__name__)
            raise


# Tool names that require execution (cannot be answered without tool use)
TOOL_DEPENDENT_NAMES = {
    "search_car_database",
    "search_car_web",
    "calculate_emi",
}


def requires_tool_execution(tools: list[dict] | None) -> bool:
    """Check if the request requires tool execution."""
    if not tools:
        return False
    for tool in tools:
        name = tool.get("function", {}).get("name")
        if name in TOOL_DEPENDENT_NAMES:
            return True
    return False


class ProviderRouter:
    """Routes requests to Groq primary with Qwen3 fallback on rate limit.

    Implements automatic recovery:
    - On HTTP 429/RateLimitError, respects Retry-After header or uses configured cooldown
    - After cooldown expires, attempts Groq again
    - If Groq returns 429 again, reactivates fallback
    - Thread-safe for concurrent requests
    """

    def __init__(
        self,
        groq_provider: GroqProvider,
        qwen3_provider: Qwen3Provider | None = None,
        cooldown_seconds: int = 60,
    ):
        self._groq = groq_provider
        self._qwen3 = qwen3_provider
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._groq_exhausted = False
        self._groq_retry_after: float | None = None  # Unix timestamp when Groq can be retried

    def _extract_retry_after(self, error: RateLimitError | APIStatusError) -> int | None:
        """Extract Retry-After seconds from error response."""
        try:
            if hasattr(error, "response") and error.response is not None:
                retry_after = error.response.headers.get("Retry-After")
                if retry_after:
                    return int(retry_after)
        except (ValueError, TypeError):
            pass
        return None

    def _is_groq_available(self) -> bool:
        """Check if Groq is available for retry (thread-safe)."""
        with self._lock:
            if not self._groq_exhausted:
                return True
            if self._groq_retry_after is not None and time.time() >= self._groq_retry_after:
                # Cooldown expired, allow retry
                self._groq_exhausted = False
                self._groq_retry_after = None
                logger.info("Groq cooldown expired, attempting Groq again")
                return True
            return False

    def _mark_groq_exhausted(self, retry_after_seconds: int | None) -> None:
        """Mark Groq as exhausted and set retry time (thread-safe)."""
        with self._lock:
            self._groq_exhausted = True
            if retry_after_seconds is not None:
                self._groq_retry_after = time.time() + retry_after_seconds
            else:
                self._groq_retry_after = time.time() + self._cooldown_seconds
            logger.warning("Groq marked as exhausted, retry after %s seconds",
                          retry_after_seconds or self._cooldown_seconds)

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
    ) -> ProviderResponse:
        """Execute request with automatic fallback on rate limit and recovery."""

        # Check if Groq is available for retry
        if self._is_groq_available():
            try:
                response = self._groq.chat_completion(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                )
                logger.info("Groq request succeeded")
                return response

            except RateLimitError as e:
                retry_after = self._extract_retry_after(e)
                self._mark_groq_exhausted(retry_after)

                if self._qwen3 is None:
                    logger.error("Qwen3 fallback not configured")
                    raise RuntimeError("Groq rate limit exceeded and no fallback provider configured") from e

                return self._handle_qwen3_fallback(
                    messages, tools, tool_choice, temperature, "Groq rate limit hit"
                )

            except APIStatusError as e:
                if e.status_code == 429:
                    retry_after = self._extract_retry_after(e)
                    self._mark_groq_exhausted(retry_after)

                    if self._qwen3 is None:
                        logger.error("Qwen3 fallback not configured")
                        raise RuntimeError("Groq rate limit exceeded and no fallback provider configured") from e

                    return self._handle_qwen3_fallback(
                        messages, tools, tool_choice, temperature, "Groq HTTP 429"
                    )
                raise

        else:
            # Groq is exhausted, use Qwen3 directly
            logger.info("Groq exhausted, using Qwen3 directly")
            if self._qwen3 is None:
                raise RuntimeError("Groq exhausted and no fallback provider configured")
            return self._handle_qwen3_fallback(
                messages, tools, tool_choice, temperature, "Groq previously exhausted"
            )

    def _handle_qwen3_fallback(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | dict,
        temperature: float,
        reason: str,
    ) -> ProviderResponse:
        """Handle Qwen3 fallback with tool support checks."""
        if tools and not self._qwen3.supports_tools():
            if requires_tool_execution(tools):
                logger.warning(
                    "Qwen3 does not support tool calling; request requires tools (%s). "
                    "Returning safe unavailable response.",
                    reason,
                )
                return ProviderResponse(
                    content=(
                        "I'm temporarily unable to access vehicle databases or calculation tools. "
                        "Please try again later for specific car specifications, prices, or EMI calculations. "
                        "For general automotive questions, I'm still here to help."
                    ),
                    tool_calls=None,
                    provider_name=self._qwen3.name,
                )
            else:
                logger.warning("Qwen3 does not support tool calling; proceeding without tools")
                tools = None
                tool_choice = "none"

        return self._qwen3.chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )

    def reset_groq_state(self) -> None:
        """Reset the Groq exhaustion state (e.g., for new session or after retry-after)."""
        with self._lock:
            self._groq_exhausted = False
            self._groq_retry_after = None


def create_groq_provider() -> GroqProvider:
    """Create Groq provider from settings."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing from environment")
    return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)


def create_qwen3_provider() -> Qwen3Provider | None:
    """Create Qwen3 provider from settings. Returns None if not configured.

    QWEN3_SPACE_ID takes precedence: when set, the provider routes through the
    Hugging Face Gradio Space. Otherwise QWEN3_ENDPOINT_URL is used for a
    local/OpenAI-compatible endpoint. If neither is set, the fallback is disabled.
    """
    space_id = settings.qwen3_space_id or None
    endpoint_url = settings.qwen3_endpoint_url or None

    if not space_id and not endpoint_url:
        logger.info("QWEN3_ENDPOINT_URL not set; Qwen3 fallback disabled")
        return None

    return Qwen3Provider(
        endpoint_url=endpoint_url,
        space_id=space_id,
        hf_token=settings.huggingface_token,
        model_id=settings.qwen3_model_id,
        timeout=settings.qwen3_timeout,
        supports_tool_calling=settings.qwen3_supports_tools,
    )


def create_provider_router() -> ProviderRouter:
    """Create the provider router with configured providers."""
    groq = create_groq_provider()
    qwen3 = create_qwen3_provider()
    return ProviderRouter(
        groq_provider=groq,
        qwen3_provider=qwen3,
        cooldown_seconds=settings.groq_cooldown_seconds,
    )
