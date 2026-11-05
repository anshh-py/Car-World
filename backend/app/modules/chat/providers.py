"""Provider abstraction for Groq primary and Qwen3 fallback inference."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from groq import APIStatusError, Groq, RateLimitError

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
        else:
            kwargs["tool_choice"] = "none"

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
    """Fine-tuned Qwen3-4B provider via Hugging Face Inference API or local endpoint.

    This provider implements the fallback path. Tool calling support depends on
    the deployed model and endpoint capabilities. If the endpoint does not support
    tool calling, tools will be ignored and only text completion will be used.
    """

    def __init__(
        self,
        endpoint_url: str,
        hf_token: str | None = None,
        model_id: str = "qwen3-4b-finetuned",
        timeout: float = 60.0,
        supports_tool_calling: bool = False,
    ):
        self._endpoint_url = endpoint_url.rstrip("/")
        self._hf_token = hf_token
        self._model_id = model_id
        self._timeout = timeout
        self._supports_tool_calling = supports_tool_calling
        self._session = None

    @property
    def name(self) -> str:
        return "qwen3_fallback"

    def supports_tools(self) -> bool:
        return self._supports_tool_calling

    def _get_session(self):
        """Lazy-initialize HTTP session."""
        if self._session is None:
            import httpx
            headers = {"Content-Type": "application/json"}
            if self._hf_token:
                headers["Authorization"] = f"Bearer {self._hf_token}"
            self._session = httpx.Client(timeout=self._timeout, headers=headers)
        return self._session

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
    ) -> ProviderResponse:
        session = self._get_session()

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

        logger.info("Calling Qwen3 fallback provider at %s", self._endpoint_url)

        try:
            response = session.post(
                f"{self._endpoint_url}/v1/chat/completions",
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
                raise RateLimitError("Qwen3 rate limit exceeded", response=e.response, body=e.response.json() if e.response.content else {})
            raise
        except Exception as e:
            logger.error("Qwen3 provider error: %s", type(e).__name__)
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
    """Create Qwen3 provider from settings. Returns None if not configured."""
    if not settings.qwen3_endpoint_url:
        logger.info("QWEN3_ENDPOINT_URL not set; Qwen3 fallback disabled")
        return None

    return Qwen3Provider(
        endpoint_url=settings.qwen3_endpoint_url,
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