"""Tests for the Qwen3 Gradio Space fallback provider.

These tests verify the real Gradio integration path (QWEN3_SPACE_ID configured)
and the legacy HTTP endpoint path (QWEN3_ENDPOINT_URL configured), plus the
existing Groq routing / cooldown / recovery behavior remains unchanged.

No real HF credentials or network access to the Space is required:
all interactions with the Space go through a mocked gradio_client.Client.
"""

from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import RateLimitError

from app.core.config import Settings
from app.modules.chat import providers as providers_module
from app.modules.chat.providers import (
    GroqProvider,
    ProviderResponse,
    ProviderRouter,
    Qwen3Provider,
    create_provider_router,
    create_qwen3_provider,
)
from app.modules.chat import routes as chat_routes


class DynamicProviderSettings:
    """Proxy that reads provider settings live from the environment.

    create_qwen3_provider() reads the module-level ``settings`` singleton that
    is built once at import time, so monkeypatch.setenv() after import would
    otherwise go unnoticed. This proxy re-evaluates Settings.from_environment()
    on every attribute access so tests that set Qwen3 env vars are reflected.
    """

    def __getattr__(self, name: str):
        return getattr(Settings.from_environment(), name)


class FakeGradioClient:
    """Deterministic mock of gradio_client.Client for testing."""

    def __init__(self, responses=None, should_fail_with=None):
        self.responses = iter(responses) if responses else None
        self.should_fail_with = should_fail_with
        self.predict_calls = []
        self.warmup_calls = 0

    def predict(self, payload_text, api_name="/chat_completion"):
        self.predict_calls.append({"payload_text": payload_text, "api_name": api_name})
        if self.should_fail_with:
            raise self.should_fail_with
        if self.responses:
            return next(self.responses)
        return {"finetuned_model_response": "Default Gradio fallback response"}


@pytest.fixture(autouse=True)
def clear_env_qwen3(monkeypatch):
    """Remove Qwen3 env vars from the environment before each test."""
    for key in ("QWEN3_ENDPOINT_URL", "QWEN3_SPACE_ID", "HUGGINGFACE_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(chat_routes, "settings", Settings.from_environment())
    # Point the providers-module settings at a live proxy of the current env so
    # that tests using monkeypatch.setenv() are reflected by create_qwen3_provider().
    monkeypatch.setattr(providers_module, "settings", DynamicProviderSettings())


class TestQwen3ProviderConfiguration:
    """Test provider wiring and environment variable handling."""

    def test_no_qwen3_config_disabled(self, monkeypatch):
        """When neither QWEN3_SPACE_ID nor QWEN3_ENDPOINT_URL is set, provider is None."""
        assert create_qwen3_provider() is None

    def test_gradio_space_id_enabled(self, monkeypatch):
        """QWEN3_SPACE_ID enables the Gradio Space path."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()
        assert provider is not None
        assert provider._space_id == "test-org/test-space"
        assert provider._http_endpoint is None

    def test_http_endpoint_enabled(self, monkeypatch):
        """QWEN3_ENDPOINT_URL still enables the legacy HTTP path."""
        monkeypatch.setenv("QWEN3_ENDPOINT_URL", "https://localhost:8080")
        provider = create_qwen3_provider()
        assert provider is not None
        assert provider._space_id is None
        assert provider._http_endpoint == "https://localhost:8080"

    def test_space_id_takes_precedence_over_endpoint(self, monkeypatch):
        """QWEN3_SPACE_ID takes precedence when both are set."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        monkeypatch.setenv("QWEN3_ENDPOINT_URL", "https://localhost:8080")
        provider = create_qwen3_provider()
        assert provider is not None
        assert provider._space_id == "org/space"
        assert provider._http_endpoint == "https://localhost:8080"
        # chat_completion routes to _call_gradio because space_id is set

    def test_hf_token_fallback_chain(self, monkeypatch):
        """HUGGINGFACE_TOKEN and HF_TOKEN fallback chain works."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_primary")
        provider = create_qwen3_provider()
        assert provider is not None
        assert provider._hf_token == "hf_primary"

        monkeypatch.delenv("HUGGINGFACE_TOKEN")
        monkeypatch.setenv("HF_TOKEN", "hf_secondary")
        provider = create_qwen3_provider()
        assert provider._hf_token == "hf_secondary"

    def test_space_id_default_empty(self, monkeypatch):
        """QWEN3_SPACE_ID defaults to empty string / None."""
        monkeypatch.delenv("QWEN3_SPACE_ID", raising=False)
        settings = Settings.from_environment()
        assert settings.qwen3_space_id is None

    def test_default_values(self, monkeypatch):
        """Default config values are independent of local environment."""
        monkeypatch.delenv("QWEN3_TIMEOUT", raising=False)
        monkeypatch.delenv("QWEN3_SPACE_ID", raising=False)
        monkeypatch.delenv("QWEN3_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("QWEN3_MODEL_ID", raising=False)

        settings = Settings.from_environment()

        assert settings.qwen3_endpoint_url is None
        assert settings.qwen3_space_id is None
        assert settings.qwen3_model_id == "qwen3-4b-finetuned"
        assert settings.qwen3_timeout == 60.0
        assert settings.qwen3_supports_tools is False

    def test_timeout_converted_to_float(self, monkeypatch):
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        monkeypatch.setenv("QWEN3_TIMEOUT", "120")
        provider = create_qwen3_provider()
        assert provider._timeout == 120.0

    def test_supports_tools_true(self, monkeypatch):
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        monkeypatch.setenv("QWEN3_SUPPORTS_TOOLS", "true")
        provider = create_qwen3_provider()
        assert provider.supports_tools() is True


class TestGradioClientLifecycle:
    """Test Gradio client instantiation and reuse."""

    def test_client_lazy_on_first_request(self, monkeypatch):
        """The Gradio Client is not created until the first request."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()
        assert provider._gradio_client is None

    def test_client_created_with_correct_args(self, monkeypatch):
        """Client constructor arguments match the supported Gradio Client API."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_x")
        monkeypatch.setenv("QWEN3_TIMEOUT", "90.0")

        provider = create_qwen3_provider()
        assert provider._gradio_client is None

        with patch("gradio_client.Client") as MockClient:
            instance = FakeGradioClient()
            MockClient.return_value = instance
            provider.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                tools=None,
                tool_choice="none",
                temperature=0.2,
            )

        MockClient.assert_called_once_with(
            src="test-org/test-space",
            token="hf_x",
            httpx_kwargs={"timeout": 90.0},
            verbose=False,
        )

    def test_client_reused_across_requests(self, monkeypatch):
        """A single Gradio Client instance is reused for every request."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch("gradio_client.Client") as MockClient:
            MockClient.return_value = FakeGradioClient()
            for _ in range(3):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    tools=None,
                    tool_choice="none",
                )

        assert MockClient.call_count == 1

    def test_client_not_created_during_import(self, monkeypatch):
        """Creating a provider does not instantiate the Gradio Client."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        with patch("gradio_client.Client") as MockClient:
            provider = create_qwen3_provider()
            assert MockClient.call_count == 0

    def test_concurrent_requests_share_client(self, monkeypatch):
        """Concurrent requests serialize safely on client initialization."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        import threading

        provider = create_qwen3_provider()
        client_lock_acquired = []
        barrier = threading.Barrier(3)

        with patch("gradio_client.Client") as MockClient:
            MockClient.return_value = FakeGradioClient()

            def work():
                barrier.wait()
                provider.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    tools=None,
                    tool_choice="none",
                )
                provider._client_lock.acquire()
                client_lock_acquired.append(True)
                provider._client_lock.release()

            threads = [threading.Thread(target=work) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert MockClient.call_count == 1


class TestGradioRequestPayload:
    """Test the exact payload sent to the Space's chat_completion function."""

    def test_messages_serialized_as_json_text(self, monkeypatch):
        """Messages are serialized into a JSON string passed to predict()."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        messages = [
            {"role": "system", "content": "You are a car assistant."},
            {"role": "user", "content": "What cars do you have?"},
        ]
        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "I have many cars."}]
            )
            provider.chat_completion(messages=messages, tools=None, tool_choice="none")

        call = MockClient.return_value.predict_calls[0]
        assert call["api_name"] == "/chat_completion"
        payload = json.loads(call["payload_text"])
        assert payload["messages"] == messages
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 1024

    def test_payload_does_not_contain_openai_stream_field(self, monkeypatch):
        """The Gradio payload does not carry the OpenAI stream parameter."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "ok"}]
            )
            provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}], tools=None, tool_choice="none"
            )

        payload = json.loads(MockClient.return_value.predict_calls[0]["payload_text"])
        assert "stream" not in payload

    def test_tools_included_when_supported(self, monkeypatch):
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        monkeypatch.setenv("QWEN3_SUPPORTS_TOOLS", "true")
        provider = create_qwen3_provider()

        tools = [
            {"type": "function", "function": {"name": "search_car_web", "parameters": {}}}
        ]
        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "ok"}]
            )
            provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                tool_choice="auto",
            )

        payload = json.loads(MockClient.return_value.predict_calls[0]["payload_text"])
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "auto"

    def test_tools_excluded_when_not_supported(self, monkeypatch):
        """Tools are never sent to the Space because supports_tools is false by default."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "ok"}]
            )
            provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto"
            )

        payload = json.loads(MockClient.return_value.predict_calls[0]["payload_text"])
        assert "tools" not in payload
        assert "tool_choice" not in payload


class TestGradioResponseMapping:
    """Test extraction of the fine-tuned response from Gradio predictions."""

    def test_successful_prediction(self, monkeypatch):
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "The Scorpio costs 14 lakh."}]
            )
            response = provider.chat_completion(
                messages=[{"role": "user", "content": "How much is the Scorpio?"}],
                tools=None,
                tool_choice="none",
            )

        assert response.content == "The Scorpio costs 14 lakh."
        assert response.tool_calls is None
        assert response.provider_name == "qwen3_fallback"

    def test_finetuned_response_extracted(self, monkeypatch):
        """Only finetuned_model_response is returned; base_model_response is ignored."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[
                    {
                        "base_model_response": "Diagnostic base-model output: this must never be exposed.",
                        "finetuned_model_response": "The base model costs less.",
                    }
                ]
            )
            response = provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}], tools=None, tool_choice="none"
            )

        assert response.content == "The base model costs less."
        assert "Diagnostic base-model output" not in response.content
        assert "[REDACTED]" not in response.content

    def test_finetuned_missing_no_base_substitution(self, monkeypatch):
        """Missing finetuned response raises; base_model_response is never substituted."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[
                    {
                        "base_model_response": "This is the base model and must not be exposed.",
                    }
                ]
            )
            with pytest.raises(RuntimeError, match="unexpected response") as excinfo:
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )
            # Verify the base-model text leaked anywhere in the raised error.
            assert "base model" not in str(excinfo.value).lower()

    def test_finetuned_empty_string_raises(self, monkeypatch):
        """An empty fine-tuned response raises; empty is treated as invalid."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": ""}]
            )
            with pytest.raises(RuntimeError, match="unexpected response"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )

    def test_finetuned_whitespace_only_raises(self, monkeypatch):
        """A whitespace-only fine-tuned response raises as invalid."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "   \n\t  "}]
            )
            with pytest.raises(RuntimeError, match="unexpected response"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )

    def test_finetuned_none_raises(self, monkeypatch):
        """A None fine-tuned response raises."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": None}]
            )
            with pytest.raises(RuntimeError, match="unexpected response"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )

    def test_malformed_prediction_result_raises(self, monkeypatch):
        """Non-dict / unexpected Gradio result raises safely."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(responses=[None])
            with pytest.raises(RuntimeError, match="unexpected response"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )

    def test_result_as_object(self, monkeypatch):
        """Attribute-style result objects are also handled."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        result = SimpleNamespace(
            finetuned_model_response="Attribute-style result works.",
            base_model_response="Ignored base output.",
        )

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(responses=[result])
            response = provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                tool_choice="none",
            )

        assert response.content == "Attribute-style result works."

    def test_content_stripped(self, monkeypatch):
        """Whitespace around the fine-tuned content is stripped."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "   trimmed answer   \n"}]
            )
            response = provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                tool_choice="none",
            )

        assert response.content == "trimmed answer"

    def test_tool_calls_none_always(self, monkeypatch):
        """QWEN3_SUPPORTS_TOOLS is false, so tool_calls is never populated."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "ok"}]
            )
            response = provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                tool_choice="none",
            )
            assert response.tool_calls is None


class TestGradioErrorHandling:
    """Test that Gradio errors are handled safely and never misclassified as Groq 429s."""

    def test_authentication_error(self, monkeypatch):
        """HF authentication failures raise a safe RuntimeError, not a 429."""
        from gradio_client.exceptions import AuthenticationError

        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_invalid_token")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                should_fail_with=AuthenticationError("Invalid token: hf_invalid_token")
            )
            with pytest.raises(RuntimeError, match="temporarily unavailable") as excinfo:
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )
            assert "hf_invalid_token" not in str(excinfo.value)

    def test_connection_error(self, monkeypatch):
        """Connection failures raise a safe RuntimeError."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                should_fail_with=ConnectionError("Space is offline")
            )
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )

    def test_timeout_error(self, monkeypatch):
        """Timeout errors raise a safe RuntimeError."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                should_fail_with=TimeoutError("Predict timed out")
            )
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )

    def test_unexpected_gradio_error(self, monkeypatch):
        """Unexpected Gradio errors still raise a safe RuntimeError."""
        from gradio_client.exceptions import AppError

        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                should_fail_with=AppError("Internal Gradio error")
            )
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )

    def test_error_not_misclassified_as_429(self, monkeypatch, caplog):
        """Gradio failures never get treated like a Groq 429."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                should_fail_with=ConnectionError("offline")
            )
            with pytest.raises(RuntimeError):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )
            assert "429" not in caplog.text

    def test_hf_token_not_exposed_in_error_logs(self, monkeypatch, caplog):
        """HF tokens are redacted from error logs."""
        from gradio_client.exceptions import AuthenticationError

        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                should_fail_with=AuthenticationError(
                    "Invalid token hf_xxxxxxxxxxxxxxxxxxxxxx in request"
                )
            )
            with pytest.raises(RuntimeError):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )
            log_text = "\n".join(record.message for record in caplog.records)
            assert "hf_xxxxxxxxxxxxxxxxxxxxxx" not in log_text
            assert "HUGGINGFACE_TOKEN_REDACTED" in log_text

    def test_api_keys_not_exposed_in_error_logs(self, monkeypatch, caplog):
        """Groq/Tavily/GitHub tokens are redacted from error logs."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                should_fail_with=ConnectionError(
                    "Bearer gsk_ABCDEFGHIJKLMNOPQRSTUVWXYZ and tvly_ABCDEFGHIJKLMNOPQRSTUVWXYZ leaked"
                )
            )
            with pytest.raises(RuntimeError):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )
            log_text = "\n".join(record.message for record in caplog.records)
            assert "gsk_" not in log_text
            assert "tvly_" not in log_text
            assert "Bearer [REDACTED]" in log_text

    def test_request_payload_not_logged(self, monkeypatch, caplog):
        """Message contents are not logged in provider errors."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()
        secret_message = "secret message with driver@example.test and 9876543210"

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                should_fail_with=ConnectionError("offline")
            )
            with pytest.raises(RuntimeError):
                provider.chat_completion(
                    messages=[{"role": "user", "content": secret_message}],
                    tools=None,
                    tool_choice="none",
                )
            log_text = "\n".join(record.message for record in caplog.records)
            assert secret_message not in log_text

    def test_import_error_message(self, monkeypatch):
        """Missing gradio_client raises a helpful RuntimeError."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "test-org/test-space")
        provider = create_qwen3_provider()

        # The provider checks the module global `gradio_client` and wraps any
        # ImportError from client construction. Simulate a missing dependency the
        # way it actually happens: the module reference is None.
        with patch.object(providers_module, "gradio_client", None):
            with pytest.raises(RuntimeError, match="gradio_client"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )


class TestHttpEndpointCompatibility:
    """Test the legacy OpenAI-compatible HTTP endpoint path still works."""

    def test_http_endpoint_used_when_space_id_unset(self, monkeypatch):
        monkeypatch.setenv("QWEN3_ENDPOINT_URL", "https://local.example.test/api")
        provider = create_qwen3_provider()
        assert provider._http_endpoint == "https://local.example.test/api"
        assert provider._gradio_client is None

    def test_http_openai_payload(self, monkeypatch, caplog):
        """The HTTP path sends the original OpenAI-compatible payload."""
        monkeypatch.setenv("QWEN3_ENDPOINT_URL", "https://local.example.test/api")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_http")
        provider = create_qwen3_provider()

        messages = [{"role": "user", "content": "Hi there"}]
        with patch("httpx.Client") as MockClient:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "HTTP response content"}}]
            }
            # The provider uses `with httpx.Client(...) as session:`, so the object
            # it posts with is Client().__enter__(). Configure that explicitly:
            # assigning to a child of a MagicMock would otherwise leave a bare
            # MagicMock as the response and .json() would not be the fixture dict.
            mock_session = MockClient.return_value.__enter__.return_value
            mock_session.post.return_value = mock_response

            response = provider.chat_completion(messages=messages, temperature=0.2)

        assert response.content == "HTTP response content"
        call = mock_session.post.call_args
        assert call[0][0] == "https://local.example.test/api/v1/chat/completions"
        payload = call[1]["json"]
        assert payload["model"] == "qwen3-4b-finetuned"
        assert payload["messages"] == messages
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 2048
        assert payload["stream"] is False

    def test_http_endpoint_429_becomes_rate_limit_error(self, monkeypatch):
        """HTTP 429 from a local endpoint is still classified as a Groq-rate-limit error."""
        monkeypatch.setenv("QWEN3_ENDPOINT_URL", "https://local.example.test/api")
        provider = create_qwen3_provider()

        with patch("httpx.Client") as MockClient:
            mock_request = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "30"}
            mock_response.content = b'{"error": "rate limit"}'
            mock_response.json.return_value = {"error": "rate limit"}
            MockClient.return_value.__enter__.return_value = mock_response
            MockClient.return_value.__enter__.return_value.post.side_effect = (
                httpx.HTTPStatusError("429", request=mock_request, response=mock_response)
            )
            with pytest.raises(RateLimitError, match="Qwen3 rate limit exceeded"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    tool_choice="none",
                )

    def test_http_tool_calls_when_supported(self, monkeypatch):
        monkeypatch.setenv("QWEN3_ENDPOINT_URL", "https://local.example.test/api")
        monkeypatch.setenv("QWEN3_SUPPORTS_TOOLS", "true")
        provider = create_qwen3_provider()

        tools = [
            {"type": "function", "function": {"name": "calc", "parameters": {}}}
        ]
        with patch("httpx.Client") as MockClient:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}
                            ],
                        }
                    }
                ]
            }
            mock_session = MockClient.return_value.__enter__.return_value
            mock_session.post.return_value = mock_response

            response = provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto"
            )

        assert response.tool_calls == [
            {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}
        ]


class TestIntegrationWithProviderRouter:
    """End-to-end routing tests: Groq primary still takes precedence."""

    @pytest.fixture
    def router_with_gradio_qwen3(self, monkeypatch):
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_for_router_test")
        monkeypatch.setattr(chat_routes, "settings", Settings.from_environment())
        with patch("app.modules.chat.providers.gradio_client.Client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[
                    {"finetuned_model_response": "Qwen3 from router test."},
                    {"finetuned_model_response": "Qwen3 from router test."},
                ]
            )
            provider = create_qwen3_provider()
            assert provider is not None
            yield provider, MockClient

    def test_groq_still_primary_over_gradio_qwen3(self, monkeypatch, router_with_gradio_qwen3):
        """Groq primary succeeds; Gradio Qwen3 is never invoked."""
        router, mock_client = router_with_gradio_qwen3
        groq = GroqProvider(api_key="sk_groq", model="openai/gpt-oss-120b")
        rr = ProviderRouter(groq_provider=groq, qwen3_provider=router)

        with patch.object(groq, "chat_completion") as MockGroq:
            MockGroq.return_value = ProviderResponse(
                content="Groq primary won.", provider_name="groq"
            )
            response = rr.chat_completion(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                tools=None,
                tool_choice="none",
            )
            assert response.content == "Groq primary won."
            assert response.provider_name == "groq"
            assert mock_client.return_value.predict_calls == []

    def test_groq_429_routes_to_gradio_qwen3(self, monkeypatch, router_with_gradio_qwen3):
        """Groq 429 routes through the Gradio Qwen3 provider."""
        router, mock_client = router_with_gradio_qwen3
        groq = GroqProvider(api_key="sk_groq", model="openai/gpt-oss-120b")
        rr = ProviderRouter(groq_provider=groq, qwen3_provider=router)

        mock_response = httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.groq.com"),
            headers={"Retry-After": "10"},
        )
        with patch.object(groq, "chat_completion") as mock_groq_chat:
            mock_groq_chat.side_effect = RateLimitError(
                "rate limit", response=mock_response, body={}
            )
            response = rr.chat_completion(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                tools=None,
                tool_choice="none",
            )

        assert response.content == "Qwen3 from router test."
        assert response.provider_name == "qwen3_fallback"
        assert len(mock_client.return_value.predict_calls) == 1
        mock_groq_chat.assert_called_once()

    def test_existing_groq_cooldown_retries_groq_after_exhaustion(self, monkeypatch, router_with_gradio_qwen3):
        """Existing cooldown/recovery behavior unchanged with Gradio Qwen3 in place."""
        import time

        rr, mock_client = router_with_gradio_qwen3
        groq = GroqProvider(api_key="sk_groq", model="openai/gpt-oss-120b")
        router = ProviderRouter(groq_provider=groq, qwen3_provider=rr, cooldown_seconds=1)

        mock_response = httpx.Response(
            429, request=httpx.Request("POST", "https://api.groq.com")
        )

        with patch.object(groq, "chat_completion") as mock_groq_chat:
            mock_groq_chat.side_effect = RateLimitError(
                "rate limit", response=mock_response, body={}
            )

            r1 = router.chat_completion(
                messages=[{"role": "user", "content": "hi"}], tools=None, tool_choice="none"
            )
            assert r1.provider_name == "qwen3_fallback"
            assert mock_groq_chat.call_count == 1

            # Groq exhausted: second call goes straight to Qwen3.
            r2 = router.chat_completion(
                messages=[{"role": "user", "content": "hi again"}], tools=None, tool_choice="none"
            )
            assert r2.provider_name == "qwen3_fallback"
            assert mock_groq_chat.call_count == 1

        time.sleep(1.1)

        # Cooldown expired: Groq is retried.
        with patch.object(groq, "chat_completion") as mock_groq_chat:
            mock_groq_chat.return_value = ProviderResponse(
                content="Groq recovered.", provider_name="groq"
            )
            r3 = router.chat_completion(
                messages=[{"role": "user", "content": "hi after cooldown"}], tools=None, tool_choice="none"
            )
            assert r3.content == "Groq recovered."
            assert r3.provider_name == "groq"
            assert mock_groq_chat.call_count == 1


class TestToolDependentFallback:
    """Tool-dependent requests still get the safe unavailable message via Gradio Qwen3."""

    def test_search_tool_returns_unavailable_via_gradio_qwen3(self, monkeypatch):
        """Tool-dependent request returns safe unavailable when Qwen3 doesn't support tools."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        provider = create_qwen3_provider()
        groq = GroqProvider(api_key="x")
        router = ProviderRouter(groq_provider=groq, qwen3_provider=provider)

        tools = [
            {"type": "function", "function": {"name": "search_car_database", "parameters": {}}}
        ]
        mock_response = httpx.Response(
            429, request=httpx.Request("POST", "https://api.groq.com")
        )
        with patch.object(groq, "chat_completion") as mock_groq_chat:
            mock_groq_chat.side_effect = RateLimitError(
                "rate limit", response=mock_response, body={}
            )
            response = router.chat_completion(
                messages=[{"role": "user", "content": "Search the database"}],
                tools=tools,
                tool_choice="auto",
            )

        assert "unable to access vehicle databases" in response.content.lower()
        assert "try again later" in response.content.lower()

    def test_emi_tool_returns_unavailable_via_gradio_qwen3(self, monkeypatch):
        """EMI tool request returns safe unavailable when Qwen3 doesn't support tools."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        provider = create_qwen3_provider()
        groq = GroqProvider(api_key="x")
        router = ProviderRouter(groq_provider=groq, qwen3_provider=provider)

        tools = [
            {"type": "function", "function": {"name": "calculate_emi", "parameters": {}}}
        ]
        mock_response = httpx.Response(
            429, request=httpx.Request("POST", "https://api.groq.com")
        )
        with patch.object(groq, "chat_completion") as mock_groq_chat:
            mock_groq_chat.side_effect = RateLimitError(
                "rate limit", response=mock_response, body={}
            )
            response = router.chat_completion(
                messages=[{"role": "user", "content": "Calculate EMI"}],
                tools=tools,
                tool_choice="auto",
            )

        assert "unable to access" in response.content.lower()
        assert "calculation" in response.content.lower() or "emi" in response.content.lower()

    def test_non_tool_dependent_works_via_gradio_qwen3(self, monkeypatch):
        """Non-tool request calls the Gradio Qwen3 fallback normally."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        provider = create_qwen3_provider()
        groq = GroqProvider(api_key="x")
        router = ProviderRouter(groq_provider=groq, qwen3_provider=provider)

        mock_response = httpx.Response(
            429, request=httpx.Request("POST", "https://api.groq.com")
        )
        with patch.object(groq, "chat_completion") as mock_groq_chat, patch.object(
            provider, "_get_gradio_client"
        ) as MockClient:
            mock_groq_chat.side_effect = RateLimitError(
                "rate limit", response=mock_response, body={}
            )
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "The Tata Nexon is a good hatchback."}]
            )
            response = router.chat_completion(
                messages=[{"role": "user", "content": "What is a good family car?"}],
                tools=None,
                tool_choice="none",
            )

        assert response.content == "The Tata Nexon is a good hatchback."
        assert response.provider_name == "qwen3_fallback"


class TestColdStartBehavior:
    """Cold-start handling for Hugging Face Spaces that spin down."""

    def test_first_predict_after_cold_start_retries_on_transient_failure(self, monkeypatch):
        """A transient Space cold-start failure still produces a safe RuntimeError, not a 429.

        We make two requests: the first raises a transient cold-start error (handled
        safely), the second succeeds. This verifies error handling does not block
        subsequent retries by the caller, and that no automatic retry loop runs here.
        """
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        provider = create_qwen3_provider()
        cold_start_counter = [0]

        def predict_with_cold_start(payload_text, api_name="/chat_completion"):
            cold_start_counter[0] += 1
            if cold_start_counter[0] == 1:
                raise ConnectionError("Space is waking up (cold start)")
            return {"finetuned_model_response": "Now it works."}

        client = FakeGradioClient()
        client.predict = predict_with_cold_start

        with patch.object(provider, "_get_gradio_client") as MockClient:
            MockClient.return_value = client
            # First call raises a safe RuntimeError (not a 429, no tokens leaked).
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                provider.chat_completion(
                    messages=[{"role": "user", "content": "hi"}], tools=None, tool_choice="none"
                )
            # Client reused; second call succeeds.
            response = provider.chat_completion(
                messages=[{"role": "user", "content": "hi again"}], tools=None, tool_choice="none"
            )
            assert response.content == "Now it works."
            assert cold_start_counter[0] == 2

    def test_gradio_client_timeout_respected(self, monkeypatch):
        """The configured timeout is passed into the Gradio Client's httpx config."""
        monkeypatch.setenv("QWEN3_SPACE_ID", "org/space")
        monkeypatch.setenv("QWEN3_TIMEOUT", "45.0")
        provider = create_qwen3_provider()
        assert provider._timeout == 45.0

        with patch("gradio_client.Client") as MockClient:
            MockClient.return_value = FakeGradioClient(
                responses=[{"finetuned_model_response": "ok"}]
            )
            provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}], tools=None, tool_choice="none"
            )
        call_kwargs = MockClient.call_args.kwargs
        assert call_kwargs["httpx_kwargs"]["timeout"] == 45.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
