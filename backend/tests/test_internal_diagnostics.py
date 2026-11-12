import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.core.config import settings
from app.modules.chat.providers import ProviderResponse
from app.api.routes import internal_diagnostics as diag_module


class FakeQwen3Provider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat_completion(self, messages, tools=None, tool_choice="auto", temperature=0.2):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
        })
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def set_secret(monkeypatch):
    monkeypatch.setattr(settings, "qwen3_diagnostic_secret", "super-secret-key")

    def _set(value):
        monkeypatch.setattr(settings, "qwen3_diagnostic_secret", value)

    return _set


@pytest.fixture
def fake_provider(monkeypatch):
    """Install a fake provider and monkeypatch the factory to return it."""
    def _install(provider):
        monkeypatch.setattr(diag_module, "create_qwen3_provider", lambda: provider)
        return provider

    return _install


async def post_diag(headers=None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/internal/diagnostics/qwen3", headers=headers or {})


@pytest.mark.asyncio
async def test_missing_diagnostic_secret_returns_404(set_secret):
    set_secret(None)
    response = await post_diag({"X-Qwen3-Diagnostic-Key": "any"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Diagnostic endpoint is not configured."


@pytest.mark.asyncio
async def test_missing_header_returns_401(set_secret):
    response = await post_diag()
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized."


@pytest.mark.asyncio
async def test_incorrect_secret_returns_401(set_secret):
    response = await post_diag({"X-Qwen3-Diagnostic-Key": "wrong-key"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized."


@pytest.mark.asyncio
async def test_correct_secret_invokes_provider_and_inference(set_secret, fake_provider):
    provider = fake_provider(FakeQwen3Provider(
        response=ProviderResponse(content="Petrol: spark-ignited. Diesel: compression-ignited. Petrol is lighter and higher-revving.",
                                 provider_name="qwen3_fallback")
    ))
    response = await post_diag({"X-Qwen3-Diagnostic-Key": "super-secret-key"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "qwen3_fallback"
    assert body["response_preview"].startswith("Petrol")

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["tools"] is None
    assert call["temperature"] == 0.2
    assert call["messages"][0]["role"] == "user"
    assert "petrol and diesel engines" in call["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_preview_limited_to_500_chars(set_secret, fake_provider):
    fake_provider(FakeQwen3Provider(
        response=ProviderResponse(content="x" * 3000, provider_name="qwen3_fallback")
    ))
    response = await post_diag({"X-Qwen3-Diagnostic-Key": "super-secret-key"})
    assert response.status_code == 200
    assert len(response.json()["response_preview"]) == 500


@pytest.mark.asyncio
async def test_empty_model_output_reported_as_failure(set_secret, fake_provider):
    fake_provider(FakeQwen3Provider(
        response=ProviderResponse(content="   ", provider_name="qwen3_fallback")
    ))
    response = await post_diag({"X-Qwen3-Diagnostic-Key": "super-secret-key"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Qwen3 diagnostic returned an empty response."


@pytest.mark.asyncio
async def test_wrong_provider_name_reported_as_failure(set_secret, fake_provider):
    fake_provider(FakeQwen3Provider(
        response=ProviderResponse(content="hello", provider_name="groq")
    ))
    response = await post_diag({"X-Qwen3-Diagnostic-Key": "super-secret-key"})
    assert response.status_code == 502
    assert "unexpected provider" in response.json()["detail"]


@pytest.mark.asyncio
async def test_provider_exception_does_not_leak_credentials(set_secret, fake_provider):
    # Simulate an exception whose message contains a Hugging Face token.
    fake_provider(FakeQwen3Provider(error=RuntimeError("hf_abc123secret token leaked")))
    response = await post_diag({"X-Qwen3-Diagnostic-Key": "super-secret-key"})

    assert response.status_code == 502
    body = response.json()
    assert "hf_" not in body["detail"]
    assert "leaked" not in body["detail"]
    assert body["detail"] == "Qwen3 diagnostic inference failed."
