"""Production-hardening tests for request-size limit and security headers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response

from app.core.middleware import RequestBodyTooLarge, RequestSizeLimitMiddleware, SecurityHeadersMiddleware
from app.main import app


class _CollectingApp:
    """ASGI app that drains the request body and records what it received."""

    def __init__(self):
        self.received = b""
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            self.received += message.get("body", b"")
            if not message.get("more_body", False):
                break
        response = Response("ok", status_code=200)
        await response(scope, receive, send)


class _MsgSource:
    """Deterministic ASGI receive source from a list of body chunks."""

    def __init__(self, chunks: list[bytes]):
        self._items = [
            {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
            for i, chunk in enumerate(chunks)
        ]
        self._items.append({"type": "http.disconnect"})
        self._idx = 0

    async def __call__(self):
        message = self._items[self._idx]
        self._idx += 1
        return message


async def _run_middleware(chunks, max_bytes=65536, headers=None):
    app = _CollectingApp()
    middleware = RequestSizeLimitMiddleware(app, max_body_bytes=max_bytes)
    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": headers or []}
    await middleware(scope, _MsgSource(chunks), send)
    return app, sent


def _status(sent):
    for message in sent:
        if message["type"] == "http.response.start":
            return message["status"]
    return None


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_oversized_request_body_is_rejected_before_processing(client):
    # 256 KiB body >> 64 KiB limit. Must be rejected with 413 by the middleware,
    # not parsed/processed by the application.
    oversized = {"message": "x" * 256_000, "history": []}
    response = client.post(
        "/api/chat",
        json=oversized,
    )
    assert response.status_code == 413


def test_normal_sized_request_is_not_blocked_by_middleware(client):
    # Empty message reaches the app layer (422 from Pydantic) rather than being
    # rejected by the body-limit or security middleware.
    response = client.post("/api/chat", json={"message": " ", "history": []})
    assert response.status_code == 422


def test_health_returns_security_headers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "no-referrer"
    assert response.headers.get("cache-control") == "no-store"


def test_v1_health_returns_security_headers(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"


def test_security_headers_present_on_error_responses(client):
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code >= 400
    assert response.headers.get("x-content-type-options") == "nosniff"


def test_request_size_limit_rejects_content_length_ge_limit(client):
    # Direct check that the boundary is enforced even when JSON is not parseable
    # (e.g. a raw oversized body).
    response = client.post(
        "/api/chat",
        content=b"x" * 70_000,
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 413


def test_streamed_body_over_limit_rejected_413():
    import asyncio

    # Two 50KB chunks = 100KB > 64KB. The middleware must abort the stream and
    # return 413 WITHOUT letting the app consume the full body.
    app, sent = asyncio.run(_run_middleware([b"x" * 50_000, b"y" * 50_000]))
    assert _status(sent) == 413
    assert app.received == b"x" * 50_000


def test_streamed_first_chunk_over_limit_rejected_413():
    import asyncio

    # A single chunk already over the limit -> immediate 413, nothing forwarded.
    app, sent = asyncio.run(_run_middleware([b"z" * 80_000]))
    assert _status(sent) == 413
    assert app.received == b""


def test_streamed_body_under_limit_passes_through():
    import asyncio

    app, sent = asyncio.run(_run_middleware([b"a" * 1000, b"b" * 1000]))
    assert _status(sent) == 200
    assert app.received == b"a" * 1000 + b"b" * 1000


def test_empty_body_reaches_app():
    import asyncio

    app, sent = asyncio.run(_run_middleware([]))
    assert _status(sent) == 200
    assert app.called is True


def test_malformed_content_length_is_not_blocked():
    import asyncio

    app, sent = asyncio.run(
        _run_middleware([b"ok"], headers=[(b"content-length", b"not-a-number")])
    )
    assert _status(sent) == 200
    assert app.received == b"ok"


def test_oversized_content_length_rejected_before_app():
    import asyncio

    app, sent = asyncio.run(
        _run_middleware([b"x" * 10], headers=[(b"content-length", b"70000")])
    )
    assert _status(sent) == 413
    assert app.called is False


def test_nested_middlewares_security_headers_on_413():
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            content=b"x" * 70_000,
            headers={"Content-Type": "text/plain"},
        )
    assert response.status_code == 413
    assert response.headers.get("x-content-type-options") == "nosniff"


def test_cors_headers_present_on_normal_response():
    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={"Origin": "http://127.0.0.1:5500"},
        )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5500"


def test_swagger_openapi_exposed_in_dev_only():
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        assert client.get("/openapi.json").status_code == 200
