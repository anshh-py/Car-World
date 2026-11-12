"""Production hardening middleware: request-body size bound and security headers."""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.responses import Response

DEFAULT_MAX_BODY_BYTES = 65536  # 64 KiB


class RequestBodyTooLarge(Exception):
    """Raised internally once a streamed request body exceeds the limit."""


class RequestSizeLimitMiddleware:
    """Reject oversized request bodies early instead of parsing them fully.

    The application's Pydantic models already cap individual fields; this
    middleware adds an early, cheap bound at the ASGI boundary so oversized
    payloads are dropped before they consume parsing/LLM resources. It honors
    the Content-Length header (covers standard HTTP clients incl. browsers and
    curl) and also bounds streamed/chunked bodies by counting received bytes.
    """

    def __init__(self, app, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    response = Response(
                        "Request body too large", status_code=413
                    )
                    await response(scope, receive, send)
                    return
            except (TypeError, ValueError):
                # Malformed Content-Length: let the server/app handle it.
                pass

        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    # Abort the stream before the app can process this oversized
                    # body. Raising here unwinds through the inner app call and
                    # lets us emit a clean 413 before any response is committed.
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            response = Response("Request body too large", status_code=413)
            await response(scope, receive, send)


class SecurityHeadersMiddleware:
    """Apply conservative security headers to every API response.

    - X-Content-Type-Options: nosniff  -> prevent MIME-sniffing.
    - X-Frame-Options: DENY             -> prevent clickjacking/framing.
    - Referrer-Policy: no-referrer      -> do not leak the page URL.
    - Cache-Control: no-store           -> do not cache personalized chat content.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message.get("headers", []))
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault("Cache-Control", "no-store")
                message["headers"] = headers.raw
            await send(message)

        await self.app(scope, receive, send_with_headers)
