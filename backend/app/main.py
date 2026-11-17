from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.modules.chat.routes import router as chat_router
from app.core.config import settings
from app.core.middleware import RequestSizeLimitMiddleware, SecurityHeadersMiddleware


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Lifecycle hooks: emit one sanitized operational summary on shutdown.

    Logging is aggregate-only (counters, latency summaries, cost/token totals)
    and never contains prompts, user messages, PII, or secrets. It runs once
    per process lifetime rather than per request to avoid log noise, and any
    failure is contained inside the logger so it cannot break shutdown.
    """
    yield
    from app.modules.chat.metrics import log_diagnostics
    log_diagnostics()

def create_app(*, carworld_env: str | None = None) -> FastAPI:
    env = (carworld_env or settings.carworld_env).strip().lower()
    is_production = env == "production"
    from app.modules.chat.semantic_cache import semantic_cache
    from app.modules.chat.usage import usage
    semantic_cache.configure(
        enabled=settings.semantic_cache_enabled,
        max_entries=settings.semantic_cache_max_entries,
        ttl_seconds=settings.semantic_cache_ttl_seconds,
        similarity_threshold=settings.semantic_cache_similarity_threshold,
    )
    usage.set_prices(
        "groq",
        settings.groq_input_price_per_1m,
        settings.groq_output_price_per_1m,
    )
    usage.set_prices(
        "qwen3_fallback",
        settings.qwen3_input_price_per_1m,
        settings.qwen3_output_price_per_1m,
    )

    application = FastAPI(
        title="CarWorld API",
        version="1.0.0",
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
        openapi_url=None if is_production else "/openapi.json",
        lifespan=lifespan,
    )
    # Registration order matters: Starlette wraps middlewares such that the FIRST
    # one registered is the OUTERMOST. CORS must be outermost so preflight and
    # error responses carry CORS headers; the request-size limit is innermost so
    # its 413 responses flow back out through the security-header and CORS layers.
    application.add_middleware(RequestSizeLimitMiddleware)
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )
    application.include_router(v1_router, prefix="/api/v1")
    application.include_router(chat_router)

    @application.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
