from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.internal_diagnostics import router as internal_diagnostics_router
from app.api.v1.router import router as v1_router
from app.modules.chat.routes import router as chat_router
from app.core.config import settings
from app.core.middleware import RequestSizeLimitMiddleware, SecurityHeadersMiddleware

def create_app(*, carworld_env: str | None = None) -> FastAPI:
    env = (carworld_env or settings.carworld_env).strip().lower()
    is_production = env == "production"
    application = FastAPI(
        title="CarWorld API",
        version="1.0.0",
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
        openapi_url=None if is_production else "/openapi.json",
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
    application.include_router(internal_diagnostics_router)

    @application.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
