"""Verification for M3 production configuration: API docs gating.

When CARWORLD_ENV=production the API documentation endpoints (Swagger UI,
ReDoc and the OpenAPI schema) must be disabled. In development they must
remain available. The tests build isolated app instances through
``create_app(carworld_env=...)`` so they do not mutate the real environment
and do not require production credentials.
"""

import httpx
import pytest

from app.main import create_app


@pytest.mark.asyncio
async def test_production_docs_endpoints_unavailable() -> None:
    """In production, /docs, /redoc and /openapi.json must be unavailable."""
    prod_app = create_app(carworld_env="production")
    transport = httpx.ASGITransport(app=prod_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        docs = await client.get("/docs")
        redoc = await client.get("/redoc")
        openapi = await client.get("/openapi.json")

    assert docs.status_code == 404
    assert redoc.status_code == 404
    assert openapi.status_code == 404


@pytest.mark.asyncio
async def test_development_docs_endpoints_available() -> None:
    """In development, all three documentation endpoints remain available."""
    dev_app = create_app(carworld_env="development")
    transport = httpx.ASGITransport(app=dev_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        docs = await client.get("/docs")
        redoc = await client.get("/redoc")
        openapi = await client.get("/openapi.json")

    assert docs.status_code == 200
    assert redoc.status_code == 200
    assert openapi.status_code == 200
    assert "swagger" in docs.text.lower()


@pytest.mark.asyncio
async def test_production_health_endpoints_work() -> None:
    """Health endpoints still work in the production configuration."""
    prod_app = create_app(carworld_env="production")
    transport = httpx.ASGITransport(app=prod_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        health = await client.get("/health")
        v1_health = await client.get("/api/v1/health")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert v1_health.status_code == 200
    assert v1_health.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_development_health_endpoints_work() -> None:
    """Health endpoints still work in the development configuration."""
    dev_app = create_app(carworld_env="development")
    transport = httpx.ASGITransport(app=dev_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        health = await client.get("/health")
        v1_health = await client.get("/api/v1/health")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert v1_health.status_code == 200
    assert v1_health.json() == {"status": "ok"}
