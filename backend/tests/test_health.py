import httpx
import pytest

from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/health")
        root_health = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert root_health.status_code == 200
    assert root_health.json() == {"status": "ok"}
