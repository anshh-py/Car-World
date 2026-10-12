from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.modules.auth import dependencies as auth_dependencies


@pytest_asyncio.fixture
async def client() -> AsyncClient:
	async with AsyncClient(
		transport=ASGITransport(app=app),
		base_url="http://test",
	) as test_client:
		yield test_client


@pytest.mark.asyncio
async def test_auth_config_exposes_only_browser_configuration(
	client: AsyncClient,
) -> None:
	response = await client.get("/api/v1/auth/config")

	assert response.status_code == 200
	assert set(response.json()) == {"url", "anon_key"}
	assert "service_role" not in response.text
	assert "secret" not in response.json()


@pytest.mark.asyncio
async def test_auth_me_without_token_returns_401(client: AsyncClient) -> None:
	response = await client.get("/api/v1/auth/me")

	assert response.status_code == 401
	assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_auth_me_with_malformed_token_returns_401(client: AsyncClient) -> None:
	response = await client.get(
		"/api/v1/auth/me",
		headers={"Authorization": "Bearer not-a-jwt"},
	)

	assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_me_with_supabase_rejected_token_returns_401(
	client: AsyncClient,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(
		auth_dependencies,
		"settings",
		SimpleNamespace(
			supabase_url="https://project.example.supabase.co",
			supabase_anon_key="public-test-key",
		),
	)

	async def rejected_token(_access_token: str) -> httpx.Response:
		return httpx.Response(
			401,
			json={"message": "Invalid token"},
			request=httpx.Request("GET", "https://project.example.supabase.co/auth/v1/user"),
		)

	monkeypatch.setattr(auth_dependencies, "request_supabase_user", rejected_token)
	response = await client.get(
		"/api/v1/auth/me",
		headers={"Authorization": "Bearer header.payload.signature"},
	)

	assert response.status_code == 401
	assert "signature" not in response.text


@pytest.mark.asyncio
async def test_auth_me_returns_verified_supabase_user(
	client: AsyncClient,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(
		auth_dependencies,
		"settings",
		SimpleNamespace(
			supabase_url="https://project.example.supabase.co",
			supabase_anon_key="public-test-key",
		),
	)

	async def accepted_token(_access_token: str) -> httpx.Response:
		return httpx.Response(
			200,
			json={
				"id": "verified-supabase-user-id",
				"email": "carworld-user@example.test",
				"role": "authenticated",
			},
			request=httpx.Request("GET", "https://project.example.supabase.co/auth/v1/user"),
		)

	monkeypatch.setattr(auth_dependencies, "request_supabase_user", accepted_token)
	response = await client.get(
		"/api/v1/auth/me",
		headers={"Authorization": "Bearer header.payload.signature"},
	)

	assert response.status_code == 200
	assert response.json() == {
		"id": "verified-supabase-user-id",
		"email": "carworld-user@example.test",
		"role": "authenticated",
	}
	assert "access_token" not in response.json()


@pytest.mark.asyncio
async def test_auth_me_without_supabase_configuration_returns_503(
	client: AsyncClient,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(
		auth_dependencies,
		"settings",
		SimpleNamespace(supabase_url=None, supabase_anon_key=None),
	)
	response = await client.get(
		"/api/v1/auth/me",
		headers={"Authorization": "Bearer header.payload.signature"},
	)

	assert response.status_code == 503