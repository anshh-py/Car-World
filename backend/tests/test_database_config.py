import pytest

from app.core.config import Settings


def clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
	for name in (
		"DATABASE_URL",
		"POSTGRES_HOST",
		"POSTGRES_PORT",
		"POSTGRES_USER",
		"POSTGRES_PASSWORD",
		"POSTGRES_DB",
		"RENDER_SERVICE_ID",
	):
		monkeypatch.delenv(name, raising=False)


def test_production_uses_render_database_url_and_asyncpg(monkeypatch: pytest.MonkeyPatch) -> None:
	clear_database_environment(monkeypatch)
	monkeypatch.setenv("CARWORLD_ENV", "production")
	monkeypatch.setenv(
		"DATABASE_URL",
		"postgres://render_user:do-not-log@pg-internal.example:5432/carworld?sslmode=require",
	)
	monkeypatch.setenv("POSTGRES_HOST", "localhost")

	url = Settings.from_environment().database_url

	assert url.drivername == "postgresql+asyncpg"
	assert url.host == "pg-internal.example"
	assert url.database == "carworld"
	assert url.query == {"ssl": "require"}
	assert "do-not-log" not in url.render_as_string(hide_password=True)


def test_production_without_database_url_fails_instead_of_using_localhost(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	clear_database_environment(monkeypatch)
	monkeypatch.setenv("CARWORLD_ENV", "production")
	monkeypatch.setenv("POSTGRES_HOST", "localhost")

	with pytest.raises(ValueError, match="DATABASE_URL is required in production"):
		Settings.from_environment()


def test_render_environment_requires_database_url_even_without_carworld_env(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	clear_database_environment(monkeypatch)
	monkeypatch.setenv("RENDER_SERVICE_ID", "srv-test")

	with pytest.raises(ValueError, match="DATABASE_URL is required in production"):
		Settings.from_environment()


def test_local_development_keeps_postgres_host_configuration(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	clear_database_environment(monkeypatch)
	monkeypatch.setenv("CARWORLD_ENV", "development")
	monkeypatch.setenv("POSTGRES_HOST", "localhost")
	monkeypatch.setenv("POSTGRES_PORT", "5433")
	monkeypatch.setenv("POSTGRES_USER", "local_user")
	monkeypatch.setenv("POSTGRES_PASSWORD", "local_password")
	monkeypatch.setenv("POSTGRES_DB", "carworld_dev")

	url = Settings.from_environment().database_url

	assert url.drivername == "postgresql+asyncpg"
	assert url.host == "localhost"
	assert url.port == 5433
	assert url.username == "local_user"
	assert url.database == "carworld_dev"
