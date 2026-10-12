import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic import BaseModel, Field
from sqlalchemy.engine import URL, make_url


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    carworld_env: str = "development"
    database_url_override: str | None = None
    chat_rate_limit_per_minute: int = Field(default=30, gt=0)
    chat_rate_limit_per_day: int = Field(default=300, gt=0)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str = "carworld"
    supabase_url: str | None = None
    supabase_anon_key: str | None = None
    frontend_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:5500", "http://localhost:5500"]
    )
    groq_api_key: str | None = None
    tavily_api_key: str | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        render_service = bool(os.environ.get("RENDER_SERVICE_ID"))
        environment = os.environ.get(
            "CARWORLD_ENV", "production" if render_service else "development"
        ).strip().lower()
        database_url_value = os.environ.get("DATABASE_URL", "").strip() or None
        if (environment == "production" or render_service) and not database_url_value:
            raise ValueError(
                "DATABASE_URL is required in production; configure the Render PostgreSQL connection string."
            )

        normalized_database_url = None
        if database_url_value:
            try:
                parsed_url = make_url(database_url_value)
            except Exception:
                raise ValueError("DATABASE_URL must be a valid PostgreSQL URL.") from None

            if parsed_url.drivername.split("+", 1)[0] not in {"postgres", "postgresql"}:
                raise ValueError("DATABASE_URL must use PostgreSQL.")

            query = dict(parsed_url.query)
            ssl_mode = query.pop("sslmode", None)
            if ssl_mode is not None:
                query.setdefault("ssl", ssl_mode)
            normalized_database_url = parsed_url.set(
                drivername="postgresql+asyncpg",
                query=query,
            ).render_as_string(hide_password=False)

        default_minute_limit, default_day_limit = (
            (5, 40) if environment == "production" else (30, 300)
        )
        return cls(
            carworld_env=environment,
            database_url_override=normalized_database_url,
            chat_rate_limit_per_minute=int(os.environ.get(
                "CHAT_RATE_LIMIT_PER_MINUTE", default_minute_limit
            )),
            chat_rate_limit_per_day=int(os.environ.get(
                "CHAT_RATE_LIMIT_PER_DAY", default_day_limit
            )),
            postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
            postgres_port=os.environ.get("POSTGRES_PORT", "5432"),
            postgres_user=os.environ.get("POSTGRES_USER") or None,
            postgres_password=os.environ.get("POSTGRES_PASSWORD") or None,
            postgres_db=os.environ.get("POSTGRES_DB", "carworld"),
            supabase_url=os.environ.get("SUPABASE_URL") or None,
            supabase_anon_key=os.environ.get("SUPABASE_ANON_KEY") or None,
            frontend_origins=[origin.strip() for origin in os.environ.get(
                "FRONTEND_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
            ).split(",") if origin.strip()],
            groq_api_key=os.environ.get("GROQ_API_KEY") or None,
            tavily_api_key=os.environ.get("TAVILY_API_KEY") or None,
        )

    @property
    def database_url(self) -> URL:
        if self.database_url_override:
            return make_url(self.database_url_override)

        if self.carworld_env == "production":
            raise ValueError("DATABASE_URL is required in production.")

        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


settings = Settings.from_environment()
