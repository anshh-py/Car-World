import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic import BaseModel, Field
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
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
        environment = os.environ.get("CARWORLD_ENV", "development").strip().lower()
        default_minute_limit, default_day_limit = (
            (5, 40) if environment == "production" else (30, 300)
        )
        return cls(
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
        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


settings = Settings.from_environment()
