from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    DATABASE_URL: str
    JWT_SECRET: str
    JWT_EXPIRE_MINUTES: int
    CORS_ORIGINS: list[str]
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_TIMEOUT_SECONDS: float = Field(default=8.0, gt=0)
    BPR_PARQUET_URL: str | None = None  # Local path or remote URL (http/https/s3/gs/az)
    BOOK_METRICS_PARQUET_URL: str | None = None  # Local path or remote URL (http/https/s3/gs/az)

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, value: str) -> str:
        if len(value.encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET must be at least 32 bytes for HS256.")
        return value


settings = Settings()
