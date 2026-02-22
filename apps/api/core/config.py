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
    EMBEDDING_SERVICE_URL: str | None = None
    EMBEDDING_SERVICE_MODEL: str = "finetuned"
    EMBEDDING_SERVICE_TIMEOUT_SECONDS: float = Field(default=8.0, gt=0)
    EMBEDDING_SERVICE_API_KEY: str | None = None
    CHATBOT_TOP_K: int = Field(default=20, ge=1, le=100)
    CHATBOT_MAX_REVIEWS: int = Field(default=30, ge=0, le=200)
    CHATBOT_MAX_BOOKS: int = Field(default=6, ge=1, le=20)
    BPR_PARQUET_URL: str | None = None  # Local path or remote URL (http/https/s3/gs/az)
    BOOK_METRICS_PARQUET_URL: str | None = None  # Local path or remote URL (http/https/s3/gs/az)

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, value: str) -> str:
        if len(value.encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET must be at least 32 bytes for HS256.")
        return value


settings = Settings()
