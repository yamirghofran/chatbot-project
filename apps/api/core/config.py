from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    DATABASE_URL: str
    JWT_SECRET: str
    JWT_EXPIRE_MINUTES: int
    CORS_ORIGINS: list[str]
    QDRANT_URL: str
    QDRANT_API_KEY: str
    BPR_PARQUET_URL: str | None = None  # Local path or remote URL (http/https/s3/gs/az)


settings = Settings()
