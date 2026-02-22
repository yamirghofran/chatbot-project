import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class _ExcludedCorsEnvSettingsSource(EnvSettingsSource):
    def _extract_field_info(
        self, field: Any, field_name: str
    ) -> list[tuple[str, str, bool]]:
        if field_name == "CORS_ORIGINS":
            return []
        return super()._extract_field_info(field, field_name)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        if field_name == "CORS_ORIGINS":
            return None, field_name, False
        return super().get_field_value(field, field_name)

    def _get_resolved_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        if field_name == "CORS_ORIGINS":
            return None, field_name, False
        return super()._get_resolved_field_value(field, field_name)


class _CorsSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)

    def get_field_value(self, field_name: str, field: Field) -> Any:
        if field_name == "CORS_ORIGINS":
            cors_val = os.environ.get("CORS_ORIGINS")
            if cors_val is not None:
                return cors_val, True, False
        return None, False, False

    def __call__(self) -> dict[str, Any]:
        cors_val = os.environ.get("CORS_ORIGINS")
        if cors_val is not None:
            return {"CORS_ORIGINS": cors_val}
        return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    DATABASE_URL: str
    JWT_SECRET: str
    JWT_EXPIRE_MINUTES: int
    CORS_ORIGINS: list[str]
    QDRANT_URL: str | None = None
    QDRANT_PORT: int | None = None
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _CorsSettingsSource(settings_cls),
            _ExcludedCorsEnvSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                raise ValueError("CORS_ORIGINS cannot be empty.")
            try:
                parsed = json.loads(raw) if raw.startswith("[") else raw.split(",")
            except json.JSONDecodeError:
                parsed = raw.split(",")
            value = parsed

        if not isinstance(value, list):
            raise ValueError("CORS_ORIGINS must be a list or comma-separated string.")

        normalized: list[str] = []
        for origin in value:
            if not isinstance(origin, str):
                raise ValueError("CORS_ORIGINS entries must be strings.")
            cleaned = origin.strip().strip('[]"\'').rstrip("/")
            if cleaned:
                normalized.append(cleaned)

        if not normalized:
            raise ValueError("CORS_ORIGINS must include at least one origin.")

        # Keep order while removing duplicates.
        return list(dict.fromkeys(normalized))

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, value: str) -> str:
        if len(value.encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET must be at least 32 bytes for HS256.")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class _LazySettings:
    def __getattr__(self, item: str) -> Any:
        return getattr(get_settings(), item)


settings = cast(Settings, _LazySettings())
