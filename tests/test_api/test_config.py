from apps.api.core.config import Settings


def _build_settings(cors_origins: object) -> Settings:
    return Settings.model_validate(
        {
            "DATABASE_URL": "postgresql+psycopg://user:pw@localhost:5432/db",
            "JWT_SECRET": "a" * 32,
            "JWT_EXPIRE_MINUTES": 60,
            "CORS_ORIGINS": cors_origins,
            "QDRANT_URL": "http://localhost:6333",
        }
    )


def test_cors_origins_normalizes_json_and_trailing_slash():
    settings = _build_settings(
        '["https://bookdb.yamirghofran.workers.dev/","http://localhost:5173"]'
    )

    assert settings.CORS_ORIGINS == [
        "https://bookdb.yamirghofran.workers.dev",
        "http://localhost:5173",
    ]


def test_cors_origins_accepts_comma_separated_values():
    settings = _build_settings(
        "https://bookdb.yamirghofran.workers.dev/, http://localhost:5173 "
    )

    assert settings.CORS_ORIGINS == [
        "https://bookdb.yamirghofran.workers.dev",
        "http://localhost:5173",
    ]
