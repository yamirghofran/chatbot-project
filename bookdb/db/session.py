import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Resolve .env relative to this file so it works regardless of CWD.
# Searches: <repo_root>/apps/api/.env, then <repo_root>/.env, then CWD/.env.
_repo_root = Path(__file__).resolve().parents[2]
for _candidate in [
    _repo_root / "apps" / "api" / ".env",
    _repo_root / ".env",
]:
    if _candidate.exists():
        load_dotenv(_candidate)
        break
else:
    load_dotenv()  # fallback to CWD


def _normalize_database_url(url: str) -> str:
    """Force psycopg v3 driver so SQLAlchemy does not try psycopg2."""
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql+psycopg://" + url[len("postgresql+psycopg2://") :]
    return url


def build_database_url() -> str:
    direct = os.getenv("DATABASE_URL")
    if direct:
        return _normalize_database_url(direct)

    user = os.getenv("DATABASE_USER", "app_user")
    password = os.getenv("DATABASE_PW", "app_pw")
    name = os.getenv("DATABASE_NAME", "app_db")
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "5433")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"


DATABASE_URL = build_database_url()

engine = create_engine(
    DATABASE_URL,
    future=True,
    connect_args={
        "keepalives": 1,
        "keepalives_idle": 10,
        "keepalives_interval": 5,
        "keepalives_count": 5,
        "connect_timeout": 30,
    },
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
