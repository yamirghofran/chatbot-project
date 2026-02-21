import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()


def build_database_url() -> str:
    direct = os.getenv("DATABASE_URL")
    if direct:
        return direct

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
