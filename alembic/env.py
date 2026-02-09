import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

from bookdb.db.base import Base
from bookdb.db import models  # noqa

load_dotenv()

DATABASE_USER = os.getenv('DATABASE_USER')
DATABASE_PW = os.getenv('DATABASE_PW')
DATABASE_NAME = os.getenv('DATABASE_NAME')

if not all([DATABASE_USER, DATABASE_PW, DATABASE_NAME]):
    raise ValueError("DATABASE_USER, DATABASE_PW, and DATABASE_NAME must be set in environment variables")

DATABASE_URL = f"postgresql+psycopg://{DATABASE_USER}:{DATABASE_PW}@localhost:5433/{DATABASE_NAME}"

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

fileConfig(config.config_file_name)
target_metadata = Base.metadata

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()