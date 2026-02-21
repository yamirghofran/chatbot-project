---
icon: lucide/book-open
---

# BookDB Docs

This documentation covers the current PostgreSQL schema, migration workflow, and dataset import process.

## Quick Start

```bash
uv sync
cp .env.example .env
make db-up
make migrate
make import-data
```

## Documentation Map

- [Database setup and migration workflow](database-setup.md)
- [Current schema and dataset mapping](context.md)
- [Qdrant integration](qdrant.md)

## Current Data Pipeline

1. Apply Alembic migrations to create schema.
2. Run `scripts/import_goodreads_to_postgres.py` (via `make import-data`).
3. Import data from parquet datasets in `data/` into PostgreSQL tables.
4. Use `interactions` as transient input only (no `interactions` table persisted).
