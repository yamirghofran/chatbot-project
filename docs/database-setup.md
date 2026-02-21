# PostgreSQL Setup, Migration, and Import Workflow

This guide is the source of truth for starting BookDB locally with the current schema and importing Goodreads parquet datasets.

## Prerequisites

- Docker + Docker Compose
- Python + `uv`
- Project checked out locally

## 1. Install Dependencies

```bash
uv sync
```

## 2. Configure Environment

Create `.env` (if missing):

```bash
cp .env.example .env
```

Default local database connection uses:

- host: `localhost`
- port: `5433`
- database: `app_db`
- user: `app_user`

## 3. Start Postgres

```bash
make db-up
```

To confirm:

```bash
docker compose ps
```

## 4. Apply Migrations

```bash
make migrate
```

This runs:

```bash
uv run alembic upgrade head
```

## 5. Import Dataset Files

Run the importer:

```bash
make import-data
```

This executes:

```bash
uv run python scripts/import_goodreads_to_postgres.py
```

### Import script defaults

- `data/raw_goodreads_book_authors.parquet`
- `data/3_goodreads_books_with_metrics.parquet`
- `data/3_goodreads_reviews_dedup_clean.parquet`
- `data/3_goodreads_interactions_reduced.parquet`

### Optional import flags

```bash
uv run python scripts/import_goodreads_to_postgres.py --limit 5000 --batch-size 1000
uv run python scripts/import_goodreads_to_postgres.py --skip-reviews --skip-interactions
```

## 6. Verify Tables

```bash
docker exec -it local-postgres psql -U app_user -d app_db -c "\dt"
```

Expected tables:

- `alembic_version`
- `users`
- `authors`
- `books`
- `book_authors`
- `lists`
- `list_books`
- `shells`
- `shell_books`
- `book_ratings`
- `reviews`
- `review_comments`
- `review_likes`

## 7. Useful Commands

- `make db-down`: stop Postgres
- `make db-reset`: reset docker volume and start fresh Postgres
- `make setup`: start DB and run migrations
- `make make-migration msg="..."`: generate migration from model changes

## Clean Reset From Scratch

Use this when you want an empty database and then re-import everything:

```bash
make db-reset
make migrate
make import-data
```

## Current Workflow for Schema Changes

1. Edit SQLAlchemy models in `bookdb/db/models.py`.
2. Generate migration:
   ```bash
   make make-migration msg="describe change"
   ```
3. Review migration in `alembic/versions/`.
4. Apply migration:
   ```bash
   make migrate
   ```
5. Re-run importer if schema changes affect ingest:
   ```bash
   make import-data
   ```

## Notes About ID Strategy

- `users` and `authors` use internal auto-increment `id` as PK.
- Both have unique `goodreads_id` populated from datasets.
- Importer resolves dataset IDs (`goodreads_id`) to internal PKs before writing FK rows.
