.PHONY: help install dev db-up db-down db-reset migrate make-migration import-data seed setup qdrant-up qdrant-down qdrant-reset qdrant-logs

help:
	@echo "Available commands:"
	@echo "  make db-up             - Start Postgres with Docker"
	@echo "  make db-down           - Stop Postgres"
	@echo "  make db-reset          - Reset Postgres volume"
	@echo "  make migrate           - Apply migrations"
	@echo "  make make-migration    - Create a migration (msg=...)"
	@echo "  make import-data       - Import parquet datasets into Postgres"
	@echo "  make seed              - Seed the database"
	@echo "  make setup             - Bring up DB and migrate"
	@echo "  make qdrant-up         - Start Qdrant with Docker"
	@echo "  make qdrant-down       - Stop Qdrant"
	@echo "  make qdrant-reset      - Reset Qdrant volume"
	@echo "  make qdrant-logs       - View Qdrant logs"

install:
	uv sync

dev:
	uv sync --all-extras

db-up:
	docker compose up -d

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d
	@echo "Waiting for database to be ready..."
	@sleep 3

migrate:
	uv run alembic upgrade head

make-migration:
	uv run alembic revision --autogenerate -m "$(msg)"

import-data:
	uv run python scripts/import_goodreads_to_postgres.py

seed:
	uv run python -m bookdb.seed

setup:
	make db-up
	@echo "Waiting for database to be ready..."
	@sleep 3
	make migrate

qdrant-up:
	docker compose up -d qdrant

qdrant-down:
	docker compose stop qdrant

qdrant-reset:
	docker compose stop qdrant || true
	docker compose rm -f qdrant || true
	docker volume rm bookdb_qdrant_data || true
	docker compose up -d qdrant

qdrant-logs:
	docker compose logs -f qdrant
