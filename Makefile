.PHONY: help install dev db-up db-down db-reset migrate make-migration seed setup

help:
	@echo "Available commands:"
	@echo "  make db-up             - Start Postgres with Docker"
	@echo "  make db-down           - Stop Postgres"
	@echo "  make db-reset          - Reset Postgres volume"
	@echo "  make migrate           - Apply migrations"
	@echo "  make make-migration    - Create a migration (msg=...)"
	@echo "  make seed              - Seed the database"
	@echo "  make setup             - Bring up DB, migrate, seed"

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

seed:
	uv run python -m bookdb.seed

setup:
	make db-up
	@echo "Waiting for database to be ready..."
	@sleep 3
	make migrate
	make seed