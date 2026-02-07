.PHONY: help install dev db-up db-down db-reset migrate make-migration seed setup chroma-up chroma-down chroma-reset chroma-logs

help:
	@echo "Available commands:"
	@echo "  make db-up             - Start Postgres with Docker"
	@echo "  make db-down           - Stop Postgres"
	@echo "  make db-reset          - Reset Postgres volume"
	@echo "  make migrate           - Apply migrations"
	@echo "  make make-migration    - Create a migration (msg=...)"
	@echo "  make seed              - Seed the database"
	@echo "  make setup             - Bring up DB, migrate, seed"
	@echo "  make chroma-up         - Start ChromaDB with Docker"
	@echo "  make chroma-down       - Stop ChromaDB"
	@echo "  make chroma-reset      - Reset ChromaDB volume"
	@echo "  make chroma-logs       - View ChromaDB logs"

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

migrate:
	alembic upgrade head

make-migration:
	alembic revision --autogenerate -m "$(msg)"

seed:
	python app/seed.py

setup:
	make db-up
	make migrate
	make seed

chroma-up:
	docker compose up -d chromadb

chroma-down:
	docker compose stop chromadb

chroma-reset:
	docker compose down chromadb
	docker volume rm bookdb_chroma_data || true
	docker compose up -d chromadb

chroma-logs:
	docker compose logs -f chromadb