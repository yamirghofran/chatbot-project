.PHONY: help install dev db-up db-down db-reset migrate make-migration seed setup

help:
	@echo "Available commands:"
	@echo "  make install           - Install dependencies"
	@echo "  make dev               - Install dev dependencies"
	@echo "  make db-up             - Start MySQL with Docker"
	@echo "  make db-down           - Stop MySQL"
	@echo "  make db-reset          - Reset MySQL volume"
	@echo "  make migrate           - Apply migrations"
	@echo "  make make-migration    - Create a new migration (msg=...)"
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

migrate:
	alembic upgrade head

make-migration:
	alembic revision --autogenerate -m "$(msg)"

seed:
	python bookdb/seed.py

setup:
	make db-up
	make migrate
	make seed