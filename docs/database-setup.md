# Local PostgreSQL Setup Guide

This guide covers setting up the local PostgreSQL database using Docker and managing migrations with Alembic.

## Prerequisites

- Docker and Docker Compose installed
- Python environment with `uv` package manager
- Project dependencies installed (`uv sync`)

## Quick Start

Run the complete setup with a single command:

```bash
make setup
```

This will:
1. Start the PostgreSQL container
2. Wait for the database to be ready
3. Run all migrations
4. Seed initial data

## Database Configuration

The database connection is configured via environment variables. Copy the example file:

```bash
cp .env.example .env
```

Default configuration (in `.env`):
```
DATABASE_URL=postgresql+psycopg://app_user:app_password@localhost:5433/app_db
```

Note: I used port **5433** to avoid conflicts with my other local dbs

## Make Commands

| Command | Description |
|---------|-------------|
| `make db-up` | Start the PostgreSQL container |
| `make db-down` | Stop the PostgreSQL container |
| `make db-reset` | Reset database (deletes all data) |
| `make migrate` | Apply pending migrations |
| `make make-migration msg="description"` | Create a new migration |
| `make seed` | Seed initial data |
| `make setup` | Full setup (db-up + migrate + seed) |

## Step-by-Step Setup

### 1. Start the Database

```bash
make db-up
```

This starts a PostgreSQL 16 container named `local-postgres`. Verify it's running:

```bash
docker ps
```

### 2. Run Migrations

Apply all database migrations:

```bash
make migrate
```

### 3. Verify the Setup

Connect to the database and check tables:

```bash
docker exec -it local-postgres psql -U app_user -d app_db -c "\dt"
```

Expected output:
```
 Schema |      Name       | Type  |  Owner
--------+-----------------+-------+----------
 public | alembic_version | table | app_user
 public | authors         | table | app_user
 public | book_authors    | table | app_user
 public | books           | table | app_user
 public | list_books      | table | app_user
 public | lists           | table | app_user
 public | users           | table | app_user
```

## Creating Migrations

When you modify models in `bookdb/db/models.py`, create a new migration:

```bash
make make-migration msg="add rating column to books"
```

This auto-generates a migration file in `alembic/versions/`. Review the generated file, then apply:

```bash
make migrate
```

### Migration Best Practices

1. **Review generated migrations** - Alembic's autogenerate is helpful but not perfect
2. **Test migrations locally** before committing
3. **Use descriptive messages** for migration names
4. **Keep migrations small** - one logical change per migration

## Resetting the Database

To completely reset the database (deletes all data):

```bash
make db-reset
make migrate
make seed
```

Or use the combined command:

```bash
make db-reset && make migrate && make seed
```

## Troubleshooting

### Port Conflict

If you see an error about port 5432/5433 being in use:

```bash
# Check what's using the port
lsof -i :5433

# Stop any conflicting services or change the port in docker-compose.yml
```

### Connection Refused

If migrations fail with "connection refused":

1. Check the container is running: `docker ps`
2. Wait a few seconds for PostgreSQL to initialize
3. Verify the port in `.env` matches `docker-compose.yml`

### Permission Denied

If you get permission errors:

```bash
# Reset with fresh volume
make db-reset
```

## Accessing the Database

### Via Docker

```bash
docker exec -it local-postgres psql -U app_user -d app_db
```

### Via Python

```python
from bookdb.db.session import SessionLocal

session = SessionLocal()
# ... do queries ...
session.close()
```

### Connection String

For external tools (DBeaver, pgAdmin, etc.):

```
Host: localhost
Port: 5433
Database: app_db
User: app_user
Password: app_password
```

## Schema Overview

```
users
├── id (PK)
├── email (unique)
├── name
└── created_at

authors
├── id (PK)
├── name (indexed)
└── created_at

books
├── id (PK)
├── title (indexed)
├── pages_number
├── publisher_name
├── publish_day/month/year
├── num_reviews
├── rating_dist_1..5
├── rating_dist_total
└── created_at

book_authors (junction table)
├── book_id (FK → books)
└── author_id (FK → authors)

lists
├── id (PK)
├── name
├── user_id (FK → users, indexed)
└── created_at

list_books (junction table)
├── list_id (FK → lists)
└── book_id (FK → books)
```
