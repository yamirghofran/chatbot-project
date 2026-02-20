# Qdrant Integration Documentation

This document covers how BookDB uses Qdrant for vector storage and retrieval.

## Overview

BookDB stores embeddings in three Qdrant collections:

- `books`
- `users`
- `reviews`

Collection lifecycle and defaults are managed by `bookdb/vector_db/collections.py`.

## Local Setup

### 1. Start Qdrant

```bash
make qdrant-up
```

Equivalent Docker command:

```bash
docker compose up -d qdrant
```

### 2. Check logs

```bash
make qdrant-logs
```

### 3. Reset local data (optional)

```bash
make qdrant-reset
```

## Configuration

Qdrant client config is loaded from environment variables in `bookdb/vector_db/config.py`.

```bash
QDRANT_MODE=server
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_HTTPS=false
QDRANT_PATH=./qdrant_data
QDRANT_TIMEOUT=10.0
QDRANT_VECTOR_SIZE=384
```

## Runtime API

Use `bookdb.vector_db` exports for client and collection access:

```python
from bookdb.vector_db import (
    get_qdrant_client,
    initialize_all_collections,
    CollectionNames,
)

client = get_qdrant_client()
manager = initialize_all_collections()
books = manager.get_collection(CollectionNames.BOOKS)
```

## CRUD Model

`BaseVectorCRUD` stores items as Qdrant points:

- point `id`: application id
- point `vector`: embedding (or zero-vector fallback if missing)
- point `payload.document`: source text
- point `payload.metadata`: structured metadata

Return shape for callers is preserved as:

- `id`
- `document`
- `metadata`
- `embedding`

## Metadata Contract

Vector metadata is intentionally aligned with SQL primary entities:

- `books` metadata:
  - none (books store only `document` + `embedding`)
- `users` metadata:
  - `user_id` (PostgreSQL `users.id`)
  - `name`
- `reviews` metadata:
  - `user_id` (PostgreSQL `users.id`)
  - `book_id` (PostgreSQL `books.id`)

Fields like `author`, `rating`, and ad-hoc vector timestamps are intentionally excluded
to avoid divergence from relational source-of-truth tables.

## Verification

Quick checks:

```bash
uv run pytest tests/test_vector_db -q
```
