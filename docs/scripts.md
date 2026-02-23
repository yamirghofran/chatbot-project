---
icon: lucide/terminal
---

# Scripts Reference

Utility scripts for data import, model training, embedding generation, and system operations.

## Data Import

### `scripts/import_goodreads_to_postgres.py`

Imports Goodreads parquet datasets into PostgreSQL.

**Usage:**

```bash
# Default import (all datasets)
uv run python scripts/import_goodreads_to_postgres.py

# With limits
uv run python scripts/import_goodreads_to_postgres.py --limit 5000 --batch-size 1000

# Skip specific datasets
uv run python scripts/import_goodreads_to_postgres.py --skip-reviews --skip-interactions
```

**Datasets:**

| File | Target Table |
|------|--------------|
| `data/raw_goodreads_book_authors.parquet` | `authors` |
| `data/3_goodreads_books_with_metrics.parquet` | `books`, `book_authors` |
| `data/3_goodreads_reviews_dedup_clean.parquet` | `reviews` |
| `data/3_goodreads_interactions_reduced.parquet` | `book_ratings`, `list_books` |

### `scripts/import_tags_from_shelves.py`

Extracts tags from Goodreads shelves and creates tag/book-tag relationships.

**Usage:**

```bash
uv run python scripts/import_tags_from_shelves.py
```

## Model Training

### `scripts/train_sar.py`

Trains the SAR (Simple Algorithm for Recommendations) model.

**Usage:**

```bash
uv run python scripts/train_sar.py
```

**Output:** Model artifacts in `models/sar/`

### `scripts/train_bpr.py`

Trains the BPR (Bayesian Personalized Ranking) model and generates recommendations.

**Usage:**

```bash
uv run python scripts/train_bpr.py
```

**Output:** Recommendations parquet at `data/bpr_model_predictions/bpr_recommendations.parquet`

## Embedding Generation

### `scripts/finetune_embeddinggemma.py`

Fine-tunes EmbeddingGemma on book data.

**Usage:**

```bash
uv run python scripts/finetune_embeddinggemma.py \
    --model_name google/embeddinggemma \
    --output_dir models/embeddinggemma_finetuned \
    --epochs 3
```

**Shell wrapper:**

```bash
bash scripts/run_embeddinggemma_finetuning.sh
```

### `scripts/generate_book_embeddings.py`

Generates embeddings for all books.

**Usage:**

```bash
uv run python scripts/generate_book_embeddings.py \
    --model_path models/embeddinggemma_finetuned \
    --input data/3_goodreads_books_with_metrics.parquet \
    --output data/book_embeddings.parquet
```

**Shell wrapper:**

```bash
bash scripts/run_generate_embeddings.sh
```

### `scripts/ingest_books_embeddings_to_qdrant.py`

Uploads book embeddings to Qdrant.

**Usage:**

```bash
uv run python scripts/ingest_books_embeddings_to_qdrant.py \
    --embeddings data/book_embeddings.parquet \
    --collection books
```

### `scripts/serve_embeddinggemma.py`

Starts HTTP embedding service.

**Usage:**

```bash
uv run python scripts/serve_embeddinggemma.py \
    --model_path models/embeddinggemma_finetuned \
    --port 8000
```

### `scripts/infer_embeddinggemma.py`

Quick inference test.

**Usage:**

```bash
uv run python scripts/infer_embeddinggemma.py \
    --model_path models/embeddinggemma_finetuned \
    --text "Your query text"
```

## Model Management

### `scripts/upload_model_to_huggingface.py`

Uploads model to HuggingFace Hub.

**Usage:**

```bash
uv run python scripts/upload_model_to_huggingface.py \
    --model_path models/embeddinggemma_finetuned \
    --repo_id username/model-name
```

## Chatbot Testing

### `scripts/try_chatbot_llm.py`

Interactive chatbot testing.

**Usage:**

```bash
uv run python scripts/try_chatbot_llm.py
```

## Utilities

### `scripts/update_user_names.py`

Updates user names in the database.

**Usage:**

```bash
uv run python scripts/update_user_names.py
```

### `scripts/check_qdrant_client.py`

Verifies Qdrant connection.

**Usage:**

```bash
uv run python scripts/check_qdrant_client.py
```

### `scripts/rag_books_test.py`

Tests RAG book search functionality.

**Usage:**

```bash
uv run python scripts/rag_books_test.py
```

## Makefile Commands

Convenient aliases for common operations:

| Command | Description |
|---------|-------------|
| `make db-up` | Start PostgreSQL |
| `make db-down` | Stop PostgreSQL |
| `make db-reset` | Reset PostgreSQL volume |
| `make migrate` | Apply Alembic migrations |
| `make make-migration msg="..."` | Create new migration |
| `make import-data` | Import parquet datasets |
| `make seed` | Seed database |
| `make setup` | Start DB and migrate |
| `make qdrant-up` | Start Qdrant |
| `make qdrant-down` | Stop Qdrant |
| `make qdrant-reset` | Reset Qdrant volume |
| `make qdrant-logs` | View Qdrant logs |
