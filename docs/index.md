---
icon: lucide/book-open
---

# BookDB Documentation

BookDB is a recommendation engine and social platform for books. This documentation covers the architecture, database schema, API, ML models, and deployment workflows.

## Project Overview

BookDB combines:

- **PostgreSQL database** for relational data (users, books, authors, reviews, ratings, lists)
- **Qdrant vector database** for semantic search and similarity-based recommendations
- **FastAPI REST API** for web and mobile clients
- **React SPA** for the user interface
- **ML recommendation models** (SAR, BPR) for personalized recommendations
- **LLM-powered chatbot** for natural language book discovery

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env

# Start PostgreSQL
make db-up

# Run migrations
make migrate

# Import dataset
make import-data

# Start Qdrant (for vector search)
make qdrant-up

# Start the API
uv run uvicorn apps.api.main:app --reload

# Start the web app (in another terminal)
cd apps/web && bun run dev
```

## Architecture

```
bookdb/
├── apps/
│   ├── api/           # FastAPI REST API
│   ├── web/           # React 19 SPA (Vite, TypeScript, TanStack)
│   └── coming-soon/   # Astro static landing page
├── bookdb/            # Python package
│   ├── db/            # SQLAlchemy models, session, CRUD
│   ├── models/        # ML models (SAR, BPR, embeddings, chatbot)
│   ├── vector_db/     # Qdrant integration
│   ├── evaluation/    # Recommender metrics (precision, recall, NDCG)
│   └── utils/         # Utilities
├── scripts/           # Training, inference, and data scripts
├── alembic/           # Database migrations
├── chatbots/          # RAG chatbot implementations
├── notebooks/         # Marimo notebooks for experimentation
└── docs/              # This documentation
```

## Documentation Map

### Core

- [Database Setup](database-setup.md) - PostgreSQL, Alembic migrations, data import
- [Schema & Dataset Mapping](context.md) - Database tables and parquet import mapping
- [Qdrant Integration](qdrant.md) - Vector database configuration and usage

### Applications

- [REST API](api.md) - FastAPI endpoints, authentication, search
- [Web App](web-app.md) - React SPA architecture and development

### ML & AI

- [Recommendation Models](ml-recommendations.md) - SAR, BPR, and recommendation pipelines
- [Embedding Models](embeddings.md) - EmbeddingGemma fine-tuning and inference
- [Chatbots & RAG](chatbots.md) - LLM-powered book discovery

### Operations

- [Scripts Reference](scripts.md) - Training, inference, and utility scripts

## Current Data Pipeline

1. **Apply migrations**: `make migrate` (Alembic)
2. **Import data**: `make import-data` (parquet to PostgreSQL)
3. **Generate embeddings**: `scripts/generate_book_embeddings.py`
4. **Ingest to Qdrant**: `scripts/ingest_books_embeddings_to_qdrant.py`
5. **Train models**: `scripts/train_sar.py`, `scripts/train_bpr.py`

## Key Technologies

| Component | Technology |
|-----------|------------|
| Database | PostgreSQL 15, SQLAlchemy 2.0, Alembic |
| Vector DB | Qdrant |
| API | FastAPI, Pydantic, python-jose (JWT) |
| Web | React 19, Vite, TypeScript, TanStack Router/Query, shadcn/ui, Tailwind CSS v4 |
| ML | PyTorch, sentence-transformers, implicit, polars |
| LLM | Groq API (Llama, Kimi) |
| Package Manager | uv (Python), bun (JS/TS) |
| Notebooks | Marimo |
