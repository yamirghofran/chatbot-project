# BookDB

A book database platform with ML-powered recommendations, vector search, and chatbot functionality. Built for discovering, organizing, and sharing books.

## Features

- **Book Database**: Comprehensive PostgreSQL database for books, authors, users, reviews, and ratings
- **Personalized Recommendations**: ML-powered recommendation engine using SAR and BPR algorithms
- **Vector Search**: Semantic book search using Qdrant with fine-tuned embeddings
- **Chatbot**: AI-powered book discovery assistant using RAG (Retrieval-Augmented Generation)
- **Modern Web App**: React SPA with TypeScript and shadcn/ui components
- **REST API**: FastAPI backend with authentication and comprehensive endpoints

## Tech Stack

| Layer                | Technologies                                                                            |
| -------------------- | --------------------------------------------------------------------------------------- |
| **Backend**          | Python 3.9+, FastAPI, Uvicorn, SQLAlchemy                                               |
| **Database**         | PostgreSQL 16, Alembic (migrations)                                                     |
| **Vector DB**        | Qdrant                                                                                  |
| **ML/AI**            | SAR, BPR, Sentence Transformers, EmbeddingGemma, MLflow                                 |
| **Frontend**         | React 19, TypeScript, Vite, TanStack Router, TanStack Query, shadcn/ui, Tailwind CSS v4 |
| **Data Processing**  | Polars, DuckDB, PyArrow                                                                 |
| **Chatbot**          | Groq API, RAG with Qdrant                                                               |
| **Package Managers** | uv (Python), Bun (JavaScript)                                                           |

## Project Structure

```
bookdb/
├── apps/
│   ├── api/              # FastAPI backend application
│   │   ├── core/         # Config, dependencies, embeddings
│   │   ├── routers/      # API endpoints (auth, books, discovery, etc.)
│   │   └── schemas/      # Pydantic models
│   ├── web/              # React web application
│   │   └── src/          # Components, routes, lib
│   └── coming-soon/      # Astro static landing page
├── bookdb/               # Core Python library
│   ├── db/               # SQLAlchemy models and CRUD operations
│   ├── vector_db/        # Qdrant client and vector CRUD
│   ├── models/           # ML models (SAR, BPR, embeddings, chatbot)
│   ├── evaluation/       # Recommendation metrics
│   ├── datasets/         # DataFrame utilities (Polars, Pandas)
│   ├── tuning/           # Hyperparameter tuning
│   └── utils/            # Constants and utilities
├── scripts/              # Utility scripts
│   ├── import_goodreads_to_postgres.py
│   ├── train_sar.py
│   ├── train_bpr.py
│   ├── generate_book_embeddings.py
│   ├── finetune_embeddinggemma.py
│   └── ...
├── alembic/              # Database migrations
│   └── versions/
├── tests/                # Test suites
│   ├── test_api/
│   ├── test_db/
│   └── test_vector_db/
├── chatbots/             # RAG implementations
│   └── backend/
├── examples/             # Notebook examples
│   ├── marimo/
│   └── models/
├── notebooks/            # EDA notebooks
├── docs/                 # Documentation
└── data/                 # Data files (DVC tracked)
```

## Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker and Docker Compose
- Bun (for web app development)

### Setup

1. **Clone and install dependencies:**

```bash
git clone <repository-url>
cd bookdb
uv sync
```

2. **Configure environment:**

```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start the database:**

```bash
make db-up
```

4. **Run migrations:**

```bash
make migrate
```

5. **Import data:**

```bash
make import-data
```

6. **Start Qdrant (for vector search):**

```bash
make qdrant-up
```

### All at once:

```bash
make setup  # Starts DB and runs migrations
```

## Development

### Available Makefile Commands

| Command                         | Description                             |
| ------------------------------- | --------------------------------------- |
| `make install`                  | Install dependencies with uv            |
| `make dev`                      | Install all extras                      |
| `make db-up`                    | Start PostgreSQL with Docker            |
| `make db-down`                  | Stop PostgreSQL                         |
| `make db-reset`                 | Reset PostgreSQL volume                 |
| `make migrate`                  | Apply Alembic migrations                |
| `make make-migration msg="..."` | Create a new migration                  |
| `make import-data`              | Import parquet datasets into PostgreSQL |
| `make seed`                     | Seed the database                       |
| `make setup`                    | Bring up DB and migrate                 |
| `make qdrant-up`                | Start Qdrant with Docker                |
| `make qdrant-down`              | Stop Qdrant                             |
| `make qdrant-reset`             | Reset Qdrant volume                     |

### Running the API

```bash
uv run uvicorn apps.api.main:app --reload
```

API will be available at `http://localhost:8000`

### Running the Web App

```bash
cd apps/web
bun install
bun run dev
```

Web app will be available at `http://localhost:5173`

### Running Tests

```bash
uv run pytest
```

## API Endpoints

| Router       | Description                      |
| ------------ | -------------------------------- |
| `/auth`      | Authentication (login, register) |
| `/books`     | Book CRUD operations             |
| `/reviews`   | Review management                |
| `/users`     | User profiles                    |
| `/lists`     | User book lists                  |
| `/me`        | Current user operations          |
| `/discovery` | Recommendations and discovery    |

### Key Discovery Endpoints

- `GET /discovery/recommendations` - Personalized book recommendations
- `GET /discovery/staff-picks` - Curated staff picks
- `GET /discovery/activity` - Activity feed

## Database Schema

The database includes the following core models:

- **Users**: User accounts with authentication
- **Authors**: Book author information
- **Books**: Book metadata (title, description, ISBN, etc.)
- **Book Authors**: Many-to-many relationship
- **Tags / Book Tags**: Book categorization
- **Lists / List Books**: User-created book lists
- **Shells / Shell Books**: User reading shells (personal collections)
- **Book Ratings**: User ratings (1-5 stars)
- **Reviews / Review Comments / Review Likes**: Review system

## ML & Recommendations

### Recommendation Algorithms

- **SAR (Simple Algorithm for Recommendations)**: Collaborative filtering based on user transaction history
- **BPR (Bayesian Personalized Ranking)**: Matrix factorization for personalized ranking

### Embedding Models

- Fine-tuned EmbeddingGemma for book semantic search
- Training scripts available in `scripts/finetune_embeddinggemma.py`

### Vector Search

- Qdrant-powered semantic search for books and reviews
- Embedding generation in `bookdb/vector_db/`

### Training Models

```bash
# Train SAR model
uv run python scripts/train_sar.py

# Train BPR model
uv run python scripts/train_bpr.py

# Generate book embeddings
uv run python scripts/generate_book_embeddings.py
```

## Chatbot

The RAG-based chatbot helps users discover books through natural language queries:

- Query rewriting using Groq LLM
- Semantic search in Qdrant vector database
- Book and review retrieval

```bash
# Test the chatbot LLM
uv run python scripts/try_chatbot_llm.py
```

## Web Application

Built with modern React ecosystem:

- **React 19** with TypeScript
- **Vite** for fast development
- **TanStack Router** for file-based routing
- **TanStack Query** for data fetching
- **shadcn/ui** for components
- **Tailwind CSS v4** for styling
- **Storybook** for component development

See `apps/web/README.md` for more details.

## Data Pipeline

1. Raw Goodreads data stored as parquet files in `data/`
2. Import script processes and loads data into PostgreSQL
3. Embeddings generated and stored in Qdrant
4. BPR model predictions cached as parquet

## Documentation

Documentation is built with Zensical. Key docs:

- `docs/database-setup.md` - Database setup guide
- `docs/context.md` - Schema and dataset mapping
- `docs/qdrant.md` - Qdrant integration

## Contributing

1. Work on feature branches
2. Submit changes via Pull Requests
3. Ensure tests pass
4. Follow the coding principles in AGENTS.md

## License

MIT License - see LICENSE file for details.
