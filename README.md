# BookDB

> **Find your next favourite book!**

BookDB is a full-stack book discovery platform that combines the best of collaborative filtering, semantic search, and conversational AI. Tell the chatbot what you're in the mood for, get personalised recommendations ranked by multiple ML models, and dive into community reviews... all in one place! Unlike generic recommendation engines that recycle the same bestsellers, BookDB learns from your reading history and taste profile to surface books you'll actually want to read, whether that's an obscure 1970s sci-fi novel or the latest literary fiction.

Built on 229 million Goodreads interactions and a catalogue of 2.3 million books, the system runs different recommendation strategies like BPR, and vector similarity, fusing their outputs with weighted scoring and RRF reranking so you always get the most relevant result. The AI chatbot understands natural language queries, rewrites them for better retrieval, and can explain _why_ it's recommending a book by pulling from real user reviews.

![BookDB Demo Video](https://pub-cb8b3df74f8941b7a14e2ba6346106cb.r2.dev/BF8C61A0-CleanShot%202026-04-21%20at%2012.48.09.mp4)

[![BookDB Demo]([https://github.com/user-attachments/assets/d2a8d58f-9988-451c-a46a-37d1a538392f](https://github.com/user-attachments/assets/9ba822b3-6495-4e13-9831-6da30ed7666b))]([https://assets.amirghofran.com/amirghofran-com-astro-public/bookdb-vid1.mp4](https://pub-cb8b3df74f8941b7a14e2ba6346106cb.r2.dev/BF8C61A0-CleanShot%202026-04-21%20at%2012.48.09.mp4))


Built on the [Goodreads dataset](https://mengtingwan.github.io/data/goodreads.html).

## What's in this repo

| Layer                             | What it does                                                                                 |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| **Data pipeline**                 | Marimo notebooks that clean, standardise, and aggregate raw Goodreads data                   |
| **ML models**                     | BPR, SAR, NCF training + tuning notebooks, exported to Parquet/MLflow                        |
| **`bookdb/` library**             | Shared Python library — data processing, EDA utilities, models, vector DB, validation        |
| **FastAPI backend** (`apps/api/`) | REST + SSE API for books, recommendations, reviews, auth, and chat                           |
| **React frontend** (`apps/web/`)  | Web UI for search, recommendations, and the AI chat interface                                |
| **AI chatbot**                    | LLM-powered chat with tool-routing (search, recommend, review RAG) via OpenAI-compatible API |
| **MCP server** (`mcp/`)           | Go-based Model Context Protocol server — exposes tools to Claude Desktop, Cursor, etc.       |

## Project structure

```
BookDB/
├── bookdb/                    # Python library
│   ├── datasets/              # Dataset loaders
│   ├── db/                    # SQLAlchemy models and DB utilities
│   ├── eda/                   # Shared EDA utilities (plots, stats)
│   ├── evaluation/            # Recommendation evaluation metrics
│   ├── models/                # BPR, SAR, NCF, embedding inference
│   ├── processing/            # Data processing helpers (book IDs, interactions, text)
│   ├── tuning/                # Hyperparameter tuning
│   ├── utils/                 # Shared utilities (paths, constants)
│   ├── validation/            # Input validation and data quality checks
│   └── vector_db/             # Qdrant vector database client
├── apps/
│   ├── api/                   # FastAPI backend
│   │   ├── routers/           # auth, books, chat, discovery, lists, reviews, users
│   │   ├── schemas/           # Pydantic request/response models
│   │   └── core/              # Config, DB session, auth middleware
│   └── web/                   # React + TypeScript frontend
├── notebooks/
│   └── data/
│       ├── eda/               # EDA notebooks (books, works, authors, interactions, reviews)
│       └── processing/        # Data pipeline notebooks
├── mcp/                       # Go MCP server
├── tests/                     # Python tests for bookdb library
├── alembic/                   # Database migrations
├── docker-compose.yml         # PostgreSQL + Qdrant services
└── pyproject.toml
```

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/) (for PostgreSQL and Qdrant)
- [Node.js 18+](https://nodejs.org/) (for the web frontend)
- [Go 1.24+](https://go.dev/dl/) (only if building the MCP server from source)

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/yamirghofran/BookDB.git
cd BookDB
uv sync
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable                                      | Description                                                                                                    |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`                                | PostgreSQL connection string, e.g. `postgresql+psycopg://user:pw@localhost:1234/bookdb`                        |
| `QDRANT_URL`                                  | Qdrant URL, e.g. `http://localhost:6333`                                                                       |
| `OPENAI_API_KEY`                              | OpenAI API key for the LLM chatbot                                                                             |
| `OPENAI_BASE_URL`                             | Base URL for the LLM provider (e.g., `https://api.groq.com` for Groq, `https://api.together.xyz` for Together) |
| `EMBEDDING_SERVICE_URL`                       | URL of the sentence-transformer embedding service                                                              |
| `HF_TOKEN`                                    | HuggingFace token (for model downloads)                                                                        |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 credentials (for DVC remote data)                                                                           |

### 3. Start infrastructure services

```bash
docker-compose up -d
```

This starts PostgreSQL (port 5432) and Qdrant (port 6333).

### 4. Run database migrations

```bash
uv run alembic upgrade head
```

### 5. Start the API

```bash
uv run uvicorn apps.api.main:app --reload --port 8000
```

### 6. Start the web frontend

```bash
cd apps/web
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

### 7. (Optional) Run data pipeline notebooks

If you have access to the raw Goodreads data, run the processing notebooks in order:

```bash
# Books pipeline
uv run marimo run notebooks/data/processing/books/1_clean_books.py
uv run marimo run notebooks/data/processing/books/2_standardize_book_ids.py
uv run marimo run notebooks/data/processing/books/3_aggregate_book_metrics.py

# Interactions pipeline
uv run marimo run notebooks/data/processing/interactions/1_merge_interactions_book_editions.py
# ... and so on
```

Or open them for editing:

```bash
uv run marimo edit notebooks/data/eda/books-works-authors/books_eda.py
```

### 8. (Optional) MCP server

One-line install:

```bash
curl -fsSL https://raw.githubusercontent.com/yamirghofran/BookDB/main/mcp/install.sh | bash
```

Or build from source:

```bash
cd mcp && go build -o bookdb-mcp .
```

See the [MCP section](#mcp-server) below for configuration.

## Running tests

```bash
uv run pytest
```

## MCP server

A Go-based [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server under `mcp/` exposes BookDB tools to LLM clients.

### Tools

| Tool                  | Description                                  |
| --------------------- | -------------------------------------------- |
| `search_books`        | Search books by title, author, or keyword    |
| `get_book`            | Get detailed book info and stats             |
| `get_related_books`   | Find semantically similar books              |
| `get_book_reviews`    | Read reviews for a book                      |
| `get_recommendations` | Personalised recommendations (requires auth) |
| `get_staff_picks`     | Curated well-rated books                     |
| `get_user_profile`    | Look up a user profile                       |
| `get_user_ratings`    | View a user's rated books                    |

### Claude Desktop config

```json
{
  "mcpServers": {
    "bookdb": {
      "command": "/path/to/bookdb-mcp",
      "env": {
        "BOOKDB_API_URL": "https://bookdb.up.railway.app",
        "BOOKDB_API_KEY": "your-jwt-token"
      }
    }
  }
}
```

Get a JWT token:

```bash
./bookdb-mcp login -email you@example.com
```

## Dataset citations

Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in _RecSys'18_.

Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in _ACL'19_.

## Development principles

**Notebooks call a library, not the other way around.** All reusable logic lives in `bookdb/` and is tested. Notebooks are for experimentation and visualisation only.

**Tests are the immune system.** Run `uv run pytest` before merging. The CI pipeline runs tests on every pull request.

**Start from something that works.** Rather than building from scratch, the project extends proven baselines from [Recommenders](https://github.com/recommenders-team/recommenders) and adapts them to the Goodreads dataset.
