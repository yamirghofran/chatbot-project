"""BookDB FastAPI application.

Run with:
    uvicorn apps.api.main:app --reload

Production-friendly entrypoint (uses PORT env fallback):
    python -m apps.api.main
"""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.embeddings import COLLECTION_NAME, get_qdrant_client
from .routers import auth, books, discovery, lists, me, reviews, users

app = FastAPI(title="BookDB API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(books.router)
app.include_router(reviews.router)
app.include_router(users.router)
app.include_router(lists.router)
app.include_router(me.router)
app.include_router(discovery.router)


@app.on_event("startup")
async def startup_event():
    print(f"CORS origins: {settings.CORS_ORIGINS}")

    try:
        qdrant = get_qdrant_client(
            settings.QDRANT_URL,
            settings.QDRANT_PORT,
            settings.QDRANT_API_KEY,
            timeout_seconds=settings.QDRANT_TIMEOUT_SECONDS,
        )
        qdrant.get_collection(COLLECTION_NAME)
        app.state.qdrant = qdrant
        print(f"Qdrant connected: {settings.QDRANT_URL} ({COLLECTION_NAME})")
    except Exception as e:
        app.state.qdrant = None
        print(f"WARNING: Qdrant unavailable: {e}. Related-books endpoint will use fallback.")

    # Use configured URL/path, or fall back to default local path
    bpr_path = settings.BPR_PARQUET_URL
    if bpr_path is None:
        bpr_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "bpr_model_predictions", "bpr_recommendations.parquet",
        )

    # Check existence for local paths; skip for remote URLs (DuckDB handles errors)
    is_remote = bpr_path.startswith(("http://", "https://", "s3://", "gs://", "az://"))
    if is_remote or os.path.exists(bpr_path):
        app.state.bpr_parquet_path = bpr_path
        print(f"BPR recommendations loaded: {bpr_path}")
    else:
        app.state.bpr_parquet_path = None
        print("WARNING: BPR recommendations parquet not found, will use cold-start fallback.")

    metrics_path = settings.BOOK_METRICS_PARQUET_URL
    if metrics_path is None:
        metrics_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "3_goodreads_books_with_metrics.parquet",
        )

    is_metrics_remote = metrics_path.startswith(("http://", "https://", "s3://", "gs://", "az://"))
    if is_metrics_remote or os.path.exists(metrics_path):
        app.state.book_metrics_parquet_path = metrics_path
        print(f"Book metrics parquet loaded: {metrics_path}")
    else:
        app.state.book_metrics_parquet_path = None
        print("WARNING: Book metrics parquet not found, popularity/ranking fallbacks will use Postgres.")


@app.get("/")
def health_check():
    return {"status": "ok", "service": "BookDB API"}


def _get_cli_arg(argv: list[str], flag: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _resolve_port(argv: list[str]) -> int:
    cli_port = _get_cli_arg(argv, "--port")
    if cli_port is not None:
        return int(cli_port)

    env_port = os.getenv("PORT")
    if env_port:
        return int(env_port)

    return 8000


def _resolve_host(argv: list[str]) -> str:
    cli_host = _get_cli_arg(argv, "--host")
    if cli_host is not None:
        return cli_host
    return os.getenv("HOST", "0.0.0.0")


if __name__ == "__main__":
    import uvicorn

    args = sys.argv[1:]
    uvicorn.run(
        "apps.api.main:app",
        host=_resolve_host(args),
        port=_resolve_port(args),
        reload="--reload" in args,
    )
