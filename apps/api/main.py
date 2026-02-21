"""BookDB FastAPI application.

Run with:
    uvicorn apps.api.main:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.embeddings import get_qdrant_client
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
    app.state.qdrant = get_qdrant_client(settings.QDRANT_URL, settings.QDRANT_API_KEY)

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


@app.get("/")
def health_check():
    return {"status": "ok", "service": "BookDB API"}
