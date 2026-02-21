"""BookDB FastAPI application.

Run with:
    uvicorn apps.api.main:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.embeddings import load_embedding_index
from .routers import auth, books, discovery, lists, me, users

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
app.include_router(users.router)
app.include_router(lists.router)
app.include_router(me.router)
app.include_router(discovery.router)


@app.on_event("startup")
async def startup_event():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    embeddings_path = os.path.join(project_root, "data", "books_finetuned_embeddings.parquet")
    app.state.embedding_index = load_embedding_index(embeddings_path)

    bpr_path = os.path.join(project_root, "data", "bpr_model_predictions", "bpr_recommendations.parquet")
    if os.path.exists(bpr_path):
        app.state.bpr_parquet_path = bpr_path
        print(f"BPR recommendations loaded: {bpr_path}")
    else:
        app.state.bpr_parquet_path = None
        print("WARNING: BPR recommendations parquet not found, will use cold-start fallback.")


@app.get("/")
def health_check():
    return {"status": "ok", "service": "BookDB API"}
