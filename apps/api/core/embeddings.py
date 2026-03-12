"""Qdrant client wrapper for book similarity lookups."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, HasIdCondition

COLLECTION_NAME = "books"

def get_qdrant_client(
    url: str,
    port: int | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 8.0,
) -> QdrantClient:
    return QdrantClient(url=url, port=port, api_key=api_key or None, timeout=timeout_seconds)


def most_similar(
    client: QdrantClient,
    goodreads_id: int,
    top_k: int = 20,
    exclude_ids: set[int] | None = None,
) -> list[int]:
    """Return goodreads_ids of the top_k most similar books via Qdrant recommend.

    Point IDs in the collection are expected to equal the book's goodreads_id.
    """
    # Always exclude the source book; add any extra exclusions.
    exclude = set(exclude_ids or set())
    exclude.add(goodreads_id)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=goodreads_id,
        limit=top_k,
        query_filter=Filter(
            must_not=[HasIdCondition(has_id=list(exclude))]
        ),
    )
    points = getattr(results, "points", results)
    return [int(hit.id) for hit in points if hit.id is not None]


def most_similar_by_vector(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Return top-k Qdrant hits for a query embedding vector.

    Each item includes:
      - id: Qdrant point id (goodreads_id in this project)
      - payload: optional payload document/metadata
      - score: vector similarity score
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        with_payload=True,
        limit=top_k,
    )
    points = getattr(results, "points", results)
    hits: list[dict[str, Any]] = []
    for hit in points:
        point_id = getattr(hit, "id", None)
        if point_id is None:
            continue
        hits.append({
            "id": point_id,
            "payload": getattr(hit, "payload", {}) or {},
            "score": float(getattr(hit, "score", 0.0) or 0.0),
        })
    return hits


REVIEWS_COLLECTION_NAME = "reviews"


def most_similar_reviews_by_vector(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """Return review_id, book_id and score for top_k review hits.
    """
    results = client.query_points(
        collection_name=REVIEWS_COLLECTION_NAME,
        query=query_vector,
        with_payload=True,
        limit=top_k,
    )
    points = getattr(results, "points", results)
    hits: list[dict[str, Any]] = []
    for hit in points:
        point_id = getattr(hit, "id", None)
        if point_id is None:
            continue
        payload = getattr(hit, "payload", {}) or {}
        book_id = payload.get("book_id")
        if book_id is None:
            continue
        hits.append({
            "review_id": point_id,
            "book_id": int(book_id),
            "score": float(getattr(hit, "score", 0.0) or 0.0),
        })
    return hits


def get_book_scores_by_ids(
    client: QdrantClient,
    book_ids: set[int],
    query_vector: list[float],
) -> dict[int, float]:
    """Get similarity scores for specific book IDs using Qdrant search.
    """
    if not book_ids:
        return {}

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=[HasIdCondition(has_id=list(book_ids))]),
        limit=len(book_ids),
        with_payload=False,
    )

    points = getattr(results, "points", results)
    scores: dict[int, float] = {}
    for hit in points:
        point_id = getattr(hit, "id", None)
        if point_id is None:
            continue
        scores[int(point_id)] = float(getattr(hit, "score", 0.0) or 0.0)

    return scores
