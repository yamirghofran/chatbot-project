"""Minimal RAG review search implementation."""

from typing import Any, Dict, List
import httpx
from sqlalchemy import select
import os

from bookdb.db.models import Book, Review
from bookdb.db.session import SessionLocal
from bookdb.vector_db.client import get_qdrant_client

EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "http://127.0.0.1:8000/embed")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "finetuned")
COLLECTION_NAME = os.getenv("REVIEWS_COLLECTION_NAME", "reviews_collection")

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text using the finetuned model endpoint."""
    response = httpx.post(
        EMBEDDING_ENDPOINT,
        json={
            "model": MODEL_NAME,
            "texts": [text],
            "normalize_embeddings": True,
            "batch_size": 1,
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"][0]


def search_similar_reviews(
    query_embedding: List[float], top_k: int = 10
) -> List[Dict[str, Any]]:
    """Search Qdrant for similar reviews."""
    client = get_qdrant_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    )

    reviews = []
    for point in results.points:
        reviews.append(
            {
                "review_id": point.payload.get("review_id"),
                "similarity_score": point.score,
            }
        )
    return reviews


# TODO: implement in proper file
def get_review_metadata(review_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Retrieve review metadata from database."""
    if not review_ids:
        return {}
    with SessionLocal() as session:
        stmt = select(Review).where(Review.goodreads_id.in_(review_ids))
        reviews = session.scalars(stmt).all()
        return {
            r.goodreads_id: {"review_text": r.review_text, "book_id": r.book_id}
            for r in reviews
            if r.goodreads_id is not None
        }


def get_book_titles(book_ids: List[int]) -> Dict[int, str]:
    """Retrieve book titles from database."""
    if not book_ids:
        return {}
    with SessionLocal() as session:
        stmt = select(Book).where(Book.id.in_(book_ids))
        books = session.scalars(stmt).all()
        return {b.id: b.title for b in books}


def search_reviews(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search for similar reviews given a text query."""
    query_embedding = generate_embedding(query_text)
    similar_reviews = search_similar_reviews(query_embedding, top_k)
    review_ids = [review["review_id"] for review in similar_reviews]
    metadata_by_id = get_review_metadata(review_ids)

    # Get book_ids and fetch titles
    book_ids = [metadata_by_id.get(rid, {}).get("book_id") for rid in review_ids]
    book_ids = [bid for bid in book_ids if bid is not None]
    book_titles = get_book_titles(book_ids)

    results = []
    for review in similar_reviews:
        review_id = review["review_id"]
        metadata = metadata_by_id.get(review_id, {})
        book_id = metadata.get("book_id")
        results.append({
            "review_id": review_id,
            "book_title": book_titles.get(book_id, "Unknown"),
            "review": metadata.get("review_text", ""),
        })

    return results
