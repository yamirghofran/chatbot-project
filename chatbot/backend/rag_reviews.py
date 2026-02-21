"""Minimal RAG review search implementation."""

from typing import Any, Dict, List

import httpx
import polars as pl
from qdrant_client import QdrantClient

# Hardcoded defaults
EMBEDDING_ENDPOINT = "http://127.0.0.1:8000/embed"
MODEL_NAME = "finetuned"  # 'base'
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "reviews_collection"
REVIEWS_PARQUET = "data/3_goodreads_reviews_dedup_clean.parquet"
BOOKS_PARQUET = "data/3_goodreads_books_with_metrics.parquet"


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
    client = QdrantClient(url=QDRANT_URL)
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


def get_review_metadata(review_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Retrieve review metadata from parquet file."""
    df = pl.read_parquet(REVIEWS_PARQUET)

    filtered_df = df.filter(pl.col("review_id").is_in(review_ids))

    metadata_by_id = {
        row["review_id"]: dict(row) for row in filtered_df.iter_rows(named=True)
    }

    return metadata_by_id


def get_book_titles(book_ids: List[str]) -> Dict[str, str]:
    """Retrieve book titles from parquet file."""
    df = pl.read_parquet(BOOKS_PARQUET)

    filtered_df = df.filter(pl.col("book_id").is_in(book_ids))

    return {
        str(row["book_id"]): row["title"] for row in filtered_df.iter_rows(named=True)
    }


def search_reviews(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search for similar reviews given a text query.

    Args:
        query_text: Text query to find similar reviews
        top_k: Number of similar reviews to return

    Returns:
        List of reviews with format: {"review_id": str, "book_title": str, "review": str}
    """
    query_embedding = generate_embedding(query_text)
    similar_reviews = search_similar_reviews(query_embedding, top_k)
    review_ids = [review["review_id"] for review in similar_reviews]
    metadata_by_id = get_review_metadata(review_ids)

    # Get book_ids and fetch titles
    book_ids = [str(metadata_by_id.get(rid, {}).get("book_id", "")) for rid in review_ids]
    book_titles = get_book_titles(book_ids)

    results = []
    for review in similar_reviews:
        review_id = review["review_id"]
        metadata = metadata_by_id.get(review_id, {})
        book_id = str(metadata.get("book_id", ""))
        results.append({
            "review_id": review_id,
            "book_title": book_titles.get(book_id, "Unknown"),
            "review": metadata.get("review_text", ""),
        })

    return results

