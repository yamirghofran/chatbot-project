"""Minimal RAG review search implementation."""

from typing import Any, Dict, List

import chromadb
import httpx
import polars as pl

# Hardcoded defaults
EMBEDDING_ENDPOINT = "http://127.0.0.1:8000/embed"
MODEL_NAME = "finetuned" #'base'
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "reviews_collection"
REVIEWS_PARQUET = "data/goodreads_reviews_dedup_clean.parquet"


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
    """Search Chroma for similar reviews."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    reviews = []
    for i, review_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i] if results["distances"] else 0.0
        reviews.append(
            {
                "review_id": review_id,
                "similarity_score": 1.0 - distance,
                "distance": distance,
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


def search_reviews(query_text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Search for similar reviews given a text query.

    Args:
        query_text: Text query to find similar reviews
        top_k: Number of similar reviews to return

    Returns:
        JSON with query, top_k, and results containing review metadata
    """
    query_embedding = generate_embedding(query_text)
    similar_reviews = search_similar_reviews(query_embedding, top_k)
    review_ids = [review["review_id"] for review in similar_reviews]
    metadata_by_id = get_review_metadata(review_ids)

    results = []
    for review in similar_reviews:
        review_id = review["review_id"]
        metadata = metadata_by_id.get(review_id, {})
        review_result = {
            **review,
            "rating": metadata.get("rating"),
            "review_text": metadata.get("review_text"),
            "book_id": metadata.get("book_id"),
            "user_id": metadata.get("user_id"),
            "n_votes": metadata.get("n_votes"),
            "n_comments": metadata.get("n_comments"),
            "date_updated": metadata.get("date_updated"),
        }
        results.append(review_result)

    return {"query": query_text, "top_k": top_k, "results": results}
