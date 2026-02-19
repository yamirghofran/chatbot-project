"""Minimal RAG book search implementation."""

from typing import Any, Dict, List

import httpx
import polars as pl

# Hardcoded defaults
EMBEDDING_ENDPOINT = "http://127.0.0.1:8000/embed"
MODEL_NAME = "finetuned"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "books_collection"
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


def search_similar_books(
    query_embedding: List[float], top_k: int = 10
) -> List[Dict[str, Any]]:
    """Search Chroma for similar books (mocked for testing)."""
    # import chromadb
    # client = chromadb.PersistentClient(path=CHROMA_DIR)
    # collection = client.get_collection(name=COLLECTION_NAME)
    # results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Mock Chroma
    import random

    mock_book_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    books = []
    for i in range(min(top_k, len(mock_book_ids))):
        distance = random.uniform(0.1, 0.9)
        books.append(
            {
                "book_id": mock_book_ids[i],
                "similarity_score": 1.0 - distance,
                "distance": distance,
            }
        )

    return books


def get_book_metadata(book_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Retrieve book metadata from parquet file."""
    df = pl.read_parquet(BOOKS_PARQUET)

    filtered_df = df.filter(pl.col("book_id").is_in(book_ids))

    metadata_by_id = {
        row["book_id"]: dict(row) for row in filtered_df.iter_rows(named=True)
    }

    return metadata_by_id


def search_books(query_text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Search for similar books given a text query.

    Args:
        query_text: Text query (rewritten by LLM to look like a book description)
        top_k: Number of similar books to return

    Returns:
        JSON with query, top_k, and results containing book metadata
    """
    query_embedding = generate_embedding(query_text)
    similar_books = search_similar_books(query_embedding, top_k)
    book_ids = [book["book_id"] for book in similar_books]
    metadata_by_id = get_book_metadata(book_ids)

    results = []
    for book in similar_books:
        book_id = book["book_id"]
        metadata = metadata_by_id.get(book_id, {})
        book_result = {
            **book,
            "title": metadata.get("title"),
            "description": metadata.get("description"),
            "authors": metadata.get("authors"),
            "publisher": metadata.get("publisher"),
            "publication_year": metadata.get("publication_year"),
            "num_pages": metadata.get("num_pages"),
            "isbn13": metadata.get("isbn13"),
            "url": metadata.get("url"),
            "image_url": metadata.get("image_url"),
            "num_interactions": metadata.get("num_interactions"),
            "num_read": metadata.get("num_read"),
            "num_ratings": metadata.get("num_ratings"),
            "num_reviews": metadata.get("num_reviews"),
            "language_code": metadata.get("language_code"),
            "format": metadata.get("format"),
            "popular_shelves": metadata.get("popular_shelves"),
            "similar_books": metadata.get("similar_books"),
        }
        results.append(book_result)

    return {"query": query_text, "top_k": top_k, "results": results}
