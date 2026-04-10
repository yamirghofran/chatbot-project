"""Minimal RAG book search implementation using Qdrant.

Includes input validation, error handling, and health checks
for production robustness.
"""

from typing import Any, Dict, List, Optional

import httpx
import polars as pl

from bookdb.vector_db import BookVectorCRUD
from bookdb.validation import (
    HealthChecker,
    QueryValidator,
    validate_search_request,
)
from bookdb.validation.data_quality import check_data_quality

# Configuration
EMBEDDING_ENDPOINT = "https://bookdb-models.up.railway.app/embed"
MODEL_NAME = "finetuned"
BOOKS_PARQUET = "data/3_goodreads_books_with_metrics.parquet"

# Fallback responses for error cases
FALLBACK_RESULTS = {
    "error": None,
    "query": "",
    "top_k": 10,
    "results": [],
    "fallback": True,
    "message": "Search could not be completed. Please try again.",
}


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
    """Search Qdrant for similar books."""
    crud = BookVectorCRUD()
    results = crud.search_similar_books(
        query_embedding=query_embedding,
        n_results=top_k,
    )
    return results


def get_book_metadata(book_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Retrieve book metadata from parquet file."""
    df = pl.read_parquet(BOOKS_PARQUET)

    filtered_df = df.filter(pl.col("book_id").is_in(book_ids))

    metadata_by_id = {
        row["book_id"]: dict(row) for row in filtered_df.iter_rows(named=True)
    }

    return metadata_by_id


def search_books(
    query_text: str,
    top_k: int = 10,
    validate: bool = True,
    fallback_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Search for similar books given a text query.

    Args:
        query_text: Text query (rewritten by LLM to look like a book description)
        top_k: Number of similar books to return (max 100)
        validate: Whether to validate input (default: True)
        fallback_on_error: Return empty results on error instead of raising (default: True)

    Returns:
        JSON with query, top_k, results, and validation info
    """
    # Input validation
    if validate:
        try:
            sanitized_query, validation_results = validate_search_request(
                query_text, top_k
            )
        except ValueError as e:
            if fallback_on_error:
                return {
                    **FALLBACK_RESULTS,
                    "query": query_text,
                    "top_k": top_k,
                    "error": f"Validation error: {str(e)}",
                }
            raise
    else:
        sanitized_query = QueryValidator.sanitize(query_text)
        validation_results = []

    try:
        query_embedding = generate_embedding(sanitized_query)
    except httpx.HTTPError as e:
        if fallback_on_error:
            return {
                **FALLBACK_RESULTS,
                "query": sanitized_query,
                "top_k": top_k,
                "error": f"Embedding service error: {str(e)}",
            }
        raise

    try:
        similar_books = search_similar_books(query_embedding, top_k)
    except Exception as e:
        if fallback_on_error:
            return {
                **FALLBACK_RESULTS,
                "query": sanitized_query,
                "top_k": top_k,
                "error": f"Search error: {str(e)}",
            }
        raise

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

    response = {
        "query": sanitized_query,
        "top_k": top_k,
        "results": results,
        "result_count": len(results),
        "fallback": False,
    }

    # Include validation info if there were warnings
    if validation_results:
        warnings = [v for v in validation_results if v.severity == "warning"]
        if warnings:
            response["warnings"] = [w.message for w in warnings]

    return response


def health_check() -> Dict[str, Any]:
    """Perform health checks on RAG pipeline components.

    Returns:
        Health status of all components
    """
    checks = {
        "embedding_service": None,
        "qdrant": None,
        "data_quality": None,
        "overall": "unknown",
    }

    # Check embedding service
    try:
        embed_health = HealthChecker.check_embedding_service(EMBEDDING_ENDPOINT)
        checks["embedding_service"] = {
            "status": "healthy" if embed_health.is_valid else "unhealthy",
            "message": embed_health.message,
        }
    except Exception as e:
        checks["embedding_service"] = {
            "status": "error",
            "message": str(e),
        }

    # Check Qdrant
    try:
        from bookdb.vector_db import get_qdrant_client

        client = get_qdrant_client()
        qdrant_health = HealthChecker.check_qdrant_connection(client)
        checks["qdrant"] = {
            "status": "healthy" if qdrant_health.is_valid else "unhealthy",
            "message": qdrant_health.message,
        }
    except Exception as e:
        checks["qdrant"] = {
            "status": "error",
            "message": str(e),
        }

    # Check data quality
    try:
        data_report = check_data_quality(BOOKS_PARQUET)
        checks["data_quality"] = {
            "status": "healthy" if data_report.quality_score > 80 else "warning",
            "quality_score": round(data_report.quality_score, 1),
            "total_books": data_report.total_books,
            "message": f"Quality score: {data_report.quality_score:.1f}/100",
        }
    except Exception as e:
        checks["data_quality"] = {
            "status": "error",
            "message": str(e),
        }

    # Determine overall health
    statuses = [c["status"] for c in checks.values() if c is not None]
    if all(s == "healthy" for s in statuses):
        checks["overall"] = "healthy"
    elif any(s == "error" for s in statuses):
        checks["overall"] = "unhealthy"
    else:
        checks["overall"] = "degraded"

    return checks
