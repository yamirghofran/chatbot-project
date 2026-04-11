"""Validation utilities for RAG pipeline.

Provides input validation, data quality checks, and error handling
for the book search RAG pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    message: str
    severity: str = "error"  # error, warning, info


class QueryValidator:
    """Validates search queries for the book RAG pipeline."""

    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 1000
    MAX_WORDS = 100

    # Common spam/malicious patterns to reject
    BLOCKED_PATTERNS = [
        r"<script",  # XSS attempts
        r"javascript:",  # JS injection
        r"DROP\s+TABLE",  # SQL injection
        r";\s*--",  # SQL comment injection
        r"\.\.\.",  # Path traversal attempt
    ]

    @classmethod
    def validate(cls, query: str) -> list[ValidationResult]:
        """Validate a search query.

        Returns list of validation results. Empty list means valid.
        """
        results = []

        # Type check
        if not isinstance(query, str):
            return [ValidationResult(False, "Query must be a string", "error")]

        # Length checks
        if len(query.strip()) < cls.MIN_QUERY_LENGTH:
            results.append(
                ValidationResult(
                    False,
                    f"Query too short (min {cls.MIN_QUERY_LENGTH} chars)",
                    "error",
                )
            )

        if len(query) > cls.MAX_QUERY_LENGTH:
            results.append(
                ValidationResult(
                    False, f"Query too long (max {cls.MAX_QUERY_LENGTH} chars)", "error"
                )
            )

        # Word count
        word_count = len(query.split())
        if word_count > cls.MAX_WORDS:
            results.append(
                ValidationResult(
                    False, f"Query too many words (max {cls.MAX_WORDS})", "error"
                )
            )

        # Security: Check for blocked patterns
        query_lower = query.lower()
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, query_lower):
                results.append(
                    ValidationResult(
                        False, "Query contains invalid characters/patterns", "error"
                    )
                )
                break

        # Warning: Mostly numbers (likely not useful)
        alpha_chars = sum(1 for c in query if c.isalpha())
        if alpha_chars < len(query) * 0.3 and len(query) > 10:
            results.append(
                ValidationResult(
                    True,
                    "Query is mostly non-alphabetic - results may be poor",
                    "warning",
                )
            )

        # Warning: Very short after cleaning
        cleaned = re.sub(r"[^\w\s]", "", query).strip()
        if len(cleaned) < 5:
            results.append(
                ValidationResult(
                    True,
                    "Query is very short after removing special characters",
                    "warning",
                )
            )

        return results

    @classmethod
    def sanitize(cls, query: str) -> str:
        """Sanitize a query for safe processing."""
        # Remove control characters
        query = "".join(char for char in query if ord(char) >= 32 or char == "\n")

        # Normalize whitespace
        query = " ".join(query.split())

        # Trim
        query = query.strip()

        return query


class EmbeddingValidator:
    """Validates embedding service and results."""

    EXPECTED_DIMENSION = 768
    MIN_SIMILARITY = -1.0
    MAX_SIMILARITY = 1.0

    @classmethod
    def validate_embedding(cls, embedding: list[float]) -> list[ValidationResult]:
        """Validate an embedding vector."""
        results = []

        if not isinstance(embedding, list):
            return [ValidationResult(False, "Embedding must be a list", "error")]

        # Dimension check
        if len(embedding) != cls.EXPECTED_DIMENSION:
            results.append(
                ValidationResult(
                    False,
                    f"Expected {cls.EXPECTED_DIMENSION} dimensions, got {len(embedding)}",
                    "error",
                )
            )

        # Check for NaN or Inf
        import math

        for i, val in enumerate(embedding[:10]):  # Check first 10
            if math.isnan(val) or math.isinf(val):
                results.append(
                    ValidationResult(
                        False, f"Embedding contains NaN/Inf at position {i}", "error"
                    )
                )
                break

        # Check normalization (if using normalized embeddings)
        norm = sum(x**2 for x in embedding) ** 0.5
        if abs(norm - 1.0) > 0.1:  # Should be close to 1 if normalized
            results.append(
                ValidationResult(
                    True,
                    f"Embedding may not be normalized (norm={norm:.3f})",
                    "warning",
                )
            )

        return results

    @classmethod
    def validate_similarity_scores(cls, scores: list[float]) -> list[ValidationResult]:
        """Validate similarity scores from retrieval."""
        results = []

        if not scores:
            return [ValidationResult(False, "Empty similarity scores", "error")]

        # Check range
        for i, score in enumerate(scores):
            if not (cls.MIN_SIMILARITY <= score <= cls.MAX_SIMILARITY):
                results.append(
                    ValidationResult(
                        False, f"Similarity score {i} out of range: {score}", "error"
                    )
                )

        # Check descending order (should be sorted by relevance)
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1] - 0.01:  # Allow small floating point errors
                results.append(
                    ValidationResult(
                        True, "Similarity scores not in descending order", "warning"
                    )
                )
                break

        # Check for all zeros (likely error)
        if all(s == 0.0 for s in scores[:5]):
            results.append(
                ValidationResult(
                    False,
                    "Top 5 scores are all zero - possible retrieval error",
                    "error",
                )
            )

        return results


class HealthChecker:
    """Health checks for RAG pipeline dependencies."""

    @staticmethod
    def check_embedding_service(
        endpoint: str, timeout: float = 5.0
    ) -> ValidationResult:
        """Check if embedding service is healthy."""
        try:
            response = httpx.get(endpoint.replace("/embed", "/health"), timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    return ValidationResult(True, "Embedding service healthy", "info")
                return ValidationResult(False, f"Service unhealthy: {data}", "error")
            return ValidationResult(
                False, f"Health check failed: HTTP {response.status_code}", "error"
            )
        except httpx.ConnectError:
            return ValidationResult(
                False, "Cannot connect to embedding service", "error"
            )
        except httpx.TimeoutException:
            return ValidationResult(False, "Embedding service timeout", "error")
        except Exception as e:
            return ValidationResult(False, f"Health check error: {str(e)}", "error")

    @staticmethod
    def check_qdrant_connection(client) -> ValidationResult:
        """Check if Qdrant is accessible."""
        try:
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if "books" in collection_names:
                # Check if collection has data
                count = client.count(collection_name="books")
                return ValidationResult(
                    True,
                    f"Qdrant connected. Books collection: {count.count} vectors",
                    "info",
                )
            return ValidationResult(
                True, f"Qdrant connected. Available: {collection_names}", "warning"
            )
        except Exception as e:
            return ValidationResult(
                False, f"Qdrant connection failed: {str(e)}", "error"
            )


def validate_search_request(
    query: str, top_k: int = 10
) -> tuple[str, list[ValidationResult]]:
    """Validate and sanitize a search request.

    Returns sanitized query and list of validation results.
    Raises ValueError if validation fails critically.
    """
    # Sanitize first
    sanitized = QueryValidator.sanitize(query)

    # Validate query
    results = QueryValidator.validate(sanitized)

    # Check for errors
    errors = [r for r in results if not r.is_valid and r.severity == "error"]
    if errors:
        raise ValueError("; ".join(r.message for r in errors))

    # Validate top_k
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        raise ValueError("top_k must be an integer between 1 and 100")

    return sanitized, results


# Convenience function for quick validation
def is_valid_query(query: str) -> bool:
    """Quick check if query is valid."""
    try:
        validate_search_request(query)
        return True
    except ValueError:
        return False
