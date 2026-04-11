"""Tests for entity extraction functionality.

Tests are designed to work with the existing test environment
and pytest configuration.
"""

import pytest
import os

# Test imports that work with project structure
from apps.api.core.entity_extraction import (
    _string_similarity,
    get_cache_stats,
    clear_entity_cache,
)


# ============================================================================
# Unit Tests: String Similarity
# ============================================================================


def test_string_similarity_exact_match():
    """Test exact match returns 1.0."""
    score = _string_similarity("Harry Potter", "Harry Potter")
    assert score == pytest.approx(1.0, abs=0.01)


def test_string_similarity_case_insensitive():
    """Test case-insensitive matching."""
    score = _string_similarity("Harry Potter", "harry potter")
    assert score == pytest.approx(1.0, abs=0.01)


def test_string_similarity_partial_match():
    """Test partial matching."""
    score = _string_similarity("Harry Potter", "Harry")
    assert score > 0.5
    assert score < 1.0


def test_string_similarity_no_match():
    """Test no match returns low score."""
    score = _string_similarity("Harry Potter", "Lord of the Rings")
    assert score < 0.3


def test_string_similarity_typo_tolerance():
    """Test typo tolerance."""
    score = _string_similarity("Harry Potter", "Hary Potter")
    assert score > 0.8


# ============================================================================
# Unit Tests: Cache Management
# ============================================================================


def test_cache_stats():
    """Get cache statistics."""
    stats = get_cache_stats()

    assert "size" in stats
    assert "maxsize" in stats
    assert "ttl" in stats

    assert stats["size"] == 0  # Empty initially
    assert stats["maxsize"] == 1000
    assert stats["ttl"] == 3600


def test_clear_cache():
    """Clear entity cache."""
    # Cache should be empty initially
    stats_before = get_cache_stats()
    assert stats_before["size"] == 0

    # Clear cache (should work even if empty)
    clear_entity_cache()

    # Verify cache is still empty
    stats_after = get_cache_stats()
    assert stats_after["size"] == 0


# ============================================================================
# Tests: Edge Cases
# ============================================================================


def test_string_similarity_empty_strings():
    """Handle empty strings."""
    score1 = _string_similarity("", "Harry Potter")
    score2 = _string_similarity("Harry Potter", "")

    assert score1 < 0.5
    assert score2 < 0.5


def test_string_similarity_special_characters():
    """Handle special characters."""
    score = _string_similarity("Book & Test", "Book and Test")
    assert score > 0.8  # Should still match well


# ============================================================================
# LLM Integration Tests (Only run if GROQ_API_KEY is set)
# ============================================================================


@pytest.mark.skipif(
    "GROQ_API_KEY" not in os.environ,
    reason="LLM tests require GROQ_API_KEY environment variable",
)
def test_extract_book_entities_basic():
    """Test basic entity extraction (requires GROQ_API_KEY)."""
    from apps.api.core.entity_extraction import extract_book_entities
    from bookdb.models.chatbot_llm import create_groq_client_sync

    if "GROQ_API_KEY" not in os.environ:
        pytest.skip("GROQ_API_KEY not set")

    client = create_groq_client_sync()
    result = extract_book_entities("I love Harry Potter", client=client)

    assert "book_titles" in result
    assert "author_names" in result
    assert "confidence" in result


@pytest.mark.skipif(
    "GROQ_API_KEY" not in os.environ,
    reason="LLM tests require GROQ_API_KEY environment variable",
)
def test_extract_book_entities_empty_query():
    """Handle empty queries (requires GROQ_API_KEY)."""
    from apps.api.core.entity_extraction import extract_book_entities
    from bookdb.models.chatbot_llm import create_groq_client_sync

    if "GROQ_API_KEY" not in os.environ:
        pytest.skip("GROQ_API_KEY not set")

    client = create_groq_client_sync()
    result = extract_book_entities("", client=client)

    assert result.get("book_titles", []) == []
    assert result.get("author_names", []) == []
    # Low confidence for empty query
    assert result.get("confidence", 0) < 0.5


# ============================================================================
# Tests: Context Generation
# ============================================================================


def test_get_book_context_without_db_session():
    """Generate context without database session."""
    from apps.api.core.entity_extraction import get_book_context_string
    from bookdb.db.models import Book

    book = Book(
        id=1,
        goodreads_id=100,
        title="Test Book",
        description="Test description",
    )
    context = get_book_context_string(book, 0.8)

    assert "TITLE: Test Book" in context
    assert "DESCRIPTION: Test description" in context
