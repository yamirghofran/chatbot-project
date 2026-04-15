"""Tests for RAG validation and robustness features."""

import pytest

from bookdb.validation import (
    EmbeddingValidator,
    HealthChecker,
    QueryValidator,
    ValidationResult,
    is_valid_query,
    validate_search_request,
)
from bookdb.validation.data_quality import BookDataValidator, DataQualityReport


class TestQueryValidator:
    """Tests for query validation."""

    def test_valid_query(self):
        """Test a normal valid query."""
        results = QueryValidator.validate("fantasy books about dragons")
        errors = [r for r in results if r.severity == "error"]
        assert len(errors) == 0

    def test_query_too_short(self):
        """Test query below minimum length."""
        results = QueryValidator.validate("ab")
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert any("too short" in r.message for r in errors)

    def test_query_too_long(self):
        """Test query exceeding maximum length."""
        results = QueryValidator.validate("a" * 1001)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert any("too long" in r.message for r in errors)

    def test_query_security_blocked(self):
        """Test query with blocked patterns."""
        results = QueryValidator.validate("<script>alert('xss')</script>")
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0

    def test_sanitize_query(self):
        """Test query sanitization."""
        dirty = "  fantasy   books   \n\n  "
        clean = QueryValidator.sanitize(dirty)
        assert clean == "fantasy books"

    def test_is_valid_query(self):
        """Test quick validation function."""
        assert is_valid_query("good query about books") is True
        assert is_valid_query("ab") is False


class TestValidateSearchRequest:
    """Tests for complete search request validation."""

    def test_valid_request(self):
        """Test valid search request."""
        query, results = validate_search_request("fantasy books", top_k=10)
        assert query == "fantasy books"
        assert len([r for r in results if r.severity == "error"]) == 0

    def test_invalid_top_k(self):
        """Test invalid top_k values."""
        with pytest.raises(ValueError, match="top_k"):
            validate_search_request("query", top_k=0)

        with pytest.raises(ValueError, match="top_k"):
            validate_search_request("query", top_k=101)

        with pytest.raises(ValueError, match="top_k"):
            validate_search_request("query", top_k="ten")

    def test_sanitization_on_validation(self):
        """Test that validation sanitizes input."""
        query, _ = validate_search_request("  dirty   query  ")
        assert query == "dirty query"


class TestEmbeddingValidator:
    """Tests for embedding validation."""

    def test_valid_embedding(self):
        """Test a valid normalized embedding."""
        import math

        # Create a normalized embedding
        embedding = [0.1] * 768
        norm = math.sqrt(sum(x**2 for x in embedding))
        normalized = [x / norm for x in embedding]

        results = EmbeddingValidator.validate_embedding(normalized)
        errors = [r for r in results if r.severity == "error"]
        assert len(errors) == 0

    def test_wrong_dimension(self):
        """Test embedding with wrong dimension."""
        embedding = [0.1] * 512  # Wrong size
        results = EmbeddingValidator.validate_embedding(embedding)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0

    def test_nan_in_embedding(self):
        """Test embedding with NaN values."""
        embedding = [float("nan")] + [0.1] * 767
        results = EmbeddingValidator.validate_embedding(embedding)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0

    def test_similarity_scores_valid(self):
        """Test valid similarity scores."""
        scores = [0.95, 0.87, 0.76, 0.65, 0.54]
        results = EmbeddingValidator.validate_similarity_scores(scores)
        errors = [r for r in results if r.severity == "error"]
        assert len(errors) == 0

    def test_similarity_scores_out_of_range(self):
        """Test scores outside valid range."""
        scores = [1.5, 0.87, -1.2]  # Out of [-1, 1] range
        results = EmbeddingValidator.validate_similarity_scores(scores)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0

    def test_all_zero_scores(self):
        """Test all-zero scores (likely error)."""
        scores = [0.0, 0.0, 0.0, 0.0, 0.0]
        results = EmbeddingValidator.validate_similarity_scores(scores)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0


class TestBookDataValidator:
    """Tests for data quality validation."""

    def test_valid_schema(self):
        """Test schema validation with valid columns."""
        import polars as pl

        df = pl.DataFrame(
            {
                "book_id": [1, 2, 3],
                "title": ["Book 1", "Book 2", "Book 3"],
                "authors": [["Author 1"], ["Author 2"], ["Author 3"]],
                "description": ["Desc 1", "Desc 2", "Desc 3"],
                "similar_books": [[2], [3], [1]],
            }
        )

        results = BookDataValidator.validate_schema(df)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0

    def test_missing_required_columns(self):
        """Test schema with missing required columns."""
        import polars as pl

        df = pl.DataFrame(
            {
                "book_id": [1, 2, 3],
                "title": ["Book 1", "Book 2", "Book 3"],
                # Missing authors, description, similar_books
            }
        )

        results = BookDataValidator.validate_schema(df)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0

    def test_data_quality_report(self):
        """Test full data quality report."""
        import polars as pl

        df = pl.DataFrame(
            {
                "book_id": [1, 2, 3],
                "title": ["Book 1", "Book 2", "Book 3"],
                "authors": [["A1"], ["A2"], ["A3"]],
                "description": ["D1", "D2", "D3"],
                "similar_books": [[2], [3], [1]],
            }
        )

        report = BookDataValidator.validate_dataset(df)

        assert isinstance(report, DataQualityReport)
        assert report.total_books == 3
        assert report.schema_valid is True
        assert report.quality_score > 0

    def test_duplicate_book_ids(self):
        """Test detection of duplicate book IDs."""
        import polars as pl

        df = pl.DataFrame(
            {
                "book_id": [1, 2, 1, 3],  # Duplicate 1
                "title": ["Book 1", "Book 2", "Book 1 again", "Book 3"],
                "authors": [["A"], ["A"], ["A"], ["A"]],
                "description": ["D", "D", "D", "D"],
                "similar_books": [[], [], [], []],
            }
        )

        results = BookDataValidator.validate_integrity(df)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert any("duplicate" in r.message.lower() for r in errors)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid result."""
        result = ValidationResult(True, "All good", "info")
        assert result.is_valid is True
        assert result.message == "All good"
        assert result.severity == "info"

    def test_error_result(self):
        """Test error result."""
        result = ValidationResult(False, "Something wrong", "error")
        assert result.is_valid is False
        assert result.severity == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
