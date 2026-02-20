"""Tests for Qdrant-backed book CRUD operations."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bookdb.vector_db.book_crud import BookVectorCRUD
from bookdb.vector_db.schemas import CollectionNames


def _record(
    id: str,
    document: str | None = None,
    metadata: dict | None = None,
    embedding: list[float] | None = None,
):
    payload = {}
    if document is not None:
        payload["document"] = document
    if metadata is not None:
        payload["metadata"] = metadata
    return SimpleNamespace(id=id, payload=payload, vector=embedding)


class TestBookVectorCRUDQdrant:
    """Tests for BookVectorCRUD against Qdrant interfaces."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def book_crud(self, mock_client):
        with patch("bookdb.vector_db.crud.get_qdrant_client", return_value=mock_client):
            yield BookVectorCRUD()

    def test_init_defaults_to_books_collection(self, book_crud):
        assert book_crud.collection_name == CollectionNames.BOOKS.value

    def test_add_book_minimal(self, book_crud, mock_client):
        mock_client.retrieve.return_value = []

        book_crud.add_book(
            book_id="book_123",
            title="Test Book",
        )

        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "books"
        point = kwargs["points"][0]
        assert point.id == "book_123"
        assert point.payload["document"] == "Test Book"
        assert point.payload["metadata"]["title"] == "Test Book"
        assert "publication_year" not in point.payload["metadata"]

    def test_add_book_with_description_and_embedding(self, book_crud, mock_client):
        mock_client.retrieve.return_value = []
        embedding = [0.1, 0.2, 0.3]

        book_crud.add_book(
            book_id="book_456",
            title="The Great Gatsby",
            description="A novel about the American Dream",
            publication_year=1925,
            embedding=embedding,
        )

        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.payload["document"] == "A novel about the American Dream"
        assert point.payload["metadata"]["publication_year"] == 1925
        assert point.vector == embedding

    def test_add_book_duplicate_id(self, book_crud, mock_client):
        mock_client.retrieve.return_value = [_record(id="book_123")]

        with pytest.raises(ValueError, match="already exists"):
            book_crud.add_book(
                book_id="book_123",
                title="Test Book",
            )

    def test_add_book_invalid_metadata(self, book_crud, mock_client):
        mock_client.retrieve.return_value = []

        with pytest.raises(ValueError):
            book_crud.add_book(
                book_id="book_123",
                title="Test Book",
                publication_year=999,
            )

        mock_client.upsert.assert_not_called()

    def test_update_book_single_field(self, book_crud, mock_client):
        mock_client.retrieve.return_value = [
            _record(
                id="book_123",
                document="Original doc",
                metadata={
                    "title": "Original Title",
                    "publication_year": 2020,
                },
                embedding=[0.1, 0.2, 0.3],
            )
        ]

        book_crud.update_book(
            book_id="book_123",
            publication_year=2023,
        )

        point = mock_client.upsert.call_args.kwargs["points"][0]
        metadata = point.payload["metadata"]
        assert point.payload["document"] == "Original doc"
        assert metadata["publication_year"] == 2023
        assert metadata["title"] == "Original Title"
        assert point.vector == [0.1, 0.2, 0.3]

    def test_update_book_with_description(self, book_crud, mock_client):
        mock_client.retrieve.return_value = [
            _record(
                id="book_123",
                document="Original doc",
                metadata={
                    "title": "Original Title",
                },
                embedding=[0.1, 0.2, 0.3],
            )
        ]

        book_crud.update_book(
            book_id="book_123",
            title="New Title",
            description="New description",
        )

        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.payload["document"] == "New description"
        assert point.payload["metadata"]["title"] == "New Title"

    def test_update_book_nonexistent(self, book_crud, mock_client):
        mock_client.retrieve.return_value = []

        with pytest.raises(ValueError, match="does not exist"):
            book_crud.update_book(
                book_id="nonexistent",
                title="New Title",
            )

    def test_search_by_metadata_single_filter(self, book_crud, mock_client):
        mock_client.scroll.return_value = ([
            _record("book_1", "Doc 1", {"title": "Book 1"}, [0.1]),
            _record("book_2", "Doc 2", {"title": "Book 1"}, [0.2]),
        ], None)

        results = book_crud.search_by_metadata(title="Book 1", limit=5)

        assert len(results) == 2
        kwargs = mock_client.scroll.call_args.kwargs
        assert kwargs["collection_name"] == "books"
        assert kwargs["limit"] == 5
        scroll_filter = kwargs["scroll_filter"]
        assert len(scroll_filter.must) == 1
        assert scroll_filter.must[0].key == "metadata.title"
        assert scroll_filter.must[0].match.value == "Book 1"

    def test_search_by_metadata_multiple_filters(self, book_crud, mock_client):
        mock_client.scroll.return_value = ([
            _record("book_1", "Doc 1", {"title": "Book 1"}, [0.1]),
        ], None)

        book_crud.search_by_metadata(
            title="Book 1",
            min_year=1900,
            max_year=2000,
        )

        scroll_filter = mock_client.scroll.call_args.kwargs["scroll_filter"]
        assert len(scroll_filter.must) == 2
        by_key = {condition.key: condition for condition in scroll_filter.must}
        assert by_key["metadata.title"].match.value == "Book 1"
        assert by_key["metadata.publication_year"].range.gte == 1900
        assert by_key["metadata.publication_year"].range.lte == 2000

    def test_search_by_metadata_without_filters(self, book_crud, mock_client):
        mock_client.scroll.return_value = ([
            _record("book_1", "Doc 1", {"title": "Book 1"}, [0.1]),
        ], None)

        results = book_crud.search_by_metadata(limit=10)

        assert len(results) == 1
        kwargs = mock_client.scroll.call_args.kwargs
        assert kwargs["scroll_filter"] is None
        assert kwargs["limit"] == 10

    def test_search_by_metadata_wraps_errors(self, book_crud, mock_client):
        mock_client.scroll.side_effect = RuntimeError("boom")

        with pytest.raises(Exception, match="Failed to search by metadata"):
            book_crud.search_by_metadata(title="Book 1")

    def test_search_similar_books_not_implemented(self, book_crud):
        result = book_crud.search_similar_books(
            query_text="test query",
            n_results=5,
        )
        assert result is None

    def test_get_book_recommendations_not_implemented(self, book_crud):
        result = book_crud.get_book_recommendations(
            book_id="book_123",
            n_results=5,
        )
        assert result is None
