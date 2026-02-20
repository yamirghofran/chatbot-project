"""Tests for Qdrant-backed collection management."""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance

from bookdb.vector_db.collections import (
    CollectionManager,
    get_books_collection,
    get_users_collection,
    initialize_all_collections,
)
from bookdb.vector_db.schemas import (
    BookMetadata,
    CollectionNames,
    UserMetadata,
    validate_book_metadata,
    validate_user_metadata,
)


class TestBookMetadata:
    """Tests for BookMetadata schema."""

    def test_book_metadata_minimal(self):
        metadata = BookMetadata(
            title="Test Book",
            author="Test Author",
        )

        assert metadata.title == "Test Book"
        assert metadata.author == "Test Author"
        assert metadata.created_at is not None

    def test_book_metadata_full(self):
        metadata = BookMetadata(
            title="The Great Gatsby",
            author="F. Scott Fitzgerald",
            genre="Fiction",
            publication_year=1925,
        )

        assert metadata.title == "The Great Gatsby"
        assert metadata.publication_year == 1925

    def test_book_metadata_invalid_year(self):
        with pytest.raises(ValueError):
            BookMetadata(
                title="Test",
                author="Test",
                publication_year=999,
            )

    def test_validate_book_metadata_function(self):
        data = {
            "title": "Test Book",
            "author": "Test Author",
            "genre": "Fiction",
        }

        metadata = validate_book_metadata(data)

        assert isinstance(metadata, BookMetadata)
        assert metadata.title == "Test Book"


class TestUserMetadata:
    """Tests for UserMetadata schema."""

    def test_user_metadata_required_fields(self):
        metadata = UserMetadata(user_id=123, name="Alice")

        assert metadata.user_id == 123
        assert metadata.name == "Alice"
        assert metadata.created_at is not None

    def test_user_metadata_invalid_user_id(self):
        with pytest.raises(ValueError):
            UserMetadata(user_id=0, name="Alice")

    def test_user_metadata_missing_name(self):
        with pytest.raises(ValueError):
            UserMetadata(user_id=123)

    def test_validate_user_metadata_function(self):
        data = {
            "user_id": 456,
            "name": "Bob",
        }

        metadata = validate_user_metadata(data)

        assert isinstance(metadata, UserMetadata)
        assert metadata.user_id == 456
        assert metadata.name == "Bob"


class TestCollectionNames:
    """Tests for CollectionNames enum."""

    def test_collection_names(self):
        assert CollectionNames.BOOKS.value == "books"
        assert CollectionNames.USERS.value == "users"
        assert CollectionNames.REVIEWS.value == "reviews"

    def test_collection_names_enum(self):
        assert isinstance(CollectionNames.BOOKS, CollectionNames)
        assert isinstance(CollectionNames.USERS, CollectionNames)
        assert isinstance(CollectionNames.REVIEWS, CollectionNames)


class TestCollectionManager:
    """Tests for CollectionManager class."""

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_collection_manager_init(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        manager = CollectionManager()

        assert manager.client == mock_client
        assert manager._collections == {}
        assert manager.vector_size == 384

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_collection_manager_init_invalid_vector_size(self, mock_get_client):
        mock_get_client.return_value = MagicMock()
        with pytest.raises(ValueError, match="vector_size must be > 0"):
            CollectionManager(vector_size=0)

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_initialize_collections_creates_missing(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.side_effect = [False, False, False]
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        manager.initialize_collections()

        assert mock_client.create_collection.call_count == 3
        calls = mock_client.create_collection.call_args_list
        expected_names = ["books", "users", "reviews"]
        for i, name in enumerate(expected_names):
            assert calls[i].kwargs["collection_name"] == name
            vectors_config = calls[i].kwargs["vectors_config"]
            assert vectors_config.size == 384
            assert vectors_config.distance == Distance.COSINE

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_initialize_collections_skips_existing(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.side_effect = [True, True, True]
        mock_client.get_collection.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        manager.initialize_collections()

        mock_client.create_collection.assert_not_called()
        assert mock_client.get_collection.call_count == 3

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_get_collection_from_cache(self, mock_get_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        manager._collections["books"] = mock_collection

        collection = manager.get_collection(CollectionNames.BOOKS)

        assert collection == mock_collection
        mock_client.get_collection.assert_not_called()

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_get_collection_from_client(self, mock_get_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        collection = manager.get_collection(CollectionNames.BOOKS)

        assert collection == mock_collection
        mock_client.collection_exists.assert_called_once_with(collection_name="books")
        mock_client.get_collection.assert_called_once_with(collection_name="books")
        assert manager._collections["books"] == mock_collection

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_get_collection_not_found(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = mock_client

        manager = CollectionManager()

        with pytest.raises(ValueError, match="does not exist"):
            manager.get_collection(CollectionNames.BOOKS)

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_reset_collection(self, mock_get_client):
        mock_client = MagicMock()
        old_mock_collection = MagicMock()
        new_mock_collection = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.get_collection.return_value = new_mock_collection
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        manager._collections["books"] = old_mock_collection

        manager.reset_collection(CollectionNames.BOOKS)

        mock_client.delete_collection.assert_called_once_with(collection_name="books")
        mock_client.create_collection.assert_called_once()
        assert manager._collections["books"] == new_mock_collection
        assert manager._collections["books"] != old_mock_collection

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_list_collections(self, mock_get_client):
        mock_client = MagicMock()
        response = MagicMock()
        col1 = MagicMock()
        col1.name = "books"
        col2 = MagicMock()
        col2.name = "users"
        response.collections = [col1, col2]
        mock_client.get_collections.return_value = response
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        collections = manager.list_collections()

        assert collections == ["books", "users"]

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_collection_exists_true(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        assert manager.collection_exists(CollectionNames.BOOKS) is True

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_collection_exists_false(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.side_effect = Exception("Not found")
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        assert manager.collection_exists(CollectionNames.BOOKS) is False

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_get_collection_count(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        count_result = MagicMock()
        count_result.count = 42
        mock_client.count.return_value = count_result
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        count = manager.get_collection_count(CollectionNames.BOOKS)

        assert count == 42
        mock_client.count.assert_called_once_with(collection_name="books", exact=True)

    @patch("bookdb.vector_db.collections.get_qdrant_client")
    def test_get_collection_count_missing_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = mock_client

        manager = CollectionManager()
        with pytest.raises(ValueError, match="does not exist"):
            manager.get_collection_count(CollectionNames.BOOKS)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch("bookdb.vector_db.collections.CollectionManager")
    def test_initialize_all_collections(self, mock_manager_class):
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        result = initialize_all_collections()

        assert result == mock_manager
        mock_manager.initialize_collections.assert_called_once()

    @patch("bookdb.vector_db.collections.CollectionManager")
    def test_get_books_collection(self, mock_manager_class):
        mock_manager = MagicMock()
        mock_collection = MagicMock()
        mock_manager.get_collection.return_value = mock_collection
        mock_manager_class.return_value = mock_manager

        result = get_books_collection()

        assert result == mock_collection
        mock_manager.get_collection.assert_called_once_with(CollectionNames.BOOKS)

    @patch("bookdb.vector_db.collections.CollectionManager")
    def test_get_users_collection(self, mock_manager_class):
        mock_manager = MagicMock()
        mock_collection = MagicMock()
        mock_manager.get_collection.return_value = mock_collection
        mock_manager_class.return_value = mock_manager

        result = get_users_collection()

        assert result == mock_collection
        mock_manager.get_collection.assert_called_once_with(CollectionNames.USERS)
