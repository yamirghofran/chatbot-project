"""Tests for vector collection management."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from bookdb.vector_db.collections import (
    CollectionManager,
    initialize_all_collections,
    get_books_collection,
    get_users_collection,
)
from bookdb.vector_db.schemas import (
    CollectionNames,
    BookMetadata,
    UserMetadata,
    validate_book_metadata,
    validate_user_metadata,
)


class TestBookMetadata:
    """Tests for BookMetadata schema."""
    
    def test_book_metadata_minimal(self):
        """Test creating book metadata with minimal required fields."""
        metadata = BookMetadata(
            title="Test Book",
            author="Test Author",
        )
        
        assert metadata.title == "Test Book"
        assert metadata.author == "Test Author"
        assert metadata.created_at is not None
    
    def test_book_metadata_full(self):
        """Test creating book metadata with all fields."""
        metadata = BookMetadata(
            title="The Great Gatsby",
            author="F. Scott Fitzgerald",
            genre="Fiction",
            publication_year=1925,
        )

        assert metadata.title == "The Great Gatsby"
        assert metadata.publication_year == 1925

    def test_book_metadata_invalid_year(self):
        """Test validation error for invalid publication year."""
        with pytest.raises(ValueError):
            BookMetadata(
                title="Test",
                author="Test",
                publication_year=999,  # Too old
            )

    def test_validate_book_metadata_function(self):
        """Test validate_book_metadata helper function."""
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
        """Test creating user metadata with required fields."""
        metadata = UserMetadata(user_id=123, name="Alice")
        
        assert metadata.user_id == 123
        assert metadata.name == "Alice"
        assert metadata.created_at is not None
    
    def test_user_metadata_invalid_user_id(self):
        """Test validation error for invalid user_id."""
        with pytest.raises(ValueError):
            UserMetadata(user_id=0, name="Alice")  # Must be >= 1
    
    def test_user_metadata_missing_name(self):
        """Test validation error when name is missing."""
        with pytest.raises(ValueError):
            UserMetadata(user_id=123)
    
    def test_validate_user_metadata_function(self):
        """Test validate_user_metadata helper function."""
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
        """Test collection name values."""
        assert CollectionNames.BOOKS.value == "books"
        assert CollectionNames.USERS.value == "users"
    
    def test_collection_names_enum(self):
        """Test that collection names are proper enum members."""
        assert isinstance(CollectionNames.BOOKS, CollectionNames)
        assert isinstance(CollectionNames.USERS, CollectionNames)


class TestCollectionManager:
    """Tests for CollectionManager class."""
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_collection_manager_init(self, mock_get_client):
        """Test CollectionManager initialization."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        
        assert manager.client == mock_client
        assert manager._collections == {}
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_initialize_collections(self, mock_get_client):
        """Test initializing all collections."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        manager.initialize_collections()
        
        # Should create all collections (books, users, reviews)
        assert mock_client.get_or_create_collection.call_count == 3
        
        # Check books collection creation
        calls = mock_client.get_or_create_collection.call_args_list
        books_call = calls[0]
        assert books_call[1]["name"] == "books"
        assert "cosine" in books_call[1]["metadata"]["hnsw:space"]
        
        # Check users collection creation
        users_call = calls[1]
        assert users_call[1]["name"] == "users"
        assert "cosine" in users_call[1]["metadata"]["hnsw:space"]
        
        # Check reviews collection creation
        reviews_call = calls[2]
        assert reviews_call[1]["name"] == "reviews"
        assert "cosine" in reviews_call[1]["metadata"]["hnsw:space"]
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_get_collection_from_cache(self, mock_get_client):
        """Test getting collection from cache."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        manager._collections["books"] = mock_collection
        
        collection = manager.get_collection(CollectionNames.BOOKS)
        
        assert collection == mock_collection
        # Should not call client since it's cached
        mock_client.get_collection.assert_not_called()
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_get_collection_from_client(self, mock_get_client):
        """Test getting collection from the backing client."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        collection = manager.get_collection(CollectionNames.BOOKS)
        
        assert collection == mock_collection
        mock_client.get_collection.assert_called_once_with(name="books")
        # Should be cached now
        assert manager._collections["books"] == mock_collection
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_get_collection_not_found(self, mock_get_client):
        """Test error when collection doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        
        with pytest.raises(ValueError, match="does not exist"):
            manager.get_collection(CollectionNames.BOOKS)
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_reset_collection(self, mock_get_client):
        """Test resetting a collection."""
        mock_client = MagicMock()
        old_mock_collection = MagicMock()
        new_mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = new_mock_collection
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        manager._collections["books"] = old_mock_collection
        
        manager.reset_collection(CollectionNames.BOOKS)
        
        # Should delete the collection
        mock_client.delete_collection.assert_called_once_with(name="books")
        # Should recreate the collection (now in cache with new instance)
        mock_client.get_or_create_collection.assert_called_once()
        # Should have new collection in cache
        assert manager._collections["books"] == new_mock_collection
        assert manager._collections["books"] != old_mock_collection
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_list_collections(self, mock_get_client):
        """Test listing all collections."""
        mock_client = MagicMock()
        mock_col1 = MagicMock()
        mock_col1.name = "books"
        mock_col2 = MagicMock()
        mock_col2.name = "users"
        mock_client.list_collections.return_value = [mock_col1, mock_col2]
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        collections = manager.list_collections()
        
        assert collections == ["books", "users"]
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_collection_exists_true(self, mock_get_client):
        """Test checking if collection exists (positive case)."""
        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock()
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        exists = manager.collection_exists(CollectionNames.BOOKS)
        
        assert exists is True
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_collection_exists_false(self, mock_get_client):
        """Test checking if collection exists (negative case)."""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        exists = manager.collection_exists(CollectionNames.BOOKS)
        
        assert exists is False
    
    @patch("bookdb.vector_db.collections.get_chroma_client")
    def test_get_collection_count(self, mock_get_client):
        """Test getting collection item count."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        
        manager = CollectionManager()
        count = manager.get_collection_count(CollectionNames.BOOKS)
        
        assert count == 42


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @patch("bookdb.vector_db.collections.CollectionManager")
    def test_initialize_all_collections(self, mock_manager_class):
        """Test initialize_all_collections convenience function."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        result = initialize_all_collections()
        
        assert result == mock_manager
        mock_manager.initialize_collections.assert_called_once()
    
    @patch("bookdb.vector_db.collections.CollectionManager")
    def test_get_books_collection(self, mock_manager_class):
        """Test get_books_collection convenience function."""
        mock_manager = MagicMock()
        mock_collection = MagicMock()
        mock_manager.get_collection.return_value = mock_collection
        mock_manager_class.return_value = mock_manager
        
        result = get_books_collection()
        
        assert result == mock_collection
        mock_manager.get_collection.assert_called_once_with(CollectionNames.BOOKS)
    
    @patch("bookdb.vector_db.collections.CollectionManager")
    def test_get_users_collection(self, mock_manager_class):
        """Test get_users_collection convenience function."""
        mock_manager = MagicMock()
        mock_collection = MagicMock()
        mock_manager.get_collection.return_value = mock_collection
        mock_manager_class.return_value = mock_manager
        
        result = get_users_collection()
        
        assert result == mock_collection
        mock_manager.get_collection.assert_called_once_with(CollectionNames.USERS)
