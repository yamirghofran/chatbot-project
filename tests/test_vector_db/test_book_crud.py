"""Tests for book CRUD operations."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from bookdb.vector_db.book_crud import BookVectorCRUD
from bookdb.vector_db.schemas import BookMetadata


class TestBookVectorCRUD:
    """Tests for BookVectorCRUD class."""
    
    @pytest.fixture
    def mock_collection(self):
        """Create a mock ChromaDB collection."""
        collection = MagicMock()
        collection.count.return_value = 0
        return collection
    
    @pytest.fixture
    def book_crud(self, mock_collection):
        """Create a BookVectorCRUD instance with mock collection."""
        return BookVectorCRUD(mock_collection)
    
    def test_init(self, mock_collection):
        """Test BookVectorCRUD initialization."""
        crud = BookVectorCRUD(mock_collection)
        assert crud.collection == mock_collection
    
    def test_add_book_minimal(self, book_crud, mock_collection):
        """Test adding a book with minimal required fields."""
        mock_collection.get.return_value = {"ids": []}  # Book doesn't exist
        
        book_crud.add_book(
            book_id="book_123",
            title="Test Book",
            author="Test Author",
        )
        
        # Verify add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        # Check arguments
        assert call_args[1]["ids"] == ["book_123"]
        assert "Test Book" in call_args[1]["documents"][0]
        assert "Test Author" in call_args[1]["documents"][0]
        
        # Check metadata
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "Test Book"
        assert metadata["author"] == "Test Author"
        assert metadata["language"] == "en"  # default
    
    def test_add_book_full(self, book_crud, mock_collection):
        """Test adding a book with all fields."""
        mock_collection.get.return_value = {"ids": []}
        
        book_crud.add_book(
            book_id="book_456",
            title="The Great Gatsby",
            author="F. Scott Fitzgerald",
            description="A novel about the American Dream",
            genre="Fiction",
            publication_year=1925,
            isbn="978-0743273565",
            language="en",
            page_count=180,
            average_rating=4.5,
        )
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        # Check metadata
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "The Great Gatsby"
        assert metadata["author"] == "F. Scott Fitzgerald"
        assert metadata["genre"] == "Fiction"
        assert metadata["publication_year"] == 1925
        assert metadata["isbn"] == "978-0743273565"
        assert metadata["page_count"] == 180
        assert metadata["average_rating"] == 4.5
    
    def test_add_book_with_embedding(self, book_crud, mock_collection):
        """Test adding a book with pre-computed embedding."""
        mock_collection.get.return_value = {"ids": []}
        embedding = [0.1, 0.2, 0.3, 0.4]
        
        book_crud.add_book(
            book_id="book_789",
            title="Test Book",
            author="Test Author",
            embedding=embedding,
        )
        
        call_args = mock_collection.add.call_args
        assert call_args[1]["embeddings"] == [embedding]
    
    def test_add_book_duplicate_id(self, book_crud, mock_collection):
        """Test error when adding book with duplicate ID."""
        mock_collection.get.return_value = {"ids": ["book_123"]}  # Already exists
        
        with pytest.raises(ValueError, match="already exists"):
            book_crud.add_book(
                book_id="book_123",
                title="Test Book",
                author="Test Author",
            )
    
    def test_add_book_invalid_metadata(self, book_crud, mock_collection):
        """Test error with invalid metadata."""
        mock_collection.get.return_value = {"ids": []}
        
        with pytest.raises(ValueError):
            book_crud.add_book(
                book_id="book_123",
                title="Test Book",
                author="Test Author",
                publication_year=999,  # Invalid year
            )
    
    def test_add_book_with_description(self, book_crud, mock_collection):
        """Test document uses description when provided."""
        mock_collection.get.return_value = {"ids": []}
        
        book_crud.add_book(
            book_id="book_123",
            title="1984",
            author="George Orwell",
            description="A dystopian novel",
            genre="Dystopian",
        )
        
        call_args = mock_collection.add.call_args
        document = call_args[1]["documents"][0]
        
        assert document == "A dystopian novel"
    
    def test_update_book_single_field(self, book_crud, mock_collection):
        """Test updating a single book field."""
        # Mock existing book - first call is from get(), second is from update()'s exists check
        mock_collection.get.side_effect = [
            {  # get() call for existing data
                "ids": ["book_123"],
                "documents": ["Original doc"],
                "metadatas": [{
                    "title": "Original Title",
                    "author": "Original Author",
                    "genre": "Fiction",
                    "publication_year": 2020,
                    "language": "en",
                }],
                "embeddings": [[0.1, 0.2]],
            },
            {"ids": ["book_123"]},  # exists check in update()
        ]
        
        book_crud.update_book(
            book_id="book_123",
            average_rating=4.8,
        )
        
        mock_collection.update.assert_called_once()
        call_args = mock_collection.update.call_args
        
        # Check that only metadata was updated, not document
        assert call_args[1]["documents"] is None
        
        metadata = call_args[1]["metadatas"][0]
        assert metadata["average_rating"] == 4.8
        assert metadata["title"] == "Original Title"  # Unchanged
    
    def test_update_book_content_fields(self, book_crud, mock_collection):
        """Test updating content fields regenerates document."""
        mock_collection.get.side_effect = [
            {  # get() call for existing data
                "ids": ["book_123"],
                "documents": ["Original doc"],
                "metadatas": [{
                    "title": "Original Title",
                    "author": "Original Author",
                    "language": "en",
                }],
                "embeddings": [[0.1]],
            },
            {"ids": ["book_123"]},  # exists check in update()
        ]
        
        book_crud.update_book(
            book_id="book_123",
            title="New Title",
            description="New description",
        )
        
        call_args = mock_collection.update.call_args
        
        # Document should be updated
        assert call_args[1]["documents"] is not None
        document = call_args[1]["documents"][0]
        assert document == "New description"
    
    def test_update_book_nonexistent(self, book_crud, mock_collection):
        """Test error when updating non-existent book."""
        mock_collection.get.return_value = {"ids": []}  # Doesn't exist
        
        with pytest.raises(ValueError, match="does not exist"):
            book_crud.update_book(
                book_id="nonexistent",
                title="New Title",
            )
    
    def test_update_book_multiple_fields(self, book_crud, mock_collection):
        """Test updating multiple fields at once."""
        mock_collection.get.side_effect = [
            {  # get() call for existing data
                "ids": ["book_123"],
                "documents": ["Doc"],
                "metadatas": [{
                    "title": "Title",
                    "author": "Author",
                    "language": "en",
                }],
                "embeddings": [[0.1]],
            },
            {"ids": ["book_123"]},  # exists check in update()
        ]
        
        book_crud.update_book(
            book_id="book_123",
            title="New Title",
            author="New Author",
            genre="New Genre",
            publication_year=2023,
        )
        
        call_args = mock_collection.update.call_args
        metadata = call_args[1]["metadatas"][0]
        
        assert metadata["title"] == "New Title"
        assert metadata["author"] == "New Author"
        assert metadata["genre"] == "New Genre"
        assert metadata["publication_year"] == 2023
    
    def test_search_by_metadata_single_filter(self, book_crud, mock_collection):
        """Test searching by single metadata field."""
        mock_collection.get.return_value = {
            "ids": ["book_1", "book_2"],
            "documents": ["Doc 1", "Doc 2"],
            "metadatas": [
                {"title": "Book 1", "genre": "Fiction"},
                {"title": "Book 2", "genre": "Fiction"},
            ],
            "embeddings": [[0.1], [0.2]],
        }
        
        results = book_crud.search_by_metadata(genre="Fiction")
        
        assert len(results) == 2
        mock_collection.get.assert_called_once()
        call_args = mock_collection.get.call_args
        assert call_args[1]["where"]["genre"] == "Fiction"
    
    def test_search_by_metadata_multiple_filters(self, book_crud, mock_collection):
        """Test searching with multiple metadata filters."""
        mock_collection.get.return_value = {
            "ids": ["book_1"],
            "documents": ["Doc 1"],
            "metadatas": [{"title": "Book 1"}],
            "embeddings": [[0.1]],
        }
        
        results = book_crud.search_by_metadata(
            genre="Fiction",
            author="Test Author",
            language="en",
            min_rating=4.0,
        )
        
        call_args = mock_collection.get.call_args
        where_filter = call_args[1]["where"]
        
        assert where_filter["genre"] == "Fiction"
        assert where_filter["author"] == "Test Author"
        assert where_filter["language"] == "en"
        assert where_filter["average_rating"]["$gte"] == 4.0
    
    def test_search_by_metadata_year_range(self, book_crud, mock_collection):
        """Test searching with year range."""
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }
        
        book_crud.search_by_metadata(
            min_year=1900,
            max_year=2000,
        )
        
        call_args = mock_collection.get.call_args
        year_filter = call_args[1]["where"]["publication_year"]
        
        assert year_filter["$gte"] == 1900
        assert year_filter["$lte"] == 2000
    
    def test_search_by_metadata_with_limit(self, book_crud, mock_collection):
        """Test search with result limit."""
        mock_collection.get.return_value = {
            "ids": ["book_1", "book_2", "book_3"],
            "documents": ["Doc 1", "Doc 2", "Doc 3"],
            "metadatas": [{"title": "B1"}, {"title": "B2"}, {"title": "B3"}],
            "embeddings": [[0.1], [0.2], [0.3]],
        }
        
        results = book_crud.search_by_metadata(
            genre="Fiction",
            limit=5,
        )
        
        assert len(results) == 3
        call_args = mock_collection.get.call_args
        assert call_args[1]["limit"] == 5
    
    def test_search_by_metadata_no_filters(self, book_crud, mock_collection):
        """Test search without any filters returns all."""
        mock_collection.get.return_value = {
            "ids": ["book_1"],
            "documents": ["Doc 1"],
            "metadatas": [{"title": "Book 1"}],
            "embeddings": [[0.1]],
        }
        
        results = book_crud.search_by_metadata(limit=10)
        
        assert len(results) == 1
        # Should call get without where filter
        call_args = mock_collection.get.call_args
        assert "where" not in call_args[1] or not call_args[1].get("where")
    
    def test_search_by_metadata_empty_results(self, book_crud, mock_collection):
        """Test search returning no results."""
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }
        
        results = book_crud.search_by_metadata(genre="NonexistentGenre")
        
        assert len(results) == 0
        assert isinstance(results, list)
    
    def test_search_similar_books_not_implemented(self, book_crud):
        """Test that search_similar_books is TODO."""
        result = book_crud.search_similar_books(
            query_text="test query",
            n_results=5,
        )
        
        # Should return None (TODO not implemented)
        assert result is None
    
    def test_get_book_recommendations_not_implemented(self, book_crud):
        """Test that get_book_recommendations is TODO."""
        result = book_crud.get_book_recommendations(
            book_id="book_123",
            n_results=5,
        )
        
        # Should return None (TODO not implemented)
        assert result is None
