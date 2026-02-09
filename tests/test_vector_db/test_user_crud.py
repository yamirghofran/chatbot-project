"""Tests for user CRUD operations."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from bookdb.vector_db.user_crud import UserVectorCRUD
from bookdb.vector_db.schemas import UserMetadata


class TestUserVectorCRUD:
    """Tests for UserVectorCRUD class."""
    
    @pytest.fixture
    def mock_collection(self):
        """Create a mock ChromaDB collection."""
        collection = MagicMock()
        collection.count.return_value = 0
        return collection
    
    @pytest.fixture
    def user_crud(self, mock_collection):
        """Create a UserVectorCRUD instance with mock collection."""
        return UserVectorCRUD(mock_collection)
    
    def test_init(self, mock_collection):
        """Test UserVectorCRUD initialization."""
        crud = UserVectorCRUD(mock_collection)
        assert crud.collection == mock_collection
    
    def test_add_user_minimal(self, user_crud, mock_collection):
        """Test adding a user with minimal required fields."""
        mock_collection.get.return_value = {"ids": []}  # User doesn't exist
        
        user_crud.add_user(user_id=12345)
        
        # Verify add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        # Check ID format
        assert call_args[1]["ids"] == ["user_12345"]
        
        # Check metadata
        metadata = call_args[1]["metadatas"][0]
        assert metadata["user_id"] == 12345
        assert metadata["num_books_read"] == 0  # default
    
    def test_add_user_full(self, user_crud, mock_collection):
        """Test adding a user with all fields."""
        mock_collection.get.return_value = {"ids": []}
        
        user_crud.add_user(
            user_id=12345,
            preferences_text="I enjoy science fiction and mystery novels",
            favorite_genres="Science Fiction,Mystery,Thriller",
            num_books_read=42,
            average_rating_given=4.2,
            reading_level="advanced",
        )
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        # Check document
        assert call_args[1]["documents"][0] == "I enjoy science fiction and mystery novels"
        
        # Check metadata
        metadata = call_args[1]["metadatas"][0]
        assert metadata["user_id"] == 12345
        assert metadata["favorite_genres"] == "Science Fiction,Mystery,Thriller"
        assert metadata["num_books_read"] == 42
        assert metadata["average_rating_given"] == 4.2
        assert metadata["reading_level"] == "advanced"
    
    def test_add_user_with_embedding(self, user_crud, mock_collection):
        """Test adding a user with pre-computed embedding."""
        mock_collection.get.return_value = {"ids": []}
        embedding = [0.1, 0.2, 0.3, 0.4]
        
        user_crud.add_user(
            user_id=456,
            preferences_text="Test preferences",
            embedding=embedding,
        )
        
        call_args = mock_collection.add.call_args
        assert call_args[1]["embeddings"] == [embedding]
    
    def test_add_user_duplicate_id(self, user_crud, mock_collection):
        """Test error when adding user with duplicate ID."""
        mock_collection.get.return_value = {"ids": ["user_12345"]}  # Already exists
        
        with pytest.raises(ValueError, match="already exists"):
            user_crud.add_user(user_id=12345)
    
    def test_add_user_invalid_metadata(self, user_crud, mock_collection):
        """Test error with invalid metadata."""
        mock_collection.get.return_value = {"ids": []}
        
        with pytest.raises(ValueError):
            user_crud.add_user(
                user_id=0,  # Invalid user_id (must be >= 1)
            )
    
    def test_add_user_default_document(self, user_crud, mock_collection):
        """Test default document when no preferences provided."""
        mock_collection.get.return_value = {"ids": []}
        
        user_crud.add_user(user_id=789)
        
        call_args = mock_collection.add.call_args
        document = call_args[1]["documents"][0]
        assert "User preferences for user 789" in document
    
    def test_update_user_preferences_single_field(self, user_crud, mock_collection):
        """Test updating a single user field."""
        # Mock existing user
        mock_collection.get.side_effect = [
            {  # get() call for existing data
                "ids": ["user_12345"],
                "documents": ["Original preferences"],
                "metadatas": [{
                    "user_id": 12345,
                    "num_books_read": 10,
                    "favorite_genres": "Fiction",
                    "average_rating_given": 4.0,
                }],
                "embeddings": [[0.1, 0.2]],
            },
            {"ids": ["user_12345"]},  # exists check in update()
        ]
        
        user_crud.update_user_preferences(
            user_id=12345,
            num_books_read=15,
        )
        
        mock_collection.update.assert_called_once()
        call_args = mock_collection.update.call_args
        
        # Check that document was not updated (no preferences_text)
        assert call_args[1]["documents"] is None
        
        # Check metadata
        metadata = call_args[1]["metadatas"][0]
        assert metadata["num_books_read"] == 15
        assert metadata["favorite_genres"] == "Fiction"  # Unchanged
    
    def test_update_user_preferences_with_text(self, user_crud, mock_collection):
        """Test updating preferences text regenerates document."""
        mock_collection.get.side_effect = [
            {  # get() call
                "ids": ["user_12345"],
                "documents": ["Old preferences"],
                "metadatas": [{
                    "user_id": 12345,
                    "num_books_read": 10,
                }],
                "embeddings": [[0.1]],
            },
            {"ids": ["user_12345"]},  # exists check
        ]
        
        user_crud.update_user_preferences(
            user_id=12345,
            preferences_text="I now prefer fantasy novels",
        )
        
        call_args = mock_collection.update.call_args
        
        # Document should be updated
        assert call_args[1]["documents"] is not None
        document = call_args[1]["documents"][0]
        assert document == "I now prefer fantasy novels"
    
    def test_update_user_preferences_nonexistent(self, user_crud, mock_collection):
        """Test error when updating non-existent user."""
        mock_collection.get.return_value = {"ids": []}  # Doesn't exist
        
        with pytest.raises(ValueError, match="does not exist"):
            user_crud.update_user_preferences(
                user_id=99999,
                num_books_read=10,
            )
    
    def test_update_user_preferences_multiple_fields(self, user_crud, mock_collection):
        """Test updating multiple fields at once."""
        mock_collection.get.side_effect = [
            {  # get() call
                "ids": ["user_12345"],
                "documents": ["Preferences"],
                "metadatas": [{
                    "user_id": 12345,
                    "num_books_read": 10,
                }],
                "embeddings": [[0.1]],
            },
            {"ids": ["user_12345"]},  # exists check
        ]
        
        user_crud.update_user_preferences(
            user_id=12345,
            num_books_read=20,
            favorite_genres="Sci-Fi,Fantasy",
            average_rating_given=4.5,
            reading_level="expert",
        )
        
        call_args = mock_collection.update.call_args
        metadata = call_args[1]["metadatas"][0]
        
        assert metadata["num_books_read"] == 20
        assert metadata["favorite_genres"] == "Sci-Fi,Fantasy"
        assert metadata["average_rating_given"] == 4.5
        assert metadata["reading_level"] == "expert"
    
    def test_get_book_recommendations_for_user_not_implemented(self, user_crud):
        """Test that get_book_recommendations_for_user is TODO."""
        result = user_crud.get_book_recommendations_for_user(
            user_id=12345,
            n_results=5,
        )
        
        # Should return None (TODO not implemented)
        assert result is None
    
    def test_find_similar_users_not_implemented(self, user_crud):
        """Test that find_similar_users is TODO."""
        result = user_crud.find_similar_users(
            user_id=12345,
            n_results=5,
        )
        
        # Should return None (TODO not implemented)
        assert result is None
    
    def test_update_from_reading_history_not_implemented(self, user_crud):
        """Test that update_from_reading_history is TODO."""
        result = user_crud.update_from_reading_history(
            user_id=12345,
            book_ids=["book_1", "book_2", "book_3"],
        )
        
        # Should return None (TODO not implemented)
        assert result is None
