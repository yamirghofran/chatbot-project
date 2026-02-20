"""Tests for base CRUD operations."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from bookdb.vector_db.crud import BaseVectorCRUD


class TestBaseVectorCRUD:
    """Tests for BaseVectorCRUD class."""
    
    @pytest.fixture
    def mock_collection(self):
        """Create a mock ChromaDB collection."""
        collection = MagicMock()
        collection.count.return_value = 0
        return collection
    
    @pytest.fixture
    def crud(self, mock_collection):
        """Create a BaseVectorCRUD instance with mock collection."""
        return BaseVectorCRUD(mock_collection)
    
    def test_init(self, mock_collection):
        """Test CRUD initialization."""
        crud = BaseVectorCRUD(mock_collection)
        assert crud.collection == mock_collection
    
    def test_add_success(self, crud, mock_collection):
        """Test adding a single item successfully."""
        mock_collection.get.return_value = {"ids": []}  # ID doesn't exist
        
        crud.add(
            id="test_1",
            document="Test document",
            metadata={"title": "Test"},
        )
        
        mock_collection.add.assert_called_once_with(
            ids=["test_1"],
            documents=["Test document"],
            metadatas=[{"title": "Test"}],
            embeddings=None,
        )
    
    def test_add_with_embedding(self, crud, mock_collection):
        """Test adding item with pre-computed embedding."""
        mock_collection.get.return_value = {"ids": []}
        embedding = [0.1, 0.2, 0.3]
        
        crud.add(
            id="test_1",
            document="Test",
            embedding=embedding,
        )
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args[1]["embeddings"] == [embedding]
    
    def test_add_duplicate_id(self, crud, mock_collection):
        """Test error when adding duplicate ID."""
        mock_collection.get.return_value = {"ids": ["test_1"]}  # ID exists
        
        with pytest.raises(ValueError, match="already exists"):
            crud.add(id="test_1", document="Test")
    
    def test_add_batch_success(self, crud, mock_collection):
        """Test adding multiple items in batch."""
        mock_collection.get.side_effect = [
            {"ids": []},  # test_1 doesn't exist
            {"ids": []},  # test_2 doesn't exist
        ]
        
        crud.add_batch(
            ids=["test_1", "test_2"],
            documents=["Doc 1", "Doc 2"],
            metadatas=[{"title": "Test 1"}, {"title": "Test 2"}],
        )
        
        mock_collection.add.assert_called_once_with(
            ids=["test_1", "test_2"],
            documents=["Doc 1", "Doc 2"],
            metadatas=[{"title": "Test 1"}, {"title": "Test 2"}],
            embeddings=None,
        )
    
    def test_add_batch_mismatched_lengths(self, crud):
        """Test error when batch input lengths don't match."""
        with pytest.raises(ValueError, match="Length of metadatas must match"):
            crud.add_batch(
                ids=["test_1", "test_2"],
                documents=["Doc 1", "Doc 2"],
                metadatas=[{"title": "Test 1"}],  # Only one metadata
            )
    
    def test_add_batch_duplicate_ids(self, crud, mock_collection):
        """Test error when batch contains existing IDs."""
        mock_collection.get.side_effect = [
            {"ids": ["test_1"]},  # test_1 exists
            {"ids": []},  # test_2 doesn't exist
        ]
        
        with pytest.raises(ValueError, match="already exist"):
            crud.add_batch(
                ids=["test_1", "test_2"],
                documents=["Doc 1", "Doc 2"],
            )
    
    def test_get_success(self, crud, mock_collection):
        """Test retrieving a single item."""
        mock_collection.get.return_value = {
            "ids": ["test_1"],
            "documents": ["Test document"],
            "metadatas": [{"title": "Test"}],
            "embeddings": [[0.1, 0.2, 0.3]],
        }
        
        result = crud.get("test_1")
        
        assert result["id"] == "test_1"
        assert result["document"] == "Test document"
        assert result["metadata"] == {"title": "Test"}
        assert result["embedding"] == [0.1, 0.2, 0.3]
    
    def test_get_not_found(self, crud, mock_collection):
        """Test retrieving non-existent item."""
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }
        
        result = crud.get("nonexistent")
        
        assert result is None
    
    def test_get_batch_success(self, crud, mock_collection):
        """Test retrieving multiple items."""
        mock_collection.get.return_value = {
            "ids": ["test_1", "test_2"],
            "documents": ["Doc 1", "Doc 2"],
            "metadatas": [{"title": "Test 1"}, {"title": "Test 2"}],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }
        
        results = crud.get_batch(["test_1", "test_2"])
        
        assert len(results) == 2
        assert results[0]["id"] == "test_1"
        assert results[1]["id"] == "test_2"
        assert results[0]["metadata"]["title"] == "Test 1"
    
    def test_get_batch_partial_results(self, crud, mock_collection):
        """Test batch get with some missing items."""
        # Only test_1 exists, test_2 doesn't
        mock_collection.get.return_value = {
            "ids": ["test_1"],
            "documents": ["Doc 1"],
            "metadatas": [{"title": "Test 1"}],
            "embeddings": [[0.1, 0.2]],
        }
        
        results = crud.get_batch(["test_1", "test_2"])
        
        assert len(results) == 1
        assert results[0]["id"] == "test_1"
    
    def test_update_success(self, crud, mock_collection):
        """Test updating an existing item."""
        mock_collection.get.return_value = {"ids": ["test_1"]}  # Item exists
        
        crud.update(
            id="test_1",
            document="Updated document",
            metadata={"title": "Updated"},
        )
        
        mock_collection.update.assert_called_once_with(
            ids=["test_1"],
            documents=["Updated document"],
            metadatas=[{"title": "Updated"}],
            embeddings=None,
        )
    
    def test_update_nonexistent(self, crud, mock_collection):
        """Test error when updating non-existent item."""
        mock_collection.get.return_value = {"ids": []}  # Item doesn't exist
        
        with pytest.raises(ValueError, match="does not exist"):
            crud.update(id="nonexistent", document="Test")
    
    def test_update_partial(self, crud, mock_collection):
        """Test updating only some fields."""
        mock_collection.get.return_value = {"ids": ["test_1"]}
        
        crud.update(id="test_1", metadata={"title": "New Title"})
        
        call_args = mock_collection.update.call_args
        assert call_args[1]["documents"] is None
        assert call_args[1]["metadatas"] == [{"title": "New Title"}]
    
    def test_delete_success(self, crud, mock_collection):
        """Test deleting a single item."""
        mock_collection.get.return_value = {"ids": ["test_1"]}  # Item exists
        
        crud.delete("test_1")
        
        mock_collection.delete.assert_called_once_with(ids=["test_1"])
    
    def test_delete_nonexistent(self, crud, mock_collection):
        """Test error when deleting non-existent item."""
        mock_collection.get.return_value = {"ids": []}  # Item doesn't exist
        
        with pytest.raises(ValueError, match="does not exist"):
            crud.delete("nonexistent")
    
    def test_delete_batch_success(self, crud, mock_collection):
        """Test deleting multiple items."""
        mock_collection.get.side_effect = [
            {"ids": ["test_1"]},  # test_1 exists
            {"ids": ["test_2"]},  # test_2 exists
        ]
        
        crud.delete_batch(["test_1", "test_2"])
        
        mock_collection.delete.assert_called_once_with(ids=["test_1", "test_2"])
    
    def test_delete_batch_missing_items(self, crud, mock_collection):
        """Test error when batch delete includes non-existent items."""
        mock_collection.get.side_effect = [
            {"ids": ["test_1"]},  # test_1 exists
            {"ids": []},  # test_2 doesn't exist
        ]
        
        with pytest.raises(ValueError, match="do not exist"):
            crud.delete_batch(["test_1", "test_2"])
    
    def test_exists_true(self, crud, mock_collection):
        """Test exists returns True for existing item."""
        mock_collection.get.return_value = {"ids": ["test_1"]}
        
        assert crud.exists("test_1") is True
    
    def test_exists_false(self, crud, mock_collection):
        """Test exists returns False for non-existent item."""
        mock_collection.get.return_value = {"ids": []}
        
        assert crud.exists("nonexistent") is False
    
    def test_exists_handles_exception(self, crud, mock_collection):
        """Test exists returns False on exception."""
        mock_collection.get.side_effect = Exception("Connection error")
        
        assert crud.exists("test_1") is False
    
    def test_count(self, crud, mock_collection):
        """Test getting item count."""
        mock_collection.count.return_value = 42
        
        count = crud.count()
        
        assert count == 42
        mock_collection.count.assert_called_once()
    
    def test_count_zero(self, crud, mock_collection):
        """Test count with empty collection."""
        mock_collection.count.return_value = 0
        
        count = crud.count()
        
        assert count == 0
    
    def test_get_all(self, crud, mock_collection):
        """Test getting all items."""
        mock_collection.get.return_value = {
            "ids": ["test_1", "test_2", "test_3"],
            "documents": ["Doc 1", "Doc 2", "Doc 3"],
            "metadatas": [
                {"title": "Test 1"},
                {"title": "Test 2"},
                {"title": "Test 3"},
            ],
            "embeddings": [[0.1], [0.2], [0.3]],
        }
        
        results = crud.get_all()
        
        assert len(results) == 3
        assert results[0]["id"] == "test_1"
        assert results[2]["metadata"]["title"] == "Test 3"
    
    def test_get_all_with_limit(self, crud, mock_collection):
        """Test getting items with limit."""
        mock_collection.get.return_value = {
            "ids": ["test_1", "test_2"],
            "documents": ["Doc 1", "Doc 2"],
            "metadatas": [{"title": "Test 1"}, {"title": "Test 2"}],
            "embeddings": [[0.1], [0.2]],
        }
        
        results = crud.get_all(limit=2)
        
        assert len(results) == 2
        mock_collection.get.assert_called_once()
        call_args = mock_collection.get.call_args
        assert call_args[1]["limit"] == 2
    
    def test_get_all_with_offset(self, crud, mock_collection):
        """Test getting items with offset."""
        mock_collection.get.return_value = {
            "ids": ["test_3"],
            "documents": ["Doc 3"],
            "metadatas": [{"title": "Test 3"}],
            "embeddings": [[0.3]],
        }
        
        results = crud.get_all(offset=2)
        
        assert len(results) == 1
        call_args = mock_collection.get.call_args
        assert call_args[1]["offset"] == 2
    
    def test_add_exception_handling(self, crud, mock_collection):
        """Test exception handling during add."""
        mock_collection.get.return_value = {"ids": []}
        mock_collection.add.side_effect = Exception("DB error")
        
        with pytest.raises(Exception, match="Failed to add item"):
            crud.add(id="test_1", document="Test")
    
    def test_update_exception_handling(self, crud, mock_collection):
        """Test exception handling during update."""
        mock_collection.get.return_value = {"ids": ["test_1"]}
        mock_collection.update.side_effect = Exception("DB error")
        
        with pytest.raises(Exception, match="Failed to update item"):
            crud.update(id="test_1", document="Test")
    
    def test_delete_exception_handling(self, crud, mock_collection):
        """Test exception handling during delete."""
        mock_collection.get.return_value = {"ids": ["test_1"]}
        mock_collection.delete.side_effect = Exception("DB error")
        
        with pytest.raises(Exception, match="Failed to delete item"):
            crud.delete("test_1")
    
    def test_count_exception_handling(self, crud, mock_collection):
        """Test exception handling during count."""
        mock_collection.count.side_effect = Exception("DB error")
        
        with pytest.raises(Exception, match="Failed to count items"):
            crud.count()
