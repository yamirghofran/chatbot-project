"""Tests for base CRUD operations."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bookdb.vector_db.crud import BaseVectorCRUD


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
    return SimpleNamespace(
        id=id,
        payload=payload,
        vector=embedding,
    )


class TestBaseVectorCRUDQdrant:
    """Tests for BaseVectorCRUD Qdrant path."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def crud(self, mock_client):
        with patch("bookdb.vector_db.crud.get_qdrant_client", return_value=mock_client):
            yield BaseVectorCRUD("test_collection", vector_size=3)

    def test_add_success(self, crud, mock_client):
        mock_client.retrieve.return_value = []

        crud.add(
            id="test_1",
            document="Test document",
            metadata={"title": "Test"},
        )

        mock_client.upsert.assert_called_once()
        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "test_collection"
        point = kwargs["points"][0]
        assert point.id == "test_1"
        assert point.payload["document"] == "Test document"
        assert point.payload["metadata"] == {"title": "Test"}

    def test_add_with_embedding(self, crud, mock_client):
        mock_client.retrieve.return_value = []
        embedding = [0.1, 0.2, 0.3]

        crud.add(
            id="test_1",
            document="Test",
            embedding=embedding,
        )

        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.vector == embedding

    def test_add_uses_zero_vector_when_embedding_missing(self, crud, mock_client):
        mock_client.retrieve.return_value = []

        crud.add(id="test_1", document="Test")

        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.vector == [0.0, 0.0, 0.0]

    def test_add_duplicate_id(self, crud, mock_client):
        mock_client.retrieve.return_value = [_record(id="test_1")]

        with pytest.raises(ValueError, match="already exists"):
            crud.add(id="test_1", document="Test")

    def test_add_batch_success(self, crud, mock_client):
        mock_client.retrieve.return_value = []

        crud.add_batch(
            ids=["test_1", "test_2"],
            documents=["Doc 1", "Doc 2"],
            metadatas=[{"title": "Test 1"}, {"title": "Test 2"}],
        )

        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "test_collection"
        assert len(kwargs["points"]) == 2
        assert kwargs["points"][0].id == "test_1"
        assert kwargs["points"][1].id == "test_2"

    def test_add_batch_mismatched_lengths(self, crud):
        with pytest.raises(ValueError, match="Length of metadatas must match"):
            crud.add_batch(
                ids=["test_1", "test_2"],
                documents=["Doc 1", "Doc 2"],
                metadatas=[{"title": "Test 1"}],
            )

    def test_add_batch_duplicate_ids(self, crud, mock_client):
        mock_client.retrieve.return_value = [_record(id="test_1")]

        with pytest.raises(ValueError, match="already exist"):
            crud.add_batch(
                ids=["test_1", "test_2"],
                documents=["Doc 1", "Doc 2"],
            )

    def test_get_success(self, crud, mock_client):
        mock_client.retrieve.return_value = [
            _record(
                id="test_1",
                document="Test document",
                metadata={"title": "Test"},
                embedding=[0.1, 0.2, 0.3],
            )
        ]

        result = crud.get("test_1")

        assert result["id"] == "test_1"
        assert result["document"] == "Test document"
        assert result["metadata"] == {"title": "Test"}
        assert result["embedding"] == [0.1, 0.2, 0.3]

    def test_get_not_found(self, crud, mock_client):
        mock_client.retrieve.return_value = []

        result = crud.get("nonexistent")

        assert result is None

    def test_get_batch_success(self, crud, mock_client):
        mock_client.retrieve.return_value = [
            _record("test_1", "Doc 1", {"title": "T1"}, [0.1, 0.2]),
            _record("test_2", "Doc 2", {"title": "T2"}, [0.3, 0.4]),
        ]

        results = crud.get_batch(["test_1", "test_2"])

        assert len(results) == 2
        assert results[0]["id"] == "test_1"
        assert results[1]["id"] == "test_2"

    def test_update_success(self, crud, mock_client):
        mock_client.retrieve.side_effect = [
            [_record("test_1", "Old", {"title": "Old"}, [0.1, 0.2, 0.3])],
            [_record("test_1", "Old", {"title": "Old"}, [0.1, 0.2, 0.3])],
        ]

        crud.update(
            id="test_1",
            document="Updated document",
            metadata={"title": "Updated"},
        )

        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.payload["document"] == "Updated document"
        assert point.payload["metadata"] == {"title": "Updated"}

    def test_update_nonexistent(self, crud, mock_client):
        mock_client.retrieve.return_value = []

        with pytest.raises(ValueError, match="does not exist"):
            crud.update(id="nonexistent", document="Test")

    def test_update_partial(self, crud, mock_client):
        mock_client.retrieve.side_effect = [
            [_record("test_1", "Old", {"title": "Old"}, [0.1, 0.2, 0.3])],
            [_record("test_1", "Old", {"title": "Old"}, [0.1, 0.2, 0.3])],
        ]

        crud.update(id="test_1", metadata={"title": "New Title"})

        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert point.payload["document"] == "Old"
        assert point.payload["metadata"] == {"title": "New Title"}
        assert point.vector == [0.1, 0.2, 0.3]

    def test_delete_success(self, crud, mock_client):
        mock_client.retrieve.return_value = [_record("test_1")]

        crud.delete("test_1")

        kwargs = mock_client.delete.call_args.kwargs
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["points_selector"].points == ["test_1"]

    def test_delete_nonexistent(self, crud, mock_client):
        mock_client.retrieve.return_value = []

        with pytest.raises(ValueError, match="does not exist"):
            crud.delete("nonexistent")

    def test_delete_batch_success(self, crud, mock_client):
        mock_client.retrieve.return_value = [_record("test_1"), _record("test_2")]

        crud.delete_batch(["test_1", "test_2"])

        kwargs = mock_client.delete.call_args.kwargs
        assert kwargs["points_selector"].points == ["test_1", "test_2"]

    def test_delete_batch_missing_items(self, crud, mock_client):
        mock_client.retrieve.return_value = [_record("test_1")]

        with pytest.raises(ValueError, match="do not exist"):
            crud.delete_batch(["test_1", "test_2"])

    def test_exists_true(self, crud, mock_client):
        mock_client.retrieve.return_value = [_record("test_1")]
        assert crud.exists("test_1") is True

    def test_exists_false(self, crud, mock_client):
        mock_client.retrieve.return_value = []
        assert crud.exists("nonexistent") is False

    def test_exists_handles_exception(self, crud, mock_client):
        mock_client.retrieve.side_effect = Exception("Connection error")
        assert crud.exists("test_1") is False

    def test_count(self, crud, mock_client):
        mock_client.count.return_value = SimpleNamespace(count=42)

        count = crud.count()

        assert count == 42
        mock_client.count.assert_called_once_with(collection_name="test_collection", exact=True)

    def test_get_all(self, crud, mock_client):
        mock_client.scroll.side_effect = [
            ([
                _record("test_1", "Doc 1", {"title": "Test 1"}, [0.1]),
                _record("test_2", "Doc 2", {"title": "Test 2"}, [0.2]),
            ], "next"),
            ([
                _record("test_3", "Doc 3", {"title": "Test 3"}, [0.3]),
            ], None),
        ]

        results = crud.get_all()

        assert len(results) == 3
        assert results[0]["id"] == "test_1"
        assert results[2]["metadata"]["title"] == "Test 3"

    def test_get_all_with_limit(self, crud, mock_client):
        mock_client.scroll.return_value = ([
            _record("test_1", "Doc 1", {"title": "Test 1"}, [0.1]),
            _record("test_2", "Doc 2", {"title": "Test 2"}, [0.2]),
            _record("test_3", "Doc 3", {"title": "Test 3"}, [0.3]),
        ], None)

        results = crud.get_all(limit=2)

        assert len(results) == 2
        assert results[1]["id"] == "test_2"

    def test_get_all_with_offset(self, crud, mock_client):
        mock_client.scroll.return_value = ([
            _record("test_1", "Doc 1", {"title": "Test 1"}, [0.1]),
            _record("test_2", "Doc 2", {"title": "Test 2"}, [0.2]),
            _record("test_3", "Doc 3", {"title": "Test 3"}, [0.3]),
        ], None)

        results = crud.get_all(offset=2)

        assert len(results) == 1
        assert results[0]["id"] == "test_3"
