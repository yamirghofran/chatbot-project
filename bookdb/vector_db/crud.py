"""Base CRUD operations for vector collections."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from qdrant_client.models import PointIdsList, PointStruct

from .client import get_qdrant_client


class BaseVectorCRUD:
    """Base CRUD operations for vector collections.

    Qdrant is the primary backend. A temporary legacy fallback is kept for
    still-unmigrated call sites that pass Chroma-like collection objects.
    """

    def __init__(
        self,
        collection: Any,
        client: Any | None = None,
        vector_size: int | None = None,
    ):
        """Initialize CRUD operations for a collection.

        Args:
            collection: Collection reference. Prefer collection name (str) for Qdrant.
            client: Optional Qdrant client override (primarily for tests).
            vector_size: Optional vector dimensionality for zero-vector fallback.
        """
        self.collection = collection
        self.collection_name = self._resolve_collection_name(collection)
        self.vector_size = vector_size or int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
        self._legacy_mode = self._is_legacy_collection(collection)
        self.client = None if self._legacy_mode else (client or get_qdrant_client())

    def add(
        self,
        id: str,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a single item to the collection."""
        if self._legacy_mode:
            self._legacy_add(id=id, document=document, metadata=metadata, embedding=embedding)
            return

        if self.exists(id):
            raise ValueError(f"Item with ID '{id}' already exists")

        point = self._make_point(
            id=id,
            document=document,
            metadata=metadata,
            embedding=embedding,
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True,
        )

    def add_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Add multiple items to the collection in batch."""
        if self._legacy_mode:
            self._legacy_add_batch(ids, documents, metadatas, embeddings)
            return

        if metadatas and len(metadatas) != len(ids):
            raise ValueError("Length of metadatas must match length of ids")
        if embeddings and len(embeddings) != len(ids):
            raise ValueError("Length of embeddings must match length of ids")
        if len(documents) != len(ids):
            raise ValueError("Length of documents must match length of ids")

        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = [str(record.id) for record in existing]
        if existing_ids:
            raise ValueError(f"Items with IDs already exist: {existing_ids}")

        points = []
        for idx, id_ in enumerate(ids):
            points.append(
                self._make_point(
                    id=id_,
                    document=documents[idx],
                    metadata=metadatas[idx] if metadatas else None,
                    embedding=embeddings[idx] if embeddings else None,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single item by ID."""
        if self._legacy_mode:
            return self._legacy_get(id)

        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None
        return self._record_to_item(records[0])

    def get_batch(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple items by IDs."""
        if self._legacy_mode:
            return self._legacy_get_batch(ids)

        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=True,
        )
        return [self._record_to_item(record) for record in records]

    def update(
        self,
        id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Update an existing item."""
        if self._legacy_mode:
            self._legacy_update(id=id, document=document, metadata=metadata, embedding=embedding)
            return

        existing = self.get(id)
        if not existing:
            raise ValueError(f"Item with ID '{id}' does not exist")

        merged_document = existing["document"] if document is None else document
        merged_metadata = existing["metadata"] if metadata is None else metadata
        merged_embedding = existing["embedding"] if embedding is None else embedding

        point = self._make_point(
            id=id,
            document=merged_document,
            metadata=merged_metadata,
            embedding=merged_embedding,
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True,
        )

    def delete(self, id: str) -> None:
        """Delete a single item from the collection."""
        if self._legacy_mode:
            self._legacy_delete(id)
            return

        if not self.exists(id):
            raise ValueError(f"Item with ID '{id}' does not exist")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[id]),
            wait=True,
        )

    def delete_batch(self, ids: List[str]) -> None:
        """Delete multiple items from the collection."""
        if self._legacy_mode:
            self._legacy_delete_batch(ids)
            return

        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(record.id) for record in existing}
        missing_ids = [id_ for id_ in ids if id_ not in existing_ids]
        if missing_ids:
            raise ValueError(f"Items with IDs do not exist: {missing_ids}")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
            wait=True,
        )

    def exists(self, id: str) -> bool:
        """Check if an item exists in the collection."""
        if self._legacy_mode:
            try:
                result = self.collection.get(ids=[id])
                return len(result["ids"]) > 0
            except Exception:
                return False

        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id],
                with_payload=False,
                with_vectors=False,
            )
            return len(records) > 0
        except Exception:
            return False

    def count(self) -> int:
        """Get the total number of items in the collection."""
        if self._legacy_mode:
            return self._legacy_count()

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return int(result.count)

    def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get all items from the collection."""
        if self._legacy_mode:
            return self._legacy_get_all(limit=limit, offset=offset)

        skip = max(0, offset or 0)
        items: List[Dict[str, Any]] = []
        skipped = 0
        next_offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=next_offset,
                limit=1000,
                with_payload=True,
                with_vectors=True,
            )

            for point in points:
                if skipped < skip:
                    skipped += 1
                    continue

                items.append(self._record_to_item(point))
                if limit is not None and len(items) >= limit:
                    return items

            if next_offset is None:
                break

        return items

    def _resolve_collection_name(self, collection: Any) -> str:
        if isinstance(collection, str):
            return collection

        name = getattr(collection, "name", None)
        if isinstance(name, str) and name:
            return name

        return str(collection)

    def _is_legacy_collection(self, collection: Any) -> bool:
        return (
            hasattr(collection, "add")
            and hasattr(collection, "get")
            and hasattr(collection, "update")
            and hasattr(collection, "delete")
            and hasattr(collection, "count")
        )

    def _make_point(
        self,
        id: str,
        document: Optional[str],
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[List[float]],
    ) -> PointStruct:
        payload: Dict[str, Any] = {}
        if document is not None:
            payload["document"] = document
        if metadata is not None:
            payload["metadata"] = metadata

        vector = embedding if embedding is not None else self._zero_vector()
        return PointStruct(id=id, vector=vector, payload=payload)

    def _zero_vector(self) -> List[float]:
        return [0.0] * self.vector_size

    def _record_to_item(self, record: Any) -> Dict[str, Any]:
        payload = record.payload or {}
        vector = record.vector
        if isinstance(vector, dict):
            vector = next(iter(vector.values()), None)

        return {
            "id": str(record.id),
            "document": payload.get("document"),
            "metadata": payload.get("metadata"),
            "embedding": vector,
        }

    # --- Temporary legacy compatibility helpers ---
    def _legacy_add(
        self,
        id: str,
        document: str,
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[List[float]],
    ) -> None:
        if self.exists(id):
            raise ValueError(f"Item with ID '{id}' already exists")
        self.collection.add(
            ids=[id],
            documents=[document],
            metadatas=[metadata] if metadata else None,
            embeddings=[embedding] if embedding else None,
        )

    def _legacy_add_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
        embeddings: Optional[List[List[float]]],
    ) -> None:
        if metadatas and len(metadatas) != len(ids):
            raise ValueError("Length of metadatas must match length of ids")
        if embeddings and len(embeddings) != len(ids):
            raise ValueError("Length of embeddings must match length of ids")
        if len(documents) != len(ids):
            raise ValueError("Length of documents must match length of ids")

        existing_items = self.collection.get(ids=ids)
        if existing_items["ids"]:
            raise ValueError(f"Items with IDs already exist: {existing_items['ids']}")

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def _legacy_get(self, id: str) -> Optional[Dict[str, Any]]:
        result = self.collection.get(
            ids=[id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "document": result["documents"][0] if result["documents"] else None,
            "metadata": result["metadatas"][0] if result["metadatas"] else None,
            "embedding": result["embeddings"][0] if result["embeddings"] else None,
        }

    def _legacy_get_batch(self, ids: List[str]) -> List[Dict[str, Any]]:
        result = self.collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"],
        )
        items: List[Dict[str, Any]] = []
        for i in range(len(result["ids"])):
            items.append({
                "id": result["ids"][i],
                "document": result["documents"][i] if result["documents"] else None,
                "metadata": result["metadatas"][i] if result["metadatas"] else None,
                "embedding": result["embeddings"][i] if result["embeddings"] else None,
            })
        return items

    def _legacy_update(
        self,
        id: str,
        document: Optional[str],
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[List[float]],
    ) -> None:
        if not self.exists(id):
            raise ValueError(f"Item with ID '{id}' does not exist")
        self.collection.update(
            ids=[id],
            documents=[document] if document else None,
            metadatas=[metadata] if metadata else None,
            embeddings=[embedding] if embedding else None,
        )

    def _legacy_delete(self, id: str) -> None:
        if not self.exists(id):
            raise ValueError(f"Item with ID '{id}' does not exist")
        self.collection.delete(ids=[id])

    def _legacy_delete_batch(self, ids: List[str]) -> None:
        missing_ids = [id_ for id_ in ids if not self.exists(id_)]
        if missing_ids:
            raise ValueError(f"Items with IDs do not exist: {missing_ids}")
        self.collection.delete(ids=ids)

    def _legacy_count(self) -> int:
        return self.collection.count()

    def _legacy_get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        result = self.collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=limit,
            offset=offset,
        )
        items: List[Dict[str, Any]] = []
        for i in range(len(result["ids"])):
            items.append({
                "id": result["ids"][i],
                "document": result["documents"][i] if result["documents"] else None,
                "metadata": result["metadatas"][i] if result["metadatas"] else None,
                "embedding": result["embeddings"][i] if result["embeddings"] else None,
            })
        return items
