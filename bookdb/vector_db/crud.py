"""Base CRUD operations for ChromaDB collections."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from chromadb import Collection
from chromadb.api.types import GetResult, QueryResult


class BaseVectorCRUD(ABC):
    """Abstract base class for CRUD operations on ChromaDB collections.
    
    This class provides common CRUD operations that can be used with any
    ChromaDB collection. Subclasses should implement collection-specific
    logic for embedding generation and metadata handling.
    
    Attributes:
        collection: ChromaDB collection instance
    """
    
    def __init__(self, collection: Collection):
        """Initialize CRUD operations for a collection.
        
        Args:
            collection: ChromaDB collection instance
        """
        self.collection = collection
    
    def add(
        self,
        id: str,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a single item to the collection.
        
        Args:
            id: Unique identifier for the item
            document: Text document associated with the embedding
            metadata: Optional metadata dictionary
            embedding: Optional pre-computed embedding vector
        
        Raises:
            ValueError: If item with ID already exists
            Exception: If addition fails
        
        Example:
            >>> crud.add(
            ...     id="book_123",
            ...     document="A great book about...",
            ...     metadata={"title": "Book Title", "author": "Author Name"},
            ... )
        """
        # Check if ID already exists
        if self.exists(id):
            raise ValueError(f"Item with ID '{id}' already exists")
        
        try:
            self.collection.add(
                ids=[id],
                documents=[document],
                metadatas=[metadata] if metadata else None,
                embeddings=[embedding] if embedding else None,
            )
        except Exception as e:
            raise Exception(f"Failed to add item '{id}': {str(e)}") from e
    
    def add_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Add multiple items to the collection in batch.
        
        Args:
            ids: List of unique identifiers
            documents: List of text documents
            metadatas: Optional list of metadata dictionaries
            embeddings: Optional list of embedding vectors
        
        Raises:
            ValueError: If lengths don't match or duplicate IDs exist
            Exception: If batch addition fails
        
        Example:
            >>> crud.add_batch(
            ...     ids=["book_1", "book_2"],
            ...     documents=["Doc 1", "Doc 2"],
            ...     metadatas=[{"title": "Book 1"}, {"title": "Book 2"}],
            ... )
        """
        # Validate input lengths
        if metadatas and len(metadatas) != len(ids):
            raise ValueError("Length of metadatas must match length of ids")
        if embeddings and len(embeddings) != len(ids):
            raise ValueError("Length of embeddings must match length of ids")
        if len(documents) != len(ids):
            raise ValueError("Length of documents must match length of ids")
        
        # Check for existing IDs
        existing_ids = [id for id in ids if self.exists(id)]
        if existing_ids:
            raise ValueError(f"Items with IDs already exist: {existing_ids}")
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as e:
            raise Exception(f"Failed to add batch: {str(e)}") from e
    
    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single item by ID.
        
        Args:
            id: Unique identifier of the item
        
        Returns:
            Dictionary containing the item data, or None if not found
            Keys include: id, document, metadata, embedding (if included)
        
        Example:
            >>> item = crud.get("book_123")
            >>> print(item["metadata"]["title"])
        """
        try:
            result = self.collection.get(
                ids=[id],
                include=["documents", "metadatas", "embeddings"],
            )
            
            if not result["ids"]:
                return None
            
            return {
                "id": result["ids"][0],
                "document": result["documents"][0] if result["documents"] is not None and len(result["documents"]) > 0 else None,
                "metadata": result["metadatas"][0] if result["metadatas"] is not None and len(result["metadatas"]) > 0 else None,
                "embedding": result["embeddings"][0] if result["embeddings"] is not None and len(result["embeddings"]) > 0 else None,
            }
        except Exception as e:
            raise Exception(f"Failed to get item '{id}': {str(e)}") from e
    
    def get_batch(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple items by IDs.
        
        Args:
            ids: List of unique identifiers
        
        Returns:
            List of dictionaries containing item data
            Only returns items that exist (may be shorter than input list)
        
        Example:
            >>> items = crud.get_batch(["book_1", "book_2", "book_3"])
            >>> for item in items:
            ...     print(item["metadata"]["title"])
        """
        try:
            result = self.collection.get(
                ids=ids,
                include=["documents", "metadatas", "embeddings"],
            )
            
            items = []
            for i in range(len(result["ids"])):
                items.append({
                    "id": result["ids"][i],
                    "document": result["documents"][i] if result["documents"] is not None and len(result["documents"]) > i else None,
                    "metadata": result["metadatas"][i] if result["metadatas"] is not None and len(result["metadatas"]) > i else None,
                    "embedding": result["embeddings"][i] if result["embeddings"] is not None and len(result["embeddings"]) > i else None,
                })
            
            return items
        except Exception as e:
            raise Exception(f"Failed to get batch: {str(e)}") from e
    
    def update(
        self,
        id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Update an existing item.
        
        Args:
            id: Unique identifier of the item to update
            document: Optional new document text
            metadata: Optional new metadata (replaces existing)
            embedding: Optional new embedding vector
        
        Raises:
            ValueError: If item doesn't exist
            Exception: If update fails
        
        Example:
            >>> crud.update(
            ...     id="book_123",
            ...     metadata={"title": "Updated Title", "author": "Author"},
            ... )
        """
        if not self.exists(id):
            raise ValueError(f"Item with ID '{id}' does not exist")
        
        try:
            self.collection.update(
                ids=[id],
                documents=[document] if document else None,
                metadatas=[metadata] if metadata else None,
                embeddings=[embedding] if embedding else None,
            )
        except Exception as e:
            raise Exception(f"Failed to update item '{id}': {str(e)}") from e
    
    def delete(self, id: str) -> None:
        """Delete a single item from the collection.
        
        Args:
            id: Unique identifier of the item to delete
        
        Raises:
            ValueError: If item doesn't exist
            Exception: If deletion fails
        
        Example:
            >>> crud.delete("book_123")
        """
        if not self.exists(id):
            raise ValueError(f"Item with ID '{id}' does not exist")
        
        try:
            self.collection.delete(ids=[id])
        except Exception as e:
            raise Exception(f"Failed to delete item '{id}': {str(e)}") from e
    
    def delete_batch(self, ids: List[str]) -> None:
        """Delete multiple items from the collection.
        
        Args:
            ids: List of unique identifiers to delete
        
        Raises:
            ValueError: If any items don't exist
            Exception: If batch deletion fails
        
        Example:
            >>> crud.delete_batch(["book_1", "book_2", "book_3"])
        """
        # Check that all IDs exist
        missing_ids = [id for id in ids if not self.exists(id)]
        if missing_ids:
            raise ValueError(f"Items with IDs do not exist: {missing_ids}")
        
        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            raise Exception(f"Failed to delete batch: {str(e)}") from e
    
    def exists(self, id: str) -> bool:
        """Check if an item exists in the collection.
        
        Args:
            id: Unique identifier to check
        
        Returns:
            True if item exists, False otherwise
        
        Example:
            >>> if crud.exists("book_123"):
            ...     print("Book exists")
        """
        try:
            result = self.collection.get(ids=[id])
            return len(result["ids"]) > 0
        except Exception:
            return False
    
    def count(self) -> int:
        """Get the total number of items in the collection.
        
        Returns:
            Integer count of items
        
        Example:
            >>> total = crud.count()
            >>> print(f"Collection has {total} items")
        """
        try:
            return self.collection.count()
        except Exception as e:
            raise Exception(f"Failed to count items: {str(e)}") from e
    
    def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get all items from the collection.
        
        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip
        
        Returns:
            List of all items in the collection
        
        Example:
            >>> all_items = crud.get_all(limit=100)
        """
        try:
            result = self.collection.get(
                include=["documents", "metadatas", "embeddings"],
                limit=limit,
                offset=offset,
            )
            
            items = []
            for i in range(len(result["ids"])):
                items.append({
                    "id": result["ids"][i],
                    "document": result["documents"][i] if result["documents"] is not None and len(result["documents"]) > i else None,
                    "metadata": result["metadatas"][i] if result["metadatas"] is not None and len(result["metadatas"]) > i else None,
                    "embedding": result["embeddings"][i] if result["embeddings"] is not None and len(result["embeddings"]) > i else None,
                })
            
            return items
        except Exception as e:
            raise Exception(f"Failed to get all items: {str(e)}") from e
