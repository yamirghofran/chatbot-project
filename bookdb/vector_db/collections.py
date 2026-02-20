"""Collection management for Qdrant collections."""

from typing import Any, Dict, List, Optional

from qdrant_client.models import Distance, VectorParams

from .client import get_qdrant_client
from .config import QdrantConfig
from .schemas import CollectionNames


DEFAULT_VECTOR_SIZE = 384

_COLLECTION_DESCRIPTIONS = {
    CollectionNames.BOOKS: "Book embeddings and metadata for semantic search",
    CollectionNames.USERS: "User preference embeddings for personalized recommendations",
    CollectionNames.REVIEWS: "Book review embeddings for sentiment analysis and recommendations",
}


class CollectionManager:
    """Manager for Qdrant collections."""

    def __init__(
        self,
        config: Optional[QdrantConfig] = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ):
        """Initialize collection manager.

        Args:
            config: Qdrant configuration. If None, loads from environment.
            vector_size: Vector dimensionality used for all collections.
        """
        if vector_size <= 0:
            raise ValueError("vector_size must be > 0")

        self.client = get_qdrant_client(config)
        self.vector_size = vector_size
        self._collections: Dict[str, Any] = {}

    def initialize_collections(self) -> None:
        """Initialize all required collections."""
        for collection_name, description in _COLLECTION_DESCRIPTIONS.items():
            self._create_collection_if_missing(
                name=collection_name.value,
                description=description,
            )

    def get_collection(self, collection_name: CollectionNames) -> Any:
        """Get collection metadata by name."""
        if collection_name.value in self._collections:
            return self._collections[collection_name.value]

        if not self.client.collection_exists(collection_name=collection_name.value):
            raise ValueError(
                f"Collection '{collection_name.value}' does not exist. "
                "Call initialize_collections() first."
            )

        collection = self.client.get_collection(collection_name=collection_name.value)
        self._collections[collection_name.value] = collection
        return collection

    def reset_collection(self, collection_name: CollectionNames) -> None:
        """Reset a collection by deleting and recreating it."""
        try:
            self.client.delete_collection(collection_name=collection_name.value)
        except Exception:
            # Collection might not exist, which is fine for reset semantics.
            pass

        self._collections.pop(collection_name.value, None)
        self._create_collection_if_missing(
            name=collection_name.value,
            description=_COLLECTION_DESCRIPTIONS[collection_name],
        )

    def list_collections(self) -> List[str]:
        """List all collection names."""
        response = self.client.get_collections()
        return [collection.name for collection in response.collections]

    def collection_exists(self, collection_name: CollectionNames) -> bool:
        """Check if a collection exists."""
        try:
            return self.client.collection_exists(collection_name=collection_name.value)
        except Exception:
            return False

    def get_collection_count(self, collection_name: CollectionNames) -> int:
        """Get the number of vectors in a collection."""
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name.value}' does not exist")

        response = self.client.count(
            collection_name=collection_name.value,
            exact=True,
        )
        return int(response.count)

    def _create_collection_if_missing(self, name: str, description: str) -> Any:
        """Create a collection if it doesn't already exist and cache its metadata."""
        if not self.client.collection_exists(collection_name=name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

        collection = self.client.get_collection(collection_name=name)
        self._collections[name] = collection
        return collection


def initialize_all_collections(config: Optional[QdrantConfig] = None) -> CollectionManager:
    """Initialize all collections and return manager."""
    manager = CollectionManager(config)
    manager.initialize_collections()
    return manager


def get_books_collection(config: Optional[QdrantConfig] = None) -> Any:
    """Get the books collection metadata."""
    manager = CollectionManager(config)
    return manager.get_collection(CollectionNames.BOOKS)


def get_users_collection(config: Optional[QdrantConfig] = None) -> Any:
    """Get the users collection metadata."""
    manager = CollectionManager(config)
    return manager.get_collection(CollectionNames.USERS)


def get_reviews_collection(config: Optional[QdrantConfig] = None) -> Any:
    """Get the reviews collection metadata."""
    manager = CollectionManager(config)
    return manager.get_collection(CollectionNames.REVIEWS)
