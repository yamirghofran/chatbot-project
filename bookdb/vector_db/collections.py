"""Collection management for Qdrant collections."""

from typing import Any, Dict, List, Optional

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
    VectorParamsDiff,
)

from .client import get_qdrant_client
from .config import QdrantConfig
from .schemas import CollectionNames


DEFAULT_VECTOR_SIZE = 768

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
        vector_size: Optional[int] = None,
    ):
        """Initialize collection manager.

        Args:
            config: Qdrant configuration. If None, loads from environment.
            vector_size: Vector dimensionality used for all collections.
                Defaults to config.vector_size.
        """
        self.config = config or QdrantConfig.from_env()
        resolved_vector_size = (
            self.config.vector_size
            if vector_size is None
            else vector_size
        )

        if resolved_vector_size <= 0:
            raise ValueError("vector_size must be > 0")

        self.client = get_qdrant_client(self.config)
        self.vector_size = resolved_vector_size
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

        if collection_name is CollectionNames.BOOKS:
            self._apply_books_collection_tuning()

        collection = self.client.get_collection(collection_name=collection_name.value)
        self._validate_collection_schema(collection_name.value, collection)
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
                vectors_config=self._build_vector_params(name),
            )
        elif name == CollectionNames.BOOKS.value:
            self._apply_books_collection_tuning()

        collection = self.client.get_collection(collection_name=name)
        self._validate_collection_schema(name, collection)
        self._collections[name] = collection
        return collection

    def _build_vector_params(self, collection_name: str) -> VectorParams:
        params = VectorParams(
            size=self.vector_size,
            distance=Distance.COSINE,
        )

        if collection_name == CollectionNames.BOOKS.value:
            params.on_disk = self.config.books_on_disk
            params.hnsw_config = HnswConfigDiff(
                on_disk=self.config.books_hnsw_on_disk,
                m=self.config.books_hnsw_m,
            )
            quantization = self._books_quantization_config()
            if quantization is not None:
                params.quantization_config = quantization

        return params

    def _books_quantization_config(self) -> Optional[ScalarQuantization]:
        if not self.config.books_int8_quantization:
            return None

        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=self.config.books_quantile,
                always_ram=self.config.books_quantization_always_ram,
            )
        )

    def _apply_books_collection_tuning(self) -> None:
        self.client.update_collection(
            collection_name=CollectionNames.BOOKS.value,
            vectors_config={
                "": VectorParamsDiff(on_disk=self.config.books_on_disk),
            },
            hnsw_config=HnswConfigDiff(
                on_disk=self.config.books_hnsw_on_disk,
                m=self.config.books_hnsw_m,
            ),
            quantization_config=self._books_quantization_config(),
        )

    def _validate_collection_schema(self, collection_name: str, collection: Any) -> None:
        vector_params = self._extract_vector_params(collection)
        if vector_params is None:
            return

        size = getattr(vector_params, "size", None)
        if isinstance(size, int) and size != self.vector_size:
            raise ValueError(
                f"Collection '{collection_name}' vector size is {size}, "
                f"but configured QDRANT_VECTOR_SIZE is {self.vector_size}. "
                "Reset or recreate the collection with matching dimensions."
            )

        distance = getattr(vector_params, "distance", None)
        if isinstance(distance, Distance) and distance != Distance.COSINE:
            raise ValueError(
                f"Collection '{collection_name}' distance is {distance}, "
                "but this application expects COSINE."
            )

    def _extract_vector_params(self, collection: Any) -> Optional[Any]:
        config = getattr(collection, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return None

        if isinstance(vectors, dict):
            if "" in vectors:
                return vectors[""]
            return next(iter(vectors.values()), None)

        return vectors


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
