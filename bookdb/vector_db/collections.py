"""Collection management for vector database collections."""

from typing import Optional, List
from chromadb import Collection
from chromadb.api.types import CollectionMetadata

from .client import get_chroma_client
from .schemas import CollectionNames
from .config import ChromaDBConfig


class CollectionManager:
    """Manager for vector database collections.
    
    This class handles collection lifecycle operations including
    creation, retrieval, and configuration.
    """
    
    def __init__(self, config: Optional[ChromaDBConfig] = None):
        """Initialize collection manager.
        
        Args:
            config: Vector DB configuration. If None, loads from environment.
        """
        self.client = get_chroma_client(config)
        self._collections = {}
    
    def initialize_collections(self) -> None:
        """Initialize all required collections.

        Creates the BOOKS, USERS, and REVIEWS collections if they don't exist.
        Configures appropriate distance functions and metadata for each.

        Note:
            This method is idempotent - it will not recreate existing collections.
        """
        # Initialize books collection
        self._get_or_create_collection(
            name=CollectionNames.BOOKS.value,
            metadata={
                "description": "Book embeddings and metadata for semantic search",
                "hnsw:space": "cosine",  # Use cosine similarity for book embeddings
            },
        )

        # Initialize users collection
        self._get_or_create_collection(
            name=CollectionNames.USERS.value,
            metadata={
                "description": "User preference embeddings for personalized recommendations",
                "hnsw:space": "cosine",  # Use cosine similarity for user embeddings
            },
        )
        
        # Initialize reviews collection
        self._get_or_create_collection(
            name=CollectionNames.REVIEWS.value,
            metadata={
                "description": "Book review embeddings for sentiment analysis and recommendations",
                "hnsw:space": "cosine",  # Use cosine similarity for review embeddings
            },
        )
    
    def get_collection(self, collection_name: CollectionNames) -> Collection:
        """Get a collection by name.
        
        Args:
            collection_name: Name of the collection to retrieve
        
        Returns:
            Collection instance
        
        Raises:
            ValueError: If collection doesn't exist
        """
        # Check cache first
        if collection_name.value in self._collections:
            return self._collections[collection_name.value]
        
        # Try to get from backing client
        try:
            collection = self.client.get_collection(name=collection_name.value)
            self._collections[collection_name.value] = collection
            return collection
        except Exception as e:
            raise ValueError(
                f"Collection '{collection_name.value}' does not exist. "
                f"Call initialize_collections() first. Error: {str(e)}"
            ) from e
    
    def reset_collection(self, collection_name: CollectionNames) -> None:
        """Reset a collection by deleting and recreating it.
        
        This is primarily useful for testing to ensure a clean state.
        
        Args:
            collection_name: Name of the collection to reset
        
        Warning:
            This will delete all data in the collection!
        """
        try:
            self.client.delete_collection(name=collection_name.value)
        except Exception:
            # Collection might not exist, which is fine
            pass
        
        # Remove from cache
        self._collections.pop(collection_name.value, None)
        
        # Recreate collection
        if collection_name == CollectionNames.BOOKS:
            self._get_or_create_collection(
                name=collection_name.value,
                metadata={
                    "description": "Book embeddings and metadata for semantic search",
                    "hnsw:space": "cosine",
                },
            )
        elif collection_name == CollectionNames.USERS:
            self._get_or_create_collection(
                name=collection_name.value,
                metadata={
                    "description": "User preference embeddings for personalized recommendations",
                    "hnsw:space": "cosine",
                },
            )
        elif collection_name == CollectionNames.REVIEWS:
            self._get_or_create_collection(
                name=collection_name.value,
                metadata={
                    "description": "Book review embeddings for sentiment analysis and recommendations",
                    "hnsw:space": "cosine",
                },
            )
    
    def list_collections(self) -> List[str]:
        """List all collection names.
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def collection_exists(self, collection_name: CollectionNames) -> bool:
        """Check if a collection exists.
        
        Args:
            collection_name: Name of the collection to check
        
        Returns:
            True if collection exists, False otherwise
        """
        try:
            self.client.get_collection(name=collection_name.value)
            return True
        except Exception:
            return False
    
    def get_collection_count(self, collection_name: CollectionNames) -> int:
        """Get the number of items in a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Number of items in the collection
        
        Raises:
            ValueError: If collection doesn't exist
        """
        collection = self.get_collection(collection_name)
        return collection.count()
    
    def _get_or_create_collection(
        self,
        name: str,
        metadata: Optional[CollectionMetadata] = None,
    ) -> Collection:
        """Get or create a collection.
        
        Args:
            name: Collection name
            metadata: Collection metadata including distance function
        
        Returns:
            Collection instance
        """
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=metadata,
        )
        
        # Cache the collection
        self._collections[name] = collection
        
        return collection


# Convenience functions for common operations

def initialize_all_collections(config: Optional[ChromaDBConfig] = None) -> CollectionManager:
    """Initialize all collections and return manager.
    
    This is a convenience function that creates a CollectionManager
    and initializes all collections in one call.
    
    Args:
        config: Vector DB configuration. If None, loads from environment.
    
    Returns:
        Initialized CollectionManager instance
    """
    manager = CollectionManager(config)
    manager.initialize_collections()
    return manager


def get_books_collection(config: Optional[ChromaDBConfig] = None) -> Collection:
    """Get the books collection.
    
    Args:
        config: Vector DB configuration. If None, loads from environment.
    
    Returns:
        Books Collection instance
    """
    manager = CollectionManager(config)
    return manager.get_collection(CollectionNames.BOOKS)


def get_users_collection(config: Optional[ChromaDBConfig] = None) -> Collection:
    """Get the users collection.
    
    Args:
        config: Vector DB configuration. If None, loads from environment.
    
    Returns:
        Users Collection instance
    """
    manager = CollectionManager(config)
    return manager.get_collection(CollectionNames.USERS)


def get_reviews_collection(config: Optional[ChromaDBConfig] = None) -> Collection:
    """Get the reviews collection.
    
    Args:
        config: Vector DB configuration. If None, loads from environment.
    
    Returns:
        Reviews Collection instance
    """
    manager = CollectionManager(config)
    return manager.get_collection(CollectionNames.REVIEWS)
