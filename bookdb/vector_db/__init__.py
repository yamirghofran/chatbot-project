"""Vector database module.

This module provides a clean interface for interacting with the vector backend,
including client management, configuration, and CRUD operations.

Example:
    >>> from bookdb.vector_db import initialize_all_collections, CollectionNames
    >>> manager = initialize_all_collections()
    >>> books = manager.get_collection(CollectionNames.BOOKS)
"""

from .client import (
    get_qdrant_client,
    reset_client,
    get_client_info,
)
from .config import QdrantConfig
from .schemas import (
    CollectionNames,
    UserMetadata,
    ReviewMetadata,
    validate_user_metadata,
    validate_review_metadata,
)
from .collections import (
    CollectionManager,
    initialize_all_collections,
    get_books_collection,
    get_users_collection,
    get_reviews_collection,
)
from .embeddings import (
    EmbeddingService,
    get_embedding_service,
)
from .crud import BaseVectorCRUD
from .book_crud import BookVectorCRUD
from .user_crud import UserVectorCRUD
from .review_crud import ReviewVectorCRUD

__all__ = [
    # Client
    "get_qdrant_client",
    "reset_client",
    "get_client_info",
    # Config
    "QdrantConfig",
    # Schemas
    "CollectionNames",
    "UserMetadata",
    "ReviewMetadata",
    "validate_user_metadata",
    "validate_review_metadata",
    # Collections
    "CollectionManager",
    "initialize_all_collections",
    "get_books_collection",
    "get_users_collection",
    "get_reviews_collection",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    # CRUD
    "BaseVectorCRUD",
    "BookVectorCRUD",
    "UserVectorCRUD",
    "ReviewVectorCRUD",
]
