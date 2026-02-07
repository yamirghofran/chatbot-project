"""Vector database module for ChromaDB integration.

This module provides a clean interface for interacting with ChromaDB,
including client management, configuration, and CRUD operations.

Example:
    >>> from bookdb.vector_db import get_chroma_client, initialize_all_collections
    >>> client = get_chroma_client()
    >>> manager = initialize_all_collections()
    >>> books = manager.get_collection(CollectionNames.BOOKS)
"""

from .client import (
    get_chroma_client,
    reset_client,
    get_client_info,
)
from .config import ChromaDBConfig
from .schemas import (
    CollectionNames,
    BookMetadata,
    UserMetadata,
    validate_book_metadata,
    validate_user_metadata,
)
from .collections import (
    CollectionManager,
    initialize_all_collections,
    get_books_collection,
    get_users_collection,
)
from .embeddings import (
    EmbeddingService,
    get_embedding_service,
)
from .crud import BaseVectorCRUD

__all__ = [
    # Client
    "get_chroma_client",
    "reset_client",
    "get_client_info",
    # Config
    "ChromaDBConfig",
    # Schemas
    "CollectionNames",
    "BookMetadata",
    "UserMetadata",
    "validate_book_metadata",
    "validate_user_metadata",
    # Collections
    "CollectionManager",
    "initialize_all_collections",
    "get_books_collection",
    "get_users_collection",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    # CRUD
    "BaseVectorCRUD",
]
