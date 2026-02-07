"""Vector database module for ChromaDB integration.

This module provides a clean interface for interacting with ChromaDB,
including client management, configuration, and CRUD operations.

Example:
    >>> from bookdb.vector_db import get_chroma_client
    >>> client = get_chroma_client()
    >>> collections = client.list_collections()
"""

from .client import (
    get_chroma_client,
    reset_client,
    get_client_info,
)
from .config import ChromaDBConfig

__all__ = [
    "get_chroma_client",
    "reset_client",
    "get_client_info",
    "ChromaDBConfig",
]
