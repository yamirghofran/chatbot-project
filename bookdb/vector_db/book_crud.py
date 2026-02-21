"""Book-specific CRUD operations for Qdrant."""

from typing import Any, Dict, List, Optional

from .crud import BaseVectorCRUD
from .schemas import CollectionNames


class BookVectorCRUD(BaseVectorCRUD):
    """CRUD operations specialized for book embeddings.
    
    This class extends BaseVectorCRUD with book-specific operations including
    document updates and semantic search hooks.
    
    Example:
        >>> from bookdb.vector_db import get_books_collection, BookVectorCRUD
        >>> collection = get_books_collection()
        >>> crud = BookVectorCRUD(collection)
        >>> crud.add_book(
        ...     book_id="123",
        ...     title="The Great Gatsby",
        ...     description="A novel about...",
        ... )
    """
    
    def __init__(self, collection: Any = CollectionNames.BOOKS.value):
        """Initialize book CRUD operations.

        Args:
            collection: Collection reference. Prefer collection name for Qdrant.
        """
        super().__init__(collection)
    
    def add_book(
        self,
        book_id: str,
        title: str,
        description: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a book.

        Args:
            book_id: Unique identifier for the book
            title: Book title (required)
            description: Book description or summary
            embedding: Optional pre-computed embedding vector

        Raises:
            ValueError: If book already exists
            Exception: If addition fails
        """
        # Use description as document, or create default
        document = description or title
        
        # TODO: Generate embedding if not provided
        # if embedding is None:
        #     from .embeddings import get_embedding_service
        #     service = get_embedding_service()
        #     embedding = service.generate_book_embedding(
        #         title=title,
        #         description=description,
        #     )
        
        self.add(
            id=book_id,
            document=document,
            metadata=None,
            embedding=embedding,
        )
    
    def update_book(
        self,
        book_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Update a book's information.

        Args:
            book_id: Unique identifier of the book to update
            title: New book title
            description: New description
            embedding: Optional new embedding vector

        Raises:
            ValueError: If book doesn't exist
            Exception: If update fails
        """
        # Get existing item to merge with updates
        existing = self.get(book_id)
        if not existing:
            raise ValueError(f"Book with ID '{book_id}' does not exist")

        # Update document when description or title changes.
        document = None
        if description is not None:
            document = description
        elif title is not None:
            document = title
            
            # TODO: Regenerate embedding if description changed
            # if embedding is None:
            #     from .embeddings import get_embedding_service
            #     service = get_embedding_service()
            #     embedding = service.generate_book_embedding(
            #         title=title or existing.get("document") or "",
            #         description=description,
            #     )
        
        self.update(
            id=book_id,
            document=document,
            metadata=None,
            embedding=embedding,
        )
    
    def search_by_metadata(
        self,
        title: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List books.

        Args:
            title: Deprecated metadata filter (unsupported).
            min_year: Deprecated metadata filter (unsupported).
            max_year: Deprecated metadata filter (unsupported).
            limit: Maximum number of results

        Returns:
            List of matching books
        """
        if title is not None or min_year is not None or max_year is not None:
            raise ValueError(
                "Book metadata filters are not supported; books no longer store metadata fields."
            )

        try:
            return self.get_all(limit=limit)
        except Exception as e:
            raise Exception(f"Failed to search by metadata: {str(e)}") from e

    def search_similar_books(
        self,
        query_text: str,
        n_results: int = 10,
        title: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for books similar to a query text using semantic search."""
        # TODO: Implement semantic search

        pass
    
    def get_book_recommendations(
        self,
        book_id: str,
        n_results: int = 10,
        title: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get book recommendations based on a given book."""
        # TODO: Implement book recommendations
        pass
