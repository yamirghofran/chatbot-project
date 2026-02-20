"""Book-specific CRUD operations for Qdrant."""

from typing import Any, Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from .crud import BaseVectorCRUD
from .schemas import BookMetadata, CollectionNames


class BookVectorCRUD(BaseVectorCRUD):
    """CRUD operations specialized for book embeddings and metadata.
    
    This class extends BaseVectorCRUD with book-specific operations including
    automatic embedding generation, metadata validation, and semantic search.
    
    Example:
        >>> from bookdb.vector_db import get_books_collection, BookVectorCRUD
        >>> collection = get_books_collection()
        >>> crud = BookVectorCRUD(collection)
        >>> crud.add_book(
        ...     book_id="123",
        ...     title="The Great Gatsby",
        ...     description="A novel about...",
        ...     author="F. Scott Fitzgerald",
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
        author: str,
        description: Optional[str] = None,
        genre: Optional[str] = None,
        publication_year: Optional[int] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a book with validated metadata.

        Args:
            book_id: Unique identifier for the book
            title: Book title (required)
            author: Book author (required)
            description: Book description or summary
            genre: Book genre/category
            publication_year: Year of publication
            embedding: Optional pre-computed embedding vector

        Raises:
            ValueError: If metadata validation fails or book already exists
            Exception: If addition fails
        """
        # Validate metadata
        metadata = BookMetadata(
            title=title,
            author=author,
            genre=genre,
            publication_year=publication_year,
        )
        
        # Use description as document, or create default
        document = description or f"{title} by {author}"
        
        # TODO: Generate embedding if not provided
        # if embedding is None:
        #     from .embeddings import get_embedding_service
        #     service = get_embedding_service()
        #     embedding = service.generate_book_embedding(
        #         title=title,
        #         description=description,
        #         author=author,
        #         genre=genre,
        #     )
        
        self.add(
            id=book_id,
            document=document,
            metadata=metadata.model_dump(exclude_none=True),
            embedding=embedding,
        )
    
    def update_book(
        self,
        book_id: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        description: Optional[str] = None,
        genre: Optional[str] = None,
        publication_year: Optional[int] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Update a book's information.

        Args:
            book_id: Unique identifier of the book to update
            title: New book title
            author: New author name
            description: New description
            genre: New genre
            publication_year: New publication year
            embedding: Optional new embedding vector

        Raises:
            ValueError: If book doesn't exist or validation fails
            Exception: If update fails
        """
        # Get existing item to merge with updates
        existing = self.get(book_id)
        if not existing:
            raise ValueError(f"Book with ID '{book_id}' does not exist")

        existing_metadata = existing["metadata"]

        # Build updated metadata (keep existing values if not provided)
        updated_data = {
            "title": title if title is not None else existing_metadata.get("title"),
            "author": author if author is not None else existing_metadata.get("author"),
            "genre": genre if genre is not None else existing_metadata.get("genre"),
            "publication_year": publication_year if publication_year is not None else existing_metadata.get("publication_year"),
        }
        
        # Validate updated metadata
        metadata = BookMetadata(**updated_data)
        
        # Update document if description changed
        document = None
        if description is not None:
            document = description
            
            # TODO: Regenerate embedding if description changed
            # if embedding is None:
            #     from .embeddings import get_embedding_service
            #     service = get_embedding_service()
            #     embedding = service.generate_book_embedding(
            #         title=metadata.title,
            #         description=description,
            #         author=metadata.author,
            #         genre=metadata.genre,
            #     )
        
        # Update in collection
        self.update(
            id=book_id,
            document=document,
            metadata=metadata.model_dump(exclude_none=True),
            embedding=embedding,
        )
    
    def search_by_metadata(
        self,
        genre: Optional[str] = None,
        author: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search books by metadata filters.

        Args:
            genre: Filter by genre
            author: Filter by author name
            min_year: Minimum publication year
            max_year: Maximum publication year
            limit: Maximum number of results

        Returns:
            List of matching books
        """
        try:
            scroll_filter = self._build_qdrant_filter(
                genre=genre,
                author=author,
                min_year=min_year,
                max_year=max_year,
            )
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            return [self._record_to_item(point) for point in points]
        except Exception as e:
            raise Exception(f"Failed to search by metadata: {str(e)}") from e

    def _build_qdrant_filter(
        self,
        genre: Optional[str],
        author: Optional[str],
        min_year: Optional[int],
        max_year: Optional[int],
    ) -> Optional[Filter]:
        must: List[FieldCondition] = []

        if genre:
            must.append(
                FieldCondition(
                    key="metadata.genre",
                    match=MatchValue(value=genre),
                )
            )

        if author:
            must.append(
                FieldCondition(
                    key="metadata.author",
                    match=MatchValue(value=author),
                )
            )

        if min_year is not None or max_year is not None:
            must.append(
                FieldCondition(
                    key="metadata.publication_year",
                    range=Range(gte=min_year, lte=max_year),
                )
            )

        return Filter(must=must) if must else None

    def search_similar_books(
        self,
        query_text: str,
        n_results: int = 10,
        genre: Optional[str] = None,
        author: Optional[str] = None,
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
        genre: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get book recommendations based on a given book."""
        # TODO: Implement book recommendations
        pass
