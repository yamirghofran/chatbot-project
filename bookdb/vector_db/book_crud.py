"""Book-specific CRUD operations for ChromaDB."""

from typing import List, Optional, Dict, Any
from chromadb import Collection

from .crud import BaseVectorCRUD
from .schemas import BookMetadata, validate_book_metadata


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
    
    def __init__(self, collection: Collection):
        """Initialize book CRUD operations.
        
        Args:
            collection: ChromaDB collection for books
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
        
        # Create document from book information
        document = self._create_book_document(
            title=title,
            author=author,
            description=description,
            genre=genre,
        )
        
        if embedding is None:
            from .embeddings import get_embedding_service
            service = get_embedding_service()
            embedding = service.generate_book_embedding(
                title=title,
                description=description,
                author=author,
                genre=genre,
            )
        self.add(
            id=book_id,
            document=document,
            metadata=metadata.model_dump(),
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
        
        # Create updated document if content changed
        document = None
        if any(x is not None for x in [title, author, description, genre]):
            document = self._create_book_document(
                title=metadata.title,
                author=metadata.author,
                description=description,
                genre=metadata.genre,
            )
            
            if embedding is None:
                from .embeddings import get_embedding_service
                service = get_embedding_service()
                embedding = service.generate_book_embedding(
                    title=metadata.title,
                    description=description,
                    author=metadata.author,
                    genre=metadata.genre,
                )
        
        # Update in collection
        self.update(
            id=book_id,
            document=document,
            metadata=metadata.model_dump(),
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
        # Build where filter for ChromaDB
        where_filter = {}

        if genre:
            where_filter["genre"] = genre

        if author:
            where_filter["author"] = author

        # ChromaDB supports comparison operators in where filters
        if min_year is not None:
            where_filter["publication_year"] = {"$gte": min_year}

        if max_year is not None:
            if "publication_year" in where_filter:
                # Combine with existing year filter
                where_filter["publication_year"]["$lte"] = max_year
            else:
                where_filter["publication_year"] = {"$lte": max_year}
        
        # Query collection with filters
        try:
            if where_filter:
                result = self.collection.get(
                    where=where_filter,
                    limit=limit,
                    include=["documents", "metadatas", "embeddings"],
                )
            else:
                result = self.collection.get(
                    limit=limit,
                    include=["documents", "metadatas", "embeddings"],
                )
            
            # Format results
            items = []
            for i in range(len(result["ids"])):
                items.append({
                    "id": result["ids"][i],
                    "document": result["documents"][i] if result["documents"] else None,
                    "metadata": result["metadatas"][i] if result["metadatas"] else None,
                    "embedding": result["embeddings"][i] if result["embeddings"] else None,
                })
            
            return items
            
        except Exception as e:
            raise Exception(f"Failed to search by metadata: {str(e)}") from e
    
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
    
    def _create_book_document(
        self,
        title: str,
        author: str,
        description: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> str:
        """Create a text document from book information.
        
        This combines book fields into a single text representation for
        storage and embedding generation.
        
        Args:
            title: Book title
            author: Book author
            description: Book description
            genre: Book genre
        
        Returns:
            Combined text document
        """
        parts = [
            f"Title: {title}",
            f"Author: {author}",
        ]
        
        if genre:
            parts.append(f"Genre: {genre}")
        
        if description:
            parts.append(f"Description: {description}")
        
        return " | ".join(parts)
