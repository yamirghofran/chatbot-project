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
        isbn: Optional[str] = None,
        language: Optional[str] = "en",
        page_count: Optional[int] = None,
        average_rating: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a book with validated metadata.
        
        This method validates the book metadata using the BookMetadata schema
        and adds it to the collection. If no embedding is provided, one should
        be generated (TODO: implement embedding generation).
        
        Args:
            book_id: Unique identifier for the book
            title: Book title (required)
            author: Book author (required)
            description: Book description or summary
            genre: Book genre/category
            publication_year: Year of publication
            isbn: ISBN number
            language: Language code (ISO 639-1)
            page_count: Number of pages
            average_rating: Average user rating (0-5)
            embedding: Optional pre-computed embedding vector
        
        Raises:
            ValueError: If metadata validation fails or book already exists
            Exception: If addition fails
        
        Example:
            >>> crud.add_book(
            ...     book_id="book_123",
            ...     title="1984",
            ...     author="George Orwell",
            ...     genre="Dystopian",
            ...     publication_year=1949,
            ... )
        """
        # Validate metadata
        metadata = BookMetadata(
            title=title,
            author=author,
            genre=genre,
            publication_year=publication_year,
            isbn=isbn,
            language=language,
            page_count=page_count,
            average_rating=average_rating,
        )
        
        # Create document from description or title
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
        
        # Add to collection
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
        isbn: Optional[str] = None,
        language: Optional[str] = None,
        page_count: Optional[int] = None,
        average_rating: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Update a book's information.
        
        This method updates an existing book. If any content fields are changed
        (title, description, author, genre), the embedding should be regenerated
        (TODO: implement embedding regeneration).
        
        Args:
            book_id: Unique identifier of the book to update
            title: New book title
            author: New author name
            description: New description
            genre: New genre
            publication_year: New publication year
            isbn: New ISBN
            language: New language code
            page_count: New page count
            average_rating: New average rating
            embedding: Optional new embedding vector
        
        Raises:
            ValueError: If book doesn't exist or validation fails
            Exception: If update fails
        
        Example:
            >>> crud.update_book(
            ...     book_id="book_123",
            ...     average_rating=4.8,
            ...     page_count=328,
            ... )
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
            "isbn": isbn if isbn is not None else existing_metadata.get("isbn"),
            "language": language if language is not None else existing_metadata.get("language"),
            "page_count": page_count if page_count is not None else existing_metadata.get("page_count"),
            "average_rating": average_rating if average_rating is not None else existing_metadata.get("average_rating"),
        }
        
        # Validate updated metadata
        metadata = BookMetadata(**updated_data)
        
        # Create updated document if content changed
        document = None
        if any(x is not None for x in [title, author, description, genre]):
            document = description or f"{metadata.title} by {metadata.author}"
            
            # TODO: Regenerate embedding if content changed
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
            metadata=metadata.model_dump(),
            embedding=embedding,
        )
    
    def search_by_metadata(
        self,
        genre: Optional[str] = None,
        author: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        language: Optional[str] = None,
        min_rating: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search books by metadata filters.
        
        This performs traditional filtering based on metadata fields without
        using vector similarity.
        
        Args:
            genre: Filter by genre
            author: Filter by author name
            min_year: Minimum publication year
            max_year: Maximum publication year
            language: Filter by language code
            min_rating: Minimum average rating
            limit: Maximum number of results
        
        Returns:
            List of matching books
        
        Example:
            >>> results = crud.search_by_metadata(
            ...     genre="Fiction",
            ...     min_year=1900,
            ...     max_year=2000,
            ...     min_rating=4.0,
            ... )
        """
        # Build where filter for ChromaDB
        where_filter = {}
        
        if genre:
            where_filter["genre"] = genre
        
        if author:
            where_filter["author"] = author
        
        if language:
            where_filter["language"] = language
        
        # ChromaDB supports comparison operators in where filters
        if min_year is not None:
            where_filter["publication_year"] = {"$gte": min_year}
        
        if max_year is not None:
            if "publication_year" in where_filter:
                # Combine with existing year filter
                where_filter["publication_year"]["$lte"] = max_year
            else:
                where_filter["publication_year"] = {"$lte": max_year}
        
        if min_rating is not None:
            where_filter["average_rating"] = {"$gte": min_rating}
        
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
        """Search for books similar to a query text using semantic search.
        
        This method performs vector similarity search to find books semantically
        similar to the query text. Optionally filters by metadata.
        
        Args:
            query_text: Text to search for (e.g., "science fiction about space")
            n_results: Number of results to return
            genre: Optional genre filter
            author: Optional author filter
            min_year: Optional minimum publication year
            max_year: Optional maximum publication year
        
        Returns:
            List of similar books with similarity scores
        
        Example:
            >>> results = crud.search_similar_books(
            ...     query_text="dystopian novels about totalitarianism",
            ...     n_results=5,
            ...     genre="Fiction",
            ... )
        """
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
        """Get book recommendations based on a given book.
        
        This finds books similar to the specified book using its embedding.
        
        Args:
            book_id: ID of the book to find similar books for
            n_results: Number of recommendations to return
            genre: Optional genre filter
            min_year: Optional minimum publication year
            max_year: Optional maximum publication year
        
        Returns:
            List of recommended books (excludes the input book)
        
        Example:
            >>> recommendations = crud.get_book_recommendations(
            ...     book_id="book_123",
            ...     n_results=5,
            ...     genre="Fiction",
            ... )
        """
        # TODO: Implement book recommendations
        # 1. Get the book by ID
        # 2. Extract its embedding
        # 3. Query collection for similar embeddings
        # 4. Filter out the original book
        # 5. Apply optional metadata filters
        # 6. Return recommendations
        pass
