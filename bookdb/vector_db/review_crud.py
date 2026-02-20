"""Review-specific CRUD operations for Qdrant."""

from typing import Any, Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue

from .crud import BaseVectorCRUD
from .schemas import CollectionNames, ReviewMetadata


class ReviewVectorCRUD(BaseVectorCRUD):
    """CRUD operations specialized for book review embeddings and metadata.
    
    This class extends BaseVectorCRUD with review-specific operations including
    metadata validation and sentiment-based search.
    """
    
    def __init__(self, collection: Any = CollectionNames.REVIEWS.value):
        """Initialize review CRUD operations.

        Args:
            collection: Collection reference. Prefer collection name for Qdrant.
        """
        super().__init__(collection)
    
    def add_review(
        self,
        review_id: str,
        user_id: int,
        book_id: int,
        review_text: str,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a review with validated metadata.
        
        This method validates the review metadata using the ReviewMetadata schema
        and adds it to the collection. If no embedding is provided, one should
        be generated (TODO: implement embedding generation).
        
        Args:
            review_id: Unique identifier for the review
            user_id: User identifier who wrote the review
            book_id: Book identifier being reviewed
            review_text: Text content of the review
            embedding: Optional pre-computed embedding vector
        
        Raises:
            ValueError: If metadata validation fails or review already exists
            Exception: If addition fails
        """
        # Validate metadata
        metadata = ReviewMetadata(
            user_id=user_id,
            book_id=book_id,
        )
        
        # Use review text as document
        document = review_text
        
        # TODO: Generate embedding if not provided
        # if embedding is None:
        #     from .embeddings import get_embedding_service
        #     service = get_embedding_service()
        #     embedding = service.generate_review_embedding(
        #         review_text=review_text,
        #     )
        
        self.add(
            id=review_id,
            document=document,
            metadata=metadata.model_dump(exclude_none=True),
            embedding=embedding,
        )
    
    def update_review(
        self,
        review_id: str,
        user_id: Optional[int] = None,
        book_id: Optional[int] = None,
        review_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Update a review's information.
        
        This method updates an existing review's metadata and optionally
        regenerates the embedding if the review text changes.
        
        Args:
            review_id: Unique identifier for the review
            user_id: Updated user identifier
            book_id: Updated book identifier
            review_text: Updated review text
            embedding: Optional new embedding vector
        
        Raises:
            ValueError: If review doesn't exist or validation fails
            Exception: If update fails
        """
        # Get existing review data
        existing = self.get(review_id)
        if not existing:
            raise ValueError(f"Review with ID '{review_id}' does not exist")
        
        existing_metadata = existing["metadata"]
        
        # Build updated metadata (keep existing values if not provided)
        updated_data = {
            "user_id": user_id if user_id is not None else existing_metadata.get("user_id"),
            "book_id": book_id if book_id is not None else existing_metadata.get("book_id"),
        }
        
        # Validate updated metadata
        metadata = ReviewMetadata(**updated_data)
        
        # Update document if review text changed
        document = None
        if review_text is not None:
            document = review_text
            
            # TODO: Regenerate embedding if review text changed
            # if embedding is None:
            #     from .embeddings import get_embedding_service
            #     service = get_embedding_service()
            #     embedding = service.generate_review_embedding(
            #         review_text=review_text,
            #     )
        
        # Update in collection
        self.update(
            id=review_id,
            document=document,
            metadata=metadata.model_dump(exclude_none=True),
            embedding=embedding,
        )
    
    def search_by_metadata(
        self,
        user_id: Optional[int] = None,
        book_id: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search reviews by metadata filters.
        
        This method allows filtering reviews based on various metadata attributes.
        Multiple filters can be combined.
        
        Args:
            user_id: Filter by user who wrote the review
            book_id: Filter by book being reviewed
            limit: Maximum number of results to return
        
        Returns:
            List of reviews matching the filters
        """
        try:
            scroll_filter = self._build_qdrant_filter(
                user_id=user_id,
                book_id=book_id,
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
        user_id: Optional[int],
        book_id: Optional[int],
    ) -> Optional[Filter]:
        must: List[FieldCondition] = []

        if user_id is not None:
            must.append(
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id),
                )
            )

        if book_id is not None:
            must.append(
                FieldCondition(
                    key="metadata.book_id",
                    match=MatchValue(value=book_id),
                )
            )

        return Filter(must=must) if must else None
    
    def search_similar_reviews(
        self,
        query_text: str,
        n_results: int = 10,
        book_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find reviews similar to a query text using semantic search.
        
        This method performs vector similarity search to find reviews
        semantically similar to the input text, optionally filtered by metadata.
        
        Args:
            query_text: Text to search for similar reviews
            n_results: Number of results to return
            book_id: Optional filter by book
        
        Returns:
            List of similar reviews
        """
        # TODO: Implement semantic search
        # This requires embedding generation to be implemented
        pass
    
    def get_reviews_by_book(
        self,
        book_id: int,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get all reviews for a specific book.
        
        Args:
            book_id: Book identifier
            limit: Maximum number of reviews to return
        
        Returns:
            List of reviews for the book
        """
        return self.search_by_metadata(
            book_id=book_id,
            limit=limit,
        )
    
    def get_reviews_by_user(
        self,
        user_id: int,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get all reviews written by a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of reviews to return
        
        Returns:
            List of reviews by the user
        """
        return self.search_by_metadata(
            user_id=user_id,
            limit=limit,
        )
