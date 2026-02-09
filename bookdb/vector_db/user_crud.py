"""User-specific CRUD operations for ChromaDB."""

from typing import List, Optional, Dict, Any
from chromadb import Collection

from .crud import BaseVectorCRUD
from .schemas import UserMetadata, validate_user_metadata


class UserVectorCRUD(BaseVectorCRUD):
    """CRUD operations specialized for user preference embeddings.
    
    This class extends BaseVectorCRUD with user-specific operations including
    preference management, personalized recommendations, and collaborative filtering.
    
    Example:
        >>> from bookdb.vector_db import get_users_collection, UserVectorCRUD
        >>> collection = get_users_collection()
        >>> crud = UserVectorCRUD(collection)
        >>> crud.add_user(
        ...     user_id=12345,
        ...     preferences_text="I enjoy science fiction and mystery novels",
        ...     favorite_genres="Sci-Fi,Mystery",
        ... )
    """
    
    def __init__(self, collection: Collection):
        """Initialize user CRUD operations.
        
        Args:
            collection: ChromaDB collection for users
        """
        super().__init__(collection)
    
    def add_user(
        self,
        user_id: int,
        preferences_text: Optional[str] = None,
        favorite_genres: Optional[str] = None,
        num_books_read: int = 0,
        average_rating_given: Optional[float] = None,
        reading_level: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a user with their preference data.
        
        This method validates the user metadata and adds it to the collection.
        If no embedding is provided, one should be generated from preferences.
        
        Args:
            user_id: Unique user identifier (from PostgreSQL)
            preferences_text: Text description of user preferences
            favorite_genres: Comma-separated favorite genres
            num_books_read: Number of books the user has read
            average_rating_given: Average rating the user gives
            reading_level: User's reading level
            embedding: Optional pre-computed embedding vector
        
        Raises:
            ValueError: If metadata validation fails or user already exists
            Exception: If addition fails
        
        Example:
            >>> crud.add_user(
            ...     user_id=12345,
            ...     preferences_text="I love sci-fi and mystery",
            ...     favorite_genres="Science Fiction,Mystery",
            ...     num_books_read=42,
            ... )
        """
        # Create unique string ID from user_id
        id_str = f"user_{user_id}"
        
        # Validate metadata
        metadata = UserMetadata(
            user_id=user_id,
            num_books_read=num_books_read,
            favorite_genres=favorite_genres,
            average_rating_given=average_rating_given,
            reading_level=reading_level,
        )
        
        # Create document from preferences
        document = preferences_text or f"User preferences for user {user_id}"
        
        # TODO: Generate embedding if not provided
        # if embedding is None:
        #     from .embeddings import get_embedding_service
        #     service = get_embedding_service()
        #     embedding = service.generate_user_embedding(
        #         preferences=preferences_text,
        #         favorite_genres=favorite_genres,
        #     )
        
        # Add to collection
        self.add(
            id=id_str,
            document=document,
            metadata=metadata.model_dump(),
            embedding=embedding,
        )
    
    def update_user_preferences(
        self,
        user_id: int,
        preferences_text: Optional[str] = None,
        favorite_genres: Optional[str] = None,
        num_books_read: Optional[int] = None,
        average_rating_given: Optional[float] = None,
        reading_level: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Update a user's preference data.
        
        This method updates an existing user's preferences and metadata.
        If content fields are changed, the embedding should be regenerated.
        
        Args:
            user_id: Unique user identifier
            preferences_text: New preferences text
            favorite_genres: New favorite genres
            num_books_read: Updated book count
            average_rating_given: Updated average rating
            reading_level: Updated reading level
            embedding: Optional new embedding vector
        
        Raises:
            ValueError: If user doesn't exist or validation fails
            Exception: If update fails
        
        Example:
            >>> crud.update_user_preferences(
            ...     user_id=12345,
            ...     num_books_read=50,
            ...     average_rating_given=4.3,
            ... )
        """
        id_str = f"user_{user_id}"
        
        # Get existing user data
        existing = self.get(id_str)
        if not existing:
            raise ValueError(f"User with ID '{user_id}' does not exist")
        
        existing_metadata = existing["metadata"]
        
        # Build updated metadata (keep existing values if not provided)
        updated_data = {
            "user_id": user_id,
            "num_books_read": num_books_read if num_books_read is not None else existing_metadata.get("num_books_read"),
            "favorite_genres": favorite_genres if favorite_genres is not None else existing_metadata.get("favorite_genres"),
            "average_rating_given": average_rating_given if average_rating_given is not None else existing_metadata.get("average_rating_given"),
            "reading_level": reading_level if reading_level is not None else existing_metadata.get("reading_level"),
        }
        
        # Validate updated metadata
        metadata = UserMetadata(**updated_data)
        
        # Create updated document if preferences changed
        document = None
        if preferences_text is not None:
            document = preferences_text
            
            # TODO: Regenerate embedding if preferences changed
            # if embedding is None:
            #     from .embeddings import get_embedding_service
            #     service = get_embedding_service()
            #     embedding = service.generate_user_embedding(
            #         preferences=preferences_text,
            #         favorite_genres=metadata.favorite_genres,
            #     )
        
        # Update in collection
        self.update(
            id=id_str,
            document=document,
            metadata=metadata.model_dump(),
            embedding=embedding,
        )
    
    def get_book_recommendations_for_user(
        self,
        user_id: int,
        n_results: int = 10,
        genre: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get personalized book recommendations for a user.
        
        This method finds books that match the user's preference embedding,
        combining user preferences with optional metadata filters.
        
        Args:
            user_id: Unique user identifier
            n_results: Number of recommendations to return
            genre: Optional genre filter
            min_year: Optional minimum publication year
            max_year: Optional maximum publication year
        
        Returns:
            List of recommended books for the user
        
        Example:
            >>> recommendations = crud.get_book_recommendations_for_user(
            ...     user_id=12345,
            ...     n_results=10,
            ...     genre="Science Fiction",
            ... )
        """
        # TODO: Implement personalized book recommendations
        # 1. Get user by ID and extract embedding
        # 2. Query books collection with user embedding
        # 3. Apply optional metadata filters
        # 4. Return top n_results books
        pass
    
    def find_similar_users(
        self,
        user_id: int,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find users with similar preferences.
        
        This method finds users with similar preference embeddings,
        useful for collaborative filtering and social features.
        
        Args:
            user_id: Unique user identifier
            n_results: Number of similar users to return
        
        Returns:
            List of similar users (excludes the input user)
        
        Example:
            >>> similar_users = crud.find_similar_users(
            ...     user_id=12345,
            ...     n_results=5,
            ... )
        """
        # TODO: Implement similar user discovery
        pass
    
    def update_from_reading_history(
        self,
        user_id: int,
        book_ids: List[str],
    ) -> None:
        """Update user embedding based on their reading history.
        
        This method recomputes the user's preference embedding based on
        the books they've read, creating a more accurate representation.
        
        Args:
            user_id: Unique user identifier
            book_ids: List of book IDs the user has read
        
        Raises:
            ValueError: If user doesn't exist
            Exception: If update fails
        
        Example:
            >>> crud.update_from_reading_history(
            ...     user_id=12345,
            ...     book_ids=["book_1", "book_2", "book_3"],
            ... )
        """
        # TODO: Implement embedding update from reading history
        pass
