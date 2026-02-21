"""User-specific CRUD operations for Qdrant."""

from typing import Any, Dict, List, Optional

from .crud import BaseVectorCRUD
from .schemas import CollectionNames, UserMetadata


class UserVectorCRUD(BaseVectorCRUD):
    """CRUD operations specialized for user embeddings and metadata.
    
    This class extends BaseVectorCRUD with user-specific operations including
    metadata validation, preference updates, and similarity search.
    
    Example:
        >>> from bookdb.vector_db import get_users_collection, UserVectorCRUD
        >>> collection = get_users_collection()
        >>> crud = UserVectorCRUD(collection)
        >>> crud.add_user(
        ...     user_id="user_1",
        ...     name="Alice",
        ...     pg_user_id=1,
        ... )
    """
    
    def __init__(self, collection: Any = CollectionNames.USERS.value):
        """Initialize user CRUD operations.

        Args:
            collection: Collection reference. Prefer collection name for Qdrant.
        """
        super().__init__(collection)
    
    def add_user(
        self,
        user_id: str,
        name: str,
        pg_user_id: int,
        description: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a user with validated metadata.
        
        Args:
            user_id: Unique identifier for the user (e.g. "user_1")
            name: User display name
            pg_user_id: User ID from PostgreSQL
            description: Optional text describing user preferences (used as document)
            embedding: Optional pre-computed embedding vector
        
        Raises:
            ValueError: If metadata validation fails or user already exists
            Exception: If addition fails
        """
        metadata = UserMetadata(
            user_id=pg_user_id,
            name=name,
        )
        
        document = description or f"User: {name}"
        
        # TODO: Generate embedding if not provided
        # if embedding is None:
        #     from .embeddings import get_embedding_service
        #     service = get_embedding_service()
        #     embedding = service.generate_user_embedding(...)
        
        self.add(
            id=user_id,
            document=document,
            metadata=metadata.model_dump(exclude_none=True),
            embedding=embedding,
        )
    
    def update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Update a user's information.
        
        Args:
            user_id: Unique identifier of the user to update
            name: New display name
            description: New description / preference text
            embedding: Optional new embedding vector
        
        Raises:
            ValueError: If user doesn't exist or validation fails
            Exception: If update fails
        """
        existing = self.get(user_id)
        if not existing:
            raise ValueError(f"User with ID '{user_id}' does not exist")
        
        existing_metadata = existing["metadata"]
        
        updated_data = {
            "user_id": existing_metadata.get("user_id"),
            "name": name if name is not None else existing_metadata.get("name"),
        }
        
        metadata = UserMetadata(**updated_data)
        
        document = None
        if description is not None:
            document = description
            
            # TODO: Regenerate embedding if description changed
            # if embedding is None:
            #     from .embeddings import get_embedding_service
            #     service = get_embedding_service()
            #     embedding = service.generate_user_embedding(...)
        
        self.update(
            id=user_id,
            document=document,
            metadata=metadata.model_dump(exclude_none=True),
            embedding=embedding,
        )
    
    def get_book_recommendations_for_user(
        self,
        user_id: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get book recommendations for a user based on their preference embedding."""
        # TODO: Implement user-based book recommendations
        pass
    
    def find_similar_users(
        self,
        user_id: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find users with similar preferences using semantic search."""
        # TODO: Implement similar user search
        pass
    
    def update_from_reading_history(
        self,
        user_id: str,
        book_ids: List[str],
    ) -> None:
        """Update a user's embedding based on their reading history."""
        # TODO: Implement reading history-based embedding update
        pass
