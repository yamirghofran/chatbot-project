"""Schemas and metadata definitions for vector collections."""

from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class CollectionNames(str, Enum):
    """Enumeration of vector collection names.

    Attributes:
        BOOKS: Collection for book embeddings and metadata
        USERS: Collection for user preference embeddings and metadata
        REVIEWS: Collection for book review embeddings and metadata
    """

    BOOKS = "books"
    USERS = "users"
    REVIEWS = "reviews"



class UserMetadata(BaseModel):
    """Metadata schema for user preference embeddings.
    
    This metadata is stored alongside user embeddings
    and can be used for filtering and analysis.
    
    Attributes:
        user_id: Unique user identifier (from PostgreSQL)
        name: User display name
    """
    
    user_id: int = Field(..., description="User ID from PostgreSQL", ge=1)
    name: str = Field(..., description="User display name")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 12345,
                "name": "Alice",
            }
        }
    )


class ReviewMetadata(BaseModel):
    """Metadata schema for book review embeddings.
    
    This metadata is stored alongside review embeddings
    and can be used for filtering and analysis.
    
    Note:
        The review_id is stored as the document ID, not in metadata.
        Access it via item["id"] when retrieving reviews.
    
    Attributes:
        user_id: User identifier who wrote the review
        book_id: Book identifier being reviewed
    """
    
    user_id: int = Field(..., description="User ID who wrote the review", ge=1)
    book_id: int = Field(..., description="Book ID being reviewed", ge=1)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 12345,
                "book_id": 789,
            }
        }
    )



def validate_user_metadata(metadata: dict) -> UserMetadata:
    """Validate and parse user metadata.
    
    Args:
        metadata: Dictionary containing user metadata
    
    Returns:
        Validated UserMetadata instance
    
    Raises:
        ValueError: If metadata validation fails
    """
    return UserMetadata(**metadata)


def validate_review_metadata(metadata: dict) -> ReviewMetadata:
    """Validate and parse review metadata.
    
    Args:
        metadata: Dictionary containing review metadata
    
    Returns:
        Validated ReviewMetadata instance
    
    Raises:
        ValueError: If metadata validation fails
    """
    return ReviewMetadata(**metadata)
