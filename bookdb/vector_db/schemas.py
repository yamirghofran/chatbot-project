"""Schemas and metadata definitions for ChromaDB collections."""

from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class CollectionNames(str, Enum):
    """Enumeration of collection names in ChromaDB.

    Attributes:
        BOOKS: Collection for book embeddings and metadata
        AUTHORS: Collection for author embeddings and metadata
        USERS: Collection for user preference embeddings and metadata
        REVIEWS: Collection for book review embeddings and metadata
    """

    BOOKS = "books"
    AUTHORS = "authors"
    USERS = "users"
    REVIEWS = "reviews"


class BookMetadata(BaseModel):
    """Metadata schema for book embeddings.

    This metadata is stored alongside book embeddings in ChromaDB
    and can be used for filtering during similarity search.

    Attributes:
        title: Book title
        author: Book author(s)
        genre: Book genre/category
        publication_year: Year the book was published
        created_at: Timestamp when embedding was created
    """

    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Book author(s)")
    genre: Optional[str] = Field(None, description="Book genre/category")
    publication_year: Optional[int] = Field(None, description="Publication year", ge=1000, le=9999)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Creation timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "genre": "Fiction",
                "publication_year": 1925,
            }
        }
    )


class AuthorMetadata(BaseModel):
    """Metadata schema for author embeddings.

    This metadata is stored alongside author embeddings in ChromaDB
    and can be used for filtering during similarity search.

    Attributes:
        name: Author name
        created_at: Timestamp when embedding was created
    """

    name: str = Field(..., description="Author name")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "F. Scott Fitzgerald",
            }
        }
    )


class UserMetadata(BaseModel):
    """Metadata schema for user preference embeddings.
    
    This metadata is stored alongside user embeddings in ChromaDB
    and can be used for filtering and analysis.
    
    Attributes:
        user_id: Unique user identifier (from PostgreSQL)
        num_books_read: Total number of books the user has read
        favorite_genres: List of user's favorite genres
        average_rating_given: Average rating the user gives to books
        last_active: Timestamp of last user activity
        reading_level: User's reading level/preference
        created_at: Timestamp when embedding was created
        updated_at: Timestamp when embedding was last updated
    """
    
    user_id: int = Field(..., description="User ID from PostgreSQL", ge=1)
    num_books_read: int = Field(default=0, description="Number of books read", ge=0)
    favorite_genres: Optional[str] = Field(None, description="Comma-separated favorite genres")
    average_rating_given: Optional[float] = Field(None, description="Average rating given", ge=0.0, le=5.0)
    last_active: Optional[str] = Field(None, description="Last activity timestamp")
    reading_level: Optional[str] = Field(None, description="Reading level/preference")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Creation timestamp")
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Last update timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 12345,
                "num_books_read": 42,
                "favorite_genres": "Fiction,Science Fiction,Mystery",
                "average_rating_given": 4.2,
                "reading_level": "advanced",
            }
        }
    )


class ReviewMetadata(BaseModel):
    """Metadata schema for book review embeddings.
    
    This metadata is stored alongside review embeddings in ChromaDB
    and can be used for filtering and analysis.
    
    Note:
        The review_id is stored as the document ID in ChromaDB, not in metadata.
        Access it via item["id"] when retrieving reviews.
    
    Attributes:
        user_id: User identifier who wrote the review
        book_id: Book identifier being reviewed
        rating: Rating given (1-5)
        date_added: Date when review was added
        date_updated: Date when review was last updated
        read_at: Date when book was read
        created_at: Timestamp when embedding was created
    """
    
    user_id: str = Field(..., description="User ID who wrote the review")
    book_id: str = Field(..., description="Book ID being reviewed")
    rating: int = Field(..., description="Rating given (1-5)", ge=1, le=5)
    date_added: Optional[str] = Field(None, description="Date when review was added")
    date_updated: Optional[str] = Field(None, description="Date when review was last updated")
    read_at: Optional[str] = Field(None, description="Date when book was read")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Creation timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "book_id": "book_789",
                "rating": 5,
                "date_added": "2024-01-15",
                "date_updated": "2024-01-20",
                "read_at": "2024-01-10",
            }
        }
    )


def validate_book_metadata(metadata: dict) -> BookMetadata:
    """Validate and parse book metadata.
    
    Args:
        metadata: Dictionary containing book metadata
    
    Returns:
        Validated BookMetadata instance
    
    Raises:
        ValueError: If metadata validation fails
    """
    return BookMetadata(**metadata)


def validate_author_metadata(metadata: dict) -> AuthorMetadata:
    """Validate and parse author metadata.

    Args:
        metadata: Dictionary containing author metadata

    Returns:
        Validated AuthorMetadata instance

    Raises:
        ValueError: If metadata validation fails
    """
    return AuthorMetadata(**metadata)


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
