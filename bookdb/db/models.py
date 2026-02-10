from datetime import date
from sqlalchemy import (
    String,
    DateTime,
    Integer,
    Float,
    Boolean,
    ForeignKey,
    Table,
    func,
    Column
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)
from .base import Base


book_authors = Table(
    "book_authors",
    Base.metadata,
    Column("book_id", ForeignKey("books.id"), primary_key=True),
    Column("author_id", ForeignKey("authors.id"), primary_key=True),
)
list_books = Table(
    "list_books",
    Base.metadata,
    Column("list_id", ForeignKey("lists.id"), primary_key=True),
    Column("book_id", ForeignKey("books.id"), primary_key=True),
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))

    lists: Mapped[list["BookList"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class BookList(Base):
    __tablename__ = "lists"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )
    user: Mapped["User"] = relationship(back_populates="lists")

    books: Mapped[list["Book"]] = relationship(
        secondary=list_books,
        back_populates="lists",
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

class Book(Base):
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True)

    isbn: Mapped[str | None] = mapped_column(String(20))
    isbn13: Mapped[str | None] = mapped_column(String(20))
    asin: Mapped[str | None] = mapped_column(String(20))
    kindle_asin: Mapped[str | None] = mapped_column(String(20))

    title: Mapped[str] = mapped_column(String(500), index=True)
    title_without_series: Mapped[str | None] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(String)
    pages_number: Mapped[int | None] = mapped_column(Integer)
    num_pages: Mapped[int | None] = mapped_column(Integer)

    publisher: Mapped[str | None] = mapped_column(String(255))
    publisher_name: Mapped[str | None] = mapped_column(String(255))
    publication_day: Mapped[int | None] = mapped_column(Integer)
    publication_month: Mapped[int | None] = mapped_column(Integer)
    publication_year: Mapped[int | None] = mapped_column(Integer)

    num_reviews: Mapped[int] = mapped_column(Integer, default=0)
    text_reviews_count: Mapped[int | None] = mapped_column(Integer)
    ratings_count: Mapped[int | None] = mapped_column(Integer)
    ratings_sum: Mapped[int | None] = mapped_column(Integer)

    rating_dist_1: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_2: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_3: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_4: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_5: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_total: Mapped[int] = mapped_column(Integer, default=0)
    average_rating: Mapped[float | None] = mapped_column(Float)

    books_count: Mapped[int | None] = mapped_column(Integer)
    media_type: Mapped[str | None] = mapped_column(String(50))
    best_book_id: Mapped[str | None] = mapped_column(String(50))
    work_id: Mapped[str | None] = mapped_column(String(50))
    original_language_id: Mapped[str | None] = mapped_column(String(50))
    default_description_language_code: Mapped[str | None] = mapped_column(String(10))
    default_chaptering_book_id: Mapped[str | None] = mapped_column(String(50))

    is_ebook: Mapped[bool | None] = mapped_column(Boolean)
    format: Mapped[str | None] = mapped_column(String(50))
    edition_information: Mapped[str | None] = mapped_column(String(255))
    language_code: Mapped[str | None] = mapped_column(String(10))
    country_code: Mapped[str | None] = mapped_column(String(10))

    url: Mapped[str | None] = mapped_column(String(1000))
    link: Mapped[str | None] = mapped_column(String(1000))
    image_url: Mapped[str | None] = mapped_column(String(1000))

    # External identifiers
    book_id: Mapped[str | None] = mapped_column(String(50))
    author_id: Mapped[str | None] = mapped_column(String(50))
    role: Mapped[str | None] = mapped_column(String(50))
    element: Mapped[str | None] = mapped_column(String(50))
    count: Mapped[int | None] = mapped_column(Integer)
    name: Mapped[str | None] = mapped_column(String(255))

    # Relationships
    authors: Mapped[list["Author"]] = relationship(
        secondary=book_authors,
        back_populates="books",
    )
    lists: Mapped[list["BookList"]] = relationship(
        secondary=list_books,
        back_populates="books",
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    average_rating: Mapped[float | None] = mapped_column(Float)
    ratings_count: Mapped[int | None] = mapped_column(Integer)
    text_reviews_count: Mapped[int | None] = mapped_column(Integer)

    books: Mapped[list["Book"]] = relationship(
        secondary=book_authors,
        back_populates="authors",
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )