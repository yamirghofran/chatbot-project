from datetime import date
from sqlalchemy import (
    String,
    DateTime,
    Integer,
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

    title: Mapped[str] = mapped_column(String(500), index=True)
    pages_number: Mapped[int | None] = mapped_column(Integer)

    publisher_name: Mapped[str | None] = mapped_column(String(255))
    publish_day: Mapped[int | None] = mapped_column(Integer)
    publish_month: Mapped[int | None] = mapped_column(Integer)
    publish_year: Mapped[int | None] = mapped_column(Integer)

    num_reviews: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_1: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_2: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_3: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_4: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_5: Mapped[int] = mapped_column(Integer, default=0)
    rating_dist_total: Mapped[int] = mapped_column(Integer, default=0)

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

    books: Mapped[list["Book"]] = relationship(
        secondary=book_authors,
        back_populates="authors",
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )