from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Integer, SmallInteger, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class TimestampMixin:
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    goodreads_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, nullable=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    username: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)

    lists: Mapped[list["BookList"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    shell: Mapped["Shell | None"] = relationship(back_populates="user", cascade="all, delete-orphan", uselist=False)
    ratings: Mapped[list["BookRating"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    reviews: Mapped[list["Review"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    review_comments: Mapped[list["ReviewComment"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    review_likes: Mapped[list["ReviewLike"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Author(TimestampMixin, Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    goodreads_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, nullable=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    books: Mapped[list["BookAuthor"]] = relationship(back_populates="author", cascade="all, delete-orphan")


class Book(TimestampMixin, Base):
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    goodreads_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    format: Mapped[str | None] = mapped_column(Text, nullable=True)
    publisher: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    isbn13: Mapped[str | None] = mapped_column(Text, nullable=True)

    authors: Mapped[list["BookAuthor"]] = relationship(back_populates="book", cascade="all, delete-orphan")
    tags: Mapped[list["BookTag"]] = relationship(back_populates="book", cascade="all, delete-orphan")
    list_entries: Mapped[list["ListBook"]] = relationship(back_populates="book", cascade="all, delete-orphan")
    shell_entries: Mapped[list["ShellBook"]] = relationship(back_populates="book", cascade="all, delete-orphan")
    ratings: Mapped[list["BookRating"]] = relationship(back_populates="book", cascade="all, delete-orphan")
    reviews: Mapped[list["Review"]] = relationship(back_populates="book", cascade="all, delete-orphan")


class BookAuthor(TimestampMixin, Base):
    __tablename__ = "book_authors"

    book_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("books.id", ondelete="CASCADE"), primary_key=True)
    author_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("authors.id", ondelete="RESTRICT"), primary_key=True)

    book: Mapped["Book"] = relationship(back_populates="authors")
    author: Mapped["Author"] = relationship(back_populates="books")


class Tag(TimestampMixin, Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    books: Mapped[list["BookTag"]] = relationship(back_populates="tag", cascade="all, delete-orphan")


class BookTag(TimestampMixin, Base):
    __tablename__ = "book_tags"

    book_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("books.id", ondelete="CASCADE"), primary_key=True)
    tag_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)

    book: Mapped["Book"] = relationship(back_populates="tags")
    tag: Mapped["Tag"] = relationship(back_populates="books")


class BookList(TimestampMixin, Base):
    __tablename__ = "lists"
    __table_args__ = (UniqueConstraint("user_id", "title", name="uq_lists_user_id_title"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="lists")
    books: Mapped[list["ListBook"]] = relationship(back_populates="list", cascade="all, delete-orphan")


class ListBook(TimestampMixin, Base):
    __tablename__ = "list_books"

    list_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("lists.id", ondelete="CASCADE"), primary_key=True)
    book_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("books.id", ondelete="CASCADE"), primary_key=True)
    added_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)

    list: Mapped["BookList"] = relationship(back_populates="books")
    book: Mapped["Book"] = relationship(back_populates="list_entries")


class Shell(TimestampMixin, Base):
    __tablename__ = "shells"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False, server_default="My Shell")

    user: Mapped["User"] = relationship(back_populates="shell")
    books: Mapped[list["ShellBook"]] = relationship(back_populates="shell", cascade="all, delete-orphan")


class ShellBook(TimestampMixin, Base):
    __tablename__ = "shell_books"

    shell_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("shells.id", ondelete="CASCADE"), primary_key=True)
    book_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("books.id", ondelete="CASCADE"), primary_key=True)
    added_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)

    shell: Mapped["Shell"] = relationship(back_populates="books")
    book: Mapped["Book"] = relationship(back_populates="shell_entries")


class BookRating(TimestampMixin, Base):
    __tablename__ = "book_ratings"
    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_book_ratings_range"),
    )

    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    book_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("books.id", ondelete="CASCADE"), primary_key=True)
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    user: Mapped["User"] = relationship(back_populates="ratings")
    book: Mapped["Book"] = relationship(back_populates="ratings")


class Review(TimestampMixin, Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    goodreads_id: Mapped[str | None] = mapped_column(Text, unique=True, nullable=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    book_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    review_text: Mapped[str] = mapped_column(Text, nullable=False)

    user: Mapped["User"] = relationship(back_populates="reviews")
    book: Mapped["Book"] = relationship(back_populates="reviews")
    comments: Mapped[list["ReviewComment"]] = relationship(back_populates="review", cascade="all, delete-orphan")
    likes: Mapped[list["ReviewLike"]] = relationship(back_populates="review", cascade="all, delete-orphan")


class ReviewComment(TimestampMixin, Base):
    __tablename__ = "review_comments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    review_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    comment_text: Mapped[str] = mapped_column(Text, nullable=False)

    review: Mapped["Review"] = relationship(back_populates="comments")
    user: Mapped["User"] = relationship(back_populates="review_comments")


class ReviewLike(Base):
    __tablename__ = "review_likes"

    review_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    review: Mapped["Review"] = relationship(back_populates="likes")
    user: Mapped["User"] = relationship(back_populates="review_likes")
