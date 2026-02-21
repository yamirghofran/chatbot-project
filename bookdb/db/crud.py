"""PostgreSQL CRUD helpers aligned with the current SQLAlchemy schema.

This module provides lightweight, explicit CRUD classes per model in
``bookdb.db.models``. All methods work with a SQLAlchemy ``Session`` and
flush on writes so IDs and relationship rows are available immediately.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    Author,
    Book,
    BookAuthor,
    BookList,
    BookRating,
    BookTag,
    ListBook,
    Review,
    ReviewComment,
    ReviewLike,
    Shell,
    ShellBook,
    Tag,
    User,
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _parse_optional_bigint(value: int | float | str | None) -> int | None:
    """Parse a value into an integer ID, returning None when empty/NaN."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    if isinstance(value, int):
        return value
    value = str(value).strip()
    if not value:
        return None
    return int(value)


def _require_non_empty(value: str | None, field_name: str) -> str:
    """Validate that a string field is not None, empty, or whitespace-only."""
    if value is None:
        raise ValueError(f"{field_name} is required")
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_email(email: str) -> str:
    """Validate basic email format and return the stripped value."""
    email = _require_non_empty(email, "email")
    if not _EMAIL_RE.match(email):
        raise ValueError(f"Invalid email format: {email!r}")
    return email


def _validate_year(year: int | None) -> int | None:
    """Validate that a publication year is within a sane range, if provided."""
    if year is None:
        return None
    if not isinstance(year, int):
        raise ValueError(f"publication_year must be an integer, got {type(year).__name__}")
    if year < 1000 or year > 9999:
        raise ValueError(f"publication_year must be between 1000 and 9999, got {year}")
    return year


def _check_unique(
    session: Session, model, field, value, label: str, exclude_id: int | None = None,
) -> None:
    """Pre-check a UNIQUE column, raising ValueError on conflict."""
    stmt = select(model).where(field == value)
    if exclude_id is not None:
        stmt = stmt.where(model.id != exclude_id)
    if session.scalar(stmt) is not None:
        raise ValueError(f"{label} {value!r} is already taken")


class AuthorCRUD:
    @staticmethod
    def get_by_id(session: Session, author_id: int) -> Author | None:
        return session.get(Author, author_id)

    @staticmethod
    def get_by_name(session: Session, name: str) -> Author | None:
        stmt = select(Author).where(Author.name == name)
        return session.scalar(stmt)

    @staticmethod
    def get_by_goodreads_id(session: Session, goodreads_id: int | str) -> Author | None:
        gid = _parse_optional_bigint(goodreads_id)
        stmt = select(Author).where(Author.goodreads_id == gid)
        return session.scalar(stmt)

    @staticmethod
    def create(session: Session, name: str, **kwargs) -> Author:
        name = _require_non_empty(name, "name")
        if "goodreads_id" in kwargs:
            gid = _parse_optional_bigint(kwargs["goodreads_id"])
            kwargs["goodreads_id"] = gid
            if gid is not None:
                _check_unique(session, Author, Author.goodreads_id, gid, "goodreads_id")
        author = Author(name=name, **kwargs)
        session.add(author)
        session.flush()
        return author

    @staticmethod
    def update(session: Session, author_id: int, **kwargs) -> Author:
        author = session.get(Author, author_id)
        if not author:
            raise ValueError(f"Author with id {author_id} not found")
        if "name" in kwargs:
            kwargs["name"] = _require_non_empty(kwargs["name"], "name")
        if "goodreads_id" in kwargs:
            gid = _parse_optional_bigint(kwargs["goodreads_id"])
            kwargs["goodreads_id"] = gid
            if gid is not None:
                _check_unique(session, Author, Author.goodreads_id, gid, "goodreads_id", exclude_id=author_id)
        for key, value in kwargs.items():
            setattr(author, key, value)
        session.flush()
        return author

    @staticmethod
    def delete(session: Session, author_id: int) -> bool:
        author = session.get(Author, author_id)
        if not author:
            return False
        session.delete(author)
        session.flush()
        return True

    @staticmethod
    def get_or_create(session: Session, name: str, **kwargs) -> Author:
        existing = AuthorCRUD.get_by_name(session, name)
        if existing:
            return existing
        return AuthorCRUD.create(session, name=name, **kwargs)

    @staticmethod
    def bulk_get_or_create_by_names(
        session: Session, names: list[str]
    ) -> dict[str, Author]:
        unique_names = list(dict.fromkeys(name for name in names if name))
        if not unique_names:
            return {}
        stmt = select(Author).where(Author.name.in_(unique_names))
        existing = {a.name: a for a in session.scalars(stmt).all()}
        to_create = [name for name in unique_names if name not in existing]
        if to_create:
            new_authors = [Author(name=name) for name in to_create]
            session.add_all(new_authors)
            session.flush()
            for author in new_authors:
                existing[author.name] = author
        return existing

    @staticmethod
    def bulk_get_by_goodreads_ids(
        session: Session, goodreads_ids: list[int | str]
    ) -> dict[int, Author]:
        normalized = {_parse_optional_bigint(gid) for gid in goodreads_ids}
        normalized.discard(None)
        if not normalized:
            return {}
        stmt = select(Author).where(Author.goodreads_id.in_(normalized))
        return {a.goodreads_id: a for a in session.scalars(stmt).all() if a.goodreads_id is not None}

    # Compatibility alias for older code paths.
    get_by_external_id = get_by_goodreads_id
    bulk_get_by_external_ids = bulk_get_by_goodreads_ids


class BookCRUD:
    @staticmethod
    def get_by_id(session: Session, book_id: int) -> Book | None:
        return session.get(Book, book_id)

    @staticmethod
    def get_by_title(session: Session, title: str) -> list[Book]:
        stmt = select(Book).where(Book.title == title)
        return session.scalars(stmt).all()

    @staticmethod
    def search_by_title(session: Session, query: str, limit: int = 100) -> list[Book]:
        stmt = select(Book).where(Book.title.ilike(f"%{query}%")).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def get_by_goodreads_id(session: Session, goodreads_id: int | str) -> Book | None:
        gid = _parse_optional_bigint(goodreads_id)
        stmt = select(Book).where(Book.goodreads_id == gid)
        return session.scalar(stmt)

    @staticmethod
    def create(session: Session, goodreads_id: int | str, **kwargs) -> Book:
        gid = _parse_optional_bigint(goodreads_id)
        if gid is None:
            raise ValueError("goodreads_id is required and must be a valid integer")
        _check_unique(session, Book, Book.goodreads_id, gid, "goodreads_id")
        if "title" in kwargs:
            kwargs["title"] = _require_non_empty(kwargs["title"], "title")
        if "publication_year" in kwargs:
            kwargs["publication_year"] = _validate_year(kwargs.get("publication_year"))
        kwargs["goodreads_id"] = gid
        book = Book(**kwargs)
        session.add(book)
        session.flush()
        return book

    @staticmethod
    def update(
        session: Session,
        book_id: int,
        **kwargs,
    ) -> Book:
        book = session.get(Book, book_id)
        if not book:
            raise ValueError(f"Book with id {book_id} not found")
        if "title" in kwargs:
            kwargs["title"] = _require_non_empty(kwargs["title"], "title")
        if "publication_year" in kwargs:
            kwargs["publication_year"] = _validate_year(kwargs["publication_year"])
        if "goodreads_id" in kwargs:
            gid = _parse_optional_bigint(kwargs["goodreads_id"])
            kwargs["goodreads_id"] = gid
            if gid is not None:
                _check_unique(session, Book, Book.goodreads_id, gid, "goodreads_id", exclude_id=book_id)
        for key, value in kwargs.items():
            setattr(book, key, value)
        session.flush()
        return book

    @staticmethod
    def delete(session: Session, book_id: int) -> bool:
        book = session.get(Book, book_id)
        if not book:
            return False
        session.delete(book)
        session.flush()
        return True

    @staticmethod
    def create_with_authors(
        session: Session, author_names: list[str], **book_kwargs
    ) -> Book:
        book = BookCRUD.create(session, **book_kwargs)
        if author_names:
            unique_names = list(dict.fromkeys(author_names))
            authors = AuthorCRUD.bulk_get_or_create_by_names(session, unique_names)
            for name in unique_names:
                author = authors[name]
                session.add(BookAuthor(book_id=book.id, author_id=author.id))
        session.flush()
        return book

    @staticmethod
    def get_by_author(session: Session, author_id: int, limit: int = 100) -> list[Book]:
        stmt = (
            select(Book)
            .join(BookAuthor, BookAuthor.book_id == Book.id)
            .where(BookAuthor.author_id == author_id)
            .limit(limit)
        )
        return session.scalars(stmt).all()

    # Compatibility alias for older code paths.
    get_by_book_id = get_by_goodreads_id


class UserCRUD:
    @staticmethod
    def get_by_id(session: Session, user_id: int) -> User | None:
        return session.get(User, user_id)

    @staticmethod
    def get_by_email(session: Session, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        return session.scalar(stmt)

    @staticmethod
    def get_by_username(session: Session, username: str) -> User | None:
        stmt = select(User).where(User.username == username)
        return session.scalar(stmt)

    @staticmethod
    def get_by_goodreads_id(session: Session, goodreads_id: int | str) -> User | None:
        gid = _parse_optional_bigint(goodreads_id)
        stmt = select(User).where(User.goodreads_id == gid)
        return session.scalar(stmt)

    @staticmethod
    def create(
        session: Session,
        email: str,
        name: str,
        username: str,
        password_hash: str,
        **kwargs,
    ) -> User:
        email = _validate_email(email)
        name = _require_non_empty(name, "name")
        username = _require_non_empty(username, "username")
        password_hash = _require_non_empty(password_hash, "password_hash")
        _check_unique(session, User, User.email, email, "email")
        _check_unique(session, User, User.username, username, "username")
        if "goodreads_id" in kwargs:
            gid = _parse_optional_bigint(kwargs["goodreads_id"])
            kwargs["goodreads_id"] = gid
            if gid is not None:
                _check_unique(session, User, User.goodreads_id, gid, "goodreads_id")
        user = User(
            email=email,
            name=name,
            username=username,
            password_hash=password_hash,
            **kwargs,
        )
        session.add(user)
        session.flush()
        return user

    @staticmethod
    def update(session: Session, user_id: int, **kwargs) -> User:
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with id {user_id} not found")
        if "email" in kwargs:
            kwargs["email"] = _validate_email(kwargs["email"])
            _check_unique(session, User, User.email, kwargs["email"], "email", exclude_id=user_id)
        if "name" in kwargs:
            kwargs["name"] = _require_non_empty(kwargs["name"], "name")
        if "username" in kwargs:
            kwargs["username"] = _require_non_empty(kwargs["username"], "username")
            _check_unique(session, User, User.username, kwargs["username"], "username", exclude_id=user_id)
        if "password_hash" in kwargs:
            kwargs["password_hash"] = _require_non_empty(kwargs["password_hash"], "password_hash")
        if "goodreads_id" in kwargs:
            gid = _parse_optional_bigint(kwargs["goodreads_id"])
            kwargs["goodreads_id"] = gid
            if gid is not None:
                _check_unique(session, User, User.goodreads_id, gid, "goodreads_id", exclude_id=user_id)
        for key, value in kwargs.items():
            setattr(user, key, value)
        session.flush()
        return user

    @staticmethod
    def delete(session: Session, user_id: int) -> bool:
        user = session.get(User, user_id)
        if not user:
            return False
        session.delete(user)
        session.flush()
        return True

    @staticmethod
    def get_or_create(
        session: Session,
        email: str,
        name: str,
        username: str,
        password_hash: str,
        **kwargs,
    ) -> User:
        existing = UserCRUD.get_by_email(session, email)
        if existing:
            return existing
        return UserCRUD.create(
            session=session,
            email=email,
            name=name,
            username=username,
            password_hash=password_hash,
            **kwargs,
        )

    @staticmethod
    def create_import_stub(session: Session, goodreads_id: int | str) -> User:
        gid = _parse_optional_bigint(goodreads_id)
        if gid is None:
            raise ValueError("goodreads_id is required to create an import stub user")
        existing = UserCRUD.get_by_goodreads_id(session, gid)
        if existing:
            return existing
        prefix = str(gid)[:12]
        return UserCRUD.create(
            session=session,
            email=f"{prefix}@import.local",
            name=f"User {prefix}",
            username=f"user_{prefix}",
            password_hash="import_stub",
            goodreads_id=gid,
        )

    @staticmethod
    def bulk_get_by_goodreads_ids(
        session: Session, goodreads_ids: list[int | str]
    ) -> dict[int, User]:
        normalized = {_parse_optional_bigint(gid) for gid in goodreads_ids}
        normalized.discard(None)
        if not normalized:
            return {}
        stmt = select(User).where(User.goodreads_id.in_(normalized))
        return {u.goodreads_id: u for u in session.scalars(stmt).all() if u.goodreads_id is not None}

    @staticmethod
    def bulk_get_or_create_from_goodreads_ids(
        session: Session, goodreads_ids: list[int | str]
    ) -> dict[int, User]:
        normalized = {_parse_optional_bigint(gid) for gid in goodreads_ids}
        normalized.discard(None)
        if not normalized:
            return {}
        existing = UserCRUD.bulk_get_by_goodreads_ids(session, list(normalized))
        missing = [gid for gid in normalized if gid not in existing]
        for gid in missing:
            user = UserCRUD.create_import_stub(session, gid)
            existing[gid] = user
        return existing

    # Compatibility alias for older code paths.
    get_by_external_id = get_by_goodreads_id
    bulk_get_by_external_ids = bulk_get_by_goodreads_ids
    bulk_get_or_create_from_external = bulk_get_or_create_from_goodreads_ids


class BookAuthorCRUD:
    @staticmethod
    def link(session: Session, book_id: int, author_id: int) -> BookAuthor:
        existing = session.get(BookAuthor, (book_id, author_id))
        if existing:
            return existing
        link = BookAuthor(book_id=book_id, author_id=author_id)
        session.add(link)
        session.flush()
        return link

    @staticmethod
    def unlink(session: Session, book_id: int, author_id: int) -> bool:
        existing = session.get(BookAuthor, (book_id, author_id))
        if not existing:
            return False
        session.delete(existing)
        session.flush()
        return True


class BookListCRUD:
    @staticmethod
    def get_by_id(session: Session, list_id: int) -> BookList | None:
        return session.get(BookList, list_id)

    @staticmethod
    def get_by_user(session: Session, user_id: int) -> list[BookList]:
        stmt = select(BookList).where(BookList.user_id == user_id)
        return session.scalars(stmt).all()

    @staticmethod
    def create(
        session: Session, user_id: int, title: str, description: str | None = None
    ) -> BookList:
        title = _require_non_empty(title, "title")
        existing = session.scalar(
            select(BookList).where(BookList.user_id == user_id, BookList.title == title)
        )
        if existing:
            raise ValueError(f"User already has a list titled {title!r}")
        book_list = BookList(user_id=user_id, title=title, description=description)
        session.add(book_list)
        session.flush()
        return book_list

    @staticmethod
    def add_book(session: Session, list_id: int, book_id: int) -> bool:
        book_list = BookListCRUD.get_by_id(session, list_id)
        book = BookCRUD.get_by_id(session, book_id)
        if not book_list or not book:
            return False
        existing = session.get(ListBook, (list_id, book_id))
        if existing:
            return True
        session.add(
            ListBook(
                list_id=list_id,
                book_id=book_id,
                added_at=datetime.now(timezone.utc),
            )
        )
        session.flush()
        return True

    @staticmethod
    def remove_book(session: Session, list_id: int, book_id: int) -> bool:
        existing = session.get(ListBook, (list_id, book_id))
        if not existing:
            return False
        session.delete(existing)
        session.flush()
        return True


class ShellCRUD:
    @staticmethod
    def get_by_id(session: Session, shell_id: int) -> Shell | None:
        return session.get(Shell, shell_id)

    @staticmethod
    def get_by_user(session: Session, user_id: int) -> Shell | None:
        stmt = select(Shell).where(Shell.user_id == user_id)
        return session.scalar(stmt)

    @staticmethod
    def create(session: Session, user_id: int, name: str = "My Shell") -> Shell:
        shell = Shell(user_id=user_id, name=name)
        session.add(shell)
        session.flush()
        return shell

    @staticmethod
    def get_or_create_for_user(session: Session, user_id: int, name: str = "My Shell") -> Shell:
        existing = ShellCRUD.get_by_user(session, user_id)
        if existing:
            return existing
        return ShellCRUD.create(session, user_id=user_id, name=name)

    @staticmethod
    def add_book(session: Session, shell_id: int, book_id: int) -> bool:
        shell = ShellCRUD.get_by_id(session, shell_id)
        book = BookCRUD.get_by_id(session, book_id)
        if not shell or not book:
            return False
        existing = session.get(ShellBook, (shell_id, book_id))
        if existing:
            return True
        session.add(
            ShellBook(
                shell_id=shell_id,
                book_id=book_id,
                added_at=datetime.now(timezone.utc),
            )
        )
        session.flush()
        return True

    @staticmethod
    def remove_book(session: Session, shell_id: int, book_id: int) -> bool:
        existing = session.get(ShellBook, (shell_id, book_id))
        if not existing:
            return False
        session.delete(existing)
        session.flush()
        return True


class BookRatingCRUD:
    @staticmethod
    def get(session: Session, user_id: int, book_id: int) -> BookRating | None:
        return session.get(BookRating, (user_id, book_id))

    @staticmethod
    def get_by_user(session: Session, user_id: int, limit: int = 100) -> list[BookRating]:
        stmt = select(BookRating).where(BookRating.user_id == user_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def get_by_book(session: Session, book_id: int, limit: int = 100) -> list[BookRating]:
        stmt = select(BookRating).where(BookRating.book_id == book_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def upsert(session: Session, user_id: int, book_id: int, rating: int) -> BookRating:
        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be between 1 and 5, got {rating}")
        existing = session.get(BookRating, (user_id, book_id))
        if existing:
            existing.rating = rating
            session.flush()
            return existing
        new_rating = BookRating(user_id=user_id, book_id=book_id, rating=rating)
        session.add(new_rating)
        session.flush()
        return new_rating

    @staticmethod
    def delete(session: Session, user_id: int, book_id: int) -> bool:
        existing = session.get(BookRating, (user_id, book_id))
        if not existing:
            return False
        session.delete(existing)
        session.flush()
        return True


class ReviewCRUD:
    @staticmethod
    def get_by_id(session: Session, review_id: int) -> Review | None:
        return session.get(Review, review_id)

    @staticmethod
    def get_by_goodreads_id(session: Session, goodreads_id: str) -> Review | None:
        stmt = select(Review).where(Review.goodreads_id == goodreads_id)
        return session.scalar(stmt)

    @staticmethod
    def get_by_user(session: Session, user_id: int, limit: int = 100) -> list[Review]:
        stmt = select(Review).where(Review.user_id == user_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def get_by_book(session: Session, book_id: int, limit: int = 100) -> list[Review]:
        stmt = select(Review).where(Review.book_id == book_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def get_by_user_and_book(session: Session, user_id: int, book_id: int) -> list[Review]:
        stmt = select(Review).where(Review.user_id == user_id, Review.book_id == book_id)
        return session.scalars(stmt).all()

    @staticmethod
    def create(
        session: Session,
        user_id: int,
        book_id: int,
        review_text: str,
        goodreads_id: str | None = None,
    ) -> Review:
        review_text = _require_non_empty(review_text, "review_text")
        if goodreads_id is not None:
            goodreads_id = _require_non_empty(goodreads_id, "goodreads_id")
            _check_unique(session, Review, Review.goodreads_id, goodreads_id, "goodreads_id")
        review = Review(
            user_id=user_id,
            book_id=book_id,
            review_text=review_text,
            goodreads_id=goodreads_id,
        )
        session.add(review)
        session.flush()
        return review

    @staticmethod
    def update(session: Session, review_id: int, **kwargs) -> Review:
        review = session.get(Review, review_id)
        if not review:
            raise ValueError(f"Review with id {review_id} not found")
        if "review_text" in kwargs:
            kwargs["review_text"] = _require_non_empty(kwargs["review_text"], "review_text")
        if "goodreads_id" in kwargs and kwargs["goodreads_id"] is not None:
            kwargs["goodreads_id"] = _require_non_empty(kwargs["goodreads_id"], "goodreads_id")
            _check_unique(session, Review, Review.goodreads_id, kwargs["goodreads_id"], "goodreads_id", exclude_id=review_id)
        for key, value in kwargs.items():
            setattr(review, key, value)
        session.flush()
        return review

    @staticmethod
    def delete(session: Session, review_id: int) -> bool:
        review = session.get(Review, review_id)
        if not review:
            return False
        session.delete(review)
        session.flush()
        return True


class ReviewCommentCRUD:
    @staticmethod
    def get_by_id(session: Session, comment_id: int) -> ReviewComment | None:
        return session.get(ReviewComment, comment_id)

    @staticmethod
    def get_by_review(
        session: Session, review_id: int, limit: int = 100
    ) -> list[ReviewComment]:
        stmt = (
            select(ReviewComment)
            .where(ReviewComment.review_id == review_id)
            .limit(limit)
        )
        return session.scalars(stmt).all()

    @staticmethod
    def create(session: Session, review_id: int, user_id: int, comment_text: str) -> ReviewComment:
        comment_text = _require_non_empty(comment_text, "comment_text")
        comment = ReviewComment(
            review_id=review_id,
            user_id=user_id,
            comment_text=comment_text,
        )
        session.add(comment)
        session.flush()
        return comment

    @staticmethod
    def delete(session: Session, comment_id: int) -> bool:
        comment = session.get(ReviewComment, comment_id)
        if not comment:
            return False
        session.delete(comment)
        session.flush()
        return True


class ReviewLikeCRUD:
    @staticmethod
    def get(session: Session, review_id: int, user_id: int) -> ReviewLike | None:
        return session.get(ReviewLike, (review_id, user_id))

    @staticmethod
    def get_by_review(session: Session, review_id: int, limit: int = 100) -> list[ReviewLike]:
        stmt = select(ReviewLike).where(ReviewLike.review_id == review_id).limit(limit)
        return session.scalars(stmt).all()

    @staticmethod
    def add(session: Session, review_id: int, user_id: int) -> ReviewLike:
        existing = session.get(ReviewLike, (review_id, user_id))
        if existing:
            return existing
        like = ReviewLike(review_id=review_id, user_id=user_id)
        session.add(like)
        session.flush()
        return like

    @staticmethod
    def remove(session: Session, review_id: int, user_id: int) -> bool:
        existing = session.get(ReviewLike, (review_id, user_id))
        if not existing:
            return False
        session.delete(existing)
        session.flush()
        return True


class TagCRUD:
    @staticmethod
    def get_by_name(session: Session, name: str) -> Tag | None:
        stmt = select(Tag).where(Tag.name == name)
        return session.scalar(stmt)

    @staticmethod
    def get_or_create_by_name(session: Session, name: str) -> Tag:
        name = _require_non_empty(name, "name")
        existing = TagCRUD.get_by_name(session, name)
        if existing:
            return existing
        tag = Tag(name=name)
        session.add(tag)
        session.flush()
        return tag

    @staticmethod
    def bulk_get_or_create(session: Session, names: list[str]) -> dict[str, Tag]:
        unique_names = list(dict.fromkeys(n for n in names if n))
        if not unique_names:
            return {}
        stmt = select(Tag).where(Tag.name.in_(unique_names))
        existing = {t.name: t for t in session.scalars(stmt).all()}
        to_create = [name for name in unique_names if name not in existing]
        if to_create:
            new_tags = [Tag(name=name) for name in to_create]
            session.add_all(new_tags)
            session.flush()
            for tag in new_tags:
                existing[tag.name] = tag
        return existing

    @staticmethod
    def get_tags_for_book(session: Session, book_id: int) -> list[Tag]:
        stmt = (
            select(Tag)
            .join(BookTag, BookTag.tag_id == Tag.id)
            .where(BookTag.book_id == book_id)
        )
        return session.scalars(stmt).all()

    @staticmethod
    def link_tags_to_book(session: Session, book_id: int, tag_names: list[str]) -> None:
        tags = TagCRUD.bulk_get_or_create(session, tag_names)
        for name in tag_names:
            tag = tags.get(name)
            if tag is None:
                continue
            existing = session.get(BookTag, (book_id, tag.id))
            if not existing:
                session.add(BookTag(book_id=book_id, tag_id=tag.id))
        session.flush()


# Compatibility alias for older callsites using RatingCRUD.
class RatingCRUD(BookRatingCRUD):
    pass

