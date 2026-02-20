"""Shared fixtures and factory helpers for PostgreSQL CRUD tests.

Uses an in-memory SQLite database — no running Postgres required.
Each test gets a completely fresh database (function-scoped engine).
"""

import pytest
from sqlalchemy import BigInteger, Integer, create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session


# SQLite only auto-increments columns declared as `INTEGER PRIMARY KEY`.
# The models use `BigInteger` which renders as `BIGINT`, disabling auto-ID.
# Override the type for the sqlite dialect so all BigInteger columns
# become `INTEGER`, restoring auto-increment behaviour in tests.
@compiles(BigInteger, "sqlite")
def _sqlite_bigint(type_, compiler, **kwargs):  # noqa: ARG001
    return "INTEGER"

from bookdb.db.base import Base
from bookdb.db.crud import (
    AuthorCRUD,
    BookCRUD,
    BookListCRUD,
    BookRatingCRUD,
    ReviewCRUD,
    ShellCRUD,
    UserCRUD,
)


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session():
    """Provide a fresh, isolated in-memory SQLite session for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine, autoflush=False) as sess:
        yield sess


# ---------------------------------------------------------------------------
# Factory helpers (plain functions, not fixtures, so tests can call them
# with custom arguments easily)
# ---------------------------------------------------------------------------


def make_user(
    session,
    email="alice@example.com",
    name="Alice",
    username="alice",
    password_hash="hashed_pw",
    **kwargs,
):
    return UserCRUD.create(
        session,
        email=email,
        name=name,
        username=username,
        password_hash=password_hash,
        **kwargs,
    )


def make_author(session, name="George Orwell", **kwargs):
    return AuthorCRUD.create(session, name=name, **kwargs)


def make_book(session, goodreads_id=1001, title="1984", **kwargs):
    return BookCRUD.create(session, goodreads_id=goodreads_id, title=title, **kwargs)


def make_review(session, user, book, text="Great read!", **kwargs):
    return ReviewCRUD.create(
        session, user_id=user.id, book_id=book.id, review_text=text, **kwargs
    )


def make_book_list(session, user, title="Favourites", **kwargs):
    return BookListCRUD.create(session, user_id=user.id, title=title, **kwargs)


def make_shell(session, user, name="My Shell"):
    return ShellCRUD.create(session, user_id=user.id, name=name)
