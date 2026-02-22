from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.models import Book, BookAuthor, BookTag

from .book_engagement import build_book_engagement_map
from .serialize import serialize_book

BOOK_LOAD_OPTIONS = (
    selectinload(Book.authors).selectinload(BookAuthor.author),
    selectinload(Book.tags).selectinload(BookTag.tag),
)


def load_books_by_ids(db: Session, book_ids: list[int]) -> list[Book]:
    if not book_ids:
        return []
    books = db.scalars(
        select(Book).where(Book.id.in_(book_ids)).options(*BOOK_LOAD_OPTIONS)
    ).all()
    by_id = {b.id: b for b in books}
    return [by_id[book_id] for book_id in book_ids if book_id in by_id]


def load_books_by_goodreads_ids(db: Session, goodreads_ids: list[int]) -> list[Book]:
    if not goodreads_ids:
        return []
    books = db.scalars(
        select(Book).where(Book.goodreads_id.in_(goodreads_ids)).options(*BOOK_LOAD_OPTIONS)
    ).all()
    by_gid = {b.goodreads_id: b for b in books}
    return [by_gid[gid] for gid in goodreads_ids if gid in by_gid]


def serialize_books_with_engagement(db: Session, books: list[Book]) -> list[dict]:
    engagement_by_id = build_book_engagement_map(db, [b.id for b in books])
    return [serialize_book(b, engagement=engagement_by_id.get(b.id)) for b in books]
