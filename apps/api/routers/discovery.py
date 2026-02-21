from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.models import (
    Book,
    BookAuthor,
    BookRating,
    BookTag,
)

from ..core.book_queries import load_books_by_ids, load_books_by_goodreads_ids, serialize_books_with_engagement
from ..core.deps import get_db, get_optional_user

router = APIRouter(prefix="/discovery", tags=["discovery"])


def _bpr_recommendations(parquet_path: str, goodreads_user_id: int, limit: int) -> list[int]:
    """Query BPR parquet for top-N recommended goodreads book IDs for a user."""
    rows = duckdb.execute(
        """
        SELECT item_id
        FROM parquet_scan(?)
        WHERE user_id = ?
        ORDER BY prediction DESC
        LIMIT ?
        """,
        [parquet_path, goodreads_user_id, limit],
    ).fetchall()
    return [row[0] for row in rows]


def _cold_start(db: Session, limit: int) -> list[Book]:
    """Return most-rated books as a fallback."""
    popular = db.execute(
        select(BookRating.book_id, func.count(BookRating.book_id).label("cnt"))
        .group_by(BookRating.book_id)
        .order_by(func.count(BookRating.book_id).desc())
        .limit(limit)
    ).all()
    book_ids = [row[0] for row in popular]
    if not book_ids:
        from ..core.book_queries import BOOK_LOAD_OPTIONS
        return db.scalars(
            select(Book)
            .options(*BOOK_LOAD_OPTIONS)
            .limit(limit)
        ).all()
    return load_books_by_ids(db, book_ids)


@router.get("/recommendations")
def get_recommendations(
    limit: int = Query(20, ge=1, le=100),
    request: Request = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    bpr_path: str | None = getattr(request.app.state, "bpr_parquet_path", None)

    # BPR recommendations: user must exist and have a goodreads_id in the parquet.
    if current_user is not None and bpr_path is not None and current_user.goodreads_id is not None:
        goodreads_ids = _bpr_recommendations(bpr_path, current_user.goodreads_id, limit)
        if goodreads_ids:
            books = load_books_by_goodreads_ids(db, goodreads_ids)
            if books:
                return serialize_books_with_engagement(db, books)

    # Cold start fallback.
    return serialize_books_with_engagement(db, _cold_start(db, limit))


@router.get("/staff-picks")
def get_staff_picks(
    limit: int = Query(6, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Return a curated set of well-rated books as staff picks."""
    from ..core.book_queries import BOOK_LOAD_OPTIONS
    popular = db.execute(
        select(BookRating.book_id, func.avg(BookRating.rating).label("avg_rating"), func.count(BookRating.rating).label("cnt"))
        .group_by(BookRating.book_id)
        .having(func.count(BookRating.rating) >= 5)
        .order_by(func.avg(BookRating.rating).desc(), func.count(BookRating.rating).desc())
        .limit(limit)
    ).all()
    book_ids = [row[0] for row in popular]
    if not book_ids:
        books = db.scalars(
            select(Book)
            .options(*BOOK_LOAD_OPTIONS)
            .limit(limit)
        ).all()
        return serialize_books_with_engagement(db, books)
    return serialize_books_with_engagement(db, load_books_by_ids(db, book_ids))


@router.get("/activity")
def get_activity_feed(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    """Return activity from followed users. Returns empty until follow system is implemented."""
    # TODO: filter by current_user's follows once follow table exists.
    return []
