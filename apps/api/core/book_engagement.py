from __future__ import annotations

import threading

from cachetools import TTLCache
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bookdb.db.models import Book, BookRating, Review, ShellBook

_engagement_cache: TTLCache = TTLCache(maxsize=2000, ttl=300)
_cache_lock = threading.Lock()


def build_book_engagement_map(db: Session, book_ids: list[int]) -> dict[int, dict]:
    if not book_ids:
        return {}

    unique_ids = list(dict.fromkeys(book_ids))
    result: dict[int, dict] = {}
    uncached_ids: list[int] = []

    with _cache_lock:
        for book_id in unique_ids:
            if book_id in _engagement_cache:
                result[book_id] = _engagement_cache[book_id]
            else:
                uncached_ids.append(book_id)

    if not uncached_ids:
        return result

    # Defaults for books with no stats (stays if book has no ratings/reviews/shells)
    for book_id in uncached_ids:
        result[book_id] = {
            "averageRating": None,
            "ratingCount": 0,
            "commentCount": 0,
            "shellCount": 0,
        }

    rating_sq = (
        select(
            BookRating.book_id,
            func.avg(BookRating.rating).label("avg_r"),
            func.count().label("rc"),
        )
        .where(BookRating.book_id.in_(uncached_ids))
        .group_by(BookRating.book_id)
        .subquery()
    )
    review_sq = (
        select(Review.book_id, func.count().label("rvc"))
        .where(Review.book_id.in_(uncached_ids))
        .group_by(Review.book_id)
        .subquery()
    )
    shell_sq = (
        select(ShellBook.book_id, func.count().label("sc"))
        .where(ShellBook.book_id.in_(uncached_ids))
        .group_by(ShellBook.book_id)
        .subquery()
    )

    rows = db.execute(
        select(
            Book.id.label("book_id"),
            rating_sq.c.avg_r,
            func.coalesce(rating_sq.c.rc, 0).label("rc"),
            func.coalesce(review_sq.c.rvc, 0).label("rvc"),
            func.coalesce(shell_sq.c.sc, 0).label("sc"),
        )
        .where(Book.id.in_(uncached_ids))
        .outerjoin(rating_sq, rating_sq.c.book_id == Book.id)
        .outerjoin(review_sq, review_sq.c.book_id == Book.id)
        .outerjoin(shell_sq, shell_sq.c.book_id == Book.id)
    ).all()

    with _cache_lock:
        for row in rows:
            entry = {
                "averageRating": round(float(row.avg_r), 2) if row.avg_r is not None else None,
                "ratingCount": int(row.rc or 0),
                "commentCount": int(row.rvc or 0),
                "shellCount": int(row.sc or 0),
            }
            _engagement_cache[int(row.book_id)] = entry
            result[int(row.book_id)] = entry

    return result
