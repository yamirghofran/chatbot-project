from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bookdb.db.models import BookRating, Review, ShellBook


def build_book_engagement_map(db: Session, book_ids: list[int]) -> dict[int, dict]:
    if not book_ids:
        return {}

    unique_ids = list(dict.fromkeys(book_ids))
    engagement: dict[int, dict] = {
        book_id: {
            "averageRating": None,
            "ratingCount": 0,
            "commentCount": 0,
            "shellCount": 0,
        }
        for book_id in unique_ids
    }

    rating_rows = db.execute(
        select(
            BookRating.book_id,
            func.avg(BookRating.rating).label("avg_rating"),
            func.count(BookRating.rating).label("rating_count"),
        )
        .where(BookRating.book_id.in_(unique_ids))
        .group_by(BookRating.book_id)
    ).all()
    for row in rating_rows:
        entry = engagement.get(int(row.book_id))
        if entry is None:
            continue
        entry["averageRating"] = round(float(row.avg_rating), 2) if row.avg_rating is not None else None
        entry["ratingCount"] = int(row.rating_count or 0)

    review_rows = db.execute(
        select(
            Review.book_id,
            func.count(Review.id).label("review_count"),
        )
        .where(Review.book_id.in_(unique_ids))
        .group_by(Review.book_id)
    ).all()
    for row in review_rows:
        entry = engagement.get(int(row.book_id))
        if entry is None:
            continue
        entry["commentCount"] = int(row.review_count or 0)

    shell_rows = db.execute(
        select(
            ShellBook.book_id,
            func.count().label("shell_count"),
        )
        .where(ShellBook.book_id.in_(unique_ids))
        .group_by(ShellBook.book_id)
    ).all()
    for row in shell_rows:
        entry = engagement.get(int(row.book_id))
        if entry is None:
            continue
        entry["shellCount"] = int(row.shell_count or 0)

    return engagement
