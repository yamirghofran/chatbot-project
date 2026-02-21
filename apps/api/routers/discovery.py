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
    Shell,
    ShellBook,
    User,
)

from ..core.deps import get_db, get_optional_user
from ..core.serialize import relative_time, serialize_book, serialize_user

router = APIRouter(prefix="/discovery", tags=["discovery"])


def _load_books_by_ids(db: Session, book_ids: list[int]) -> list[Book]:
    if not book_ids:
        return []
    books = db.scalars(
        select(Book)
        .where(Book.id.in_(book_ids))
        .options(
            selectinload(Book.authors).selectinload(BookAuthor.author),
            selectinload(Book.tags).selectinload(BookTag.tag),
        )
    ).all()
    id_to_book = {b.id: b for b in books}
    return [id_to_book[bid] for bid in book_ids if bid in id_to_book]


def _load_books_by_goodreads_ids(db: Session, goodreads_ids: list[int]) -> list[Book]:
    if not goodreads_ids:
        return []
    books = db.scalars(
        select(Book)
        .where(Book.goodreads_id.in_(goodreads_ids))
        .options(
            selectinload(Book.authors).selectinload(BookAuthor.author),
            selectinload(Book.tags).selectinload(BookTag.tag),
        )
    ).all()
    id_to_book = {b.goodreads_id: b for b in books}
    return [id_to_book[gid] for gid in goodreads_ids if gid in id_to_book]


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
        return db.scalars(
            select(Book)
            .options(
                selectinload(Book.authors).selectinload(BookAuthor.author),
                selectinload(Book.tags).selectinload(BookTag.tag),
            )
            .limit(limit)
        ).all()
    return _load_books_by_ids(db, book_ids)


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
            books = _load_books_by_goodreads_ids(db, goodreads_ids)
            if books:
                return [serialize_book(b) for b in books]

    # Cold start fallback.
    return [serialize_book(b) for b in _cold_start(db, limit)]


@router.get("/staff-picks")
def get_staff_picks(
    limit: int = Query(6, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Return a curated set of well-rated books as staff picks."""
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
            .options(
                selectinload(Book.authors).selectinload(BookAuthor.author),
                selectinload(Book.tags).selectinload(BookTag.tag),
            )
            .limit(limit)
        ).all()
        return [serialize_book(b) for b in books]
    return [serialize_book(b) for b in _load_books_by_ids(db, book_ids)]


@router.get("/activity")
def get_activity_feed(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Return a global activity feed across all users."""
    items = []

    # Recent ratings.
    recent_ratings = db.scalars(
        select(BookRating)
        .options(
            selectinload(BookRating.book)
            .selectinload(Book.authors)
            .selectinload(BookAuthor.author),
            selectinload(BookRating.book)
            .selectinload(Book.tags)
            .selectinload(BookTag.tag),
            selectinload(BookRating.user),
        )
        .order_by(BookRating.updated_at.desc())
        .limit(limit)
    ).all()

    for r in recent_ratings:
        items.append({
            "id": f"rating-{r.user_id}-{r.book_id}",
            "user": serialize_user(r.user),
            "type": "rating",
            "book": serialize_book(r.book),
            "rating": r.rating,
            "listName": None,
            "timestamp": relative_time(r.updated_at),
            "_dt": r.updated_at,
        })

    # Recent shell additions.
    recent_shell = db.scalars(
        select(ShellBook)
        .join(Shell, Shell.id == ShellBook.shell_id)
        .options(
            selectinload(ShellBook.book)
            .selectinload(Book.authors)
            .selectinload(BookAuthor.author),
            selectinload(ShellBook.book)
            .selectinload(Book.tags)
            .selectinload(BookTag.tag),
            selectinload(ShellBook.shell)
            .selectinload(Shell.user),
        )
        .order_by(ShellBook.added_at.desc())
        .limit(limit)
    ).all()

    for sb in recent_shell:
        items.append({
            "id": f"shell-{sb.shell_id}-{sb.book_id}",
            "user": serialize_user(sb.shell.user),
            "type": "shell_add",
            "book": serialize_book(sb.book),
            "rating": None,
            "listName": None,
            "timestamp": relative_time(sb.added_at),
            "_dt": sb.added_at,
        })

    # Sort by datetime desc.
    from datetime import timezone
    def sort_key(x):
        dt = x.get("_dt")
        if dt is None:
            return 0
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    items.sort(key=sort_key, reverse=True)
    # Remove internal _dt field before returning.
    for item in items:
        item.pop("_dt", None)
    return items[:limit]
