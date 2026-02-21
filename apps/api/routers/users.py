from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import UserCRUD
from bookdb.db.models import (
    Book,
    BookAuthor,
    BookList,
    BookRating,
    BookTag,
    ListBook,
    Shell,
    ShellBook,
    User,
)

from ..core.deps import get_db
from ..core.serialize import relative_time, serialize_book, serialize_list, serialize_user

router = APIRouter(prefix="/users", tags=["users"])


def _get_user_or_404(db: Session, user_id: int) -> User:
    user = UserCRUD.get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.get("/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = _get_user_or_404(db, user_id)
    return serialize_user(user)


@router.get("/{user_id}/ratings")
def get_user_ratings(
    user_id: int,
    limit: int = Query(50, ge=1, le=200),
    sort: str = Query("recent", pattern="^(recent|rating)$"),
    db: Session = Depends(get_db),
):
    _get_user_or_404(db, user_id)
    stmt = (
        select(BookRating)
        .where(BookRating.user_id == user_id)
        .options(
            selectinload(BookRating.book)
            .selectinload(Book.authors)
            .selectinload(BookAuthor.author),
            selectinload(BookRating.book)
            .selectinload(Book.tags)
            .selectinload(BookTag.tag),
        )
        .limit(limit)
    )
    if sort == "rating":
        stmt = stmt.order_by(BookRating.rating.desc())
    else:
        stmt = stmt.order_by(BookRating.updated_at.desc())

    ratings = db.scalars(stmt).all()
    return [
        {
            "book": serialize_book(r.book),
            "rating": r.rating,
            "ratedAt": r.updated_at.date().isoformat() if r.updated_at else None,
        }
        for r in ratings
    ]


@router.get("/{user_id}/lists")
def get_user_lists(user_id: int, db: Session = Depends(get_db)):
    _get_user_or_404(db, user_id)
    lists = db.scalars(
        select(BookList)
        .where(BookList.user_id == user_id)
        .options(
            selectinload(BookList.user),
            selectinload(BookList.books)
            .selectinload(ListBook.book)
            .selectinload(Book.authors)
            .selectinload(BookAuthor.author),
            selectinload(BookList.books)
            .selectinload(ListBook.book)
            .selectinload(Book.tags)
            .selectinload(BookTag.tag),
        )
    ).all()
    return [serialize_list(lst) for lst in lists]


@router.get("/{user_id}/activity")
def get_user_activity(
    user_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    user = _get_user_or_404(db, user_id)

    # Collect recent rating events.
    ratings = db.scalars(
        select(BookRating)
        .where(BookRating.user_id == user_id)
        .options(
            selectinload(BookRating.book)
            .selectinload(Book.authors)
            .selectinload(BookAuthor.author),
            selectinload(BookRating.book)
            .selectinload(Book.tags)
            .selectinload(BookTag.tag),
        )
        .order_by(BookRating.updated_at.desc())
        .limit(limit)
    ).all()

    items = []
    for r in ratings:
        items.append({
            "id": f"rating-{user_id}-{r.book_id}",
            "user": serialize_user(user),
            "type": "rating",
            "book": serialize_book(r.book),
            "rating": r.rating,
            "listName": None,
            "_dt": r.updated_at,
        })

    # Collect recent shell_add events.
    shell = db.scalar(select(Shell).where(Shell.user_id == user_id))
    if shell:
        shell_books = db.scalars(
            select(ShellBook)
            .where(ShellBook.shell_id == shell.id)
            .options(
                selectinload(ShellBook.book)
                .selectinload(Book.authors)
                .selectinload(BookAuthor.author),
                selectinload(ShellBook.book)
                .selectinload(Book.tags)
                .selectinload(BookTag.tag),
            )
            .order_by(ShellBook.added_at.desc())
            .limit(limit)
        ).all()
        for sb in shell_books:
            items.append({
                "id": f"shell-{user_id}-{sb.book_id}",
                "user": serialize_user(user),
                "type": "shell_add",
                "book": serialize_book(sb.book),
                "rating": None,
                "listName": None,
                "_dt": sb.added_at,
            })

    # Sort by datetime desc and limit.
    from datetime import timezone

    def sort_key(x):
        dt = x.get("_dt")
        if dt is None:
            return 0.0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    items.sort(key=sort_key, reverse=True)
    items = items[:limit]

    # Add relative timestamp and remove internal _dt field.
    for item in items:
        dt = item.pop("_dt", None)
        item["timestamp"] = relative_time(dt) if dt else "just now"

    return items
