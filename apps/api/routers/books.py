from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import BookCRUD, ReviewCRUD
from bookdb.db.models import Book, BookAuthor, BookRating, BookTag, ShellBook

from ..core.deps import get_db, get_optional_user
from ..core.embeddings import most_similar
from ..core.serialize import serialize_book, serialize_review

router = APIRouter(prefix="/books", tags=["books"])


def _load_book(db: Session, book_id: int) -> Book:
    book = db.scalar(
        select(Book)
        .where(Book.id == book_id)
        .options(
            selectinload(Book.authors).selectinload(BookAuthor.author),
            selectinload(Book.tags).selectinload(BookTag.tag),
        )
    )
    if book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return book


def _book_stats(db: Session, book_id: int) -> dict:
    rating_row = db.execute(
        select(func.avg(BookRating.rating), func.count(BookRating.rating))
        .where(BookRating.book_id == book_id)
    ).one()
    avg_rating = float(rating_row[0]) if rating_row[0] is not None else None
    rating_count = int(rating_row[1])

    shell_count = db.scalar(
        select(func.count()).where(ShellBook.book_id == book_id)
    ) or 0

    return {
        "averageRating": round(avg_rating, 2) if avg_rating is not None else None,
        "ratingCount": rating_count,
        "shellCount": shell_count,
    }


@router.get("/search")
def search_books(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    books = db.scalars(
        select(Book)
        .where(Book.title.ilike(f"%{q}%"))
        .options(
            selectinload(Book.authors).selectinload(BookAuthor.author),
            selectinload(Book.tags).selectinload(BookTag.tag),
        )
        .limit(limit)
    ).all()
    return [serialize_book(b) for b in books]


@router.get("/{book_id}")
def get_book(book_id: int, db: Session = Depends(get_db)):
    book = _load_book(db, book_id)
    data = serialize_book(book)
    data["stats"] = _book_stats(db, book_id)
    return data


@router.get("/{book_id}/reviews")
def get_book_reviews(
    book_id: int,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    _load_book(db, book_id)
    from bookdb.db.models import Review, ReviewComment
    reviews = db.scalars(
        select(Review)
        .where(Review.book_id == book_id)
        .options(
            selectinload(Review.user),
            selectinload(Review.likes),
            selectinload(Review.comments).selectinload(ReviewComment.user),
        )
        .limit(limit)
    ).all()
    return [serialize_review(r) for r in reviews]


@router.get("/{book_id}/related")
def get_related_books(
    book_id: int,
    limit: int = Query(6, ge=1, le=20),
    request: Request = None,
    db: Session = Depends(get_db),
):
    book = _load_book(db, book_id)
    qdrant = getattr(request.app.state, "qdrant", None)

    if qdrant is not None and book.goodreads_id is not None:
        try:
            similar_goodreads_ids = most_similar(qdrant, book.goodreads_id, top_k=limit)
            if similar_goodreads_ids:
                related = db.scalars(
                    select(Book)
                    .where(Book.goodreads_id.in_(similar_goodreads_ids))
                    .options(
                        selectinload(Book.authors).selectinload(BookAuthor.author),
                        selectinload(Book.tags).selectinload(BookTag.tag),
                    )
                ).all()
                return [serialize_book(b) for b in related]
        except Exception as e:
            print(f"Qdrant recommend failed for book {book_id}: {e}")

    # Fallback: popular books.
    fallback = db.scalars(
        select(Book)
        .where(Book.id != book_id)
        .options(
            selectinload(Book.authors).selectinload(BookAuthor.author),
            selectinload(Book.tags).selectinload(BookTag.tag),
        )
        .limit(limit)
    ).all()
    return [serialize_book(b) for b in fallback]
