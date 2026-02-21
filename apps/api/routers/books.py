from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import BookCRUD, ReviewCRUD
from bookdb.db.models import Author, Book, BookAuthor, BookRating, BookTag, ShellBook

from ..core.book_engagement import build_book_engagement_map
from ..core.book_queries import BOOK_LOAD_OPTIONS, load_books_by_ids, load_books_by_goodreads_ids, serialize_books_with_engagement
from ..core.deps import get_db, get_optional_user
from ..core.embeddings import most_similar
from ..core.serialize import serialize_book, serialize_review

router = APIRouter(prefix="/books", tags=["books"])


def _load_book(db: Session, book_id: int) -> Book:
    book = db.scalar(
        select(Book)
        .where(Book.id == book_id)
        .options(*BOOK_LOAD_OPTIONS)
    )
    if book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return book


@router.get("/search")
def search_books(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = q.strip().lower()
    if not query:
        return []

    title_l = func.lower(Book.title)
    rating_counts = (
        select(
            BookRating.book_id.label("book_id"),
            func.count(BookRating.book_id).label("rating_count"),
        )
        .group_by(BookRating.book_id)
        .subquery()
    )

    author_match = (
        select(1)
        .select_from(BookAuthor)
        .join(Author, Author.id == BookAuthor.author_id)
        .where(
            BookAuthor.book_id == Book.id,
            func.lower(Author.name).like(f"%{query}%"),
        )
        .exists()
    )

    # Lightweight ranking: intent first, popularity second.
    rank_score = case(
        (title_l == query, 1000),
        (title_l.like(f"{query}%"), 800),
        (title_l.like(f"% {query}%"), 650),
        (title_l.like(f"%{query}%"), 500),
        (author_match, 300),
        else_=0,
    )
    popularity = func.coalesce(rating_counts.c.rating_count, 0)

    rows = db.execute(
        select(
            Book.id,
            rank_score.label("score"),
            func.length(Book.title).label("title_len"),
        )
        .outerjoin(rating_counts, rating_counts.c.book_id == Book.id)
        .where(
            or_(
                title_l.like(f"%{query}%"),
                author_match,
            )
        )
        .order_by(rank_score.desc(), popularity.desc(), func.length(Book.title).asc(), Book.id.asc())
        .limit(limit)
    ).all()

    ranked_ids = [int(row.id) for row in rows]
    books = load_books_by_ids(db, ranked_ids)
    return serialize_books_with_engagement(db, books)


@router.get("/{book_id}")
def get_book(book_id: int, db: Session = Depends(get_db)):
    book = _load_book(db, book_id)
    engagement = build_book_engagement_map(db, [book_id]).get(book_id, {})
    data = serialize_book(book)
    data["stats"] = {
        "averageRating": engagement.get("averageRating"),
        "ratingCount": int(engagement.get("ratingCount", 0) or 0),
        "commentCount": int(engagement.get("commentCount", 0) or 0),
        "shellCount": int(engagement.get("shellCount", 0) or 0),
    }
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
                related = load_books_by_goodreads_ids(db, similar_goodreads_ids)
                return serialize_books_with_engagement(db, related)
        except Exception as e:
            print(f"Qdrant recommend failed for book {book_id}: {e}")

    # Fallback: popular books.
    fallback = db.scalars(
        select(Book)
        .where(Book.id != book_id)
        .options(*BOOK_LOAD_OPTIONS)
        .limit(limit)
    ).all()
    return serialize_books_with_engagement(db, fallback)
