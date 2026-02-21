from __future__ import annotations

import threading

from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import case, func, literal, or_, select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import BookCRUD, ReviewCRUD
from bookdb.db.models import Author, Book, BookAuthor, BookRating, BookTag, Review, ReviewComment, ReviewLike, ShellBook, User

from ..core.book_engagement import build_book_engagement_map
from ..core.book_queries import BOOK_LOAD_OPTIONS, load_books_by_ids, load_books_by_goodreads_ids, serialize_books_with_engagement
from ..core.deps import get_current_user, get_db, get_optional_user
from ..core.embeddings import most_similar
from ..core.serialize import serialize_book, serialize_review
from ..schemas.review import CreateReviewRequest

router = APIRouter(prefix="/books", tags=["books"])

_qdrant_cache: TTLCache = TTLCache(maxsize=500, ttl=1800)
_qdrant_lock = threading.Lock()


def _load_book(db: Session, book_id: int) -> Book:
    book = db.scalar(
        select(Book)
        .where(Book.id == book_id)
        .options(*BOOK_LOAD_OPTIONS)
    )
    if book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return book


def _check_book_exists(db: Session, book_id: int) -> None:
    if db.scalar(select(Book.id).where(Book.id == book_id).limit(1)) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")


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

    author_name_l = func.lower(Author.name)
    author_scores = (
        select(
            BookAuthor.book_id.label("book_id"),
            func.max(
                case(
                    (author_name_l == query, 1200),
                    (author_name_l.like(f"{query}%"), 950),
                    (author_name_l.like(f"% {query}%"), 800),
                    (author_name_l.like(f"%{query}%"), 700),
                    else_=0,
                )
            ).label("author_score"),
        )
        .select_from(BookAuthor)
        .join(Author, Author.id == BookAuthor.author_id)
        .group_by(BookAuthor.book_id)
        .subquery()
    )

    # Lightweight ranking: intent first, popularity second.
    title_score = case(
        (title_l == query, 1000),
        (title_l.like(f"{query}%"), 800),
        (title_l.like(f"% {query}%"), 650),
        (title_l.like(f"%{query}%"), 500),
        else_=0,
    )
    author_score = func.coalesce(author_scores.c.author_score, 0)
    rank_score = func.greatest(title_score, author_score)
    popularity = func.coalesce(rating_counts.c.rating_count, 0)
    combined_score = rank_score + func.log(func.greatest(popularity, 1)) * 50

    # TODO: add pg_trgm GIN indexes on lower(books.title) and lower(authors.name)
    #       so LIKE '%...%' stops full-scanning.
    rows = db.execute(
        select(
            Book.id,
            combined_score.label("score"),
            func.length(Book.title).label("title_len"),
        )
        .outerjoin(rating_counts, rating_counts.c.book_id == Book.id)
        .outerjoin(author_scores, author_scores.c.book_id == Book.id)
        .where(
            or_(
                title_l.like(f"%{query}%"),
                author_scores.c.author_score > 0,
            )
        )
        .order_by(combined_score.desc(), func.length(Book.title).asc(), Book.id.asc())
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
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    _check_book_exists(db, book_id)
    total: int = db.scalar(
        select(func.count()).select_from(Review).where(Review.book_id == book_id)
    ) or 0
    uid = current_user.id if current_user else None

    # Own review always surfaces at the top of page 0 so the user always sees it.
    order_clauses = []
    if uid:
        order_clauses.append(case((Review.user_id == uid, 1), else_=0).desc())
    order_clauses.append(Review.created_at.desc())

    likes_count_sq = (
        select(func.count())
        .where(ReviewLike.review_id == Review.id)
        .correlate(Review)
        .scalar_subquery()
        .label("likes_count")
    )
    is_liked_sq = (
        select(func.count())
        .where(ReviewLike.review_id == Review.id, ReviewLike.user_id == uid)
        .correlate(Review)
        .scalar_subquery()
        .label("is_liked_by_me")
    ) if uid else literal(0).label("is_liked_by_me")

    rows = db.execute(
        select(Review, likes_count_sq, is_liked_sq)
        .where(Review.book_id == book_id)
        .options(
            selectinload(Review.user),
            selectinload(Review.comments).selectinload(ReviewComment.user),
        )
        .order_by(*order_clauses)
        .offset(offset)
        .limit(limit)
    ).all()

    items = [
        serialize_review(
            row.Review,
            uid,
            likes_count=int(row.likes_count or 0),
            is_liked=bool(row.is_liked_by_me),
        )
        for row in rows
    ]
    return {"items": items, "total": total}


@router.post("/{book_id}/reviews")
def post_book_review(
    book_id: int,
    body: CreateReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_book_exists(db, book_id)
    existing = ReviewCRUD.get_by_user_and_book(db, current_user.id, book_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="You have already reviewed this book",
        )
    review = ReviewCRUD.create(db, current_user.id, book_id, body.text)
    db.commit()
    review = db.scalar(
        select(Review)
        .where(Review.id == review.id)
        .options(
            selectinload(Review.user),
            selectinload(Review.comments).selectinload(ReviewComment.user),
        )
    )
    return serialize_review(review, current_user.id)


# TODO: improve related books retrieval
#   Happy path (Qdrant available):
#   1. Fetch Book.goodreads_id from Postgres for the requested book_id
#   2. Check the in-process TTL cache (30 min) — if the goodreads_id has been looked up recently,
#      return cached goodreads_id list immediately
#   3. Otherwise call Qdrant's recommend API — passing the book's goodreads_id as a positive example,
#      Qdrant finds the top_k nearest vectors in the "books" collection using its stored embeddings
#   4. The returned IDs are goodreads_ids — fetch full book rows from Postgres, then attach engagement stats
#   Issues: fallback is insertion-order (useless); books missing from Qdrant silently hit the fallback;
#   Qdrant recommend throws if the source point doesn't exist in the collection
@router.get("/{book_id}/related")
def get_related_books(
    book_id: int,
    limit: int = Query(6, ge=1, le=20),
    request: Request = None,
    db: Session = Depends(get_db),
):
    row = db.execute(select(Book.id, Book.goodreads_id).where(Book.id == book_id)).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    goodreads_id = row.goodreads_id

    qdrant = getattr(request.app.state, "qdrant", None)

    if qdrant is not None and goodreads_id is not None:
        with _qdrant_lock:
            cached_ids = _qdrant_cache.get(goodreads_id)

        if cached_ids is not None:
            related = load_books_by_goodreads_ids(db, cached_ids)
            return serialize_books_with_engagement(db, related)

        try:
            similar_goodreads_ids = most_similar(qdrant, goodreads_id, top_k=limit)
            if similar_goodreads_ids:
                with _qdrant_lock:
                    _qdrant_cache[goodreads_id] = similar_goodreads_ids
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
