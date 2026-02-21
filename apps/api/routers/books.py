from __future__ import annotations

import math
import threading

from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import case, func, literal, or_, select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import ReviewCRUD
from bookdb.db.models import Author, Book, BookAuthor, BookRating, Review, ReviewComment, ReviewLike, User

from ..core.book_engagement import build_book_engagement_map
from ..core.book_metrics import get_metrics_for_goodreads_ids
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
    request: Request = None,
    db: Session = Depends(get_db),
):
    query = q.strip().lower()
    if not query:
        return []

    title_l = func.lower(Book.title)

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
        .where(author_name_l.like(f"%{query}%"))
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

    candidate_limit = min(1000, max(limit * 12, 200))
    rows = db.execute(
        select(
            Book.id,
            Book.goodreads_id,
            rank_score.label("rank_score"),
            func.length(Book.title).label("title_len"),
        )
        .outerjoin(author_scores, author_scores.c.book_id == Book.id)
        .where(
            or_(
                title_l.like(f"%{query}%"),
                author_scores.c.author_score > 0,
            )
        )
        .order_by(rank_score.desc(), func.length(Book.title).asc(), Book.id.asc())
        .limit(candidate_limit)
    ).all()

    if not rows:
        return []

    candidate_ids = [int(row.id) for row in rows]
    metrics_parquet_path: str | None = None
    if request is not None:
        metrics_parquet_path = getattr(request.app.state, "book_metrics_parquet_path", None)
    popularity_by_goodreads_id: dict[int, int] = {}

    if metrics_parquet_path is not None:
        goodreads_ids = [int(row.goodreads_id) for row in rows if row.goodreads_id is not None]
        metrics_by_id = get_metrics_for_goodreads_ids(metrics_parquet_path, goodreads_ids)
        popularity_by_goodreads_id = {
            gid: int(metrics.get("num_ratings", 0) or 0)
            for gid, metrics in metrics_by_id.items()
        }
    else:
        rating_rows = db.execute(
            select(
                BookRating.book_id,
                func.count(BookRating.book_id).label("rating_count"),
            )
            .where(BookRating.book_id.in_(candidate_ids))
            .group_by(BookRating.book_id)
        ).all()
        rating_count_by_book_id = {int(row.book_id): int(row.rating_count or 0) for row in rating_rows}
        popularity_by_goodreads_id = {
            int(row.goodreads_id): rating_count_by_book_id.get(int(row.id), 0)
            for row in rows
            if row.goodreads_id is not None
        }

    scored = []
    for row in rows:
        popularity = popularity_by_goodreads_id.get(int(row.goodreads_id), 0) if row.goodreads_id is not None else 0
        combined_score = float(row.rank_score or 0) + math.log(max(popularity, 1)) * 50.0
        scored.append((combined_score, int(row.title_len or 0), int(row.id)))

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    ranked_ids = [book_id for _, _, book_id in scored[:limit]]
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
        select(
            ReviewLike.review_id.label("review_id"),
            func.count().label("likes_count"),
        )
        .join(Review, Review.id == ReviewLike.review_id)
        .where(Review.book_id == book_id)
        .group_by(ReviewLike.review_id)
        .subquery()
    )
    stmt = (
        select(
            Review,
            func.coalesce(likes_count_sq.c.likes_count, 0).label("likes_count"),
        )
        .where(Review.book_id == book_id)
        .outerjoin(likes_count_sq, likes_count_sq.c.review_id == Review.id)
        .options(
            selectinload(Review.user),
            selectinload(Review.comments).selectinload(ReviewComment.user),
        )
        .order_by(*order_clauses)
        .offset(offset)
        .limit(limit)
    )

    if uid:
        my_likes_sq = (
            select(ReviewLike.review_id.label("review_id"))
            .join(Review, Review.id == ReviewLike.review_id)
            .where(ReviewLike.user_id == uid, Review.book_id == book_id)
            .subquery()
        )
        stmt = stmt.add_columns(
            case((my_likes_sq.c.review_id.is_not(None), 1), else_=0).label("is_liked_by_me")
        ).outerjoin(my_likes_sq, my_likes_sq.c.review_id == Review.id)
    else:
        stmt = stmt.add_columns(literal(0).label("is_liked_by_me"))

    rows = db.execute(stmt).all()

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
