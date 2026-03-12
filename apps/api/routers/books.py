from __future__ import annotations

import math
import threading
from typing import Any

from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
import requests
from sqlalchemy import case, func, literal, or_, select
from sqlalchemy.orm import Session, selectinload

from bookdb.db.crud import ReviewCRUD
from bookdb.db.models import Author, Book, BookAuthor, BookRating, Review, ReviewComment, ReviewLike, User
from bookdb.models.chatbot_llm import create_groq_client_sync, generate_response_sync, rewrite_query_sync

from ..core.book_engagement import build_book_engagement_map
from ..core.book_metrics import get_metrics_for_goodreads_ids
from ..core.book_queries import BOOK_LOAD_OPTIONS, load_books_by_ids, load_books_by_goodreads_ids, serialize_books_with_engagement
from ..core.config import settings
from ..core.deps import get_current_user, get_db, get_optional_user
from ..core.embeddings import most_similar, most_similar_by_vector, most_similar_reviews_by_vector, get_book_scores_by_ids
from ..core.reranker import compute_review_features, build_candidates, rerank_candidates
from ..core.serialize import serialize_book, serialize_review
from ..schemas.review import CreateReviewRequest

router = APIRouter(prefix="/books", tags=["books"])

_qdrant_cache: TTLCache = TTLCache(maxsize=500, ttl=1800)
_qdrant_failure_cache: TTLCache = TTLCache(maxsize=500, ttl=60)
_qdrant_lock = threading.Lock()


def _empty_search_response() -> dict[str, Any]:
    return {
        "directHit": None,
        "keywordResults": [],
        "aiNarrative": None,
        "aiBooks": [],
    }


def _search_response_from_ranked_books(db: Session, books: list[Book]) -> dict[str, Any]:
    serialized = serialize_books_with_engagement(db, books)
    return {
        "directHit": serialized[0] if serialized else None,
        "keywordResults": serialized[1:] if len(serialized) > 1 else [],
        "aiNarrative": None,
        "aiBooks": [],
    }


def _book_author_names(book: Book) -> str:
    names = [ba.author.name for ba in book.authors if ba.author]
    return ", ".join(names) if names else "Unknown"


def _payload_to_book_context(book: Book, payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    document = payload.get("document")
    document = document.strip() if isinstance(document, str) else ""
    tags = [bt.tag.name for bt in book.tags if bt.tag]

    title = str(metadata.get("title") or book.title)
    author = str(metadata.get("author") or _book_author_names(book))
    shelves = str(metadata.get("shelves") or ", ".join(tags[:5]) or "unspecified")
    description = document or (book.description or "")
    description = description.strip() or "No description available."

    return (
        f"TITLE: {title}\n"
        f"AUTHOR: {author}\n"
        f"SHELVES: {shelves}\n"
        f"DESCRIPTION: {description}\n"
    )


def _embed_text_via_service(text: str) -> list[float]:
    service_url = settings.EMBEDDING_SERVICE_URL
    if not service_url:
        return []

    endpoint = service_url.rstrip("/") + "/embed"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.EMBEDDING_SERVICE_API_KEY:
        headers["Authorization"] = f"Bearer {settings.EMBEDDING_SERVICE_API_KEY}"

    body = {
        "texts": [text],
        "model": settings.EMBEDDING_SERVICE_MODEL,
        "normalize_embeddings": True,
    }

    try:
        response = requests.post(
            endpoint,
            json=body,
            headers=headers,
            timeout=settings.EMBEDDING_SERVICE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Embedding request failed: {e}")
        return []

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        return []
    first = embeddings[0]
    if not isinstance(first, list) or not first:
        return []

    vector: list[float] = []
    for value in first:
        try:
            vector.append(float(value))
        except (TypeError, ValueError):
            return []
    return vector


def _run_chatbot_search_pipeline(
    *,
    query: str,
    request: Request | None,
    db: Session,
) -> tuple[str | None, list[dict[str, Any]]]:
    qdrant = getattr(request.app.state, "qdrant", None) if request is not None else None
    if qdrant is None:
        return None, []

    try:
        groq_client = create_groq_client_sync()
        rewritten_description, rewritten_review = rewrite_query_sync(groq_client, query)
    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return None, []

    # Generate separate embeddings for books and reviews
    book_embedding = _embed_text_via_service(rewritten_description) if rewritten_description else []
    review_embedding = _embed_text_via_service(rewritten_review) if rewritten_review else []

    # Fallback: if one is empty, use the other
    if not book_embedding and not review_embedding:
        return None, []
    if not book_embedding:
        book_embedding = review_embedding
    if not review_embedding:
        review_embedding = book_embedding

    # Search books
    try:
        book_hits = most_similar_by_vector(
            qdrant,
            book_embedding,
            top_k=settings.CHATBOT_TOP_K,
        )
    except Exception as e:
        print(f"Qdrant book search failed: {e}")
        return None, []

    # Search reviews
    review_hits: list[dict[str, Any]] = []
    try:
        review_hits = most_similar_reviews_by_vector(
            qdrant,
            review_embedding,
            top_k=settings.CHATBOT_REVIEWS_TOP_K,
        )
    except Exception as e:
        print(f"Qdrant review search failed (continuing without reviews): {e}")

    # Compute review features grouped by book_id
    review_features = compute_review_features(review_hits)

    # Find books that came only from reviews (not in book search)
    book_ids_from_books = {hit["id"] for hit in book_hits}
    book_ids_from_reviews = set(review_features.keys())
    review_only_ids = book_ids_from_reviews - book_ids_from_books

    # Get book scores for review-only books
    additional_book_scores: dict[int, float] = {}
    if review_only_ids:
        try:
            additional_book_scores = get_book_scores_by_ids(
                qdrant,
                review_only_ids,
                book_embedding,
            )
        except Exception as e:
            print(f"Failed to get scores for review-only books: {e}")

    # Build candidates and rerank
    candidates = build_candidates(book_hits, review_features, additional_book_scores)
    if not candidates:
        return None, []

    weights = {
        "book": settings.RERANK_WEIGHT_BOOK,
        "review_max": settings.RERANK_WEIGHT_REVIEW_MAX,
        "review_top2_mean": settings.RERANK_WEIGHT_REVIEW_TOP2_MEAN,
    }
    ranked_candidates = rerank_candidates(candidates, weights)

    # Extract goodreads_ids in reranked order
    goodreads_ids = [c["book_id"] for c in ranked_candidates]
    if not goodreads_ids:
        return None, []

    # Load books from PostgreSQL
    qdrant_books = load_books_by_goodreads_ids(db, goodreads_ids)
    if not qdrant_books:
        return None, []

    books_by_goodreads_id = {int(book.goodreads_id): book for book in qdrant_books}

    # Build ranked books list and LLM context
    ranked_books: list[Book] = []
    llm_books: list[dict[str, Any]] = []
    for candidate in ranked_candidates:
        gid = candidate["book_id"]
        book = books_by_goodreads_id.get(gid)
        if book is None:
            continue
        ranked_books.append(book)
        llm_books.append({
            "book_id": int(book.id),
            "description": _payload_to_book_context(book, candidate.get("payload", {})),
        })

    if not ranked_books:
        return None, []

    # Get semantically relevant reviews for LLM 
    # Group by book to ensure coverage across recommended books (top 2 per book)
    reviews: list[dict[str, Any]] = []
    if settings.CHATBOT_MAX_REVIEWS > 0 and review_hits:
        ranked_book_ids = {book.id for book in ranked_books}
        reviews_per_book: dict[int, list[int]] = {}
        max_reviews_per_book = 2

        for hit in review_hits:
            goodreads_id = hit.get("book_id")
            book = books_by_goodreads_id.get(goodreads_id)
            if book and book.id in ranked_book_ids:
                if book.id not in reviews_per_book:
                    reviews_per_book[book.id] = []
                if len(reviews_per_book[book.id]) < max_reviews_per_book:
                    reviews_per_book[book.id].append(hit["review_id"])

        # Flatten: prioritize books in ranked order, take their reviews
        relevant_review_ids: list[int] = []
        for book in ranked_books:
            if book.id in reviews_per_book:
                for rid in reviews_per_book[book.id]:
                    if len(relevant_review_ids) >= settings.CHATBOT_MAX_REVIEWS:
                        break
                    relevant_review_ids.append(rid)
            if len(relevant_review_ids) >= settings.CHATBOT_MAX_REVIEWS:
                break

        if relevant_review_ids:
            review_rows = db.execute(
                select(
                    Review.id,
                    Review.review_text,
                    Book.title.label("book_title"),
                )
                .join(Book, Book.id == Review.book_id)
                .where(Review.id.in_(relevant_review_ids))
            ).all()
            
            # Preserve semantic order from relevant_review_ids
            rows_by_id = {int(row.id): row for row in review_rows}
            reviews = [
                {
                    "review_id": rid,
                    "book_title": str(rows_by_id[rid].book_title or ""),
                    "review": str(rows_by_id[rid].review_text or ""),
                }
                for rid in relevant_review_ids
                if rid in rows_by_id
            ]

    # Fallback to recent reviews if no semantic matches
    if not reviews and settings.CHATBOT_MAX_REVIEWS > 0:
        ranked_book_ids = [book.id for book in ranked_books]
        review_rows = db.execute(
            select(
                Review.id,
                Review.review_text,
                Book.title.label("book_title"),
            )
            .join(Book, Book.id == Review.book_id)
            .where(Review.book_id.in_(ranked_book_ids))
            .order_by(Review.created_at.desc())
            .limit(settings.CHATBOT_MAX_REVIEWS)
        ).all()
        reviews = [
            {
                "review_id": int(row.id),
                "book_title": str(row.book_title or ""),
                "review": str(row.review_text or ""),
            }
            for row in review_rows
        ]

    try:
        llm_response = generate_response_sync(groq_client, query, llm_books, reviews)
    except Exception as e:
        print(f"Chatbot response generation failed: {e}")
        llm_response = {
            "response": "",
            "referenced_book_ids": [],
        }

    ai_narrative = str(llm_response.get("response", "")).strip() or None
    referenced_ids: list[int] = []
    for raw_id in llm_response.get("referenced_book_ids", []):
        try:
            referenced_ids.append(int(raw_id))
        except (TypeError, ValueError):
            continue

    max_books = settings.CHATBOT_MAX_BOOKS
    ai_books: list[Book] = []
    ranked_ids = {book.id for book in ranked_books}

    if referenced_ids:
        ordered_referenced_ids: list[int] = []
        seen_ids: set[int] = set()
        for rid in referenced_ids:
            if rid in ranked_ids and rid not in seen_ids:
                ordered_referenced_ids.append(rid)
                seen_ids.add(rid)

        ai_books = load_books_by_ids(db, ordered_referenced_ids[:max_books])
    if not ai_books:
        ai_books = ranked_books[:max_books]

    return ai_narrative, serialize_books_with_engagement(db, ai_books)


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
    query_raw = q.strip()
    query = query_raw.lower()
    if not query_raw:
        return _empty_search_response()

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
        ai_narrative, ai_books = _run_chatbot_search_pipeline(
            query=query_raw,
            request=request,
            db=db,
        )
        return {
            "directHit": None,
            "keywordResults": [],
            "aiNarrative": ai_narrative,
            "aiBooks": ai_books,
        }

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
    return _search_response_from_ranked_books(db, books)


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
            recent_failure = _qdrant_failure_cache.get(goodreads_id, False)

        if cached_ids is not None:
            related = load_books_by_goodreads_ids(db, cached_ids)
            return serialize_books_with_engagement(db, related)

        if recent_failure:
            qdrant = None

        try:
            if qdrant is not None:
                similar_goodreads_ids = most_similar(qdrant, goodreads_id, top_k=limit)
                if similar_goodreads_ids:
                    with _qdrant_lock:
                        _qdrant_cache[goodreads_id] = similar_goodreads_ids
                    related = load_books_by_goodreads_ids(db, similar_goodreads_ids)
                    return serialize_books_with_engagement(db, related)
        except Exception as e:
            with _qdrant_lock:
                _qdrant_failure_cache[goodreads_id] = True
            print(f"Qdrant recommend failed for book {book_id}: {e}")

    # Fallback: popular books.
    fallback = db.scalars(
        select(Book)
        .outerjoin(BookRating, Book.id == BookRating.book_id)
        .where(Book.id != book_id)
        .group_by(Book.id)
        .order_by(func.count(BookRating.user_id).desc(), Book.id.asc())
        .options(*BOOK_LOAD_OPTIONS)
        .limit(limit)
    ).all()
    return serialize_books_with_engagement(db, fallback)
