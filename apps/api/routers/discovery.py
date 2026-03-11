from __future__ import annotations

import duckdb
import threading
from collections import defaultdict
from itertools import zip_longest
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bookdb.db.models import (
    Book,
    BookList,
    BookRating,
    ListBook,
    Shell,
    ShellBook,
)

from bookdb.vector_db.clustering import cluster_seeds_by_embedding
from bookdb.vector_db.reranking import reciprocal_rank_fusion

from ..core.book_queries import load_books_by_ids, load_books_by_goodreads_ids, serialize_books_with_engagement
from ..core.book_metrics import get_top_popular_goodreads_ids, get_top_staff_pick_goodreads_ids
from ..core.embeddings import most_similar, most_similar_by_vector, get_vectors_by_ids
from ..core.deps import get_db, get_optional_user

router = APIRouter(prefix="/discovery", tags=["discovery"])

MIN_SEEDS_FOR_CLUSTERING = 6

_bpr_lock = threading.Lock()


def _bpr_recommendations(parquet_path: str, goodreads_user_id: int, limit: int) -> list[int]:
    """Query BPR parquet for top-N recommended goodreads book IDs for a user."""
    with _bpr_lock:
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


def _cold_start(db: Session, limit: int, metrics_parquet_path: str | None = None) -> list[Book]:
    """Return most-rated books as a fallback."""
    if metrics_parquet_path is not None:
        # Request more than needed so we can tolerate IDs that may not exist in Postgres.
        candidate_ids = get_top_popular_goodreads_ids(metrics_parquet_path, limit=max(limit * 5, 100))
        if candidate_ids:
            books = load_books_by_goodreads_ids(db, candidate_ids)
            if books:
                return books[:limit]

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


def _append_unique_books(target: list[Book], books: list[Book], *, limit: int) -> None:
    if len(target) >= limit:
        return
    seen_ids = {book.id for book in target}
    for book in books:
        if len(target) >= limit:
            break
        if book.id in seen_ids:
            continue
        target.append(book)
        seen_ids.add(book.id)


def _collect_interaction_seed_scores(db: Session, user_id: int) -> dict[int, float]:
    """Build weighted seed goodreads IDs from user interactions."""
    seed_scores: dict[int, float] = defaultdict(float)

    rating_rows = db.execute(
        select(Book.goodreads_id, BookRating.rating)
        .select_from(BookRating)
        .join(Book, Book.id == BookRating.book_id)
        .where(
            BookRating.user_id == user_id,
            BookRating.rating >= 4,
            Book.goodreads_id.is_not(None),
        )
        .order_by(BookRating.rating.desc(), BookRating.updated_at.desc())
        .limit(60)
    ).all()
    for row in rating_rows:
        goodreads_id = row.goodreads_id
        rating = int(row.rating)
        if goodreads_id is None:
            continue
        # Strong positive ratings should dominate seed selection.
        seed_scores[int(goodreads_id)] += 2.5 if rating >= 5 else 2.0

    shell_rows = db.execute(
        select(Book.goodreads_id)
        .select_from(ShellBook)
        .join(Book, Book.id == ShellBook.book_id)
        .join(Shell, Shell.id == ShellBook.shell_id)
        .where(
            Shell.user_id == user_id,
            Book.goodreads_id.is_not(None),
        )
        .order_by(ShellBook.added_at.desc())
        .limit(60)
    ).all()
    for row in shell_rows:
        if row.goodreads_id is not None:
            seed_scores[int(row.goodreads_id)] += 1.6

    list_rows = db.execute(
        select(Book.goodreads_id)
        .select_from(ListBook)
        .join(Book, Book.id == ListBook.book_id)
        .join(BookList, BookList.id == ListBook.list_id)
        .where(
            BookList.user_id == user_id,
            Book.goodreads_id.is_not(None),
        )
        .order_by(ListBook.added_at.desc())
        .limit(80)
    ).all()
    for row in list_rows:
        if row.goodreads_id is not None:
            seed_scores[int(row.goodreads_id)] += 1.2

    return dict(seed_scores)


def _cluster_vector_recommendations(db: Session, user_id: int, qdrant_client, limit: int, exclude_ids: set[int] | None = None) -> list[int]:
    """Cluster the user's full interaction history and recommend per cluster"""
    if limit <= 0:
        return []

    seed_scores = _collect_interaction_seed_scores(db, user_id)
    if len(seed_scores) < MIN_SEEDS_FOR_CLUSTERING:
        return []

    excluded = set(exclude_ids or set())
    excluded.update(seed_scores.keys())

    try:
        vector_map = get_vectors_by_ids(qdrant_client, list(seed_scores.keys()))
    except Exception:
        return []

    valid_seeds = {gid: vec for gid, vec in vector_map.items() if gid in seed_scores}
    if len(valid_seeds) < MIN_SEEDS_FOR_CLUSTERING:
        return []

    n_clusters = max(2, min(len(valid_seeds) // 3, 5))
    try:
        clusters = cluster_seeds_by_embedding(valid_seeds, seed_scores, n_clusters)
    except Exception:
        return []

    if not clusters:
        return []

    # Proportional slot allocation, minimum 1 per cluster
    total_weight = sum(sum(w for _, w in members) for _, members in clusters)
    per_cluster_limits = []
    for _, members in clusters:
        cw = sum(w for _, w in members)
        raw = (cw / total_weight) * limit if total_weight > 0 else limit / len(clusters)
        per_cluster_limits.append(max(1, round(raw)))
    # Fix rounding drift on the heaviest cluster
    diff = limit - sum(per_cluster_limits)
    if diff != 0:
        heaviest = max(range(len(clusters)), key=lambda k: sum(w for _, w in clusters[k][1]))
        per_cluster_limits[heaviest] += diff

    seen: set[int] = set()
    per_cluster_hits: list[list[int]] = []

    for cluster_idx, (centroid, _) in enumerate(clusters):
        cluster_excluded = excluded | seen
        try:
            hits = most_similar_by_vector(
                qdrant_client,
                query_vector=centroid.tolist(),
                top_k=per_cluster_limits[cluster_idx],
                exclude_ids=cluster_excluded,
            )
        except Exception:
            per_cluster_hits.append([])
            continue

        cluster_hits = [(int(hit["id"]), hit["score"]) for hit in hits]
        per_cluster_hits.append(cluster_hits)
        seen.update(gid for gid, _ in cluster_hits)

    results: list[int] = []
    for round_hits in zip_longest(*per_cluster_hits):
        round_sorted = sorted(
            (item for item in round_hits if item is not None),
            key=lambda x: -x[1],
        )
        for gid, _ in round_sorted:
            results.append(gid)

    return results[:limit]


def _interaction_vector_recommendations(
    db: Session,
    user_id: int,
    qdrant_client,
    limit: int,
    exclude_ids: set[int] | None = None,
) -> list[int]:
    if limit <= 0:
        return []

    seed_scores = _collect_interaction_seed_scores(db, user_id)
    if not seed_scores:
        return []

    excluded = set(exclude_ids or set())
    excluded.update(seed_scores.keys())
    per_seed_top_k = max(limit * 3, 30)
    max_seed_books = 14

    candidate_scores: dict[int, float] = defaultdict(float)
    ranked_seeds = sorted(seed_scores.items(), key=lambda item: (-item[1], item[0]))[:max_seed_books]
    for seed_goodreads_id, seed_weight in ranked_seeds:
        try:
            similar_goodreads_ids = most_similar(
                qdrant_client,
                seed_goodreads_id,
                top_k=per_seed_top_k,
                exclude_ids=excluded,
            )
        except Exception:
            continue

        for rank, candidate_id in enumerate(similar_goodreads_ids, start=1):
            if candidate_id in excluded:
                continue
            candidate_scores[candidate_id] += seed_weight / rank

    if not candidate_scores:
        return []

    ranked_candidates = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    return [goodreads_id for goodreads_id, _ in ranked_candidates[:limit]]


@router.get("/recommendations")
def get_recommendations(
    limit: int = Query(20, ge=1, le=100),
    request: Request = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    bpr_path: str | None = None
    metrics_parquet_path: str | None = None
    qdrant = None
    if request is not None:
        bpr_path = getattr(request.app.state, "bpr_parquet_path", None)
        metrics_parquet_path = getattr(request.app.state, "book_metrics_parquet_path", None)
        qdrant = getattr(request.app.state, "qdrant", None)

    bpr_goodreads_ids: list[int] = []
    if current_user is not None and bpr_path is not None and current_user.goodreads_id is not None:
        bpr_goodreads_ids = _bpr_recommendations(
            bpr_path,
            current_user.goodreads_id,
            limit=max(limit * 5, 100),
        )

    interaction_goodreads_ids: list[int] = []
    if current_user is not None and qdrant is not None:
        interaction_goodreads_ids = _cluster_vector_recommendations(
            db,
            current_user.id,
            qdrant_client=qdrant,
            limit=max(limit * 4, 80),
        )
        if not interaction_goodreads_ids:
            interaction_goodreads_ids = _interaction_vector_recommendations(
                db,
                current_user.id,
                qdrant_client=qdrant,
                limit=max(limit * 4, 80),
            )

    recommendations: list[Book] = []
    if bpr_goodreads_ids or interaction_goodreads_ids:
        ranked_lists = [l for l in [bpr_goodreads_ids, interaction_goodreads_ids] if l]
        merged_ids = reciprocal_rank_fusion(ranked_lists)
        merged_books = load_books_by_goodreads_ids(db, merged_ids[: limit * 2])
        _append_unique_books(recommendations, merged_books, limit=limit)

    if len(recommendations) < limit:
        cold_start_books = _cold_start(db, limit, metrics_parquet_path)
        _append_unique_books(recommendations, cold_start_books, limit=limit)
    if recommendations:
        return serialize_books_with_engagement(db, recommendations)

    # Cold start fallback.
    return serialize_books_with_engagement(db, _cold_start(db, limit, metrics_parquet_path))


@router.get("/staff-picks")
def get_staff_picks(
    limit: int = Query(6, ge=1, le=50),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """Return a curated set of well-rated books as staff picks."""
    metrics_parquet_path: str | None = None
    if request is not None:
        metrics_parquet_path = getattr(request.app.state, "book_metrics_parquet_path", None)

    if metrics_parquet_path is not None:
        candidate_ids = get_top_staff_pick_goodreads_ids(
            metrics_parquet_path,
            limit=max(limit * 5, 100),
            min_ratings=5,
        )
        if candidate_ids:
            books = load_books_by_goodreads_ids(db, candidate_ids)
            if books:
                return serialize_books_with_engagement(db, books[:limit])

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
