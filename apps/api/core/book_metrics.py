from __future__ import annotations

import threading

import duckdb
from cachetools import TTLCache

_metrics_cache: TTLCache = TTLCache(maxsize=20_000, ttl=1800)
_top_cache: TTLCache = TTLCache(maxsize=128, ttl=1800)
_cache_lock = threading.Lock()


def _query_rows(query: str, params: list) -> list[tuple]:
    with _cache_lock:
        return duckdb.execute(query, params).fetchall()


def get_metrics_for_goodreads_ids(
    parquet_path: str | None,
    goodreads_ids: list[int],
) -> dict[int, dict]:
    if parquet_path is None or not goodreads_ids:
        return {}

    unique_ids = [int(gid) for gid in dict.fromkeys(goodreads_ids)]
    result: dict[int, dict] = {}
    missing: list[int] = []

    with _cache_lock:
        for gid in unique_ids:
            key = ("book_metrics", parquet_path, gid)
            if key in _metrics_cache:
                result[gid] = _metrics_cache[key]
            else:
                missing.append(gid)

    if missing:
        placeholders = ", ".join(["?"] * len(missing))
        query = (
            "SELECT book_id, COALESCE(num_ratings, 0), COALESCE(num_reviews, 0), avg_rating "
            f"FROM parquet_scan(?) WHERE book_id IN ({placeholders})"
        )
        try:
            rows = _query_rows(query, [parquet_path, *missing])
        except Exception:
            rows = []
        fetched: dict[int, dict] = {
            int(book_id): {
                "num_ratings": int(num_ratings or 0),
                "num_reviews": int(num_reviews or 0),
                "avg_rating": float(avg_rating) if avg_rating is not None else None,
            }
            for book_id, num_ratings, num_reviews, avg_rating in rows
        }

        with _cache_lock:
            for gid in missing:
                key = ("book_metrics", parquet_path, gid)
                entry = fetched.get(gid, {"num_ratings": 0, "num_reviews": 0, "avg_rating": None})
                _metrics_cache[key] = entry
                result[gid] = entry

    return result


def _get_top_cached(cache_key: tuple, query: str, params: list) -> list[int]:
    with _cache_lock:
        cached = _top_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        rows = _query_rows(query, params)
    except Exception:
        return []
    ids = [int(row[0]) for row in rows if row and row[0] is not None]
    with _cache_lock:
        _top_cache[cache_key] = ids
    return ids


def get_top_popular_goodreads_ids(parquet_path: str | None, limit: int) -> list[int]:
    if parquet_path is None or limit <= 0:
        return []
    cache_key = ("top_popular", parquet_path, int(limit))
    return _get_top_cached(
        cache_key=cache_key,
        query=(
            "SELECT book_id "
            "FROM parquet_scan(?) "
            "WHERE book_id IS NOT NULL "
            "ORDER BY COALESCE(num_ratings, 0) DESC, "
            "COALESCE(num_interactions, 0) DESC, "
            "COALESCE(avg_rating, 0) DESC, "
            "book_id ASC "
            "LIMIT ?"
        ),
        params=[parquet_path, int(limit)],
    )


def get_top_staff_pick_goodreads_ids(
    parquet_path: str | None,
    limit: int,
    min_ratings: int = 5,
) -> list[int]:
    if parquet_path is None or limit <= 0:
        return []
    cache_key = ("top_staff", parquet_path, int(limit), int(min_ratings))
    return _get_top_cached(
        cache_key=cache_key,
        query=(
            "SELECT book_id "
            "FROM parquet_scan(?) "
            "WHERE book_id IS NOT NULL AND COALESCE(num_ratings, 0) >= ? "
            "ORDER BY COALESCE(avg_rating, 0) DESC, "
            "COALESCE(num_ratings, 0) DESC, "
            "COALESCE(num_reviews, 0) DESC, "
            "book_id ASC "
            "LIMIT ?"
        ),
        params=[parquet_path, int(min_ratings), int(limit)],
    )
