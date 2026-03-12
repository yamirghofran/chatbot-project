"""Reranking module for combining book and review signals."""

from collections import defaultdict
from typing import Any


def compute_review_features(review_hits: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Compute RAW review features per book (no normalization).

    Returns:
        Dict mapping book_id to features:
            - review_max: highest review score
            - review_top2_mean: mean of top 2 review scores
            - review_count: number of matching reviews
            - has_reviews: True
    """
    by_book: dict[int, list[float]] = defaultdict(list)
    for hit in review_hits:
        book_id = hit.get("book_id")
        score = hit.get("score", 0.0)
        if book_id is not None:
            by_book[book_id].append(score)

    features: dict[int, dict[str, Any]] = {}
    for book_id, scores in by_book.items():
        sorted_scores = sorted(scores, reverse=True)
        top2 = sorted_scores[:2]
        features[book_id] = {
            "review_max": sorted_scores[0],
            "review_top2_mean": sum(top2) / len(top2),
            "review_count": len(scores),
            "has_reviews": True,
        }
    return features


def normalize_feature(values: list[float]) -> list[float]:
    """Min-max normalize a list of values to [0, 1].
    """
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [1.0] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


def build_candidates(
    book_hits: list[dict[str, Any]],
    review_features: dict[int, dict[str, Any]],
    additional_book_scores: dict[int, float],
) -> list[dict[str, Any]]:
    """Build unified candidate list from book hits and review features.
    """
    candidates: list[dict[str, Any]] = []
    seen_book_ids: set[int] = set()

    # Add books from book search
    for hit in book_hits:
        book_id = hit.get("id")
        if book_id is None:
            continue
        book_id = int(book_id)
        if book_id in seen_book_ids:
            continue
        seen_book_ids.add(book_id)

        candidate = {
            "book_id": book_id,
            "book_score": hit.get("score", 0.0),
            "payload": hit.get("payload", {}),
        }

        # Add review features if available
        if book_id in review_features:
            candidate.update(review_features[book_id])
        else:
            candidate["has_reviews"] = False

        candidates.append(candidate)

    # Add books found only via reviews
    for book_id, book_score in additional_book_scores.items():
        if book_id in seen_book_ids:
            continue

        candidate = {
            "book_id": book_id,
            "book_score": book_score,
            "payload": {},
        }
        candidate.update(review_features[book_id])
        candidates.append(candidate)

    return candidates


def rerank_candidates(
    candidates: list[dict[str, Any]],
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    """Normalize features and compute final scores for reranking.
    Books without reviews only use book_score (no penalty).
    """
    if not candidates:
        return []

    # Separate candidates with and without reviews
    with_reviews = [c for c in candidates if c.get("has_reviews")]

    # Extract raw features for normalization (all candidates for book_score)
    all_book_scores = [c["book_score"] for c in candidates]
    book_scores_norm = normalize_feature(all_book_scores)

    # Normalize review features only for candidates with reviews
    if with_reviews:
        review_max_values = [c["review_max"] for c in with_reviews]
        review_top2_values = [c["review_top2_mean"] for c in with_reviews]
        review_max_norm = normalize_feature(review_max_values)
        review_top2_norm = normalize_feature(review_top2_values)
    else:
        review_max_norm = []
        review_top2_norm = []

    # Compute final scores
    book_norm_map = {c["book_id"]: book_scores_norm[i] for i, c in enumerate(candidates)}

    review_idx = 0
    for candidate in candidates:
        book_id = candidate["book_id"]
        book_norm = book_norm_map[book_id]

        if candidate.get("has_reviews"):
            final_score = (
                weights["book"] * book_norm +
                weights["review_max"] * review_max_norm[review_idx] +
                weights["review_top2_mean"] * review_top2_norm[review_idx]
            )
            review_idx += 1
        else:
            # No penalty for books without reviews
            final_score = book_norm

        candidate["final_score"] = final_score
        candidate["book_score_norm"] = book_norm

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
