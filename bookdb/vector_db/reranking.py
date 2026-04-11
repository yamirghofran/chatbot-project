from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[int]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion with optional per-source weights."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[int, float] = defaultdict(float)
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] += weight / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo if hi > lo else 1.0
    return {k: (v - lo) / span for k, v in scores.items()}


def hybrid_fusion(
    bpr_scored: list[tuple[int, float]],
    vector_ranked: list[int],
    bpr_weight: float = 0.6,
    rrf_k: int = 60,
) -> list[int]:
    """
    Fuse BPR predictions (with actual confidence scores) and vector recommendations
    (rank-only) into a single ranked list.

    Both sources are normalised to [0, 1] before combining, so bpr_weight is a true
    fractional contribution (vector gets 1 - bpr_weight).
    """
    combined: dict[int, float] = defaultdict(float)
    vector_weight = 1.0 - bpr_weight

    if bpr_scored:
        bpr_norm = _normalize({item_id: score for item_id, score in bpr_scored})
        for item_id, norm_score in bpr_norm.items():
            combined[item_id] += bpr_weight * norm_score

    if vector_ranked:
        rrf_raw = {
            item_id: 1.0 / (rrf_k + rank + 1)
            for rank, item_id in enumerate(vector_ranked)
        }
        rrf_norm = _normalize(rrf_raw)
        for item_id, norm_score in rrf_norm.items():
            combined[item_id] += vector_weight * norm_score

    return sorted(combined, key=combined.__getitem__, reverse=True)
