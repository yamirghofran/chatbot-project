from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion"""
    scores: dict[int, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)
