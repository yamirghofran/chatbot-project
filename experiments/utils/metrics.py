from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable

# Per-query metric functions
def recall_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Fraction of relevant items found in top-k retrieved results."""
    if not ground_truth:
        return 0.0
    hits = len(set(retrieved[:k]) & set(ground_truth))
    return hits / len(ground_truth)


def precision_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Fraction of top-k retrieved results that are relevant."""
    if k == 0:
        return 0.0
    hits = len(set(retrieved[:k]) & set(ground_truth))
    return hits / k


def ndcg_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    if not ground_truth:
        return 0.0
    gt_set = set(ground_truth)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved[:k])
        if doc_id in gt_set
    )
    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved: list[int], ground_truth: list[int]) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant result."""
    gt_set = set(ground_truth)
    for i, doc_id in enumerate(retrieved):
        if doc_id in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """1 if at least one relevant item appears in top-k, else 0."""
    if not ground_truth:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(ground_truth) else 0.0


@dataclass
class QueryResult:
    """Retrieval outcome for a single query."""

    query_id: str
    query_text: str
    query_type: str
    ground_truth: list[int]
    retrieved_ids: list[int]
    retrieved_scores: list[float]
    latency_ms: float
    condition: str = "baseline"  #"baseline", "rewritten", "reranked"
    rewritten_text: str | None = None


@dataclass
class AggregatedMetrics:
    """Mean metrics across all queries for one experimental condition."""

    condition: str
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    precision_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    mrr_score: float = 0.0
    hit_rate_at_10: float = 0.0
    mean_latency_ms: float = 0.0
    query_count: int = 0
    per_query: list[dict] = field(default_factory=list)


def aggregate(results: list[QueryResult], condition: str) -> AggregatedMetrics:
    """Compute mean metrics from a list of per-query results."""
    if not results:
        return AggregatedMetrics(condition=condition)

    per_query = []
    r5, r10, r20, p10, n10, mrrs, hr10, lats = [], [], [], [], [], [], [], []

    for qr in results:
        r = qr.retrieved_ids
        g = qr.ground_truth
        r5.append(recall_at_k(r, g, 5))
        r10.append(recall_at_k(r, g, 10))
        r20.append(recall_at_k(r, g, 20))
        p10.append(precision_at_k(r, g, 10))
        n10.append(ndcg_at_k(r, g, 10))
        mrrs.append(mrr(r, g))
        hr10.append(hit_rate_at_k(r, g, 10))
        lats.append(qr.latency_ms)
        per_query.append({
            "query_id": qr.query_id,
            "query_type": qr.query_type,
            "recall_at_10": r10[-1],
            "ndcg_at_10": n10[-1],
            "mrr": mrrs[-1],
            "latency_ms": qr.latency_ms,
        })

    return AggregatedMetrics(
        condition=condition,
        recall_at_5=statistics.mean(r5),
        recall_at_10=statistics.mean(r10),
        recall_at_20=statistics.mean(r20),
        precision_at_10=statistics.mean(p10),
        ndcg_at_10=statistics.mean(n10),
        mrr_score=statistics.mean(mrrs),
        hit_rate_at_10=statistics.mean(hr10),
        mean_latency_ms=statistics.mean(lats),
        query_count=len(results),
        per_query=per_query,
    )


def paired_ttest(
    a_results: list[QueryResult],
    b_results: list[QueryResult],
    metric_fn: Callable[[list[int], list[int]], float],
) -> tuple[float, float]:
    """Paired t-test comparing metric values between two conditions"""
    from scipy import stats  # lazy import - scipy is a dependency

    # Align by query_id - drop queries missing from either condition
    a_map = {r.query_id: r for r in a_results}
    b_map = {r.query_id: r for r in b_results}
    common_ids = sorted(set(a_map) & set(b_map))
    a_scores = [metric_fn(a_map[qid].retrieved_ids, a_map[qid].ground_truth) for qid in common_ids]
    b_scores = [metric_fn(b_map[qid].retrieved_ids, b_map[qid].ground_truth) for qid in common_ids]
    t_stat, p_val = stats.ttest_rel(b_scores, a_scores)
    return float(t_stat), float(p_val)


def print_comparison_table(
    baseline: AggregatedMetrics,
    *treatments: AggregatedMetrics,
) -> None:
    """Pretty-print a side-by-side metric comparison table."""
    metrics = [
        ("Recall@5","recall_at_5"),
        ("Recall@10","recall_at_10"),
        ("Recall@20", "recall_at_20"),
        ("Precision@10","precision_at_10"),
        ("NDCG@10", "ndcg_at_10"),
        ("MRR", "mrr_score"),
        ("HitRate@10", "hit_rate_at_10"),
        ("Latency(ms)", "mean_latency_ms"),
    ]

    conditions = [baseline, *treatments]
    header = f"{'Metric':<16}" + "".join(f"{c.condition:>14}" for c in conditions)
    print(header)

    for label, attr in metrics:
        row = f"{label:<16}"
        base_val = getattr(baseline, attr)
        for cond in conditions:
            val = getattr(cond, attr)
            if cond is baseline:
                row += f"{val:>14.4f}"
            else:
                delta = val - base_val
                sign = "+" if delta >= 0 else ""
                row += f"{val:>9.4f}({sign}{delta:.3f})"
        print(row)
