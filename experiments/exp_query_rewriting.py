from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import httpx
from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from bookdb.models.chatbot_llm import create_llm_client, rewrite_query_sync
from bookdb.vector_db import BookVectorCRUD
from experiments.utils.metrics import (
    QueryResult,
    AggregatedMetrics,
    aggregate,
    hit_rate_at_k,
    ndcg_at_k,
    paired_ttest,
    print_comparison_table,
    recall_at_k,
)

# Configuration

EMBEDDING_ENDPOINT = os.environ["EMBEDDING_SERVICE_URL"].rstrip("/") + "/embed"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_SERVICE_MODEL", "finetuned")
TEMPORAL_BENCHMARK = Path("data/temporal_benchmark_ground_truth.json")
FALLBACK_BENCHMARK = Path("data/benchmark_ground_truth.json")
RESULTS_PATH = Path("data/exp_query_rewriting_results.json")

TOP_K_SEARCH = 50   # candidates fetched from Qdrant
TOP_K_EVAL = 10   # k used for primary reported metrics


# Vague-query extractor

def _make_vague_query(query_text: str) -> str:
    """Strip description from an embedding text, keeping only header fields"""
    lines = query_text.strip().splitlines()
    header = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(TITLE|GENRE|AUTHOR)S?:", stripped, re.IGNORECASE):
            header.append(stripped)
        elif re.match(r"^DESCRIPTION:", stripped, re.IGNORECASE):
            break  # stop before the description block
    return "\n".join(header) if header else query_text[:120].strip()


# helpers

def _get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    resp = httpx.post(
        EMBEDDING_ENDPOINT,
        json={
            "model": model,
            "texts": [text],
            "normalize_embeddings": True,
            "batch_size": 1,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _search_qdrant(embedding: list[float], top_k: int = TOP_K_SEARCH) -> list[dict[str, Any]]:
    crud = BookVectorCRUD()
    return crud.search_similar_books(query_embedding=embedding, n_results=top_k)


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["queries"]


def _load_user_taste_pool(min_ratings: int = 4, max_users: int = 500) -> list[str | None]:
    """Load a pool of taste profiles from real users in the DB.

    Returns a list of taste profile strings (or None if a user has no
    qualifying ratings). The list is shuffled for random assignment.
    """
    try:
        from bookdb.db.session import SessionLocal
        from bookdb.db.models import BookRating
        from sqlalchemy import select, func
        from apps.api.core.book_queries import get_user_taste_profile

        with SessionLocal() as db:
            user_ids = db.scalars(
                select(BookRating.user_id)
                .group_by(BookRating.user_id)
                .having(func.count() >= min_ratings)
                .limit(max_users)
            ).all()

            profiles = []
            for uid in user_ids:
                profile = get_user_taste_profile(db, uid)
                if profile:
                    profiles.append(profile)

        random.shuffle(profiles)
        print(f"  Loaded {len(profiles)} taste profiles from DB")
        return profiles
    except Exception as e:
        print(f"  Warning: could not load taste profiles from DB: {e}")
        return []


# Per-condition evaluator

def evaluate_condition(
    queries: list[dict[str, Any]],
    condition: str,
    *,
    llm_client=None,
    taste_pool: list[str] | None = None,
    rng_seed: int = 42,
) -> list[QueryResult]:
    """Run all queries under a given retrieval condition.

    Conditions:
        vague        – title + genres + authors only (no description)
        rewritten    – vague query expanded by LLM (generic, no user context)
        personalized – vague query expanded by LLM + random user taste profile
        full         – complete embedding text (ideal upper bound)
    """
    rng = random.Random(rng_seed)
    results: list[QueryResult] = []

    for i, q in enumerate(queries):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{condition}] {i + 1}/{len(queries)}", end="\r")

        full_query_text: str = q["query_text"]
        ground_truth = [int(x) for x in q["ground_truth_ids"]]
        rewritten_text: str | None = None

        try:
            t0 = time.perf_counter()

            if condition == "vague":
                # Header-only: title + genres + authors (no description)
                search_text = _make_vague_query(full_query_text)

            elif condition == "rewritten":
                # LLM rewrites the vague query into a richer description
                assert llm_client is not None, "llm_client required for rewriting"
                vague = _make_vague_query(full_query_text)
                desc, _ = rewrite_query_sync(llm_client, vague)
                search_text = desc or vague
                rewritten_text = desc

            elif condition == "personalized":
                # LLM rewrites with a randomly sampled user taste profile injected
                assert llm_client is not None, "llm_client required for rewriting"
                pool = taste_pool or []
                taste = rng.choice(pool) if pool else None
                vague = _make_vague_query(full_query_text)
                desc, _ = rewrite_query_sync(llm_client, vague, taste_profile=taste)
                search_text = desc or vague
                rewritten_text = desc

            elif condition == "full":
                # Ideal: use the complete embedding text
                search_text = full_query_text

            else:
                raise ValueError(f"Unknown condition: {condition!r}")

            embedding = _get_embedding(search_text)
            hits = _search_qdrant(embedding, top_k=TOP_K_SEARCH)
            latency_ms = (time.perf_counter() - t0) * 1000

            retrieved_ids = [int(h["book_id"]) for h in hits]
            retrieved_scores = [float(h.get("score", 0.0)) for h in hits]

        except Exception as exc:
            print(f"\n  Warning: query {q['query_id']} failed ({exc}); skipping")
            continue

        results.append(QueryResult(
            query_id=q["query_id"],
            query_text=full_query_text,
            query_type=q.get("query_type", "unknown"),
            ground_truth=ground_truth,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            latency_ms=latency_ms,
            condition=condition,
            rewritten_text=rewritten_text,
        ))

    print(f"  [{condition}] {len(results)}/{len(queries)} queries evaluated  ")
    return results


# Main experiment runner

def run_experiment(
    queries: list[dict[str, Any]],
    run_rewriting: bool = True,
    run_personalized: bool = True,
) -> dict[str, Any]:
    """Run the query-rewriting experiment and return a structured results dict."""

    # Condition 1: vague query (baseline for query rewriting)
    print("\nCondition 1/4: vague (title + genres + authors only)")
    vague_results = evaluate_condition(queries, "vague")

    # Condition 2: LLM-rewritten query (generic)
    rewritten_results: list[QueryResult] = []
    llm_client = None
    if run_rewriting:
        llm_client = create_llm_client()
        print("\nCondition 2/4: rewritten (LLM expands vague query — no user context)")
        rewritten_results = evaluate_condition(queries, "rewritten", llm_client=llm_client)

    # Condition 3: LLM-rewritten query + user taste profile (personalized)
    personalized_results: list[QueryResult] = []
    if run_personalized and run_rewriting:
        print("\nCondition 3/4: personalized (LLM rewrite + user taste profile)")
        print("  Loading taste profiles from DB...")
        taste_pool = _load_user_taste_pool()
        if taste_pool:
            personalized_results = evaluate_condition(
                queries, "personalized",
                llm_client=llm_client or create_llm_client(),
                taste_pool=taste_pool,
            )
        else:
            print("  Skipping personalized condition: no taste profiles available")

    # Condition 4: full embedding text (ideal upper bound)
    print("\nCondition 4/4: full (complete embedding text: ideal)")
    full_results = evaluate_condition(queries, "full")

    # Metrics
    vague_metrics = aggregate(vague_results, "vague")
    rewritten_metrics = aggregate(rewritten_results, "rewritten") if rewritten_results else None
    personalized_metrics = aggregate(personalized_results, "personalized") if personalized_results else None
    full_metrics = aggregate(full_results, "full")

    # Statistical tests (paired t-test vs vague baseline)
    stat_tests: dict[str, Any] = {}

    for label, treatment_results in [
        ("rewritten_vs_vague", rewritten_results),
        ("personalized_vs_vague", personalized_results),
        ("full_vs_vague", full_results),
    ]:
        if not treatment_results:
            continue
        t_r, p_r = paired_ttest(
            vague_results, treatment_results,
            lambda r, g: recall_at_k(r, g, TOP_K_EVAL),
        )
        t_n, p_n = paired_ttest(
            vague_results, treatment_results,
            lambda r, g: ndcg_at_k(r, g, TOP_K_EVAL),
        )
        stat_tests[label] = {
            "recall_at_10": {"t": round(t_r, 4), "p": round(p_r, 6)},
            "ndcg_at_10": {"t": round(t_n, 4), "p": round(p_n, 6)},
        }

    # Personalized vs rewritten (direct comparison)
    if personalized_results and rewritten_results:
        t_r, p_r = paired_ttest(
            rewritten_results, personalized_results,
            lambda r, g: recall_at_k(r, g, TOP_K_EVAL),
        )
        t_n, p_n = paired_ttest(
            rewritten_results, personalized_results,
            lambda r, g: ndcg_at_k(r, g, TOP_K_EVAL),
        )
        stat_tests["personalized_vs_rewritten"] = {
            "recall_at_10": {"t": round(t_r, 4), "p": round(p_r, 6)},
            "ndcg_at_10": {"t": round(t_n, 4), "p": round(p_n, 6)},
        }

    def _metrics_dict(m: AggregatedMetrics | None) -> dict:
        if m is None:
            return {}
        return {
            "recall_at_5": round(m.recall_at_5, 4),
            "recall_at_10": round(m.recall_at_10, 4),
            "recall_at_20": round(m.recall_at_20, 4),
            "precision_at_10": round(m.precision_at_10, 4),
            "ndcg_at_10": round(m.ndcg_at_10, 4),
            "mrr": round(m.mrr_score, 4),
            "hit_rate_at_10": round(m.hit_rate_at_10, 4),
            "mean_latency_ms": round(m.mean_latency_ms, 2),
            "query_count": m.query_count,
        }

    results: dict[str, Any] = {
        "metadata": {
            "experiment": "query_rewriting",
            "design": "vague vs rewritten vs personalized vs full-ideal",
            "top_k_search": TOP_K_SEARCH,
            "top_k_eval": TOP_K_EVAL,
            "total_queries": len(queries),
        },
        "conditions": {
            "vague": _metrics_dict(vague_metrics),
            "rewritten": _metrics_dict(rewritten_metrics),
            "personalized": _metrics_dict(personalized_metrics),
            "full": _metrics_dict(full_metrics),
        },
        "statistical_tests": stat_tests,
        "per_query": {
            "vague": vague_metrics.per_query,
            "rewritten": rewritten_metrics.per_query if rewritten_metrics else [],
            "personalized": personalized_metrics.per_query if personalized_metrics else [],
            "full": full_metrics.per_query,
        },
    }

    return results


def print_summary(results: dict[str, Any]) -> None:
    vague = results["conditions"].get("vague", {})
    rew = results["conditions"].get("rewritten", {})
    pers = results["conditions"].get("personalized", {})
    full = results["conditions"].get("full", {})

    print("\n" + "=" * 80)
    print("QUERY REWRITING EXPERIMENT RESULTS")
    print("  Design: vague  ->  rewritten (generic)  ->  personalized  ->  full ideal")
    print("=" * 80)

    metrics = [
        ("Recall@5",      "recall_at_5"),
        ("Recall@10",     "recall_at_10"),
        ("Recall@20",     "recall_at_20"),
        ("Precision@10",  "precision_at_10"),
        ("NDCG@10",       "ndcg_at_10"),
        ("MRR",           "mrr"),
        ("HitRate@10",    "hit_rate_at_10"),
        ("Latency(ms)",   "mean_latency_ms"),
    ]

    print(f"{'Metric':<16} {'Vague':>10} {'Rewritten':>12} {'Personalized':>14} {'Full(ideal)':>13}")
    print("-" * 68)
    for label, key in metrics:
        v = vague.get(key, float("nan"))
        r = rew.get(key, float("nan"))
        p = pers.get(key, float("nan"))
        f = full.get(key, float("nan"))
        print(f"{label:<16} {v:>10.4f} {r:>12.4f} {p:>14.4f} {f:>13.4f}")

    print("\nStatistical significance (paired t-test):")
    print("=" * 80)
    for test_name, test_res in results.get("statistical_tests", {}).items():
        print(f"  {test_name}:")
        for metric_name, vals in test_res.items():
            sig = "SIGNIFICANT" if vals["p"] < 0.05 else "not significant"
            print(f"    {metric_name}: t={vals['t']:.3f}, p={vals['p']:.4f} ({sig})")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query rewriting retrieval experiment")
    parser.add_argument("--max-queries", type=int, default=-1)
    parser.add_argument("--no-rewriting", action="store_true",
                        help="Skip the LLM rewriting conditions (only vague vs full)")
    parser.add_argument("--no-personalized", action="store_true",
                        help="Skip the personalized condition")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--output", default=str(RESULTS_PATH))
    args = parser.parse_args()

    if args.benchmark:
        bench_path = Path(args.benchmark)
    elif TEMPORAL_BENCHMARK.exists():
        bench_path = TEMPORAL_BENCHMARK
        print(f"Using temporal benchmark: {bench_path}")
    elif FALLBACK_BENCHMARK.exists():
        bench_path = FALLBACK_BENCHMARK
        print(f"Temporal benchmark not found; using fallback: {bench_path}")
    else:
        print("No benchmark found.  Run: uv run python experiments/utils/temporal_split.py")
        sys.exit(1)

    queries = _load_benchmark(bench_path)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]

    print(f"\nRunning query-rewriting experiment on {len(queries)} queries")
    print("=" * 60)

    results = run_experiment(
        queries,
        run_rewriting=not args.no_rewriting,
        run_personalized=not args.no_personalized,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {out}")

    print_summary(results)


if __name__ == "__main__":
    main()
