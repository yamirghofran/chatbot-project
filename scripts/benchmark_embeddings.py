"""Benchmark embedding models for book retrieval.

Compares base vs fine-tuned model performance on 200 benchmark queries.
Uses deployed embedding endpoint and Qdrant vector database.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from scipy import stats

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bookdb.vector_db import BookVectorCRUD


EMBEDDING_ENDPOINT = "https://bookdb-models.up.railway.app/embed"
BENCHMARK_DATASET_PATH = Path("data/benchmark_ground_truth.json")
RESULTS_PATH = Path("data/benchmark_results.json")


@dataclass
class BenchmarkResult:
    """Result for a single query."""

    query_id: str
    query_text: str
    query_type: str
    ground_truth: list[int]
    retrieved_ids: list[int]
    retrieved_scores: list[float]
    latency_ms: float


@dataclass
class Metrics:
    """Aggregated metrics for a model."""

    recall_at_10: float
    recall_at_20: float
    recall_at_50: float
    ndcg_at_10: float
    mrr: float
    precision_at_10: float
    hit_rate_at_10: float
    mean_latency_ms: float
    query_count: int


def get_embedding(text: str, model: str = "finetuned") -> list[float]:
    """Get embedding from deployed endpoint."""
    response = httpx.post(
        EMBEDDING_ENDPOINT,
        json={
            "model": model,
            "texts": [text],
            "normalize_embeddings": True,
            "batch_size": 1,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"][0]


def search_books(query_embedding: list[float], top_k: int = 50) -> list[dict[str, Any]]:
    """Search Qdrant for similar books."""
    crud = BookVectorCRUD()
    results = crud.search_similar_books(
        query_embedding=query_embedding,
        n_results=top_k,
    )
    return results


def calculate_recall(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Calculate recall@k."""
    if not ground_truth:
        return 0.0
    retrieved_set = set(retrieved[:k])
    ground_truth_set = set(ground_truth)
    hits = len(retrieved_set.intersection(ground_truth_set))
    return hits / len(ground_truth_set)


def calculate_precision(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Calculate precision@k."""
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved[:k])
    ground_truth_set = set(ground_truth)
    hits = len(retrieved_set.intersection(ground_truth_set))
    return hits / k


def calculate_mrr(retrieved: list[int], ground_truth: list[int]) -> float:
    """Calculate Mean Reciprocal Rank."""
    ground_truth_set = set(ground_truth)
    for i, doc_id in enumerate(retrieved):
        if doc_id in ground_truth_set:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Calculate NDCG@k."""
    if not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)

    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in ground_truth_set:
            # Position i+1, so discount is log2(i+2)
            dcg += 1.0 / np.log2(i + 2)

    # Calculate ideal DCG
    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_hit_rate(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Calculate hit rate@k (at least one relevant in top-k)."""
    if not ground_truth:
        return 0.0
    retrieved_set = set(retrieved[:k])
    ground_truth_set = set(ground_truth)
    return 1.0 if retrieved_set.intersection(ground_truth_set) else 0.0


def evaluate_query(
    query_data: dict[str, Any],
    model: str,
) -> BenchmarkResult | None:
    """Evaluate a single query."""
    try:
        query_text = query_data["query_text"]
        ground_truth = [int(x) for x in query_data["ground_truth_ids"]]

        # Time the embedding generation
        start_time = time.time()
        query_embedding = get_embedding(query_text, model=model)

        # Search Qdrant
        search_results = search_books(query_embedding, top_k=50)
        end_time = time.time()

        retrieved_ids = [int(r["book_id"]) for r in search_results]
        retrieved_scores = [float(r.get("score", 0.0)) for r in search_results]
        latency_ms = (end_time - start_time) * 1000

        return BenchmarkResult(
            query_id=query_data["query_id"],
            query_text=query_text,
            query_type=query_data["query_type"],
            ground_truth=ground_truth,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            latency_ms=latency_ms,
        )
    except Exception as e:
        print(f"  Error evaluating query {query_data['query_id']}: {e}")
        return None


def aggregate_metrics(results: list[BenchmarkResult]) -> Metrics:
    """Aggregate metrics across all queries."""
    if not results:
        raise ValueError("No results to aggregate")

    recalls_10 = []
    recalls_20 = []
    recalls_50 = []
    ndcgs = []
    mrrs = []
    precisions = []
    hit_rates = []
    latencies = []

    for result in results:
        recalls_10.append(
            calculate_recall(result.retrieved_ids, result.ground_truth, 10)
        )
        recalls_20.append(
            calculate_recall(result.retrieved_ids, result.ground_truth, 20)
        )
        recalls_50.append(
            calculate_recall(result.retrieved_ids, result.ground_truth, 50)
        )
        ndcgs.append(calculate_ndcg(result.retrieved_ids, result.ground_truth, 10))
        mrrs.append(calculate_mrr(result.retrieved_ids, result.ground_truth))
        precisions.append(
            calculate_precision(result.retrieved_ids, result.ground_truth, 10)
        )
        hit_rates.append(
            calculate_hit_rate(result.retrieved_ids, result.ground_truth, 10)
        )
        latencies.append(result.latency_ms)

    return Metrics(
        recall_at_10=statistics.mean(recalls_10),
        recall_at_20=statistics.mean(recalls_20),
        recall_at_50=statistics.mean(recalls_50),
        ndcg_at_10=statistics.mean(ndcgs),
        mrr=statistics.mean(mrrs),
        precision_at_10=statistics.mean(precisions),
        hit_rate_at_10=statistics.mean(hit_rates),
        mean_latency_ms=statistics.mean(latencies),
        query_count=len(results),
    )


def calculate_improvement(base_val: float, finetuned_val: float) -> tuple[float, float]:
    """Calculate absolute and relative improvement."""
    abs_improvement = finetuned_val - base_val
    rel_improvement = (abs_improvement / base_val * 100) if base_val > 0 else 0.0
    return abs_improvement, rel_improvement


def run_statistical_test(
    base_results: list[BenchmarkResult],
    finetuned_results: list[BenchmarkResult],
    metric_fn,
) -> tuple[float, float]:
    """Run paired t-test on a metric."""
    base_scores = [metric_fn(r.retrieved_ids, r.ground_truth) for r in base_results]
    finetuned_scores = [
        metric_fn(r.retrieved_ids, r.ground_truth) for r in finetuned_results
    ]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(finetuned_scores, base_scores)

    return t_stat, p_value


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run full benchmark."""
    print("=" * 60)
    print("BOOK EMBEDDING MODEL BENCHMARK")
    print("=" * 60)

    # Load benchmark dataset
    if not BENCHMARK_DATASET_PATH.exists():
        print(f"Error: Benchmark dataset not found at {BENCHMARK_DATASET_PATH}")
        print("Run: python scripts/generate_benchmark_dataset.py")
        return {}

    with open(BENCHMARK_DATASET_PATH) as f:
        benchmark_data = json.load(f)

    queries = benchmark_data["queries"]

    if args.max_queries > 0:
        queries = queries[: args.max_queries]

    print(f"\nBenchmark Configuration:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Embedding endpoint: {EMBEDDING_ENDPOINT}")
    print(f"  Results will be saved to: {RESULTS_PATH}")
    print()

    # Evaluate base model
    print("Evaluating BASE model...")
    base_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{len(queries)}", end="\r")
        result = evaluate_query(query, model="base")
        if result:
            base_results.append(result)
    print(f"  Progress: {len(base_results)}/{len(queries)} queries evaluated  ")

    # Evaluate fine-tuned model
    print("\nEvaluating FINE-TUNED model...")
    finetuned_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{len(queries)}", end="\r")
        result = evaluate_query(query, model="finetuned")
        if result:
            finetuned_results.append(result)
    print(f"  Progress: {len(finetuned_results)}/{len(queries)} queries evaluated  ")

    if len(base_results) != len(finetuned_results):
        print(
            f"Warning: Mismatch in result counts ({len(base_results)} vs {len(finetuned_results)})"
        )
        min_len = min(len(base_results), len(finetuned_results))
        base_results = base_results[:min_len]
        finetuned_results = finetuned_results[:min_len]

    # Aggregate metrics
    print("\nAggregating metrics...")
    base_metrics = aggregate_metrics(base_results)
    finetuned_metrics = aggregate_metrics(finetuned_results)

    # Statistical tests
    print("Running statistical tests...")
    recall_t, recall_p = run_statistical_test(
        base_results, finetuned_results, lambda r, g: calculate_recall(r, g, 10)
    )
    ndcg_t, ndcg_p = run_statistical_test(
        base_results, finetuned_results, lambda r, g: calculate_ndcg(r, g, 10)
    )

    # Compile results
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(queries),
            "successful_queries": len(base_results),
            "embedding_endpoint": EMBEDDING_ENDPOINT,
        },
        "base_model": {
            "recall_at_10": round(base_metrics.recall_at_10, 4),
            "recall_at_20": round(base_metrics.recall_at_20, 4),
            "recall_at_50": round(base_metrics.recall_at_50, 4),
            "ndcg_at_10": round(base_metrics.ndcg_at_10, 4),
            "mrr": round(base_metrics.mrr, 4),
            "precision_at_10": round(base_metrics.precision_at_10, 4),
            "hit_rate_at_10": round(base_metrics.hit_rate_at_10, 4),
            "mean_latency_ms": round(base_metrics.mean_latency_ms, 2),
        },
        "finetuned_model": {
            "recall_at_10": round(finetuned_metrics.recall_at_10, 4),
            "recall_at_20": round(finetuned_metrics.recall_at_20, 4),
            "recall_at_50": round(finetuned_metrics.recall_at_50, 4),
            "ndcg_at_10": round(finetuned_metrics.ndcg_at_10, 4),
            "mrr": round(finetuned_metrics.mrr, 4),
            "precision_at_10": round(finetuned_metrics.precision_at_10, 4),
            "hit_rate_at_10": round(finetuned_metrics.hit_rate_at_10, 4),
            "mean_latency_ms": round(finetuned_metrics.mean_latency_ms, 2),
        },
        "improvements": {},
        "statistical_tests": {
            "recall_at_10": {
                "t_statistic": round(recall_t, 4),
                "p_value": round(recall_p, 6),
            },
            "ndcg_at_10": {
                "t_statistic": round(ndcg_t, 4),
                "p_value": round(ndcg_p, 6),
            },
        },
        "query_results": {
            "base": [
                {
                    "query_id": r.query_id,
                    "query_type": r.query_type,
                    "recall_at_10": calculate_recall(
                        r.retrieved_ids, r.ground_truth, 10
                    ),
                    "ndcg_at_10": calculate_ndcg(r.retrieved_ids, r.ground_truth, 10),
                    "latency_ms": r.latency_ms,
                }
                for r in base_results
            ],
            "finetuned": [
                {
                    "query_id": r.query_id,
                    "query_type": r.query_type,
                    "recall_at_10": calculate_recall(
                        r.retrieved_ids, r.ground_truth, 10
                    ),
                    "ndcg_at_10": calculate_ndcg(r.retrieved_ids, r.ground_truth, 10),
                    "latency_ms": r.latency_ms,
                }
                for r in finetuned_results
            ],
        },
    }

    # Calculate improvements
    for metric_name in [
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "ndcg_at_10",
        "mrr",
        "precision_at_10",
        "hit_rate_at_10",
    ]:
        base_val = results["base_model"][metric_name]
        ft_val = results["finetuned_model"][metric_name]
        abs_imp, rel_imp = calculate_improvement(base_val, ft_val)
        results["improvements"][metric_name] = {
            "absolute": round(abs_imp, 4),
            "relative_percent": round(rel_imp, 2),
        }

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(results: dict[str, Any]):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    print("\nRetrieval Performance:")
    print(f"{'Metric':<20} {'Base':>10} {'Fine-tuned':>12} {'Improvement':>12}")
    print("-" * 60)

    for metric_name in [
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "ndcg_at_10",
        "mrr",
        "precision_at_10",
        "hit_rate_at_10",
    ]:
        base_val = results["base_model"][metric_name]
        ft_val = results["finetuned_model"][metric_name]
        improvement = results["improvements"][metric_name]["relative_percent"]

        # Format metric name
        display_name = metric_name.replace("_", " ").title()

        print(
            f"{display_name:<20} {base_val:>10.4f} {ft_val:>12.4f} {improvement:>+11.1f}%"
        )

    print("\nLatency:")
    base_lat = results["base_model"]["mean_latency_ms"]
    ft_lat = results["finetuned_model"]["mean_latency_ms"]
    print(f"  Base model:      {base_lat:.2f} ms")
    print(f"  Fine-tuned model: {ft_lat:.2f} ms")

    print("\nStatistical Significance (p < 0.05 indicates significant improvement):")
    for metric_name, test_results in results["statistical_tests"].items():
        p_val = test_results["p_value"]
        significant = "✓ YES" if p_val < 0.05 else "✗ NO"
        print(f"  {metric_name}: p={p_val:.6f} ({significant})")

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("=" * 60)


def main():
    global EMBEDDING_ENDPOINT

    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=-1,
        help="Maximum queries to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--endpoint",
        default=EMBEDDING_ENDPOINT,
        help="Embedding service endpoint",
    )
    args = parser.parse_args()

    EMBEDDING_ENDPOINT = args.endpoint

    results = run_benchmark(args)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
