from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from qdrant_client import QdrantClient
from experiments.utils.metrics import ndcg_at_k, recall_at_k, hit_rate_at_k
from bookdb.vector_db.clustering import cluster_seeds_by_embedding
from apps.api.core.embeddings import get_vectors_by_ids, most_similar_by_vector

INTERACTIONS_PATH = Path("data/3_goodreads_interactions_reduced.parquet")
TEMPORAL_BENCHMARK = Path("data/temporal_benchmark_ground_truth.json")
RESULTS_PATH = Path("data/exp_clustering_results.json")

COLLECTION_NAME = "books"
TOP_K_EVAL = 10
MAX_SEEDS = 14
MIN_SEEDS = 10 
N_USERS = 200
RNG_SEED = 42


# Helpers
def _val_cutoff_from_benchmark(path: Path) -> int:
    with open(path) as f:
        data = json.load(f)
    val_date = data["split_summary"]["val_end_date"]
    return int(datetime.strptime(val_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _seed_scores(rows: list[dict]) -> dict[int, float]:
    """Weight seeds by rating, mirroring discovery.py interaction scoring."""
    scores: dict[int, float] = defaultdict(float)
    for r in rows:
        gid = int(r["book_id"])
        rating = int(r.get("rating") or 0)
        scores[gid] += 2.5 if rating >= 5 else (2.0 if rating >= 4 else 1.2 if rating >= 1 else 1.0)
    top = sorted(scores.items(), key=lambda x: -x[1])[:MAX_SEEDS]
    return dict(top)


def _query_qdrant(
    client: QdrantClient,
    query_vector: list[float],
    limit: int,
    exclude_ids: set[int],
) -> list[int]:
    """Thin wrapper around the production most_similar_by_vector."""
    hits = most_similar_by_vector(client, query_vector, top_k=limit, exclude_ids=exclude_ids or None)
    return [int(h["id"]) for h in hits]


def _intra_list_diversity(client: QdrantClient, book_ids: list[int]) -> float:
    """Mean pairwise cosine distance of recommended books (higher = more diverse)."""
    if len(book_ids) < 2:
        return 0.0
    vecs = get_vectors_by_ids(client, book_ids)
    if len(vecs) < 2:
        return 0.0
    mat = np.array(list(vecs.values()), dtype=np.float32)
    # Normalise rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    mat = mat / norms
    sim = mat @ mat.T
    n = len(mat)
    # Mean off-diagonal cosine distance
    distances = 1.0 - sim
    total = distances.sum() - np.trace(distances)
    return float(total / (n * (n - 1)))


# Single-centroid baseline

def _single_centroid_recommend(
    client: QdrantClient,
    seed_scores: dict[int, float],
    vector_map: dict[int, list[float]],
    limit: int,
    exclude_ids: set[int],
) -> list[int]:
    """Average all seed vectors into one centroid and query Qdrant once."""
    valid = {gid: np.array(vec, dtype=np.float32)
             for gid, vec in vector_map.items() if gid in seed_scores}
    if not valid:
        return []
    weights = np.array([seed_scores[gid] for gid in valid], dtype=np.float32)
    vecs    = np.array(list(valid.values()), dtype=np.float32)
    centroid = (vecs * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
    return _query_qdrant(client, centroid.tolist(), limit, exclude_ids)


# Clustered (production logic)

def _clustered_recommend(
    client: QdrantClient,
    seed_scores: dict[int, float],
    vector_map: dict[int, list[float]],
    limit: int,
    exclude_ids: set[int],
) -> list[int]:
    """Mirror _cluster_vector_recommendations from discovery.py."""
    valid_seeds = {gid: vec for gid, vec in vector_map.items() if gid in seed_scores}
    if len(valid_seeds) < 2:
        return []

    n_clusters = max(2, min(len(valid_seeds) // 3, 5))
    try:
        clusters = cluster_seeds_by_embedding(valid_seeds, seed_scores, n_clusters)
    except Exception:
        return []
    if not clusters:
        return []

    # Proportional slot allocation (mirrors discovery.py exactly)
    total_weight = sum(sum(w for _, w in members) for _, members in clusters)
    per_cluster_limits = []
    for _, members in clusters:
        cw = sum(w for _, w in members)
        raw = (cw / total_weight) * limit if total_weight > 0 else limit / len(clusters)
        per_cluster_limits.append(max(1, round(raw)))
    diff = limit - sum(per_cluster_limits)
    if diff != 0:
        heaviest = max(range(len(clusters)), key=lambda k: sum(w for _, w in clusters[k][1]))
        per_cluster_limits[heaviest] += diff

    seen: set[int] = set(exclude_ids)
    per_cluster_hits: list[list[int]] = []
    for cluster_idx, (centroid, _) in enumerate(clusters):
        hits = _query_qdrant(client, centroid.tolist(), per_cluster_limits[cluster_idx], seen)
        per_cluster_hits.append(hits)
        seen.update(hits)

    # Round-robin interleave across clusters (mirrors discovery.py)
    results: list[int] = []
    for round_hits in zip_longest(*per_cluster_hits):
        for gid in round_hits:
            if gid is not None:
                results.append(gid)
    return results[:limit]


# User sampling

def sample_users(
    interactions_path: Path,
    val_cutoff: int,
    n_users: int,
    rng_seed: int,
) -> list[dict]:
    """Sample users with ≥MIN_SEEDS train interactions AND test interactions."""
    print("  Scanning interactions…", flush=True)
    lf = pl.scan_parquet(interactions_path)

    train_users = (
        lf.filter(pl.col("timestamp") <= val_cutoff)
        .group_by("user_id")
        .agg(pl.len().alias("n_train"))
        .filter(pl.col("n_train") >= MIN_SEEDS)
    )
    test_users = (
        lf.filter(pl.col("timestamp") > val_cutoff)
        .group_by("user_id")
        .agg(pl.col("book_id").alias("test_books"))
    )
    qualifying = (
        train_users.join(test_users, on="user_id", how="inner")
        .select("user_id").collect()
    )
    all_ids = qualifying["user_id"].to_list()
    print(f"  {len(all_ids):,} qualifying users", flush=True)

    import random
    sampled = sorted(random.Random(rng_seed).sample(all_ids, min(n_users, len(all_ids))))

    rows_df = (
        lf.filter(pl.col("user_id").is_in(sampled))
        .select(["user_id", "book_id", "rating", "timestamp"])
        .collect()
    )

    users = []
    for uid in sampled:
        u = rows_df.filter(pl.col("user_id") == uid)
        train = (
            u.filter(pl.col("timestamp") <= val_cutoff)
            .sort("timestamp", descending=True)
            .select(["book_id", "rating"]).to_dicts()
        )
        test_ids = u.filter(pl.col("timestamp") > val_cutoff)["book_id"].to_list()
        if len(train) >= MIN_SEEDS and test_ids:
            users.append({
                "user_id":       int(uid),
                "train":         train,
                "test_book_ids": [int(x) for x in test_ids],
            })

    print(f"  {len(users)} users kept", flush=True)
    return users


# Experiment runner

def run_experiment(users: list[dict], top_k: int = TOP_K_EVAL) -> dict[str, Any]:
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, port=None)

    single_rows: list[dict] = []
    cluster_rows: list[dict] = []

    for i, user in enumerate(users):
        uid = user["user_id"]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(users)}] user {uid}", end="\r", flush=True)

        seed_scores = _seed_scores(user["train"])
        ground_truth = user["test_book_ids"]
        exclude = set(seed_scores.keys())

        # Fetch seed vectors from Qdrant once, reuse for both conditions
        try:
            vector_map = get_vectors_by_ids(client, list(seed_scores.keys()))
        except Exception as e:
            print(f"\n  Warning: vector fetch failed for user {uid} ({e}); skipping")
            continue
        if len(vector_map) < 2:
            continue

        def _row(recs: list[int], latency_ms: float) -> dict:
            return {
                "user_id": uid,
                "recall":recall_at_k(recs, ground_truth, top_k),
                "ndcg": ndcg_at_k(recs, ground_truth, top_k),
                "hit_rate": hit_rate_at_k(recs, ground_truth, top_k),
                "diversity": _intra_list_diversity(client, recs),
                "latency_ms": latency_ms,
            }

        try:
            t0 = time.perf_counter()
            single_recs = _single_centroid_recommend(client, seed_scores, vector_map, top_k, exclude)
            single_ms = (time.perf_counter() - t0) * 1000
            single_rows.append(_row(single_recs, single_ms))
        except Exception as e:
            print(f"\n  Warning: single centroid failed for user {uid} ({e}); skipping")
            continue

        try:
            t0 = time.perf_counter()
            cluster_recs = _clustered_recommend(client, seed_scores, vector_map, top_k, exclude)
            cluster_ms = (time.perf_counter() - t0) * 1000
            cluster_rows.append(_row(cluster_recs, cluster_ms))
        except Exception as e:
            print(f"\n  Warning: clustered failed for user {uid} ({e}); skipping")
            # still keep single row — pop it to keep alignment
            single_rows.pop()
            continue

    print(f"\n  Evaluated {len(single_rows)} users", flush=True)

    def _agg(rows: list[dict]) -> dict:
        if not rows:
            return {}
        return {
            "recall_at_10":round(float(np.mean([r["recall"]    for r in rows])), 4),
            "ndcg_at_10": round(float(np.mean([r["ndcg"]      for r in rows])), 4),
            "hit_rate_at_10": round(float(np.mean([r["hit_rate"]  for r in rows])), 4),
            "diversity": round(float(np.mean([r["diversity"] for r in rows])), 4),
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in rows])), 2),
            "user_count": len(rows),
        }

    from scipy import stats
    uids   = {r["user_id"] for r in single_rows} & {r["user_id"] for r in cluster_rows}
    s_map  = {r["user_id"]: r for r in single_rows}
    c_map  = {r["user_id"]: r for r in cluster_rows}
    shared = sorted(uids)

    stat_tests: dict[str, Any] = {}
    for metric in ("recall", "ndcg", "hit_rate", "diversity"):
        s_vals = [s_map[u][metric] for u in shared]
        c_vals = [c_map[u][metric] for u in shared]
        t, p   = stats.ttest_rel(c_vals, s_vals)
        stat_tests[metric] = {
            "clustered_vs_single": {"t": round(float(t), 4), "p": round(float(p), 6)}
        }

    return {
        "metadata": {
            "experiment": "seed_clustering_vs_single_centroid",
            "top_k":top_k,
            "min_seeds":MIN_SEEDS,
            "max_seeds": MAX_SEEDS,
            "n_users_evaluated":len(single_rows),
        },
        "conditions": {
            "single_centroid": _agg(single_rows),
            "clustered": _agg(cluster_rows),
        },
        "statistical_tests": stat_tests,
        "per_user": {
            "single_centroid": single_rows,
            "clustered": cluster_rows,
        },
    }


def print_summary(results: dict[str, Any]) -> None:
    s = results["conditions"].get("single_centroid", {})
    c = results["conditions"].get("clustered", {})
    n = results["metadata"]["n_users_evaluated"]

    print("\n" + "=" * 68)
    print("SEED CLUSTERING EXPERIMENT — RESULTS")
    print(f"  Users evaluated: {n}  |  min seeds: {results['metadata']['min_seeds']}")
    print("=" * 68)
    print(f"{'Metric':<18} {'Single centroid':>16} {'Clustered':>12}")
    print("-" * 50)
    for label, key in [
        ("Recall@10", "recall_at_10"),
        ("NDCG@10", "ndcg_at_10"),
        ("HitRate@10", "hit_rate_at_10"),
        ("Diversity", "diversity"),
        ("Latency(ms)", "mean_latency_ms"),
    ]:
        sv = s.get(key, float("nan"))
        cv = c.get(key, float("nan"))
        print(f"{label:<18} {sv:>16.4f} {cv:>12.4f}")

    print("\nStatistical significance (paired t-test, clustered vs single):")
    for metric, tests in results.get("statistical_tests", {}).items():
        for name, vals in tests.items():
            sig = "SIGNIFICANT" if vals["p"] < 0.05 else "not significant"
            print(f"  {metric:12} {name}: t={vals['t']:.3f}, p={vals['p']:.4f} ({sig})")
    print("=" * 68)


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Seed clustering vs single-centroid experiment")
    parser.add_argument("--n-users", type=int, default=N_USERS)
    parser.add_argument("--top-k", type=int, default=TOP_K_EVAL)
    parser.add_argument("--output", default=str(RESULTS_PATH))
    args = parser.parse_args()

    if not TEMPORAL_BENCHMARK.exists():
        print("Run: uv run python experiments/utils/temporal_split.py")
        sys.exit(1)

    val_cutoff = _val_cutoff_from_benchmark(TEMPORAL_BENCHMARK)
    print(f"Val cutoff: {datetime.fromtimestamp(val_cutoff, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"Sampling {args.n_users} users (min {MIN_SEEDS} seeds)…")

    users = sample_users(INTERACTIONS_PATH, val_cutoff, args.n_users, RNG_SEED)
    if not users:
        print("No qualifying users found.")
        sys.exit(1)

    print(f"\nRunning experiment on {len(users)} users…")
    results = run_experiment(users, top_k=args.top_k)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {out}")
    print_summary(results)


if __name__ == "__main__":
    main()
