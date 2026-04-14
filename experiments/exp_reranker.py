from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import duckdb
import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from qdrant_client import QdrantClient
from experiments.utils.metrics import ndcg_at_k, recall_at_k, hit_rate_at_k
from apps.api.core.embeddings import most_similar

# Configuration
INTERACTIONS_PATH = Path("data/3_goodreads_interactions_reduced.parquet")
TEMPORAL_BENCHMARK = Path("data/temporal_benchmark_ground_truth.json")
RESULTS_PATH = Path("data/exp_reranker_results.json")

TOP_K_EVAL = 10
MAX_SEEDS = 14
MIN_SEEDS = 3 # minimum training interactions to include user
N_USERS = 200
RNG_SEED = 42


# Helpers

def _val_cutoff_from_benchmark(path: Path) -> int:
    with open(path) as f:
        data = json.load(f)
    val_date = data["split_summary"]["val_end_date"]  # "2017-04-19"
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


# BPR recommendations

def _bpr_recommend(bpr_url: str, user_id: int, limit: int) -> list[int]:
    """Query BPR parquet for top-limit book recommendations for a user."""
    rows = duckdb.execute(
        f"SELECT item_id FROM parquet_scan('{bpr_url}') "
        f"WHERE user_id = {user_id} ORDER BY prediction DESC LIMIT {limit}"
    ).fetchall()
    return [int(r[0]) for r in rows]


def _bpr_recommend_batch(bpr_url: str, user_ids: list[int], limit: int) -> dict[int, list[int]]:
    """Fetch BPR recs for all users in a single DuckDB query (avoids per-user round trips)."""
    id_list = ", ".join(str(uid) for uid in user_ids)
    rows = duckdb.execute(
        f"SELECT user_id, item_id, prediction FROM parquet_scan('{bpr_url}') "
        f"WHERE user_id IN ({id_list}) ORDER BY user_id, prediction DESC"
    ).fetchall()
    result: dict[int, list[int]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    for user_id, item_id, _ in rows:
        if counts[user_id] < limit:
            result[user_id].append(int(item_id))
            counts[user_id] += 1
    return dict(result)


# Vector recommendations 

def _vector_recommend(client, seed_scores: dict[int, float], limit: int) -> list[int]:
    """Qdrant-based recommendations aggregated from seed books."""
    excluded = set(seed_scores.keys())
    per_seed_top_k = max(limit * 3, 30)
    candidate_scores: dict[int, float] = defaultdict(float)

    for seed_gid, seed_weight in sorted(seed_scores.items(), key=lambda x: -x[1]):
        try:
            similar = most_similar(client, seed_gid, top_k=per_seed_top_k, exclude_ids=excluded)
        except Exception:
            continue
        for rank, cid in enumerate(similar, start=1):
            candidate_scores[cid] += seed_weight / rank

    ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
    return [gid for gid, _ in ranked[:limit]]


# Fused recommendations

def _fused_recommend(
    bpr_recs: list[int],
    vector_recs: list[int],
    limit: int,
) -> list[int]:
    """Combine BPR and vector results using the discovery.py quota logic."""
    interaction_quota = max(1, min(limit // 3, 8)) if limit >= 3 else 0
    interaction_reserved = min(interaction_quota, len(vector_recs))
    bpr_target = max(limit - interaction_reserved, 0)

    result: list[int] = []
    seen: set[int] = set()

    def _add(items: list[int], cap: int) -> None:
        for bid in items:
            if len(result) >= cap:
                break
            if bid not in seen:
                result.append(bid)
                seen.add(bid)

    _add(bpr_recs,bpr_target) # BPR fills most slots
    _add(vector_recs,limit) # vector fills the remainder
    _add(bpr_recs,limit) # BPR fills any remaining gaps
    return result[:limit]


# User sampling

def sample_users(
    interactions_path: Path,
    val_cutoff: int,
    n_users: int,
    rng_seed: int,
) -> list[dict]:
    """Sample users with train AND test interactions."""
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
                "user_id": int(uid),
                "train": train,
                "test_book_ids": [int(x) for x in test_ids],
            })

    print(f"  {len(users)} users kept", flush=True)
    return users


# Experiment runner

def run_experiment(users: list[dict], top_k: int = TOP_K_EVAL) -> dict[str, Any]:
    bpr_url = os.environ["BPR_PARQUET_URL"]
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, port=None)

    bpr_rows, vec_rows, fused_rows = [], [], []

    # Pre-fetch BPR recs for all users in one batch to avoid 200 round trips
    print("  Pre-fetching BPR recommendations (single batch query)…", flush=True)
    t_bpr0 = time.perf_counter()
    all_user_ids = [u["user_id"] for u in users]
    try:
        bpr_batch = _bpr_recommend_batch(bpr_url, all_user_ids, limit=top_k * 5)
        bpr_batch_ms = (time.perf_counter() - t_bpr0) * 1000
        print(f"  BPR batch done in {bpr_batch_ms/1000:.1f}s ({len(bpr_batch)} users with recs)", flush=True)
    except Exception as e:
        print(f"  BPR batch failed ({e}); falling back to per-user queries", flush=True)
        bpr_batch = {}
    per_user_bpr_ms = bpr_batch_ms / max(len(bpr_batch), 1)

    for i, user in enumerate(users):
        uid = user["user_id"]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(users)}] user {uid}", end="\r", flush=True)

        seed_scores = _seed_scores(user["train"])
        ground_truth = user["test_book_ids"]
        exclude = set(seed_scores)

        try:
            if uid in bpr_batch:
                bpr_recs = bpr_batch[uid]
                bpr_ms = per_user_bpr_ms
            else:
                t0 = time.perf_counter()
                bpr_recs = _bpr_recommend(bpr_url, uid, limit=top_k * 5)
                bpr_ms = (time.perf_counter() - t0) * 1000
            if not bpr_recs:
                raise ValueError("no BPR predictions for this user")
        except Exception as e:
            print(f"\n  Warning: BPR failed for user {uid} ({e}); skipping")
            continue

        try:
            t0 = time.perf_counter()
            vec_recs = _vector_recommend(client, seed_scores, limit=top_k)
            vec_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"\n  Warning: vector failed for user {uid} ({e}); skipping")
            continue

        # Filter BPR results: remove already-seen training items
        bpr_filtered = [b for b in bpr_recs if b not in exclude]
        fused_recs   = _fused_recommend(bpr_filtered, vec_recs, limit=top_k)

        def _row(recs: list[int], latency: float) -> dict:
            return {
                "user_id":uid,
                "recall":recall_at_k(recs, ground_truth, top_k),
                "ndcg":ndcg_at_k(recs, ground_truth, top_k),
                "hit_rate":hit_rate_at_k(recs, ground_truth, top_k),
                "latency_ms":latency,
            }

        bpr_rows.append(_row(bpr_filtered[:top_k], bpr_ms))
        vec_rows.append(_row(vec_recs, vec_ms))
        fused_rows.append(_row(fused_recs, bpr_ms + vec_ms))

    print(f"\n  Evaluated {len(bpr_rows)} users", flush=True)

    def _agg(rows: list[dict]) -> dict:
        if not rows:
            return {}
        return {
            "recall_at_10":round(float(np.mean([r["recall"] for r in rows])), 4),
            "ndcg_at_10":round(float(np.mean([r["ndcg"] for r in rows])), 4),
            "hit_rate_at_10":round(float(np.mean([r["hit_rate"] for r in rows])), 4),
            "mean_latency_ms":round(float(np.mean([r["latency_ms"] for r in rows])), 2),
            "user_count":len(rows),
        }

    # Paired t-test (only for users present in all three conditions)
    from scipy import stats
    uids = ({r["user_id"] for r in bpr_rows}
            & {r["user_id"] for r in vec_rows}
            & {r["user_id"] for r in fused_rows})
    b_map = {r["user_id"]: r for r in bpr_rows}
    v_map = {r["user_id"]: r for r in vec_rows}
    f_map = {r["user_id"]: r for r in fused_rows}
    shared = sorted(uids)

    stat_tests: dict[str, Any] = {}
    for metric in ("recall", "ndcg", "hit_rate"):
        b_vals = [b_map[u][metric] for u in shared]
        v_vals = [v_map[u][metric] for u in shared]
        f_vals = [f_map[u][metric] for u in shared]
        t_fv, p_fv = stats.ttest_rel(f_vals, v_vals)
        t_fb, p_fb = stats.ttest_rel(f_vals, b_vals)
        stat_tests[metric] = {
            "fused_vs_vector":{"t": round(float(t_fv), 4), "p": round(float(p_fv), 6)},
            "fused_vs_bpr":{"t": round(float(t_fb), 4), "p": round(float(p_fb), 6)},
        }

    return {
        "metadata": {
            "experiment":"bpr_vector_fusion",
            "top_k":top_k,
            "n_users_evaluated": len(bpr_rows),
        },
        "conditions": {
            "bpr_only": _agg(bpr_rows),
            "vector_only":_agg(vec_rows),
            "fused":_agg(fused_rows),
        },
        "statistical_tests": stat_tests,
        "per_user": {
            "bpr_only":bpr_rows,
            "vector_only":vec_rows,
            "fused":fused_rows,
        },
    }


def print_summary(results: dict[str, Any]) -> None:
    bpr = results["conditions"].get("bpr_only", {})
    vec = results["conditions"].get("vector_only", {})
    fused = results["conditions"].get("fused", {})
    n = results["metadata"]["n_users_evaluated"]

    print("\n" + "=" * 70)
    print("BPR + VECTOR FUSION EXPERIMENT — RESULTS")
    print(f"  Users evaluated: {n}")
    print("=" * 70)
    print(f"{'Metric':<16} {'BPR-only':>10} {'Vector-only':>12} {'Fused':>8}")
    print("-" * 50)
    for label, key in [("Recall@10", "recall_at_10"), ("NDCG@10", "ndcg_at_10"),
                       ("HitRate@10", "hit_rate_at_10"), ("Latency(ms)", "mean_latency_ms")]:
        b = bpr.get(key, float("nan"))
        v = vec.get(key, float("nan"))
        f = fused.get(key, float("nan"))
        print(f"{label:<16} {b:>10.4f} {v:>12.4f} {f:>8.4f}")

    print("\nStatistical significance (paired t-test):")
    for metric, tests in results.get("statistical_tests", {}).items():
        for name, vals in tests.items():
            sig = "SIGNIFICANT" if vals["p"] < 0.05 else "not significant"
            print(f"  {metric} {name}: t={vals['t']:.3f}, p={vals['p']:.4f} ({sig})")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="BPR + vector fusion experiment")
    parser.add_argument("--n-users", type=int, default=N_USERS)
    parser.add_argument("--top-k", type=int, default=TOP_K_EVAL)
    parser.add_argument("--output", default=str(RESULTS_PATH))
    args = parser.parse_args()

    if not TEMPORAL_BENCHMARK.exists():
        print("Run: uv run python experiments/utils/temporal_split.py")
        sys.exit(1)

    val_cutoff = _val_cutoff_from_benchmark(TEMPORAL_BENCHMARK)
    print(f"Val cutoff: {datetime.fromtimestamp(val_cutoff, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"Sampling {args.n_users} users…")

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
