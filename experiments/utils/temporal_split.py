from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import polars as pl


INTERACTIONS_PATH = Path("data/3_goodreads_interactions_reduced.parquet")
BOOKS_PATH = Path("data/3_goodreads_books_with_metrics.parquet")
EMBEDDING_TEXTS_PATH = Path("data/books_embedding_texts.parquet")
TEMPORAL_BENCHMARK_PATH = Path("data/temporal_benchmark_ground_truth.json")

@dataclass
class TemporalSplit:
    test: pl.DataFrame
    train_cutoff: int
    val_cutoff: int
    total_interactions: int
    train_interactions: int
    val_interactions: int
    train_unique_users: int
    val_unique_users: int
    test_unique_users: int
    train_unique_books: int
    val_unique_books: int
    test_unique_books: int

    @property
    def train_end_date(self) -> str:
        return datetime.fromtimestamp(self.train_cutoff, tz=timezone.utc).strftime("%Y-%m-%d")

    @property
    def val_end_date(self) -> str:
        return datetime.fromtimestamp(self.val_cutoff, tz=timezone.utc).strftime("%Y-%m-%d")

    @property
    def test_start_date(self) -> str:
        ts = self.test.select(pl.col("timestamp").min()).item()
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")

    def summary(self) -> dict[str, Any]:
        return {
            "total_interactions":self.total_interactions,
            "train_interactions":self.train_interactions,
            "val_interactions": self.val_interactions,
            "test_interactions": self.test.height,
            "train_fraction": round(self.train_interactions/self.total_interactions, 4),
            "val_fraction": round(self.val_interactions/self.total_interactions, 4),
            "test_fraction": round(self.test.height/self.total_interactions, 4),
            "train_end_date": self.train_end_date,
            "val_end_date": self.val_end_date,
            "test_start_date": self.test_start_date,
            "train_unique_users": self.train_unique_users,
            "val_unique_users": self.val_unique_users,
            "test_unique_users": self.test_unique_users,
            "train_unique_books": self.train_unique_books,
            "val_unique_books": self.val_unique_books,
            "test_unique_books": self.test_unique_books,
        }


def build_temporal_split(
    interactions_path: str | Path = INTERACTIONS_PATH,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> TemporalSplit:
    
    if val_frac + test_frac >= 1.0:
        raise ValueError(f"val_frac + test_frac must be < 1, got {val_frac + test_frac}")

    path = Path(interactions_path)
    if not path.exists():
        raise FileNotFoundError(f"Interactions parquet not found: {path}")

    lf = pl.scan_parquet(path).filter(pl.col("timestamp").is_not_null())

    # compute split cutoffs via quantile (no full sort needed)
    train_q = 1.0 - val_frac - test_frac
    val_q   = 1.0 - test_frac

    print("  Computing split timestamps…", flush=True)
    cutoffs = (
        lf.select(
            pl.col("timestamp").quantile(train_q, interpolation="lower").alias("train_cutoff"),
            pl.col("timestamp").quantile(val_q,   interpolation="lower").alias("val_cutoff"),
            pl.len().alias("total"),
        )
        .collect()
    )
    train_cutoff = int(cutoffs["train_cutoff"][0])
    val_cutoff = int(cutoffs["val_cutoff"][0])
    total = int(cutoffs["total"][0])


    # Step 2: materialise only the test slice
    print("  Loading test slice…", flush=True)
    test = lf.filter(pl.col("timestamp") > val_cutoff).collect()

    # Step 3: compute train/val stats with cheap lazy aggregations
    print("  Computing train/val statistics…", flush=True)
    train_lf = lf.filter(pl.col("timestamp") <= train_cutoff)
    val_lf   = lf.filter(
        (pl.col("timestamp") > train_cutoff) & (pl.col("timestamp") <= val_cutoff)
    )

    train_stats = train_lf.select(
        pl.len().alias("n"),
        pl.col("user_id").n_unique().alias("u"),
        pl.col("book_id").n_unique().alias("b"),
    ).collect()
    val_stats = val_lf.select(
        pl.len().alias("n"),
        pl.col("user_id").n_unique().alias("u"),
        pl.col("book_id").n_unique().alias("b"),
    ).collect()

    return TemporalSplit(
        test=test,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        total_interactions=total,
        train_interactions=int(train_stats["n"][0]),
        val_interactions=int(val_stats["n"][0]),
        train_unique_users=int(train_stats["u"][0]),
        val_unique_users=int(val_stats["u"][0]),
        test_unique_users=test.select("user_id").n_unique(),
        train_unique_books=int(train_stats["b"][0]),
        val_unique_books=int(val_stats["b"][0]),
        test_unique_books=test.select("book_id").n_unique(),
    )


# Query generation from the test split
def make_retrieval_eval_queries(
    test_interactions: pl.DataFrame,
    books_df: pl.DataFrame,
    embedding_texts_df: pl.DataFrame | None = None,
    n_queries: int = 200,
    rng_seed: int = 42,
    min_similar_books: int = 3,
    min_description_chars: int = 80,
) -> list[dict[str, Any]]:
    """Generate retrieval evaluation queries from the test interaction set.

    Uses Polars joins instead of Python dicts to keep memory usage low even
    on large book catalogs.
    """
    rng = random.Random(rng_seed)

    # Unique book IDs seen in the test period
    test_book_ids = test_interactions.select("book_id").unique()

    # Filter books_df down to only test-period books - avoids a 1.5 M-row dict
    relevant = books_df.join(test_book_ids, on="book_id", how="inner")

    # Apply quality filters with Polars (vectorised)
    relevant = relevant.filter(
        pl.col("similar_books").is_not_null()
        & (pl.col("similar_books").list.len() >= min_similar_books)
        & pl.col("description").is_not_null()
        & (pl.col("description").str.len_chars() >= min_description_chars)
    )

    if relevant.height == 0:
        return []

    # Optionally join embedding texts.
    if embedding_texts_df is not None:
        relevant = relevant.join(
            embedding_texts_df.select(["book_id", "book_embedding_text"]),
            on="book_id",
            how="left",
        )
    else:
        relevant = relevant.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("book_embedding_text")
        )

    # Sample up to n_queries rows.
    sample_n = min(n_queries, relevant.height)
    sampled  = relevant.sample(sample_n, seed=rng_seed)

    queries: list[dict[str, Any]] = []
    for i, row in enumerate(sampled.iter_rows(named=True)):
        book_id = int(row["book_id"])
        emb_text = row.get("book_embedding_text")
        desc = row.get("description") or ""

        query_text = emb_text if (emb_text and len(emb_text) > 10) else desc[:300].strip()

        similar_ids = [int(x) for x in (row.get("similar_books") or [])[:20]]
        ground_truth = [book_id] + similar_ids

        shelves = [str(s).lower() for s in (row.get("popular_shelves") or [])[:5]]
        if any(kw in s for s in shelves for kw in ("fantasy", "sci-fi", "science-fiction", "horror")):
            query_type = "genre"
        elif any(kw in s for s in shelves for kw in ("romance", "mystery", "thriller", "crime")):
            query_type = "genre"
        elif row.get("authors"):
            query_type = "author_style"
        else:
            query_type = "description"

        queries.append({
            "query_id": f"temporal_{i:03d}",
            "query_text": query_text,
            "query_type": query_type,
            "source_book_id": book_id,
            "ground_truth_ids": ground_truth,
            "difficulty": "medium",
        })

    return queries


def build_and_save_temporal_benchmark(
    interactions_path: str | Path = INTERACTIONS_PATH,
    books_path: str | Path = BOOKS_PATH,
    embedding_texts_path: str | Path = EMBEDDING_TEXTS_PATH,
    output_path: str | Path = TEMPORAL_BENCHMARK_PATH,
    n_queries: int = 200,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Build temporal split, generate benchmark queries, and save to JSON."""
    print("TEMPORAL SPLIT: BENCHMARK GENERATION", flush=True)

    print("\nBuilding temporal split…", flush=True)
    split = build_temporal_split(interactions_path, val_frac=val_frac, test_frac=test_frac)
    summary = split.summary()

    print(f"Train: {summary['train_interactions']:>10,}  (up to {split.train_end_date})", flush=True)
    print(f"Val: {summary['val_interactions']:>10,}  (up to {split.val_end_date})",   flush=True)
    print(f"Test: {summary['test_interactions']:>10,}  (from {split.test_start_date})", flush=True)
    print(f"\nTrain users : {summary['train_unique_users']:,}  |  "
          f"Val users: {summary['val_unique_users']:,}  |  "
          f"Test users: {summary['test_unique_users']:,}", flush=True)
    print(f"Train books: {summary['train_unique_books']:,}  |  "
          f"Val books: {summary['val_unique_books']:,}  |  "
          f"Test books: {summary['test_unique_books']:,}", flush=True)

    print("\nLoading book metadata…", flush=True)
    books_df = pl.read_parquet(books_path)
    print(f" {books_df.height:,} books loaded", flush=True)

    emb_texts_df: pl.DataFrame | None = None
    emb_path = Path(embedding_texts_path)
    if emb_path.exists():
        emb_texts_df = pl.read_parquet(emb_path)
        print(f"{emb_texts_df.height:,} embedding texts loaded", flush=True)
    else:
        print(f"Embedding texts not found at {emb_path}; using description excerpts", flush=True)

    print(f"\nGenerating {n_queries} evaluation queries from test split…", flush=True)
    queries = make_retrieval_eval_queries(
        split.test, books_df, emb_texts_df,
        n_queries=n_queries, rng_seed=rng_seed,
    )
    print(f"Generated {len(queries)} queries", flush=True)

    type_counts: dict[str, int] = {}
    for q in queries:
        type_counts[q["query_type"]] = type_counts.get(q["query_type"], 0) + 1
    for qtype, count in sorted(type_counts.items()):
        print(f"{qtype}: {count}", flush=True)

    benchmark = {
        "metadata": {
            "type":"temporal_split",
            "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "total_queries": len(queries),
            "rng_seed":rng_seed,
            "val_frac":val_frac,
            "test_frac":test_frac,
        },
        "split_summary": summary,
        "queries": queries,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(benchmark, fh, indent=2)
    print(f"\nSaved to {out}", flush=True)
    print("=" * 60, flush=True)
    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build temporal benchmark dataset")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(TEMPORAL_BENCHMARK_PATH))
    args = parser.parse_args()

    try:
        build_and_save_temporal_benchmark(
            n_queries=args.n_queries,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            rng_seed=args.seed,
            output_path=args.output,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}", flush=True)
        print("Make sure you are running from the project root and data/ is populated.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nUnexpected error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
