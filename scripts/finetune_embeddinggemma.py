from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import polars as pl
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers

try:
    from unsloth import FastSentenceTransformer, is_bf16_supported

    UNSLOTH_AVAILABLE = True
    UNSLOTH_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    FastSentenceTransformer = None

    def is_bf16_supported() -> bool:
        return False

    UNSLOTH_AVAILABLE = False
    UNSLOTH_IMPORT_ERROR = exc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_path = project_root / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Login to HuggingFace Hub if token is available
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    logger.info("Logged in to HuggingFace Hub")


# Verify MLflow environment variables are set (without exposing secrets)
def _check_mlflow_env() -> bool:
    """Check that required MLflow environment variables are set."""
    required_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_S3_ENDPOINT_URL",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.warning(f"Missing MLflow environment variables: {missing}")
        logger.warning(f"Make sure .env file exists at {env_path}")
    return len(missing) == 0

standardized_books_path = os.path.join(
    project_root, "data", "3_goodreads_books_with_metrics.parquet"
)
books_texts_path = os.path.join(project_root, "data", "books_embedding_texts.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finetune embedding models with Unsloth + SentenceTransformers "
            "MultipleNegativesRankingLoss."
        )
    )

    parser.add_argument("--raw-books-source", default=standardized_books_path)
    parser.add_argument("--book-texts-source", default=books_texts_path)
    parser.add_argument("--download-inputs", action="store_true")
    parser.add_argument("--cache-dir", default="data/cache")

    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--min-text-chars", type=int, default=120)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)

    # Book selection parameters
    parser.add_argument(
        "--top-popular-books",
        type=int,
        default=10000,
        help="Number of top popular books (by num_interactions) to include",
    )
    parser.add_argument(
        "--random-sampled-books",
        type=int,
        default=10000,
        help="Number of randomly sampled books (excluding top popular) to include",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max optimizer steps (-1 uses full epochs).",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=10.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient-checkpointing-mode",
        default="unsloth",
        choices=["unsloth", "torch"],
        help='Checkpointing backend when enabled. "unsloth" is most memory efficient.',
    )
    parser.add_argument(
        "--full-finetuning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train full model params instead of LoRA adapters.",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-bias", default="none")
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA.",
    )
    parser.add_argument(
        "--use-rslora",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lora-random-state", type=int, default=3407)
    parser.add_argument(
        "--save-merged-16bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save merged 16-bit model artifact for deployment.",
    )
    parser.add_argument(
        "--merged-save-method",
        default="merged_16bit",
        help="Unsloth merged save method (for save_pretrained_merged).",
    )
    parser.add_argument("--num-workers", type=int, default=-1)

    parser.add_argument(
        "--model-name", default="google/embeddinggemma-300m", help="HF model id"
    )
    parser.add_argument("--output-dir", default="data/models/embeddinggemma_mnrl")

    parser.add_argument(
        "--eval-max-queries",
        type=int,
        default=-1,
        help="Max queries to evaluate (-1 = auto, use all validation pairs)",
    )
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument(
        "--eval-query-batch-size",
        type=int,
        default=0,
        help="Queries scored per similarity batch (0 = auto)",
    )
    parser.add_argument(
        "--eval-similarity-matrix-mb",
        type=int,
        default=256,
        help="Target max memory (MB) for each query x corpus similarity matrix",
    )
    parser.add_argument(
        "--eval-baseline",
        action="store_true",
        default=True,
        help="Evaluate baseline model for comparison (default: True)",
    )
    parser.add_argument(
        "--no-eval-baseline",
        action="store_false",
        dest="eval_baseline",
        help="Skip baseline model evaluation",
    )

    # MLflow settings
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.yousef.gg"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="EmbeddingGemma_Finetuning",
        help="MLflow experiment name",
    )

    return parser.parse_args()


def maybe_download(url: str, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    with urllib.request.urlopen(url) as response, dest_path.open("wb") as out_file:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out_file.write(chunk)
    return dest_path


def resolve_source(path_or_url: str, download_inputs: bool, cache_dir: Path) -> str:
    parsed = urllib.parse.urlparse(path_or_url)
    is_remote = parsed.scheme in {"http", "https"}
    if not download_inputs or not is_remote:
        return path_or_url

    filename = Path(parsed.path).name or "input.parquet"
    local_path = cache_dir / filename
    maybe_download(path_or_url, local_path)
    return str(local_path)


def detect_runtime(num_workers_arg: int) -> dict[str, object]:
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        pin_memory = True
        prefetch_factor = 4
        auto_workers = min(16, os.cpu_count() or 1)
        eval_batch_size = 256
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        pin_memory = False
        prefetch_factor = 2
        auto_workers = 0
        eval_batch_size = 64
    else:
        device = "cpu"
        pin_memory = False
        prefetch_factor = 2
        auto_workers = min(8, os.cpu_count() or 1)
        eval_batch_size = 64

    num_workers = auto_workers if num_workers_arg < 0 else num_workers_arg
    bf16 = device == "cuda" and bool(is_bf16_supported())
    fp16 = device == "cuda" and not bf16

    return {
        "device": device,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
        "num_workers": num_workers,
        "default_eval_batch_size": eval_batch_size,
        "bf16": bf16,
        "fp16": fp16,
    }


def build_training_frames(args: argparse.Namespace, raw_source: str, text_source: str):
    # Load books with num_interactions for filtering
    raw_books_lf = pl.scan_parquet(raw_source).select(
        ["book_id", "similar_books", "num_interactions"]
    )
    book_texts_lf = pl.scan_parquet(text_source).select(["book_id", "book_embedding_text"])

    valid_texts_lf = (
        book_texts_lf.with_columns(
            pl.col("book_id").cast(pl.Utf8),
            pl.col("book_embedding_text")
            .fill_null("")
            .str.strip_chars()
            .alias("book_embedding_text"),
        )
        .filter(pl.col("book_embedding_text").str.len_chars() >= args.min_text_chars)
        .select(["book_id", "book_embedding_text"])
    )

    valid_book_ids_lf = valid_texts_lf.select("book_id").unique()

    # Filter books by popularity and random sampling
    # Get valid book_ids with their num_interactions
    books_with_interactions_lf = (
        raw_books_lf.select(["book_id", "num_interactions"])
        .with_columns(pl.col("book_id").cast(pl.Utf8))
        .join(valid_book_ids_lf, on="book_id", how="inner")
        .with_columns(
            pl.col("num_interactions").fill_null(0)
        )  # Handle nulls by defaulting to 0
    )

    # Get top X popular books (by num_interactions)
    top_popular_books_lf = books_with_interactions_lf.sort(
        "num_interactions", descending=True
    ).limit(args.top_popular_books)

    # Get the remaining books (excluding top popular)
    remaining_books_lf = books_with_interactions_lf.join(
        top_popular_books_lf.select("book_id"), on="book_id", how="anti"
    )

    # Randomly sample Y books from the remaining
    # Use a hash-based approach for reproducible random sampling
    remaining_count = remaining_books_lf.select(pl.len().alias("n")).collect().item()
    random_sample_size = min(args.random_sampled_books, remaining_count)

    if random_sample_size > 0:
        random_sampled_books_lf = (
            remaining_books_lf.with_columns(
                pl.col("book_id").hash(seed=int(args.seed)).alias("_sample_key")
            )
            .sort("_sample_key")
            .limit(random_sample_size)
            .drop("_sample_key")
        )
    else:
        random_sampled_books_lf = remaining_books_lf.limit(0)  # Empty dataframe

    # Combine selected book_ids
    selected_book_ids_lf = pl.concat(
        [top_popular_books_lf.select("book_id"), random_sampled_books_lf.select("book_id")]
    ).unique("book_id")

    # Collect to get count for logging
    selected_book_count = selected_book_ids_lf.select(pl.len().alias("n")).collect().item()
    logger.info(f"Selected {selected_book_count} books for training:")
    logger.info(f"  - Top popular: {args.top_popular_books}")
    logger.info(f"  - Random sampled: {random_sample_size}")

    # Filter pairs to only include books from the selected set
    candidate_pairs_lf = (
        raw_books_lf.filter(pl.col("similar_books").list.len() > 0)
        .explode("similar_books")
        .rename({"book_id": "source_book_id", "similar_books": "target_book_id"})
        .with_columns(
            pl.col("source_book_id").cast(pl.Utf8),
            pl.col("target_book_id").cast(pl.Utf8),
        )
        .filter(
            pl.col("target_book_id").is_not_null() & (pl.col("target_book_id") != "")
        )
        .filter(pl.col("source_book_id") != pl.col("target_book_id"))
        # Filter to only include pairs where both books are in the selected set
        .join(
            selected_book_ids_lf.rename({"book_id": "source_book_id"}),
            on="source_book_id",
            how="inner",
        )
        .join(
            selected_book_ids_lf.rename({"book_id": "target_book_id"}),
            on="target_book_id",
            how="inner",
        )
        .with_columns(
            [
                pl.min_horizontal("source_book_id", "target_book_id").alias(
                    "book_id_left"
                ),
                pl.max_horizontal("source_book_id", "target_book_id").alias(
                    "book_id_right"
                ),
            ]
        )
        .group_by(["book_id_left", "book_id_right"])
        .agg(pl.len().alias("pair_support"))
        .filter(pl.col("pair_support") >= args.min_support)
    )

    candidate_pair_count = candidate_pairs_lf.select(pl.len().alias("n")).collect().item()
    if candidate_pair_count == 0:
        raise RuntimeError("No candidate pairs found. Lower min_support/min_text_chars.")

    sample_size = min(args.max_pairs, candidate_pair_count)
    if sample_size < candidate_pair_count:
        sampled_pairs_lf = (
            candidate_pairs_lf.with_columns(
                pl.struct(["book_id_left", "book_id_right"])
                .hash(seed=int(args.seed))
                .alias("_sample_key")
            )
            .sort("_sample_key")
            .limit(sample_size)
            .drop("_sample_key")
        )
    else:
        sampled_pairs_lf = candidate_pairs_lf

    sampled_pairs_df = sampled_pairs_lf.collect()

    parent: dict[str, str] = {}
    rank: dict[str, int] = {}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while node != root:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(a: str, b: str) -> None:
        if a not in parent:
            parent[a] = a
            rank[a] = 0
        if b not in parent:
            parent[b] = b
            rank[b] = 0

        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for left_id, right_id in sampled_pairs_df.select(
        ["book_id_left", "book_id_right"]
    ).iter_rows():
        union(left_id, right_id)

    root_to_component: dict[str, int] = {}
    pair_component_ids: list[int] = []
    next_component = 0

    for left_id in sampled_pairs_df["book_id_left"].to_list():
        root = find(left_id)
        if root not in root_to_component:
            root_to_component[root] = next_component
            next_component += 1
        pair_component_ids.append(root_to_component[root])

    pairs_with_components_df = sampled_pairs_df.with_columns(
        pl.Series(name="component_id", values=pair_component_ids)
    )

    split_component_ids = pairs_with_components_df["component_id"].unique().to_list()
    rng = random.Random(int(args.seed))
    rng.shuffle(split_component_ids)

    val_ratio = max(0.0, min(100.0, args.val_fraction)) / 100.0
    val_component_count = int(len(split_component_ids) * val_ratio)
    if val_ratio > 0 and len(split_component_ids) > 1 and val_component_count == 0:
        val_component_count = 1

    val_component_ids = set(split_component_ids[:val_component_count])
    train_pairs_df = pairs_with_components_df.filter(
        ~pairs_with_components_df["component_id"].is_in(val_component_ids)
    )
    val_pairs_df = pairs_with_components_df.filter(
        pairs_with_components_df["component_id"].is_in(val_component_ids)
    )

    if train_pairs_df.height == 0:
        raise RuntimeError(
            "No training pairs after split. Increase max_pairs or lower val_fraction."
        )

    left_text_lf = valid_texts_lf.rename(
        {"book_id": "book_id_left", "book_embedding_text": "anchor_text"}
    )
    right_text_lf = valid_texts_lf.rename(
        {"book_id": "book_id_right", "book_embedding_text": "positive_text"}
    )

    train_pairs_text_df = (
        train_pairs_df.lazy()
        .join(left_text_lf, on="book_id_left", how="inner")
        .join(right_text_lf, on="book_id_right", how="inner")
        .select(
            [
                "book_id_left",
                "book_id_right",
                "pair_support",
                "component_id",
                "anchor_text",
                "positive_text",
            ]
        )
        .collect()
    )

    val_pairs_text_df = (
        val_pairs_df.lazy()
        .join(left_text_lf, on="book_id_left", how="inner")
        .join(right_text_lf, on="book_id_right", how="inner")
        .select(
            [
                "book_id_left",
                "book_id_right",
                "pair_support",
                "component_id",
                "anchor_text",
                "positive_text",
            ]
        )
        .collect()
    )

    stats = {
        "candidate_pair_count": candidate_pair_count,
        "sampled_pair_count": sample_size,
        "component_count": len(split_component_ids),
        "train_pairs": train_pairs_text_df.height,
        "val_pairs": val_pairs_text_df.height,
        "train_components": train_pairs_text_df["component_id"].n_unique(),
        "val_components": len(val_component_ids),
        "selected_books_count": selected_book_count,
        "top_popular_books": args.top_popular_books,
        "random_sampled_books": random_sample_size,
    }

    return train_pairs_text_df, val_pairs_text_df, stats


def build_eval_graph(
    val_pairs_text_df: pl.DataFrame,
    seed: int,
) -> tuple[list[str], dict[str, set[str]], dict[str, str]]:
    eval_neighbors_by_id: dict[str, set[str]] = defaultdict(set)
    eval_text_by_id: dict[str, str] = {}

    for left_id, right_id, anchor_text, positive_text in val_pairs_text_df.select(
        ["book_id_left", "book_id_right", "anchor_text", "positive_text"]
    ).iter_rows():
        eval_neighbors_by_id[left_id].add(right_id)
        eval_neighbors_by_id[right_id].add(left_id)
        if left_id not in eval_text_by_id:
            eval_text_by_id[left_id] = anchor_text
        if right_id not in eval_text_by_id:
            eval_text_by_id[right_id] = positive_text

    eval_candidate_ids = [
        book_id for book_id in eval_text_by_id if len(eval_neighbors_by_id[book_id]) > 0
    ]
    eval_rng = random.Random(seed)
    eval_rng.shuffle(eval_candidate_ids)
    return eval_candidate_ids, eval_neighbors_by_id, eval_text_by_id


def build_ir_evaluator(
    candidate_ids: list[str],
    neighbors_by_id: dict[str, set[str]],
    text_by_id: dict[str, str],
    batch_size: int,
) -> Optional[InformationRetrievalEvaluator]:
    if not candidate_ids:
        return None

    candidate_set = set(candidate_ids)
    queries = {book_id: text_by_id[book_id] for book_id in candidate_ids}
    corpus = dict(queries)
    relevant_docs = {
        book_id: sorted(neighbors_by_id[book_id].intersection(candidate_set))
        for book_id in candidate_ids
    }
    relevant_docs = {k: v for k, v in relevant_docs.items() if v}
    if not relevant_docs:
        return None

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=False,
        batch_size=batch_size,
        name="validation_ir",
    )


def evaluate_retrieval_model(
    model: SentenceTransformer,
    candidate_ids: list[str],
    neighbors_by_id: dict[str, set[str]],
    text_by_id: dict[str, str],
    max_queries: int,
    k: int,
    encode_batch_size: int,
    query_batch_size: int,
    similarity_matrix_mb: int,
) -> dict[str, float]:
    def empty_metrics(corpus_size: int) -> dict[str, float]:
        return {
            "queries_evaluated": 0,
            "corpus_size": int(corpus_size),
            "k": int(k),
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "mean_first_positive_rank": 0.0,
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
            "precision_at_k": 0.0,
            "hit_rate_at_k": 0.0,
        }

    corpus_embeddings: Optional[np.ndarray] = None
    similarities: Optional[np.ndarray] = None
    top_indices: Optional[np.ndarray] = None

    try:
        corpus_ids = list(candidate_ids)
        if not corpus_ids:
            return empty_metrics(corpus_size=0)

        corpus_texts = [text_by_id[book_id] for book_id in corpus_ids]
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        # Keep evaluation arrays compact and contiguous.
        corpus_embeddings = np.asarray(corpus_embeddings, dtype=np.float32, order="C")

        top_k = min(k, len(corpus_ids) - 1)
        if top_k <= 0:
            return empty_metrics(corpus_size=len(corpus_ids))

        id_to_index = {book_id: idx for idx, book_id in enumerate(corpus_ids)}

        # max_queries < 0 => use all queries.
        query_count = len(corpus_ids) if max_queries < 0 else min(max_queries, len(corpus_ids))
        query_ids = corpus_ids[:query_count]

        eval_targets: list[tuple[int, np.ndarray, int]] = []
        for query_id in query_ids:
            relevant_indices = np.fromiter(
                (
                    id_to_index[neighbor_id]
                    for neighbor_id in neighbors_by_id[query_id]
                    if neighbor_id in id_to_index
                ),
                dtype=np.int32,
            )
            if relevant_indices.size == 0:
                continue
            eval_targets.append((id_to_index[query_id], relevant_indices, int(relevant_indices.size)))

        if not eval_targets:
            return empty_metrics(corpus_size=len(corpus_ids))

        # Bound per-batch memory for the [query_batch, corpus_size] similarity matrix.
        if query_batch_size > 0:
            actual_query_batch_size = query_batch_size
        else:
            target_bytes = max(1, int(similarity_matrix_mb)) * 1024 * 1024
            bytes_per_row = max(1, len(corpus_ids) * np.dtype(np.float32).itemsize)
            actual_query_batch_size = max(1, target_bytes // bytes_per_row)
            actual_query_batch_size = min(actual_query_batch_size, 512)

        estimated_similarity_mb = (
            actual_query_batch_size
            * len(corpus_ids)
            * np.dtype(np.float32).itemsize
        ) / (1024 * 1024)
        if estimated_similarity_mb > max(1, int(similarity_matrix_mb)):
            logger.warning(
                "Estimated similarity matrix memory (%.2f MB) exceeds target budget (%s MB). "
                "Consider lowering --eval-query-batch-size.",
                estimated_similarity_mb,
                similarity_matrix_mb,
            )

        logger.info(
            "Evaluation config: corpus_size=%s query_count=%s scored_queries=%s top_k=%s "
            "encode_batch_size=%s query_batch_size=%s similarity_matrix_mb=%s "
            "estimated_similarity_mb=%.2f",
            len(corpus_ids),
            query_count,
            len(eval_targets),
            top_k,
            encode_batch_size,
            actual_query_batch_size,
            similarity_matrix_mb,
            estimated_similarity_mb,
        )

        recall_hits = 0
        mrr_sum = 0.0
        first_rank_sum = 0.0
        evaluated_queries = 0
        ndcg_sum = 0.0
        map_sum = 0.0
        precision_sum = 0.0
        hit_rate_hits = 0
        discount_factors = 1.0 / np.log2(np.arange(2, top_k + 2, dtype=np.float64))

        for start in range(0, len(eval_targets), actual_query_batch_size):
            batch_targets = eval_targets[start : start + actual_query_batch_size]
            batch_query_indices = np.fromiter(
                (target[0] for target in batch_targets),
                dtype=np.int32,
                count=len(batch_targets),
            )
            batch_query_embeddings = corpus_embeddings[batch_query_indices]
            similarities = batch_query_embeddings @ corpus_embeddings.T
            similarities[np.arange(len(batch_targets)), batch_query_indices] = -np.inf

            if top_k < similarities.shape[1]:
                top_indices = np.argpartition(-similarities, top_k - 1, axis=1)[:, :top_k]
                top_scores = np.take_along_axis(similarities, top_indices, axis=1)
                ranking_order = np.argsort(-top_scores, axis=1)
                top_indices = np.take_along_axis(top_indices, ranking_order, axis=1)
            else:
                top_indices = np.argsort(-similarities, axis=1)

            for row_idx, (_, relevant_indices, relevant_count) in enumerate(batch_targets):
                retrieved_indices = top_indices[row_idx, :top_k]
                relevance_hits = np.isin(retrieved_indices, relevant_indices, assume_unique=False)

                hits_in_top_k = int(np.count_nonzero(relevance_hits))
                recall_hits += int(hits_in_top_k > 0)
                hit_rate_hits += int(hits_in_top_k > 0)
                precision_sum += hits_in_top_k / top_k

                # Rank of first relevant document over the full corpus.
                row_similarities = similarities[row_idx]
                best_relevant_similarity = float(np.max(row_similarities[relevant_indices]))
                first_relevant_rank = int(np.count_nonzero(row_similarities > best_relevant_similarity) + 1)
                mrr_sum += 1.0 / first_relevant_rank
                first_rank_sum += first_relevant_rank

                # NDCG@k (binary relevance).
                dcg = float(np.sum(discount_factors[relevance_hits]))
                ideal_dcg = float(np.sum(discount_factors[: min(relevant_count, top_k)]))
                ndcg_sum += dcg / ideal_dcg if ideal_dcg > 0 else 0.0

                # AP@k contribution to MAP@k.
                if hits_in_top_k > 0:
                    hit_ranks = np.flatnonzero(relevance_hits) + 1
                    precision_at_hits = np.arange(1, hits_in_top_k + 1, dtype=np.float64) / hit_ranks
                    ap = float(np.sum(precision_at_hits) / relevant_count)
                else:
                    ap = 0.0
                map_sum += ap

                evaluated_queries += 1

            similarities = None
            top_indices = None

        if evaluated_queries == 0:
            return empty_metrics(corpus_size=len(corpus_ids))

        return {
            "queries_evaluated": int(evaluated_queries),
            "corpus_size": int(len(corpus_ids)),
            "k": int(k),
            "recall_at_k": float(recall_hits / evaluated_queries),
            "mrr": float(mrr_sum / evaluated_queries),
            "mean_first_positive_rank": float(first_rank_sum / evaluated_queries),
            "ndcg_at_k": float(ndcg_sum / evaluated_queries),
            "map_at_k": float(map_sum / evaluated_queries),
            "precision_at_k": float(precision_sum / evaluated_queries),
            "hit_rate_at_k": float(hit_rate_hits / evaluated_queries),
        }
    finally:
        corpus_embeddings = None
        similarities = None
        top_indices = None
        gc.collect()


def log_to_mlflow(
    args: argparse.Namespace,
    runtime: dict[str, object],
    stats: dict[str, int],
    training_time: float,
    finetuned_metrics: Optional[Dict[str, float]],
    baseline_metrics: Optional[Dict[str, float]],
    output_path: Path,
    trainer_metrics: Optional[Dict[str, float]] = None,
    dataset_paths: Optional[list[Path]] = None,
    run_name: Optional[str] = None,
) -> str:
    """
    Log all training artifacts to MLFlow.

    Args:
        args: Command line arguments
        runtime: Runtime configuration dictionary
        stats: Data statistics dictionary
        training_time: Training time in seconds
        finetuned_metrics: Evaluation metrics for finetuned model (can be None)
        baseline_metrics: Evaluation metrics for baseline model (can be None)
        output_path: Path to saved model
        dataset_paths: Optional list of dataset file paths to log as artifacts
        run_name: Optional run name

    Returns:
        MLflow run ID
    """
    logger.info("Logging to MLFlow...")

    # Set up MLFlow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log model hyperparameters
        mlflow.log_params(
            {
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler_type": args.lr_scheduler_type,
                "max_seq_length": args.max_seq_length,
                "gradient_checkpointing": args.gradient_checkpointing,
                "gradient_checkpointing_mode": args.gradient_checkpointing_mode,
                "seed": args.seed,
                "full_finetuning": args.full_finetuning,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_bias": args.lora_bias,
                "lora_target_modules": args.lora_target_modules,
                "use_rslora": args.use_rslora,
                "lora_random_state": args.lora_random_state,
                "save_merged_16bit": args.save_merged_16bit,
                "merged_save_method": args.merged_save_method,
            }
        )

        # Log data processing parameters
        mlflow.log_params({
            "max_pairs": args.max_pairs,
            "min_text_chars": args.min_text_chars,
            "min_support": args.min_support,
            "val_fraction": args.val_fraction,
            "top_popular_books": args.top_popular_books,
            "random_sampled_books": args.random_sampled_books,
        })

        # Log evaluation parameters
        mlflow.log_params(
            {
                "eval_max_queries": args.eval_max_queries,
                "eval_k": args.eval_k,
                "eval_batch_size": args.eval_batch_size,
                "eval_query_batch_size": args.eval_query_batch_size,
                "eval_similarity_matrix_mb": args.eval_similarity_matrix_mb,
                "eval_baseline": args.eval_baseline,
                "trainer_eval_steps": args.eval_steps,
                "trainer_logging_steps": args.logging_steps,
                "trainer_save_steps": args.save_steps,
                "trainer_save_total_limit": args.save_total_limit,
            }
        )

        # Log runtime configuration
        mlflow.log_params(
            {
                "device": runtime["device"],
                "num_workers": runtime["num_workers"],
                "bf16": runtime["bf16"],
                "fp16": runtime["fp16"],
            }
        )

        # Log dataset statistics
        mlflow.log_params(
            {
                "candidate_pair_count": stats["candidate_pair_count"],
                "sampled_pair_count": stats["sampled_pair_count"],
                "component_count": stats["component_count"],
                "train_pairs": stats["train_pairs"],
                "val_pairs": stats["val_pairs"],
                "train_components": stats["train_components"],
                "val_components": stats["val_components"],
                "selected_books_count": stats["selected_books_count"],
            }
        )

        # Log training time
        mlflow.log_metric("training_time_seconds", training_time)
        if trainer_metrics:
            for metric_name, metric_value in trainer_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"trainer_{metric_name}", float(metric_value))

        # Log baseline evaluation metrics if available
        if baseline_metrics:
            for metric_name, metric_value in baseline_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"baseline_{metric_name}", float(metric_value))

        # Log finetuned evaluation metrics if available
        if finetuned_metrics:
            for metric_name, metric_value in finetuned_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"finetuned_{metric_name}", float(metric_value))

        # Calculate and log improvement deltas (finetuned - baseline)
        if baseline_metrics and finetuned_metrics:
            improvement_metrics = {}
            for metric_name in baseline_metrics.keys():
                if metric_name in finetuned_metrics:
                    baseline_val = baseline_metrics[metric_name]
                    finetuned_val = finetuned_metrics[metric_name]
                    if isinstance(baseline_val, (int, float)) and isinstance(finetuned_val, (int, float)):
                        # Absolute improvement
                        improvement = finetuned_val - baseline_val
                        mlflow.log_metric(f"improvement_{metric_name}", float(improvement))
                        improvement_metrics[metric_name] = {
                            "baseline": baseline_val,
                            "finetuned": finetuned_val,
                            "improvement": improvement,
                        }

            logger.info("=== Improvement Summary ===")
            for metric_name, values in improvement_metrics.items():
                logger.info(f"  {metric_name}: {values['baseline']:.4f} -> {values['finetuned']:.4f} (Δ={values['improvement']:+.4f})")

        # Create artifacts directory
        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        def safe_log_artifact(
            path: Path,
            artifact_path: Optional[str] = None,
            description: Optional[str] = None,
        ) -> None:
            """Safely log an artifact, handling missing boto3 for S3 backends.

            Args:
                path: Path to the artifact file
                artifact_path: Subdirectory in MLFlow (e.g., "model", "dataset")
                description: Optional description for logging
            """
            if not path.exists():
                logger.warning(f"Artifact path does not exist, skipping: {path}")
                return

            upload_start = time.time()
            size_mb = path.stat().st_size / (1024 * 1024)
            artifact_label = description or str(path)
            logger.info(
                f"Uploading artifact: {artifact_label} ({size_mb:.2f} MB) to "
                f"{artifact_path or 'root'}"
            )
            try:
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
                upload_time = time.time() - upload_start
                logger.info(
                    f"Logged artifact: {artifact_label} in {upload_time:.2f} seconds"
                )
            except Exception as e:
                logger.warning(f"Could not log artifact {path}: {e}")
                logger.warning("Artifact saved locally but not uploaded to MLFlow")

        def safe_log_artifact_dir(
            path: Path,
            artifact_path: Optional[str] = None,
            description: Optional[str] = None,
        ) -> None:
            if not path.exists() or not path.is_dir():
                logger.warning(f"Artifact directory does not exist, skipping: {path}")
                return

            artifact_label = description or str(path)
            logger.info(
                "Uploading artifact directory: %s to %s",
                artifact_label,
                artifact_path or "root",
            )
            try:
                mlflow.log_artifacts(str(path), artifact_path=artifact_path)
                logger.info("Logged artifact directory: %s", artifact_label)
            except Exception as e:
                logger.warning(f"Could not log artifact directory {path}: {e}")
                logger.warning("Artifacts saved locally but not uploaded to MLFlow")

        # Log the saved model directory
        if output_path.exists():
            safe_log_artifact_dir(
                output_path,
                artifact_path="model",
                description="model_output",
            )

        # Log dataset files if provided
        if dataset_paths:
            for dataset_path in dataset_paths:
                if dataset_path.exists():
                    safe_log_artifact(
                        dataset_path,
                        artifact_path="dataset",
                        description=f"dataset/{dataset_path.name}"
                    )

        # Save training configuration
        config_path = artifacts_dir / "training_config.json"
        config_data = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "max_seq_length": args.max_seq_length,
            "gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_mode": args.gradient_checkpointing_mode,
            "full_finetuning": args.full_finetuning,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_bias": args.lora_bias,
            "lora_target_modules": args.lora_target_modules,
            "use_rslora": args.use_rslora,
            "seed": args.seed,
            "device": runtime["device"],
            "bf16": runtime["bf16"],
            "fp16": runtime["fp16"],
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        safe_log_artifact(config_path, artifact_path="config", description="config/training_config.json")

        # Save data statistics
        stats_path = artifacts_dir / "data_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        safe_log_artifact(stats_path, artifact_path="stats", description="stats/data_stats.json")

        # Save baseline evaluation metrics if available
        if baseline_metrics:
            baseline_metrics_path = artifacts_dir / "baseline_metrics.json"
            with open(baseline_metrics_path, "w") as f:
                json.dump(baseline_metrics, f, indent=2)
            safe_log_artifact(baseline_metrics_path, artifact_path="metrics", description="metrics/baseline_metrics.json")

        # Save finetuned evaluation metrics if available
        if finetuned_metrics:
            finetuned_metrics_path = artifacts_dir / "finetuned_metrics.json"
            with open(finetuned_metrics_path, "w") as f:
                json.dump(finetuned_metrics, f, indent=2)
            safe_log_artifact(finetuned_metrics_path, artifact_path="metrics", description="metrics/finetuned_metrics.json")

        # Save comparison summary if both are available
        if baseline_metrics and finetuned_metrics:
            comparison = {
                "baseline": baseline_metrics,
                "finetuned": finetuned_metrics,
            }
            comparison_path = artifacts_dir / "metrics_comparison.json"
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)
            safe_log_artifact(comparison_path, artifact_path="metrics", description="metrics/metrics_comparison.json")

        # Log the training script
        safe_log_artifact(Path(__file__), artifact_path="code", description="code/finetune_embeddinggemma.py")

        # Log tags
        mlflow.set_tags(
            {
                "model_type": "EmbeddingGemma",
                "framework": "unsloth+sentence-transformers",
                "training_date": datetime.now().isoformat(),
                "device": str(runtime["device"]),
            }
        )

        # Get run ID safely
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else "unknown"
        logger.info(f"MLFlow run ID: {run_id}")

        # Clean up artifacts directory
        for artifact in artifacts_dir.glob("*"):
            if artifact.is_file():
                artifact.unlink()
        if artifacts_dir.exists():
            artifacts_dir.rmdir()

        return run_id


def main() -> None:
    # Check MLflow environment before starting
    _check_mlflow_env()
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError(
            "Unsloth is required but could not be imported. "
            "Install it in this environment, then rerun.\n"
            f"Import error: {UNSLOTH_IMPORT_ERROR}"
        )

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    runtime = detect_runtime(args.num_workers)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    raw_source = resolve_source(args.raw_books_source, args.download_inputs, cache_dir)
    text_source = resolve_source(args.book_texts_source, args.download_inputs, cache_dir)

    print("=== Runtime ===")
    print(
        f"device={runtime['device']} bf16={runtime['bf16']} fp16={runtime['fp16']}"
    )
    print(
        f"num_workers={runtime['num_workers']} pin_memory={runtime['pin_memory']} prefetch_factor={runtime['prefetch_factor']}"
    )

    train_pairs_text_df, val_pairs_text_df, stats = build_training_frames(
        args, raw_source, text_source
    )
    print("=== Data stats ===")
    print(json.dumps(stats, indent=2))

    # Determine evaluation batch sizes.
    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size > 0
        else int(runtime["default_eval_batch_size"])
    )

    # Determine if we should run evaluation
    # eval_max_queries == -1 means "auto" (evaluate if validation pairs exist)
    # eval_max_queries == 0 means "skip evaluation"
    # eval_max_queries > 0 means "evaluate with specific query limit"
    should_run_eval = args.eval_max_queries != 0 and val_pairs_text_df.height > 0

    candidate_ids: list[str] = []
    neighbors_by_id: dict[str, set[str]] = {}
    text_by_id: dict[str, str] = {}
    if should_run_eval:
        candidate_ids, neighbors_by_id, text_by_id = build_eval_graph(
            val_pairs_text_df=val_pairs_text_df,
            seed=int(args.seed),
        )

    logger.info("Loading model with Unsloth...")
    model = FastSentenceTransformer.from_pretrained(
        model_name=args.model_name,
        max_seq_length=int(args.max_seq_length),
        full_finetuning=bool(args.full_finetuning),
    )
    model.max_seq_length = int(args.max_seq_length)

    baseline_metrics: Optional[Dict[str, float]] = None
    if should_run_eval and candidate_ids and args.eval_baseline:
        logger.info("=== Evaluating Baseline Model ===")
        baseline_eval_start = time.time()
        baseline_metrics = evaluate_retrieval_model(
            model=model,
            candidate_ids=candidate_ids,
            neighbors_by_id=neighbors_by_id,
            text_by_id=text_by_id,
            max_queries=int(args.eval_max_queries),
            k=int(args.eval_k),
            encode_batch_size=eval_batch_size,
            query_batch_size=int(args.eval_query_batch_size),
            similarity_matrix_mb=int(args.eval_similarity_matrix_mb),
        )
        baseline_eval_time = time.time() - baseline_eval_start
        logger.info("Baseline evaluation completed in %.2f seconds", baseline_eval_time)
        print("=== Baseline eval ===")
        print(json.dumps(baseline_metrics, indent=2))

    if not args.full_finetuning:
        target_modules = [
            module.strip()
            for module in str(args.lora_target_modules).split(",")
            if module.strip()
        ]
        if not target_modules:
            raise ValueError("At least one LoRA target module must be provided.")

        if args.gradient_checkpointing:
            gc_mode: bool | str = (
                "unsloth" if args.gradient_checkpointing_mode == "unsloth" else True
            )
        else:
            gc_mode = False

        logger.info("Attaching LoRA adapters with Unsloth...")
        model = FastSentenceTransformer.get_peft_model(
            model,
            r=int(args.lora_r),
            target_modules=target_modules,
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias=str(args.lora_bias),
            use_gradient_checkpointing=gc_mode,
            random_state=int(args.lora_random_state),
            use_rslora=bool(args.use_rslora),
            loftq_config=None,
            task_type="FEATURE_EXTRACTION",
        )

    print("=== Training config ===")
    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "output_dir": str(output_path),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "max_steps": int(args.max_steps),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "learning_rate": float(args.learning_rate),
                "warmup_ratio": float(args.warmup_ratio),
                "max_seq_length": int(args.max_seq_length),
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "gradient_checkpointing_mode": str(args.gradient_checkpointing_mode),
                "full_finetuning": bool(args.full_finetuning),
                "lora_r": int(args.lora_r),
                "lora_alpha": int(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "lora_bias": str(args.lora_bias),
                "lr_scheduler_type": str(args.lr_scheduler_type),
                "bf16": bool(runtime["bf16"]),
                "fp16": bool(runtime["fp16"]),
            },
            indent=2,
        )
    )

    train_dataset = Dataset.from_dict(
        {
            "anchor_text": train_pairs_text_df["anchor_text"].to_list(),
            "positive_text": train_pairs_text_df["positive_text"].to_list(),
        }
    )
    eval_dataset = None
    if val_pairs_text_df.height > 0:
        eval_dataset = Dataset.from_dict(
            {
                "anchor_text": val_pairs_text_df["anchor_text"].to_list(),
                "positive_text": val_pairs_text_df["positive_text"].to_list(),
            }
        )

    trainer_evaluator = None
    if should_run_eval and candidate_ids:
        trainer_evaluator = build_ir_evaluator(
            candidate_ids=candidate_ids,
            neighbors_by_id=neighbors_by_id,
            text_by_id=text_by_id,
            batch_size=eval_batch_size,
        )

    prompt_map: dict[str, str] = {}
    model_prompts = getattr(model, "prompts", None)
    if isinstance(model_prompts, dict):
        query_prompt = model_prompts.get("query")
        document_prompt = model_prompts.get("document")
        if isinstance(query_prompt, str):
            prompt_map["anchor_text"] = query_prompt
        if isinstance(document_prompt, str):
            prompt_map["positive_text"] = document_prompt

    eval_strategy = (
        "steps"
        if trainer_evaluator is not None and eval_dataset is not None
        else "no"
    )
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_path / "trainer"),
        "num_train_epochs": float(args.epochs),
        "max_steps": int(args.max_steps),
        "per_device_train_batch_size": int(args.batch_size),
        "per_device_eval_batch_size": int(eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "warmup_ratio": float(args.warmup_ratio) / 100.0,
        "logging_steps": int(args.logging_steps),
        "eval_steps": int(args.eval_steps),
        "save_strategy": "steps",
        "save_steps": max(1, int(args.save_steps)),
        "save_total_limit": max(1, int(args.save_total_limit)),
        "bf16": bool(runtime["bf16"]),
        "fp16": bool(runtime["fp16"]),
        "report_to": "none",
        "lr_scheduler_type": str(args.lr_scheduler_type),
        "dataloader_num_workers": int(runtime["num_workers"]),
        "dataloader_pin_memory": bool(runtime["pin_memory"]),
        "remove_unused_columns": False,
        "seed": int(args.seed),
    }
    training_args_signature = inspect.signature(
        SentenceTransformerTrainingArguments.__init__
    )
    if "eval_strategy" in training_args_signature.parameters:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy
    optional_kwargs = {
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "prompts": prompt_map,
    }
    for key, value in optional_kwargs.items():
        if key in training_args_signature.parameters:
            training_kwargs[key] = value
    # Keep only kwargs supported by the installed SentenceTransformers version.
    training_kwargs = {
        key: value
        for key, value in training_kwargs.items()
        if key in training_args_signature.parameters
    }

    training_args = SentenceTransformerTrainingArguments(**training_kwargs)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        args=training_args,
        evaluator=trainer_evaluator,
    )

    logger.info("Starting model training...")
    start_time = time.time()
    trainer_stats = trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    trainer_metrics = (
        dict(trainer_stats.metrics)
        if trainer_stats is not None and getattr(trainer_stats, "metrics", None)
        else {}
    )

    gpu_peak_reserved_gb = 0.0
    if torch.cuda.is_available():
        gpu_peak_reserved_gb = float(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
        )
        trainer_metrics["gpu_peak_reserved_gb"] = gpu_peak_reserved_gb

    lora_output_path = output_path / "lora"
    lora_output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving LoRA adapter model to %s", lora_output_path)
    model.save_pretrained(str(lora_output_path))
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.save_pretrained(str(lora_output_path))

    merged_output_path: Optional[Path] = None
    if args.save_merged_16bit:
        merged_output_path = output_path / "merged_16bit"
        merged_output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving merged model to %s", merged_output_path)
        try:
            model.save_pretrained_merged(
                str(merged_output_path),
                tokenizer=model.tokenizer,
                save_method=str(args.merged_save_method),
            )
        except Exception as exc:
            logger.warning("Could not save merged model artifact: %s", exc)
            merged_output_path = None

    deployment_manifest = {
        "base_model": args.model_name,
        "lora_model_path": str(lora_output_path),
        "merged_model_path": str(merged_output_path) if merged_output_path else None,
        "max_seq_length": int(args.max_seq_length),
        "unsloth_loader_example": [
            "from unsloth import FastSentenceTransformer",
            "model = FastSentenceTransformer.from_pretrained('PATH_TO_LORA_OR_MERGED_MODEL')",
            "embeddings = model.encode(['your text'], convert_to_numpy=True)",
        ],
        "sentence_transformers_loader_example": [
            "from sentence_transformers import SentenceTransformer",
            "model = SentenceTransformer('PATH_TO_ARTIFACT')",
            "embeddings = model.encode(['your text'], normalize_embeddings=True)",
        ],
    }
    deployment_manifest_path = output_path / "deployment_manifest.json"
    with deployment_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(deployment_manifest, f, indent=2)

    # Persist a checkpoint summary before evaluation so training metadata survives
    # even if evaluation or remote artifact logging fails.
    pre_eval_summary = {
        "args": vars(args),
        "runtime": runtime,
        "data_stats": stats,
        "training_time_seconds": training_time,
        "model_saved_to": str(output_path),
        "lora_model_path": str(lora_output_path),
        "merged_model_path": str(merged_output_path) if merged_output_path else None,
        "trainer_metrics": trainer_metrics,
        "summary_stage": "post_training_pre_evaluation",
        "timestamp": datetime.now().isoformat(),
    }
    pre_eval_summary_path = output_path / "run_summary_pre_eval.json"
    with pre_eval_summary_path.open("w", encoding="utf-8") as f:
        json.dump(pre_eval_summary, f, indent=2)
    logger.info(f"Saved pre-evaluation checkpoint summary to {pre_eval_summary_path}")

    finetuned_metrics: Optional[Dict[str, float]] = None
    if should_run_eval:
        if candidate_ids:
            # Evaluate finetuned model
            logger.info("=== Evaluating Finetuned Model ===")
            finetuned_eval_start = time.time()
            finetuned_metrics = evaluate_retrieval_model(
                model=model,
                candidate_ids=candidate_ids,
                neighbors_by_id=neighbors_by_id,
                text_by_id=text_by_id,
                max_queries=int(args.eval_max_queries),
                k=int(args.eval_k),
                encode_batch_size=eval_batch_size,
                query_batch_size=int(args.eval_query_batch_size),
                similarity_matrix_mb=int(args.eval_similarity_matrix_mb),
            )
            finetuned_eval_time = time.time() - finetuned_eval_start
            logger.info(f"Finetuned evaluation completed in {finetuned_eval_time:.2f} seconds")
            print("=== Finetuned eval ===")
            print(json.dumps(finetuned_metrics, indent=2))

            # Print comparison summary
            if baseline_metrics and finetuned_metrics:
                print("=== Improvement Summary ===")
                for metric_name in baseline_metrics.keys():
                    if metric_name in finetuned_metrics:
                        baseline_val = baseline_metrics[metric_name]
                        finetuned_val = finetuned_metrics[metric_name]
                        if isinstance(baseline_val, (int, float)) and isinstance(finetuned_val, (int, float)):
                            improvement = finetuned_val - baseline_val
                            print(f"  {metric_name}: {baseline_val:.4f} -> {finetuned_val:.4f} (Δ={improvement:+.4f})")

    # Log to MLFlow
    run_name = (
        f"embeddinggemma_unsloth_e{args.epochs}_bs{args.batch_size}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    dataset_paths = [Path(raw_source), Path(text_source)]
    logger.info("Starting MLflow logging...")
    mlflow_start = time.time()
    mlflow_run_id = log_to_mlflow(
        args=args,
        runtime=runtime,
        stats=stats,
        training_time=training_time,
        finetuned_metrics=finetuned_metrics,
        baseline_metrics=baseline_metrics,
        output_path=output_path,
        trainer_metrics=trainer_metrics,
        dataset_paths=dataset_paths,
        run_name=run_name,
    )
    mlflow_time = time.time() - mlflow_start
    logger.info(f"MLflow logging completed in {mlflow_time:.2f} seconds")

    run_summary = {
        "args": vars(args),
        "runtime": runtime,
        "data_stats": stats,
        "baseline_eval": baseline_metrics,
        "finetuned_eval": finetuned_metrics,
        "trainer_metrics": trainer_metrics,
        "lora_model_path": str(lora_output_path),
        "merged_model_path": str(merged_output_path) if merged_output_path else None,
        "deployment_manifest_path": str(deployment_manifest_path),
        "mlflow_run_id": mlflow_run_id,
        "training_time_seconds": training_time,
    }
    with (output_path / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info(f"MLFlow Run ID: {mlflow_run_id}")
    logger.info("=" * 60)

    print("=== Done ===")
    print(f"model_saved_to={output_path}")
    print(f"summary_saved_to={output_path / 'run_summary.json'}")


if __name__ == "__main__":
    main()
