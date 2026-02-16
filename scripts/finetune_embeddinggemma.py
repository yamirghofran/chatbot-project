from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import polars as pl
import torch
from dotenv import load_dotenv
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader, Dataset, Sampler

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

raw_books_url = "https://pub-eecdafb53cc84b659949b513e40369d2.r2.dev/files/md5/68/4227dbfdbc026e431d64df236e3428"
book_texts_url = "https://pub-eecdafb53cc84b659949b513e40369d2.r2.dev/files/md5/46/22eb2357c0cdca856808f638ac5726"


class PairDataset(Dataset):
    def __init__(self, pairs_df: pl.DataFrame) -> None:
        self.examples = [
            InputExample(texts=[anchor, positive])
            for anchor, positive in pairs_df.select(
                ["anchor_text", "positive_text"]
            ).iter_rows()
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self.examples[idx]


class ComponentBatchSampler(Sampler[list[int]]):
    def __init__(self, component_ids: list[int], batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.component_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, component_id in enumerate(component_ids):
            self.component_to_indices[component_id].append(idx)
        self.num_examples = len(component_ids)

    def __iter__(self):
        rng = random.Random(self.seed)
        pools: dict[int, deque[int]] = {}
        for component_id, indices in self.component_to_indices.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            pools[component_id] = deque(shuffled)

        active_components = [cid for cid, q in pools.items() if len(q) > 0]
        while active_components:
            rng.shuffle(active_components)
            selected_components = active_components[: self.batch_size]
            batch = [pools[cid].popleft() for cid in selected_components]
            yield batch
            active_components = [cid for cid in active_components if len(pools[cid]) > 0]

    def __len__(self) -> int:
        return math.ceil(self.num_examples / self.batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune google/embeddinggemma-300m with component-safe MNRL batching."
    )

    parser.add_argument("--raw-books-source", default=raw_books_url)
    parser.add_argument("--book-texts-source", default=book_texts_url)
    parser.add_argument("--download-inputs", action="store_true")
    parser.add_argument("--cache-dir", default="data/cache")

    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--min-text-chars", type=int, default=120)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=10.0)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        use_amp = True
        pin_memory = True
        prefetch_factor = 4
        auto_workers = min(16, os.cpu_count() or 1)
        eval_batch_size = 256
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        use_amp = False
        pin_memory = False
        prefetch_factor = 2
        auto_workers = 0
        eval_batch_size = 64
    else:
        device = "cpu"
        use_amp = False
        pin_memory = False
        prefetch_factor = 2
        auto_workers = min(8, os.cpu_count() or 1)
        eval_batch_size = 64

    num_workers = auto_workers if num_workers_arg < 0 else num_workers_arg

    return {
        "device": device,
        "use_amp": use_amp,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
        "num_workers": num_workers,
        "default_eval_batch_size": eval_batch_size,
    }


def build_training_frames(args: argparse.Namespace, raw_source: str, text_source: str):
    raw_books_lf = pl.scan_parquet(raw_source).select(["book_id", "similar_books"])
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
        .join(
            valid_book_ids_lf.rename({"book_id": "source_book_id"}),
            on="source_book_id",
            how="inner",
        )
        .join(
            valid_book_ids_lf.rename({"book_id": "target_book_id"}),
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


def evaluate_retrieval_model(
    model_name_or_path: str | Path,
    candidate_ids: list[str],
    neighbors_by_id: dict[str, set[str]],
    text_by_id: dict[str, str],
    max_queries: int,
    k: int,
    device: str,
    encode_batch_size: int,
) -> dict[str, float]:
    model = SentenceTransformer(str(model_name_or_path), device=device)

    corpus_ids = list(candidate_ids)
    corpus_texts = [text_by_id[book_id] for book_id in corpus_ids]
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=encode_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    id_to_index = {book_id: idx for idx, book_id in enumerate(corpus_ids)}
    # Handle max_queries: -1 means use all queries, otherwise limit to specified value
    if max_queries < 0:
        query_count = len(corpus_ids)
    else:
        query_count = min(max_queries, len(corpus_ids))
    query_ids = corpus_ids[:query_count]

    recall_hits = 0
    mrr_sum = 0.0
    first_rank_sum = 0.0
    evaluated_queries = 0

    # Additional metric accumulators
    ndcg_sum = 0.0
    map_sum = 0.0
    precision_sum = 0.0
    hit_rate_hits = 0

    for query_id in query_ids:
        query_idx = id_to_index[query_id]
        relevant_indices = [
            id_to_index[neighbor_id]
            for neighbor_id in neighbors_by_id[query_id]
            if neighbor_id in id_to_index
        ]
        if not relevant_indices:
            continue

        similarities = corpus_embeddings @ corpus_embeddings[query_idx]
        similarities[query_idx] = -1.0
        top_k = min(k, len(similarities) - 1)
        if top_k <= 0:
            continue

        if top_k < len(similarities):
            top_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        else:
            top_indices = np.argsort(-similarities)

        relevant_set = set(relevant_indices)
        recall_hits += int(any(index in relevant_set for index in top_indices[:top_k]))

        best_relevant_similarity = float(np.max(similarities[relevant_indices]))
        first_relevant_rank = int(np.sum(similarities > best_relevant_similarity) + 1)
        mrr_sum += 1.0 / first_relevant_rank
        first_rank_sum += first_relevant_rank

        # Calculate NDCG@k (Normalized Discounted Cumulative Gain)
        # Using binary relevance (1 if relevant, 0 otherwise)
        dcg = 0.0
        for rank, idx in enumerate(top_indices[:top_k], start=1):
            if idx in relevant_set:
                dcg += 1.0 / np.log2(rank + 1)
        # Ideal DCG: all relevant items at the top
        ideal_dcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant_indices), top_k) + 1))
        ndcg_sum += dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        # Calculate AP (Average Precision) for MAP
        # AP = sum of (precision at rank k * relevance at rank k) / total relevant docs
        num_relevant_found = 0
        precision_at_positions = 0.0
        for rank, idx in enumerate(top_indices[:top_k], start=1):
            if idx in relevant_set:
                num_relevant_found += 1
                precision_at_positions += num_relevant_found / rank
        # Consider relevant items beyond top_k as not retrieved
        ap = precision_at_positions / len(relevant_indices) if relevant_indices else 0.0
        map_sum += ap

        # Calculate Precision@k
        hits_in_top_k = sum(1 for idx in top_indices[:top_k] if idx in relevant_set)
        precision_sum += hits_in_top_k / top_k

        # Calculate Hit Rate (whether any relevant item is in top-k, same as recall@k for binary)
        hit_rate_hits += int(hits_in_top_k > 0)

        evaluated_queries += 1

    if evaluated_queries == 0:
        return {
            "queries_evaluated": 0,
            "corpus_size": len(corpus_ids),
            "k": int(k),
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "mean_first_positive_rank": 0.0,
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
            "precision_at_k": 0.0,
            "hit_rate_at_k": 0.0,
        }

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


def log_to_mlflow(
    args: argparse.Namespace,
    runtime: dict[str, object],
    stats: dict[str, int],
    training_time: float,
    finetuned_metrics: Optional[Dict[str, float]],
    baseline_metrics: Optional[Dict[str, float]],
    output_path: Path,
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
        mlflow.log_params({
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "max_seq_length": args.max_seq_length,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
        })

        # Log data processing parameters
        mlflow.log_params({
            "max_pairs": args.max_pairs,
            "min_text_chars": args.min_text_chars,
            "min_support": args.min_support,
            "val_fraction": args.val_fraction,
        })

        # Log runtime configuration
        mlflow.log_params({
            "device": runtime["device"],
            "use_amp": runtime["use_amp"],
            "num_workers": runtime["num_workers"],
        })

        # Log dataset statistics
        mlflow.log_params({
            "candidate_pair_count": stats["candidate_pair_count"],
            "sampled_pair_count": stats["sampled_pair_count"],
            "component_count": stats["component_count"],
            "train_pairs": stats["train_pairs"],
            "val_pairs": stats["val_pairs"],
            "train_components": stats["train_components"],
            "val_components": stats["val_components"],
        })

        # Log training time
        mlflow.log_metric("training_time_seconds", training_time)

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
            description: Optional[str] = None
        ):
            """Safely log an artifact, handling missing boto3 for S3 backends.

            Args:
                path: Path to the artifact file
                artifact_path: Subdirectory in MLFlow (e.g., "model", "dataset")
                description: Optional description for logging
            """
            try:
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
                if description:
                    logger.info(f"Logged artifact: {description}")
            except Exception as e:
                logger.warning(f"Could not log artifact {path}: {e}")
                logger.warning("Artifact saved locally but not uploaded to MLFlow")

        # Log the saved model directory
        if output_path.exists():
            for artifact_file in output_path.glob("*"):
                if artifact_file.is_file():
                    safe_log_artifact(
                        artifact_file,
                        artifact_path="model",
                        description=f"model/{artifact_file.name}"
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
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "max_seq_length": args.max_seq_length,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
            "device": runtime["device"],
            "use_amp": runtime["use_amp"],
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
        mlflow.set_tags({
            "model_type": "EmbeddingGemma",
            "framework": "sentence-transformers",
            "training_date": datetime.now().isoformat(),
            "device": str(runtime["device"]),
        })

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
    print(f"device={runtime['device']} use_amp={runtime['use_amp']}")
    print(
        f"num_workers={runtime['num_workers']} pin_memory={runtime['pin_memory']} prefetch_factor={runtime['prefetch_factor']}"
    )

    train_pairs_text_df, val_pairs_text_df, stats = build_training_frames(
        args, raw_source, text_source
    )
    print("=== Data stats ===")
    print(json.dumps(stats, indent=2))

    model = SentenceTransformer(args.model_name, device=runtime["device"])
    model.max_seq_length = int(args.max_seq_length)

    gradient_checkpointing_enabled = False
    if args.gradient_checkpointing:
        try:
            first_module = model._first_module()
            auto_model = getattr(first_module, "auto_model", None)
            if auto_model is not None and hasattr(
                auto_model, "gradient_checkpointing_enable"
            ):
                auto_model.gradient_checkpointing_enable()
                gradient_checkpointing_enabled = True
        except Exception:
            gradient_checkpointing_enabled = False

    train_dataset = PairDataset(train_pairs_text_df)
    component_sampler = ComponentBatchSampler(
        component_ids=train_pairs_text_df["component_id"].to_list(),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    dataloader_kwargs: dict[str, object] = {}
    if runtime["num_workers"] > 0:
        dataloader_kwargs["num_workers"] = int(runtime["num_workers"])
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = int(runtime["prefetch_factor"])
    if runtime["pin_memory"]:
        dataloader_kwargs["pin_memory"] = True

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=component_sampler,
        collate_fn=model.smart_batching_collate,
        **dataloader_kwargs,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    steps_per_epoch = len(component_sampler)
    total_steps = steps_per_epoch * int(args.epochs)
    warmup_steps = int(total_steps * (float(args.warmup_ratio) / 100.0))

    print("=== Training config ===")
    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "output_dir": str(output_path),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "warmup_ratio": float(args.warmup_ratio),
                "warmup_steps": warmup_steps,
                "max_seq_length": int(args.max_seq_length),
                "gradient_checkpointing": gradient_checkpointing_enabled,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
            },
            indent=2,
        )
    )

    logger.info("Starting model training...")
    start_time = time.time()
    model.old_fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=int(args.epochs),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(args.learning_rate)},
        output_path=str(output_path),
        use_amp=bool(runtime["use_amp"]),
        show_progress_bar=True,
    )
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    baseline_metrics = None
    finetuned_metrics = None
    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size > 0
        else int(runtime["default_eval_batch_size"])
    )

    # Determine if we should run evaluation
    # eval_max_queries == -1 means "auto" (evaluate if validation pairs exist)
    # eval_max_queries == 0 means "skip evaluation"
    # eval_max_queries > 0 means "evaluate with specific query limit"
    should_run_eval = (
        args.eval_max_queries != 0 and val_pairs_text_df.height > 0
    )

    if should_run_eval:
        candidate_ids, neighbors_by_id, text_by_id = build_eval_graph(
            val_pairs_text_df=val_pairs_text_df,
            seed=int(args.seed),
        )
        if candidate_ids:
            # Evaluate baseline model first (if enabled)
            if args.eval_baseline:
                logger.info("=== Evaluating Baseline Model ===")
                baseline_metrics = evaluate_retrieval_model(
                    model_name_or_path=args.model_name,
                    candidate_ids=candidate_ids,
                    neighbors_by_id=neighbors_by_id,
                    text_by_id=text_by_id,
                    max_queries=int(args.eval_max_queries),
                    k=int(args.eval_k),
                    device=str(runtime["device"]),
                    encode_batch_size=eval_batch_size,
                )
                print("=== Baseline eval ===")
                print(json.dumps(baseline_metrics, indent=2))

            # Evaluate finetuned model
            logger.info("=== Evaluating Finetuned Model ===")
            finetuned_metrics = evaluate_retrieval_model(
                model_name_or_path=output_path,
                candidate_ids=candidate_ids,
                neighbors_by_id=neighbors_by_id,
                text_by_id=text_by_id,
                max_queries=int(args.eval_max_queries),
                k=int(args.eval_k),
                device=str(runtime["device"]),
                encode_batch_size=eval_batch_size,
            )
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
    run_name = f"embeddinggemma_e{args.epochs}_bs{args.batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_paths = [Path(raw_source), Path(text_source)]
    mlflow_run_id = log_to_mlflow(
        args=args,
        runtime=runtime,
        stats=stats,
        training_time=training_time,
        finetuned_metrics=finetuned_metrics,
        baseline_metrics=baseline_metrics,
        output_path=output_path,
        dataset_paths=dataset_paths,
        run_name=run_name,
    )

    run_summary = {
        "args": vars(args),
        "runtime": runtime,
        "data_stats": stats,
        "baseline_eval": baseline_metrics,
        "finetuned_eval": finetuned_metrics,
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
