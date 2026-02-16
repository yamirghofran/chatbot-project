from __future__ import annotations

import argparse
import json
import math
import os
import random
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader, Dataset, Sampler

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

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=10.0)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-workers", type=int, default=-1)

    parser.add_argument(
        "--model-name", default="google/embeddinggemma-300m", help="HF model id"
    )
    parser.add_argument("--output-dir", default="data/models/embeddinggemma_mnrl")

    parser.add_argument("--eval-max-queries", type=int, default=0)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=0)

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
    query_count = min(max_queries, len(corpus_ids))
    query_ids = corpus_ids[:query_count]

    recall_hits = 0
    mrr_sum = 0.0
    first_rank_sum = 0.0
    evaluated_queries = 0

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
        evaluated_queries += 1

    if evaluated_queries == 0:
        return {
            "queries_evaluated": 0,
            "corpus_size": len(corpus_ids),
            "k": int(k),
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "mean_first_positive_rank": 0.0,
        }

    return {
        "queries_evaluated": int(evaluated_queries),
        "corpus_size": int(len(corpus_ids)),
        "k": int(k),
        "recall_at_k": float(recall_hits / evaluated_queries),
        "mrr": float(mrr_sum / evaluated_queries),
        "mean_first_positive_rank": float(first_rank_sum / evaluated_queries),
    }


def main() -> None:
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

    model.old_fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=int(args.epochs),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(args.learning_rate)},
        output_path=str(output_path),
        use_amp=bool(runtime["use_amp"]),
        show_progress_bar=True,
    )

    metrics = None
    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size > 0
        else int(runtime["default_eval_batch_size"])
    )
    if args.eval_max_queries > 0 and val_pairs_text_df.height > 0:
        candidate_ids, neighbors_by_id, text_by_id = build_eval_graph(
            val_pairs_text_df=val_pairs_text_df,
            seed=int(args.seed),
        )
        if candidate_ids:
            metrics = evaluate_retrieval_model(
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
            print(json.dumps(metrics, indent=2))

    run_summary = {
        "args": vars(args),
        "runtime": runtime,
        "data_stats": stats,
        "finetuned_eval": metrics,
    }
    with (output_path / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("=== Done ===")
    print(f"model_saved_to={output_path}")
    print(f"summary_saved_to={output_path / 'run_summary.json'}")


if __name__ == "__main__":
    main()
