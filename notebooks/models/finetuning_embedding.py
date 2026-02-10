import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util as importlib_util
    import math
    import os
    import random
    from collections import defaultdict, deque
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    return (
        Path,
        defaultdict,
        deque,
        importlib_util,
        math,
        mo,
        np,
        pl,
        plt,
        random,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Finetuning `embeddinggemma` with safe `MultipleNegativesRankingLoss`

    This notebook builds positive book pairs from Goodreads `similar_books`, then finetunes
    `google/embeddinggemma-300m` using `sentence-transformers`.

    The key constraint for `MultipleNegativesRankingLoss` is handled explicitly:
    each batch is sampled with **at most one pair per similarity component** to avoid known false negatives.
    """
    )
    return


@app.cell
def _(Path):
    project_root = Path(__file__).resolve().parents[2]
    raw_books_path = project_root / "data" / "raw_goodreads_books.parquet"
    book_texts_path = project_root / "data" / "books_embedding_texts.parquet"
    default_output_path = project_root / "data" / "models" / "embeddinggemma_mnrl"
    return book_texts_path, default_output_path, project_root, raw_books_path


@app.cell
def _(book_texts_path, mo, raw_books_path):
    missing_paths = [
        str(path) for path in [raw_books_path, book_texts_path] if not path.exists()
    ]
    mo.stop(
        len(missing_paths) > 0,
        mo.md(
            "## Missing input files\n"
            f"Could not find: `{missing_paths}`.\n"
            "Make sure both parquet files exist under `data/` before running this notebook."
        ),
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 1. Configure sampling and training

    Use these controls to keep experiments lightweight while preserving safe batching behavior.
    """
    )
    return


@app.cell
def _(mo):
    max_pairs_ui = mo.ui.slider(
        start=2000,
        stop=300000,
        value=50000,
        step=2000,
        label="Max unique positive pairs",
    )
    min_text_chars_ui = mo.ui.slider(
        start=40,
        stop=800,
        value=120,
        step=20,
        label="Minimum text length",
    )
    min_support_ui = mo.ui.slider(
        start=1,
        stop=5,
        value=1,
        step=1,
        label="Minimum pair support",
    )
    val_fraction_ui = mo.ui.slider(
        start=0,
        stop=30,
        value=10,
        step=1,
        label="Validation components (%)",
    )
    seed_ui = mo.ui.number(value=42, label="Random seed")
    preview_rows_ui = mo.ui.slider(
        start=5,
        stop=30,
        value=10,
        step=1,
        label="Preview rows",
    )

    batch_size_ui = mo.ui.slider(
        start=8,
        stop=128,
        value=32,
        step=8,
        label="Batch size (unique components)",
    )
    epochs_ui = mo.ui.slider(start=1, stop=10, value=2, step=1, label="Epochs")
    learning_rate_ui = mo.ui.number(value=2e-5, label="Learning rate")
    warmup_ratio_ui = mo.ui.slider(
        start=0,
        stop=30,
        value=10,
        step=1,
        label="Warmup ratio (%)",
    )
    model_name_ui = mo.ui.text(
        value="google/embeddinggemma-300m", label="Base model name"
    )
    output_dir_ui = mo.ui.text(
        value="data/models/embeddinggemma_mnrl", label="Output directory"
    )
    run_training_ui = mo.ui.run_button(label="Run finetuning")
    eval_queries_ui = mo.ui.slider(
        start=50,
        stop=5000,
        value=800,
        step=50,
        label="Evaluation queries",
    )
    eval_k_ui = mo.ui.slider(start=1, stop=20, value=10, step=1, label="Recall@k")
    run_baseline_eval_ui = mo.ui.run_button(label="Run baseline evaluation")
    run_finetuned_eval_ui = mo.ui.run_button(label="Run finetuned evaluation")

    mo.vstack(
        [
            mo.md("### Pair construction"),
            mo.hstack(
                [
                    max_pairs_ui,
                    min_text_chars_ui,
                    min_support_ui,
                    val_fraction_ui,
                    preview_rows_ui,
                ]
            ),
            mo.hstack([seed_ui]),
            mo.md("### Training"),
            mo.hstack(
                [
                    batch_size_ui,
                    epochs_ui,
                    learning_rate_ui,
                    warmup_ratio_ui,
                ]
            ),
            mo.hstack([model_name_ui, output_dir_ui]),
            run_training_ui,
            mo.md("### Evaluation"),
            mo.hstack([eval_queries_ui, eval_k_ui]),
            mo.hstack([run_baseline_eval_ui, run_finetuned_eval_ui]),
        ]
    )
    return (
        batch_size_ui,
        epochs_ui,
        eval_k_ui,
        eval_queries_ui,
        learning_rate_ui,
        max_pairs_ui,
        min_support_ui,
        min_text_chars_ui,
        model_name_ui,
        output_dir_ui,
        preview_rows_ui,
        run_baseline_eval_ui,
        run_finetuned_eval_ui,
        run_training_ui,
        seed_ui,
        val_fraction_ui,
        warmup_ratio_ui,
    )


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 2. Load and validate source data

    We keep operations lazy with Polars and only materialize what we need for previews
    and sampled training subsets.
    """
    )
    return


@app.cell
def _(book_texts_path, pl, raw_books_path):
    raw_books_lf = pl.scan_parquet(raw_books_path).select(["book_id", "similar_books"])
    book_texts_lf = pl.scan_parquet(book_texts_path).select(
        ["book_id", "book_embedding_text"]
    )
    return book_texts_lf, raw_books_lf


@app.cell
def _(book_texts_lf, pl, raw_books_lf):
    dataset_stats_df = (
        raw_books_lf.select(
            [
                pl.len().alias("raw_rows"),
                (pl.col("similar_books").list.len() > 0)
                .sum()
                .alias("raw_rows_with_similar_books"),
                pl.col("similar_books")
                .list.len()
                .mean()
                .alias("mean_similar_books_per_row"),
            ]
        )
        .join(
            book_texts_lf.select(
                [
                    pl.len().alias("text_rows"),
                    (pl.col("book_embedding_text").fill_null("").str.len_chars() > 0)
                    .sum()
                    .alias("non_empty_text_rows"),
                ]
            ),
            how="cross",
        )
        .collect()
    )
    dataset_stats_df
    return


@app.cell
def _(book_texts_lf, min_text_chars_ui, pl, raw_books_lf):
    valid_texts_lf = (
        book_texts_lf.with_columns(
            pl.col("book_embedding_text")
            .fill_null("")
            .str.strip_chars()
            .alias("book_embedding_text")
        )
        .filter(
            pl.col("book_embedding_text").str.len_chars() >= min_text_chars_ui.value
        )
        .select(["book_id", "book_embedding_text"])
    )

    valid_book_ids_lf = valid_texts_lf.select("book_id").unique()

    candidate_pairs_lf = (
        raw_books_lf.filter(pl.col("similar_books").list.len() > 0)
        .explode("similar_books")
        .rename({"book_id": "source_book_id", "similar_books": "target_book_id"})
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
    )
    return candidate_pairs_lf, valid_texts_lf


@app.cell
def _(candidate_pairs_lf, min_support_ui, pl, preview_rows_ui):
    candidate_pairs_filtered_lf = candidate_pairs_lf.filter(
        pl.col("pair_support") >= min_support_ui.value
    )
    candidate_pair_count = (
        candidate_pairs_filtered_lf.select(pl.len().alias("n")).collect().item()
    )
    candidate_pairs_preview_df = (
        candidate_pairs_filtered_lf.sort("pair_support", descending=True)
        .head(preview_rows_ui.value)
        .collect()
    )
    return (
        candidate_pair_count,
        candidate_pairs_filtered_lf,
        candidate_pairs_preview_df,
    )


@app.cell
def _(candidate_pair_count, candidate_pairs_preview_df, mo):
    mo.vstack(
        [
            mo.md(
                f"**Candidate undirected positive pairs:** `{candidate_pair_count:,}`"
            ),
            mo.ui.table(candidate_pairs_preview_df),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 3. Sample pairs and compute similarity components

    We build connected components over sampled positive edges.
    During training, each batch uses at most one edge from each component.
    """
    )
    return


@app.cell
def _(
    candidate_pair_count,
    candidate_pairs_filtered_lf,
    max_pairs_ui,
    pl,
    seed_ui,
):
    sample_size = min(max_pairs_ui.value, candidate_pair_count)
    sampled_pairs_lf = (
        candidate_pairs_filtered_lf.with_columns(
            pl.struct(["book_id_left", "book_id_right"])
            .hash(seed=int(seed_ui.value))
            .alias("_sample_key")
        )
        .sort("_sample_key")
        .limit(sample_size)
        .drop("_sample_key")
        if sample_size < candidate_pair_count
        else candidate_pairs_filtered_lf
    )
    sampled_pairs_df = sampled_pairs_lf.collect()
    return sample_size, sampled_pairs_df


@app.cell
def _(defaultdict, pl, sampled_pairs_df):
    parent = {}
    rank = {}

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while node != root:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(a, b):
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

    root_to_component = {}
    component_members = defaultdict(int)
    pair_component_ids = []
    next_component = 0

    for left_id in sampled_pairs_df["book_id_left"].to_list():
        root = find(left_id)
        if root not in root_to_component:
            root_to_component[root] = next_component
            next_component += 1
        component_id = root_to_component[root]
        pair_component_ids.append(component_id)
        component_members[component_id] += 1

    pairs_with_components_df = sampled_pairs_df.with_columns(
        pl.Series(name="component_id", values=pair_component_ids)
    )
    component_sizes_df = pl.DataFrame(
        {
            "component_id": list(component_members.keys()),
            "pairs_in_component": list(component_members.values()),
        }
    ).sort("pairs_in_component", descending=True)
    return component_sizes_df, pairs_with_components_df


@app.cell
def _(component_sizes_df, mo, preview_rows_ui, sample_size):
    mo.vstack(
        [
            mo.md(f"**Sampled pairs kept for this run:** `{sample_size:,}`"),
            mo.md(
                f"**Connected components:** `{component_sizes_df.height:,}` "
                "(used to enforce one component per batch)"
            ),
            mo.ui.table(component_sizes_df.head(preview_rows_ui.value)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 4. Split by component and attach texts

    Splitting by component prevents near-duplicate positives from leaking between train and validation.
    """
    )
    return


@app.cell
def _(pairs_with_components_df, random, seed_ui, val_fraction_ui):
    split_component_ids = pairs_with_components_df["component_id"].unique().to_list()
    rng = random.Random(int(seed_ui.value))
    rng.shuffle(split_component_ids)

    val_ratio = val_fraction_ui.value / 100
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
    return train_pairs_df, val_component_ids, val_pairs_df


@app.cell
def _(mo, train_pairs_df):
    mo.stop(
        train_pairs_df.height == 0,
        mo.md(
            "No training pairs were produced. Increase `Max unique positive pairs` or lower "
            "`Validation components (%)`."
        ),
    )
    return


@app.cell
def _(pl, train_pairs_df, val_component_ids, val_pairs_df):
    split_stats_df = pl.DataFrame(
        {
            "split": ["train", "validation"],
            "pairs": [train_pairs_df.height, val_pairs_df.height],
            "components": [
                train_pairs_df["component_id"].n_unique(),
                len(val_component_ids),
            ],
        }
    )
    split_stats_df
    return (split_stats_df,)


@app.cell
def _(train_pairs_df, val_pairs_df, valid_texts_lf):
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
    return train_pairs_text_df, val_pairs_text_df


@app.cell
def _(
    mo,
    preview_rows_ui,
    split_stats_df,
    train_pairs_text_df,
    val_pairs_text_df,
):
    mo.vstack(
        [
            mo.ui.table(split_stats_df),
            mo.md("### Training pair preview"),
            mo.ui.table(train_pairs_text_df.head(preview_rows_ui.value)),
            mo.md("### Validation pair preview"),
            mo.ui.table(val_pairs_text_df.head(preview_rows_ui.value)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 5. Baseline retrieval evaluation

    Evaluate the base `embeddinggemma` model on held-out validation components
    before finetuning so we can compare fairly.
    """
    )
    return


@app.cell
def _(Path, default_output_path, output_dir_ui, project_root):
    output_path_value = output_dir_ui.value.strip()
    resolved_output_path = (
        Path(output_path_value)
        if output_path_value and Path(output_path_value).is_absolute()
        else project_root / output_path_value
    )
    if output_path_value == "":
        resolved_output_path = default_output_path
    return (resolved_output_path,)


@app.cell
def _(defaultdict, random, seed_ui, val_pairs_text_df):
    eval_neighbors_by_id = defaultdict(set)
    eval_text_by_id = {}

    for (
        eval_left_id,
        eval_right_id,
        eval_anchor_text,
        eval_positive_text,
    ) in val_pairs_text_df.select(
        ["book_id_left", "book_id_right", "anchor_text", "positive_text"]
    ).iter_rows():
        eval_neighbors_by_id[eval_left_id].add(eval_right_id)
        eval_neighbors_by_id[eval_right_id].add(eval_left_id)
        if eval_left_id not in eval_text_by_id:
            eval_text_by_id[eval_left_id] = eval_anchor_text
        if eval_right_id not in eval_text_by_id:
            eval_text_by_id[eval_right_id] = eval_positive_text

    eval_candidate_ids = [
        book_id for book_id in eval_text_by_id if len(eval_neighbors_by_id[book_id]) > 0
    ]
    eval_rng = random.Random(int(seed_ui.value))
    eval_rng.shuffle(eval_candidate_ids)
    return eval_candidate_ids, eval_neighbors_by_id, eval_text_by_id


@app.cell
def _(np):
    def evaluate_retrieval_model(
        model_name_or_path,
        candidate_ids,
        neighbors_by_id,
        text_by_id,
        max_queries,
        k,
    ):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(model_name_or_path))
        corpus_ids = list(candidate_ids)
        corpus_texts = [text_by_id[book_id] for book_id in corpus_ids]
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=64,
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
            if len(relevant_indices) == 0:
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
            recall_hits += int(
                any(index in relevant_set for index in top_indices[:top_k])
            )

            best_relevant_similarity = float(np.max(similarities[relevant_indices]))
            first_relevant_rank = int(
                np.sum(similarities > best_relevant_similarity) + 1
            )
            mrr_sum += 1.0 / first_relevant_rank
            first_rank_sum += first_relevant_rank
            evaluated_queries += 1

        if evaluated_queries == 0:
            return {
                "queries_evaluated": 0,
                "corpus_size": len(corpus_ids),
                "k": k,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "mean_first_positive_rank": 0.0,
            }

        return {
            "queries_evaluated": evaluated_queries,
            "corpus_size": len(corpus_ids),
            "k": k,
            "recall_at_k": recall_hits / evaluated_queries,
            "mrr": mrr_sum / evaluated_queries,
            "mean_first_positive_rank": first_rank_sum / evaluated_queries,
        }

    return (evaluate_retrieval_model,)


@app.cell
def _(
    eval_candidate_ids,
    eval_k_ui,
    eval_neighbors_by_id,
    eval_queries_ui,
    eval_text_by_id,
    evaluate_retrieval_model,
    missing_packages,
    mo,
    model_name_ui,
    pl,
    run_baseline_eval_ui,
):
    baseline_eval_metrics_df = pl.DataFrame(
        schema={
            "model_stage": pl.String,
            "queries_evaluated": pl.Int64,
            "corpus_size": pl.Int64,
            "k": pl.Int64,
            "recall_at_k": pl.Float64,
            "mrr": pl.Float64,
            "mean_first_positive_rank": pl.Float64,
        }
    )

    if missing_packages:
        baseline_eval_status_md = mo.md(
            "Install dependencies first: `uv add sentence-transformers torch`."
        )
    elif len(eval_candidate_ids) == 0:
        baseline_eval_status_md = mo.md(
            "No validation candidates are available for evaluation. Increase validation split."
        )
    elif not run_baseline_eval_ui.value:
        baseline_eval_status_md = mo.md(
            "Press **Run baseline evaluation** to compute base-model metrics."
        )
    else:
        try:
            baseline_metrics = evaluate_retrieval_model(
                model_name_or_path=model_name_ui.value,
                candidate_ids=eval_candidate_ids,
                neighbors_by_id=eval_neighbors_by_id,
                text_by_id=eval_text_by_id,
                max_queries=eval_queries_ui.value,
                k=eval_k_ui.value,
            )
            baseline_eval_metrics_df = pl.DataFrame([baseline_metrics]).with_columns(
                pl.lit("baseline").alias("model_stage")
            )
            baseline_eval_status_md = mo.md("Baseline evaluation completed.")
        except Exception as exc:
            baseline_eval_status_md = mo.md(
                "Baseline evaluation failed.\n" f"```text\n{exc}\n```"
            )

    mo.vstack([baseline_eval_status_md, mo.ui.table(baseline_eval_metrics_df)])
    return (baseline_eval_metrics_df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 6. Check dependencies and run finetuning

    Required packages: `sentence-transformers` and `torch`.
    """
    )
    return


@app.cell
def _(importlib_util, mo):
    missing_packages = []
    if importlib_util.find_spec("sentence_transformers") is None:
        missing_packages.append("sentence-transformers")
    if importlib_util.find_spec("torch") is None:
        missing_packages.append("torch")

    dependency_status_md = None
    if missing_packages:
        dependency_status_md = mo.md(
            "### Missing dependencies\n"
            f"Install: `{', '.join(missing_packages)}`\n\n"
            "Run:\n"
            "```bash\n"
            "uv add sentence-transformers torch\n"
            "```\n"
            "Then re-run this notebook."
        )
    else:
        dependency_status_md = mo.md("All training dependencies are available.")
    dependency_status_md
    return (missing_packages,)


@app.cell
def _(missing_packages, mo):
    mo.stop(len(missing_packages) > 0)
    return


@app.cell
def _(
    batch_size_ui,
    defaultdict,
    deque,
    math,
    missing_packages,
    mo,
    random,
    seed_ui,
    train_pairs_text_df,
):
    mo.stop(len(missing_packages) > 0)

    from sentence_transformers import InputExample
    from torch.utils.data import DataLoader, Dataset, Sampler

    class PairDataset(Dataset):
        def __init__(self, pairs_df):
            self.examples = [
                InputExample(texts=[anchor, positive])
                for anchor, positive in pairs_df.select(
                    ["anchor_text", "positive_text"]
                ).iter_rows()
            ]

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    class ComponentBatchSampler(Sampler):
        def __init__(self, component_ids, batch_size, seed):
            self.batch_size = batch_size
            self.seed = seed
            self.component_to_indices = defaultdict(list)
            for idx, component_id in enumerate(component_ids):
                self.component_to_indices[component_id].append(idx)
            self.num_examples = len(component_ids)

        def __iter__(self):
            rng = random.Random(self.seed)
            pools = {}
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
                active_components = [
                    cid for cid in active_components if len(pools[cid]) > 0
                ]

        def __len__(self):
            return math.ceil(self.num_examples / self.batch_size)

    train_dataset = PairDataset(train_pairs_text_df)
    component_sampler = ComponentBatchSampler(
        component_ids=train_pairs_text_df["component_id"].to_list(),
        batch_size=batch_size_ui.value,
        seed=int(seed_ui.value),
    )
    return DataLoader, component_sampler, train_dataset


@app.cell
def _(mo):
    mo.md(
        """
    ### Why this sampler avoids known false negatives

    `MultipleNegativesRankingLoss` treats all non-matching items in the batch as negatives.
    By ensuring one sample per connected similarity component in each batch, we avoid placing
    known positives together as negatives.
    """
    )
    return


@app.cell
def _(
    DataLoader,
    component_sampler,
    epochs_ui,
    learning_rate_ui,
    mo,
    model_name_ui,
    resolved_output_path,
    run_training_ui,
    train_dataset,
    warmup_ratio_ui,
):
    mo.stop(
        not run_training_ui.value,
        mo.md("Press **Run finetuning** to train and save the model."),
    )

    from sentence_transformers import SentenceTransformer, losses

    model = SentenceTransformer(model_name_ui.value)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=component_sampler,
        collate_fn=model.smart_batching_collate,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    output_path = resolved_output_path
    output_path.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(component_sampler)
    total_steps = steps_per_epoch * epochs_ui.value
    warmup_steps = int(total_steps * (warmup_ratio_ui.value / 100))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs_ui.value,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(learning_rate_ui.value)},
        output_path=str(output_path),
        show_progress_bar=True,
    )

    mo.md(
        "### Finetuning complete\n"
        f"- Saved to: `{output_path}`\n"
        f"- Steps per epoch: `{steps_per_epoch}`\n"
        f"- Total steps: `{total_steps}`\n"
        f"- Warmup steps: `{warmup_steps}`"
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 7. Evaluate the finetuned model
    """
    )
    return


@app.cell
def _(
    eval_candidate_ids,
    eval_k_ui,
    eval_neighbors_by_id,
    eval_queries_ui,
    eval_text_by_id,
    evaluate_retrieval_model,
    missing_packages,
    mo,
    pl,
    resolved_output_path,
    run_finetuned_eval_ui,
):
    finetuned_eval_metrics_df = pl.DataFrame(
        schema={
            "model_stage": pl.String,
            "queries_evaluated": pl.Int64,
            "corpus_size": pl.Int64,
            "k": pl.Int64,
            "recall_at_k": pl.Float64,
            "mrr": pl.Float64,
            "mean_first_positive_rank": pl.Float64,
        }
    )

    if missing_packages:
        finetuned_eval_status_md = mo.md(
            "Install dependencies first: `uv add sentence-transformers torch`."
        )
    elif len(eval_candidate_ids) == 0:
        finetuned_eval_status_md = mo.md(
            "No validation candidates are available for evaluation. Increase validation split."
        )
    elif not run_finetuned_eval_ui.value:
        finetuned_eval_status_md = mo.md(
            "Press **Run finetuned evaluation** after training to compute metrics."
        )
    elif not resolved_output_path.exists():
        finetuned_eval_status_md = mo.md(
            f"Finetuned model path does not exist: `{resolved_output_path}`. Train first."
        )
    else:
        try:
            finetuned_metrics = evaluate_retrieval_model(
                model_name_or_path=resolved_output_path,
                candidate_ids=eval_candidate_ids,
                neighbors_by_id=eval_neighbors_by_id,
                text_by_id=eval_text_by_id,
                max_queries=eval_queries_ui.value,
                k=eval_k_ui.value,
            )
            finetuned_eval_metrics_df = pl.DataFrame([finetuned_metrics]).with_columns(
                pl.lit("finetuned").alias("model_stage")
            )
            finetuned_eval_status_md = mo.md("Finetuned-model evaluation completed.")
        except Exception as exc:
            finetuned_eval_status_md = mo.md(
                "Finetuned-model evaluation failed.\n" f"```text\n{exc}\n```"
            )

    mo.vstack([finetuned_eval_status_md, mo.ui.table(finetuned_eval_metrics_df)])
    return (finetuned_eval_metrics_df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 8. Compare baseline vs finetuned
    """
    )
    return


@app.cell
def _(baseline_eval_metrics_df, finetuned_eval_metrics_df, mo, pl):
    comparison_eval_metrics_df = pl.DataFrame(
        schema={
            "model_stage": pl.String,
            "queries_evaluated": pl.Int64,
            "corpus_size": pl.Int64,
            "k": pl.Int64,
            "recall_at_k": pl.Float64,
            "mrr": pl.Float64,
            "mean_first_positive_rank": pl.Float64,
        }
    )

    if baseline_eval_metrics_df.height > 0 and finetuned_eval_metrics_df.height > 0:
        comparison_eval_metrics_df = pl.concat(
            [baseline_eval_metrics_df, finetuned_eval_metrics_df],
            how="vertical",
        )
        comparison_status_md = mo.md(
            "Comparison metrics are ready. Plot below uses `Recall@k` and `MRR`."
        )
    else:
        comparison_status_md = mo.md(
            "Run both baseline and finetuned evaluations to enable comparison."
        )

    mo.vstack([comparison_status_md, mo.ui.table(comparison_eval_metrics_df)])
    return (comparison_eval_metrics_df,)


@app.cell
def _(comparison_eval_metrics_df, mo, np, pl, plt):
    comparison_plot_output = None

    if comparison_eval_metrics_df.height < 2:
        comparison_plot_output = mo.md(
            "Comparison plot will appear after both evaluations finish."
        )
    else:
        baseline_rows = comparison_eval_metrics_df.filter(
            pl.col("model_stage") == "baseline"
        ).to_dicts()
        finetuned_rows = comparison_eval_metrics_df.filter(
            pl.col("model_stage") == "finetuned"
        ).to_dicts()

        if len(baseline_rows) == 0 or len(finetuned_rows) == 0:
            comparison_plot_output = mo.md(
                "Comparison plot will appear after both evaluations finish."
            )
        else:
            baseline_row = baseline_rows[0]
            finetuned_row = finetuned_rows[0]

            metric_labels = [f"Recall@{int(baseline_row['k'])}", "MRR"]
            baseline_values = [baseline_row["recall_at_k"], baseline_row["mrr"]]
            finetuned_values = [finetuned_row["recall_at_k"], finetuned_row["mrr"]]

            x_positions = np.arange(len(metric_labels))
            bar_width = 0.35
            _, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(
                x_positions - bar_width / 2,
                baseline_values,
                width=bar_width,
                label="Baseline",
            )
            ax.bar(
                x_positions + bar_width / 2,
                finetuned_values,
                width=bar_width,
                label="Finetuned",
            )
            ax.set_ylim(0, 1)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(metric_labels)
            ax.set_ylabel("Score")
            ax.set_title("Base vs Finetuned Retrieval Metrics")
            ax.legend()
            comparison_plot_output = ax

    comparison_plot_output
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step 9. Next checks

    1. Evaluate retrieval quality on held-out components.
    2. Export embeddings for all books and compare nearest-neighbor quality before/after finetuning.
    """
    )
    return


if __name__ == "__main__":
    app.run()
