import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util as importlib_util
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    return Path, importlib_util, mo, np, pa, pl, pq, time


@app.cell
def _(mo):
    mo.md("""
    # Generate Embeddings with the Finetuned `embeddinggemma`

    This notebook loads a finetuned SentenceTransformer model, streams all rows from
    `books_embedding_texts.parquet`, generates embeddings, and writes them to a parquet file
    for later vector-database upload.
    """)
    return


@app.cell
def _(Path):
    project_root = Path(__file__).resolve().parents[2]
    default_model_path = project_root / "data" / "models" / "embeddinggemma_mnrl"
    default_texts_path = project_root / "data" / "books_embedding_texts.parquet"
    default_embeddings_path = project_root / "data" / "books_finetuned_embeddings.parquet"
    return default_embeddings_path, default_model_path, default_texts_path


@app.cell
def _(default_embeddings_path, default_model_path, default_texts_path, mo):
    model_path_ui = mo.ui.text(
        value=str(default_model_path), label="Finetuned model path"
    )
    texts_path_ui = mo.ui.text(
        value=str(default_texts_path), label="Input texts parquet path"
    )
    output_embeddings_path_ui = mo.ui.text(
        value=str(default_embeddings_path), label="Output embeddings parquet path"
    )
    encode_batch_size_ui = mo.ui.slider(
        start=16,
        stop=1024,
        value=256,
        step=16,
        label="Model encode batch size",
    )
    parquet_read_batch_size_ui = mo.ui.slider(
        start=256,
        stop=16384,
        value=4096,
        step=256,
        label="Parquet read batch size",
    )
    normalize_embeddings_ui = mo.ui.checkbox(label="Normalize embeddings", value=True)
    max_rows_ui = mo.ui.number(
        value=0, label="Max rows to process (0 means all rows)"
    )
    embedding_column_name_ui = mo.ui.text(
        value="embedding", label="Embedding column name"
    )
    run_generation_ui = mo.ui.run_button(label="Generate embeddings parquet")

    mo.vstack(
        [
            mo.md("## Step 1. Configure paths and generation settings"),
            mo.hstack([model_path_ui, texts_path_ui]),
            mo.hstack([output_embeddings_path_ui]),
            mo.hstack(
                [
                    encode_batch_size_ui,
                    parquet_read_batch_size_ui,
                    max_rows_ui,
                ]
            ),
            mo.hstack([normalize_embeddings_ui, embedding_column_name_ui]),
            run_generation_ui,
        ]
    )
    return (
        embedding_column_name_ui,
        encode_batch_size_ui,
        max_rows_ui,
        model_path_ui,
        normalize_embeddings_ui,
        output_embeddings_path_ui,
        parquet_read_batch_size_ui,
        run_generation_ui,
        texts_path_ui,
    )


@app.cell
def _(Path, model_path_ui, output_embeddings_path_ui, texts_path_ui):
    resolved_model_path = Path(model_path_ui.value.strip()).expanduser()
    resolved_texts_path = Path(texts_path_ui.value.strip()).expanduser()
    resolved_output_embeddings_path = Path(
        output_embeddings_path_ui.value.strip()
    ).expanduser()
    return (
        resolved_model_path,
        resolved_output_embeddings_path,
        resolved_texts_path,
    )


@app.cell
def _(mo, resolved_model_path, resolved_texts_path):
    missing_required_paths = []
    if not resolved_model_path.exists():
        missing_required_paths.append(str(resolved_model_path))
    if not resolved_texts_path.exists():
        missing_required_paths.append(str(resolved_texts_path))

    path_status_md = (
        mo.md("All required input paths are available.")
        if len(missing_required_paths) == 0
        else mo.md(
            "## Missing input path(s)\n"
            f"Could not find: `{missing_required_paths}`"
        )
    )
    path_status_md
    return (missing_required_paths,)


@app.cell
def _(pl, resolved_texts_path):
    input_stats_df = (
        pl.scan_parquet(resolved_texts_path)
        .select(
            [
                pl.len().alias("rows"),
                (pl.col("book_embedding_text").fill_null("").str.len_chars() > 0)
                .sum()
                .alias("non_empty_text_rows"),
                pl.col("book_embedding_text")
                .fill_null("")
                .str.len_chars()
                .mean()
                .alias("mean_text_length"),
            ]
        )
        .collect()
    )
    input_preview_df = (
        pl.scan_parquet(resolved_texts_path)
        .select(["book_id", "book_embedding_text"])
        .head(5)
        .collect()
    )
    return input_preview_df, input_stats_df


@app.cell
def _(input_preview_df, input_stats_df, mo):
    mo.vstack(
        [
            mo.md("## Step 2. Inspect input text dataset"),
            mo.ui.table(input_stats_df),
            mo.ui.table(input_preview_df),
        ]
    )
    return


@app.cell
def _(importlib_util, mo):
    missing_dependency_packages = []
    if importlib_util.find_spec("sentence_transformers") is None:
        missing_dependency_packages.append("sentence-transformers")
    if importlib_util.find_spec("torch") is None:
        missing_dependency_packages.append("torch")

    dependency_status_output = (
        mo.md("All generation dependencies are available.")
        if len(missing_dependency_packages) == 0
        else mo.md(
            "## Missing dependencies\n"
            f"Install: `{', '.join(missing_dependency_packages)}`\n\n"
            "```bash\n"
            "uv add sentence-transformers torch\n"
            "```"
        )
    )
    dependency_status_output
    return (missing_dependency_packages,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 3. Generate and save embeddings

    On click, this streams the input parquet in batches, runs model encoding,
    and writes `book_id` plus embedding vectors to a parquet output file.
    """)
    return


@app.cell
def _(
    embedding_column_name_ui,
    encode_batch_size_ui,
    max_rows_ui,
    missing_dependency_packages,
    missing_required_paths,
    mo,
    normalize_embeddings_ui,
    np,
    pa,
    parquet_read_batch_size_ui,
    pl,
    pq,
    resolved_model_path,
    resolved_output_embeddings_path,
    resolved_texts_path,
    run_generation_ui,
    time,
):
    mo.stop(
        not run_generation_ui.value,
        mo.md("Press **Generate embeddings parquet** to run."),
    )
    mo.stop(
        len(missing_dependency_packages) > 0,
        mo.md("Install missing dependencies first."),
    )
    mo.stop(
        len(missing_required_paths) > 0,
        mo.md("Fix missing path(s) first."),
    )
    mo.stop(
        embedding_column_name_ui.value.strip() == "",
        mo.md("Embedding column name cannot be empty."),
    )
    mo.stop(
        resolved_output_embeddings_path.name in {"", "."}
        or resolved_output_embeddings_path.is_dir(),
        mo.md("Output embeddings path must be a parquet file path, not a directory."),
    )

    from sentence_transformers import SentenceTransformer

    generation_model = SentenceTransformer(str(resolved_model_path))
    parquet_input_reader = pq.ParquetFile(str(resolved_texts_path))
    resolved_output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = resolved_output_embeddings_path.with_suffix(
        resolved_output_embeddings_path.suffix + ".tmp"
    )
    if temp_output_path.exists():
        temp_output_path.unlink()

    output_writer = None
    generated_rows_total = 0
    skipped_rows_total = 0
    generated_embedding_dim = None
    processed_batches_total = 0
    generation_started_at = time.perf_counter()
    max_rows_limit = int(max_rows_ui.value)
    embedding_column_name = embedding_column_name_ui.value.strip()

    try:
        for input_record_batch in parquet_input_reader.iter_batches(
            columns=["book_id", "book_embedding_text"],
            batch_size=int(parquet_read_batch_size_ui.value),
        ):
            if max_rows_limit > 0 and generated_rows_total >= max_rows_limit:
                break

            processed_batches_total += 1
            batch_frame = pl.from_arrow(input_record_batch).with_columns(
                pl.col("book_embedding_text")
                .fill_null("")
                .str.strip_chars()
                .alias("book_embedding_text")
            )
            batch_frame = batch_frame.filter(pl.col("book_embedding_text").str.len_chars() > 0)

            if max_rows_limit > 0:
                remaining_rows_allowed = max_rows_limit - generated_rows_total
                if remaining_rows_allowed <= 0:
                    break
                if batch_frame.height > remaining_rows_allowed:
                    batch_frame = batch_frame.head(remaining_rows_allowed)

            kept_rows_count = batch_frame.height
            skipped_rows_total += input_record_batch.num_rows - kept_rows_count
            if kept_rows_count == 0:
                continue

            batch_book_ids = batch_frame["book_id"].to_list()
            batch_text_values = batch_frame["book_embedding_text"].to_list()

            batch_embeddings_np = generation_model.encode(
                batch_text_values,
                batch_size=int(encode_batch_size_ui.value),
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings_ui.value,
                show_progress_bar=False,
            )
            batch_embeddings_np = np.asarray(batch_embeddings_np, dtype=np.float32)

            if generated_embedding_dim is None:
                generated_embedding_dim = int(batch_embeddings_np.shape[1])

            embedding_array_arrow = pa.array(
                batch_embeddings_np.tolist(),
                type=pa.list_(pa.float32(), generated_embedding_dim),
            )
            output_batch_table = pa.table(
                {
                    "book_id": pa.array(batch_book_ids, type=pa.string()),
                    embedding_column_name: embedding_array_arrow,
                }
            )

            if output_writer is None:
                output_writer = pq.ParquetWriter(
                    str(temp_output_path),
                    output_batch_table.schema,
                    compression="zstd",
                )
            output_writer.write_table(output_batch_table)
            generated_rows_total += kept_rows_count
    finally:
        if output_writer is not None:
            output_writer.close()

    mo.stop(
        generated_rows_total == 0,
        mo.md("No rows were encoded. Check filters and input data."),
    )

    if resolved_output_embeddings_path.exists():
        resolved_output_embeddings_path.unlink()
    temp_output_path.replace(resolved_output_embeddings_path)

    generation_elapsed_seconds = time.perf_counter() - generation_started_at
    generation_summary_df = pl.DataFrame(
        {
            "output_path": [str(resolved_output_embeddings_path)],
            "rows_written": [generated_rows_total],
            "rows_skipped_empty_text": [skipped_rows_total],
            "embedding_dim": [generated_embedding_dim],
            "batches_processed": [processed_batches_total],
            "elapsed_seconds": [round(generation_elapsed_seconds, 2)],
        }
    )
    mo.vstack(
        [
            mo.md("Embeddings generation completed."),
            mo.ui.table(generation_summary_df),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 4. Preview saved embeddings parquet
    """)
    return


@app.cell
def _(pl, resolved_output_embeddings_path):
    if resolved_output_embeddings_path.exists():
        output_preview_df = pl.read_parquet(
            resolved_output_embeddings_path, n_rows=5
        )
        output_stats_df = (
            pl.scan_parquet(resolved_output_embeddings_path)
            .select([pl.len().alias("rows")])
            .collect()
        )
    else:
        output_preview_df = pl.DataFrame()
        output_stats_df = pl.DataFrame()
    return output_preview_df, output_stats_df


@app.cell
def _(mo, output_preview_df, output_stats_df):
    output_preview_layout = (
        mo.vstack([mo.ui.table(output_stats_df), mo.ui.table(output_preview_df)])
        if output_preview_df.height > 0
        else mo.md("Run generation to see output preview.")
    )
    output_preview_layout
    return


if __name__ == "__main__":
    app.run()
