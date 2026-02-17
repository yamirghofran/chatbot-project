import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import os

    return mo, os, pl


@app.cell
def _(mo):
    mo.md("""
    # Add Timestamps to interactions_merged (Chunked Processing)

    Process in chunks to avoid memory issues.

    **Join key**: `(user_id, book_id, rating, is_read)`

    **Validation**:
    - If `is_reviewed == 1`, require `review_text_incomplete` is non-empty
    - If `is_reviewed == 0`, require `review_text_incomplete` is empty

    **Timestamp**: Take the latest of `(date_added, date_updated, read_at, started_at)` as Unix timestamp
    """)
    return


@app.cell
def _(os, pl):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    data_dir = os.path.join(project_root, "data")

    merged_path = os.path.join(data_dir, "goodreads_interactions_merged.parquet")
    dedup_path = os.path.join(data_dir, "goodreads_interactions_dedup_merged.parquet")
    output_path = os.path.join(data_dir, "goodreads_interactions_merged_timestamps.parquet")
    temp_dir = os.path.join(data_dir, "temp_chunks")

    # Create temp directory for chunks
    os.makedirs(temp_dir, exist_ok=True)

    return data_dir, dedup_path, merged_path, output_path, temp_dir


@app.cell
def _(dedup_path, merged_path, mo, pl):
    # Configuration
    CHUNK_SIZE = 5_000_000  # Process 10M rows at a time

    # Get total row count
    merged_lf = pl.scan_parquet(merged_path)
    total_rows = merged_lf.select(pl.len()).collect().item()
    num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division

    mo.md(f"""
    ## Processing Configuration
    - **Total rows**: {total_rows:,}
    - **Chunk size**: {CHUNK_SIZE:,}
    - **Number of chunks**: {num_chunks}
    """)
    return CHUNK_SIZE, merged_lf, num_chunks, total_rows


@app.cell
def _(CHUNK_SIZE, dedup_path, merged_path, mo, num_chunks, os, pl, temp_dir):
    # Prepare dedup dataset (parse timestamps once, stays lazy)
    dedup_lf = pl.scan_parquet(dedup_path)

    dedup_prepared = (
        dedup_lf
        .select([
            "user_id",
            "book_id",
            "rating",
            "is_read",
            "review_text_incomplete",
            "date_added",
            "date_updated",
            "read_at",
            "started_at",
        ])
        # Parse timestamps to Unix format
        .with_columns([
            pl.col("date_added").str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False).dt.epoch("s").alias("ts_added"),
            pl.col("date_updated").str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False).dt.epoch("s").alias("ts_updated"),
            pl.col("read_at").str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False).dt.epoch("s").alias("ts_read_at"),
            pl.col("started_at").str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False).dt.epoch("s").alias("ts_started_at"),
        ])
        # Find maximum timestamp
        .with_columns(
            pl.max_horizontal(["ts_added", "ts_updated", "ts_read_at", "ts_started_at"]).alias("timestamp")
        )
        .select([
            "user_id",
            "book_id",
            "rating",
            "is_read",
            "review_text_incomplete",
            "timestamp",
        ])
    )

    # Process chunks
    chunk_files = []
    for i in range(num_chunks):
        offset = i * CHUNK_SIZE
        chunk_output = os.path.join(temp_dir, f"chunk_{i:04d}.parquet")

        # Read chunk from merged
        merged_chunk = (
            pl.scan_parquet(merged_path)
            .slice(offset, CHUNK_SIZE)
            .with_columns(pl.col("is_read").cast(pl.Boolean).alias("is_read_bool"))
        )

        # Join with dedup and apply validation
        result_chunk = (
            merged_chunk
            .join(
                dedup_prepared,
                left_on=["user_id", "book_id", "rating", "is_read_bool"],
                right_on=["user_id", "book_id", "rating", "is_read"],
                how="inner",
            )
            # Validation filter
            .filter(
                (
                    (pl.col("is_reviewed") == 1) &
                    (pl.col("review_text_incomplete").is_not_null()) &
                    (pl.col("review_text_incomplete") != "")
                ) |
                (
                    (pl.col("is_reviewed") == 0) &
                    (
                        (pl.col("review_text_incomplete").is_null()) |
                        (pl.col("review_text_incomplete") == "")
                    )
                )
            )
            .drop(["is_read_bool", "review_text_incomplete"])
            .select([
                "user_id",
                "book_id",
                "is_read",
                "rating",
                "is_reviewed",
                "timestamp",
            ])
        )

        # Write chunk to disk using streaming
        result_chunk.sink_parquet(chunk_output, engine="streaming")
        chunk_files.append(chunk_output)

        print(f"✓ Processed chunk {i+1}/{num_chunks}")

    mo.md(f"""
    **Chunk processing complete!**

    Created {len(chunk_files)} chunk files
    """)
    return chunk_files, dedup_lf, dedup_prepared, i


@app.cell
def _(chunk_files, mo, output_path, pl):
    # Concatenate all chunks into final output
    all_chunks = [pl.scan_parquet(f) for f in chunk_files]
    final_lf = pl.concat(all_chunks)

    # Write final result
    final_lf.sink_parquet(output_path)

    # Get final row count
    final_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()

    mo.md(f"""
    **Final merge complete!**

    Output: `goodreads_interactions_merged_timestamps.parquet`
    Final row count: {final_count:,}
    """)
    return all_chunks, final_count, final_lf


@app.cell
def _(mo, output_path, pl):
    # Verify output
    output_sample = pl.read_parquet(output_path, n_rows=10)

    mo.vstack([
        mo.md("## Sample Output"),
        mo.ui.table(output_sample),
    ])
    return (output_sample,)


if __name__ == "__main__":
    app.run()
