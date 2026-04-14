import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import json
    import os

    return json, mo, os, pl


@app.cell
def _(mo):
    mo.md("""
    Process review files to standardize IDs:
    1. Map Goodreads book IDs to CSV integer IDs
    2. Replace book IDs with canonical best book ID (in CSV space)
    3. Map user IDs to CSV integer IDs
    """)
    return


@app.cell
def _(mo, os, pl):
    from bookdb.utils.paths import find_project_root
    data_dir = os.path.join(find_project_root(), "data")

    book_id_map = pl.read_parquet(os.path.join(data_dir, "raw_book_id_map.parquet"))
    user_id_map = pl.read_parquet(os.path.join(data_dir, "raw_user_id_map.parquet"))

    mo.md(f"""
    ### Mapping files loaded
    - `book_id_map`: {book_id_map.shape[0]:,} rows
    - `user_id_map`: {user_id_map.shape[0]:,} rows
    """)
    return book_id_map, data_dir, user_id_map


@app.cell
def _(data_dir, mo):
    from bookdb.processing.book_ids import build_book_edition_lookup

    book_edition_lookup = build_book_edition_lookup(data_dir)

    mo.md(f"""
    ### Edition to Best Book lookup (CSV space)
    - Translated to CSV space (excluding self-maps): {book_edition_lookup.shape[0]:,}
    """)
    return (book_edition_lookup,)


@app.cell
def _(book_id_map, mo):
    # Preview book_id_map structure
    mo.vstack([
        mo.md("### Book ID Map structure"),
        book_id_map.head(5),
    ])
    return


@app.cell
def _(mo, user_id_map):
    # Preview user_id_map structure
    mo.vstack([
        mo.md("### User ID Map structure"),
        user_id_map.head(5),
    ])
    return


@app.cell
def _(book_edition_lookup, book_id_map, data_dir, mo, os, pl, user_id_map):
    # Process raw_goodreads_reviews_dedup.parquet
    _input_path = os.path.join(data_dir, "raw_goodreads_reviews_dedup.parquet")
    _output_path = os.path.join(data_dir, "1_goodreads_reviews_dedup_merged.parquet")

    _lf = pl.scan_parquet(_input_path)
    _n_in = _lf.select(pl.len()).collect().item()

    # 1. Map book_id (Goodreads) to book_id_csv
    # 2. Replace with best_csv_id if exists
    # 3. Map user_id (Goodreads) to user_id_csv

    dedup_merged = (
        _lf
        # Join book_id map
        .join(
            book_id_map.lazy().select(
                pl.col("book_id").cast(pl.String),
                pl.col("book_id_csv"),
            ),
            on="book_id",
            how="left",
        )
        # Join edition -> best book lookup
        .join(
            book_edition_lookup.lazy().rename({"edition_csv_id": "book_id_csv"}),
            on="book_id_csv",
            how="left",
        )
        # Use best_csv_id if available, else keep book_id_csv
        .with_columns(
            pl.coalesce("best_csv_id", "book_id_csv").alias("book_id_csv")
        )
        .drop("best_csv_id", "book_id")
        # Join user_id map
        .join(
            user_id_map.lazy().select(
                pl.col("user_id"),
                pl.col("user_id_csv"),
            ),
            on="user_id",
            how="left",
        )
        .drop("user_id")
        # Rename to final column names
        .rename({
            "book_id_csv": "book_id",
            "user_id_csv": "user_id",
        })
    )

    dedup_merged.sink_parquet(_output_path)

    _stats_df = pl.scan_parquet(_output_path).select(
        pl.len().alias("n_out"),
        pl.col("book_id").is_null().sum().alias("null_books"),
        pl.col("user_id").is_null().sum().alias("null_users"),
    ).collect()
    _n_out = _stats_df.item(0, "n_out")
    _null_books = _stats_df.item(0, "null_books")
    _null_users = _stats_df.item(0, "null_users")

    mo.md(f"""
    ### raw_goodreads_reviews_dedup.parquet
    - Input: {_n_in:,} rows
    - Output: {_n_out:,} rows - `{os.path.basename(_output_path)}`
    - Null book_ids (no mapping): {_null_books:,}
    - Null user_ids (no mapping): {_null_users:,}
    """)
    return


@app.cell
def _(book_edition_lookup, book_id_map, data_dir, mo, os, pl, user_id_map):
    # Process raw_goodreads_reviews_spoiler.parquet
    _input_path = os.path.join(data_dir, "raw_goodreads_reviews_spoiler.parquet")
    _output_path = os.path.join(data_dir, "1_goodreads_reviews_spoiler_merged.parquet")

    _lf = pl.scan_parquet(_input_path)
    _n_in = _lf.select(pl.len()).collect().item()

    _merged = (
        _lf
        # Join book_id map
        .join(
            book_id_map.lazy().select(
                pl.col("book_id").cast(pl.String),
                pl.col("book_id_csv"),
            ),
            on="book_id",
            how="left",
        )
        # Join edition -> best book lookup
        .join(
            book_edition_lookup.lazy().rename({"edition_csv_id": "book_id_csv"}),
            on="book_id_csv",
            how="left",
        )
        # Use best_csv_id if available, else keep book_id_csv
        .with_columns(
            pl.coalesce("best_csv_id", "book_id_csv").alias("book_id_csv")
        )
        .drop("best_csv_id", "book_id")
        # Join user_id map
        .join(
            user_id_map.lazy().select(
                pl.col("user_id"),
                pl.col("user_id_csv"),
            ),
            on="user_id",
            how="left",
        )
        .drop("user_id")
        # Rename to final column names
        .rename({
            "book_id_csv": "book_id",
            "user_id_csv": "user_id",
        })
    )

    _merged.sink_parquet(_output_path)

    _stats_df = pl.scan_parquet(_output_path).select(
        pl.len().alias("n_out"),
        pl.col("book_id").is_null().sum().alias("null_books"),
        pl.col("user_id").is_null().sum().alias("null_users"),
    ).collect()
    _n_out = _stats_df.item(0, "n_out")
    _null_books = _stats_df.item(0, "null_books")
    _null_users = _stats_df.item(0, "null_users")

    mo.md(f"""
    ### raw_goodreads_reviews_spoiler.parquet
    - Input: {_n_in:,} rows
    - Output: {_n_out:,} rows - `{os.path.basename(_output_path)}`
    - Null book_ids (no mapping): {_null_books:,}
    - Null user_ids (no mapping): {_null_users:,}
    """)
    return


@app.cell
def _(data_dir, mo, os, pl):
    # Summary: compare input vs output schemas
    _dedup_in = pl.read_parquet(os.path.join(data_dir, "raw_goodreads_reviews_dedup.parquet"), n_rows=1)
    _dedup_out = pl.read_parquet(os.path.join(data_dir, "1_goodreads_reviews_dedup_merged.parquet"), n_rows=1)

    mo.md(f"""
    ### Schema comparison (reviews_dedup)
    **Input columns:** {list(_dedup_in.columns)}
    **Output columns:** {list(_dedup_out.columns)}
    """)
    return


if __name__ == "__main__":
    app.run()
