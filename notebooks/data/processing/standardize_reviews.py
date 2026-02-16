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
def _(json, mo, os, pl):
    _project_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    data_dir = os.path.join(_project_root, "data")

    book_id_map = pl.read_parquet(os.path.join(data_dir, "raw_book_id_map.parquet"))
    user_id_map = pl.read_parquet(os.path.join(data_dir, "raw_user_id_map.parquet"))

    with open(os.path.join(data_dir, "best_book_id_map.json")) as f:
        best_book_id_map = json.load(f)

    mo.md(f"""
    ### Mapping files loaded
    - `book_id_map`: {book_id_map.shape[0]:,} rows
    - `user_id_map`: {user_id_map.shape[0]:,} rows
    - `best_book_id_map`: {len(best_book_id_map):,} best books
    """)
    return best_book_id_map, book_id_map, data_dir, user_id_map


@app.cell
def _(best_book_id_map, book_id_map, mo, pl):
    # Build edition to best book lookup in CSV space

    # Invert: edition_id to best_id (Goodreads IDs)
    _edition_to_best_gr = {
        int(eid): int(best_id)
        for best_id, edition_ids in best_book_id_map.items()
        for eid in edition_ids
    }

    _gr_lookup = pl.DataFrame({
        "edition_book_id": list(_edition_to_best_gr.keys()),
        "best_book_id": list(_edition_to_best_gr.values()),
    })

    # Translate both sides to CSV ID space
    book_edition_lookup = (
        _gr_lookup
        .join(
            book_id_map.select(
                pl.col("book_id").alias("edition_book_id"),
                pl.col("book_id_csv").alias("edition_csv_id"),
            ),
            on="edition_book_id",
            how="inner",
        )
        .join(
            book_id_map.select(
                pl.col("book_id").alias("best_book_id"),
                pl.col("book_id_csv").alias("best_csv_id"),
            ),
            on="best_book_id",
            how="inner",
        )
        .select("edition_csv_id", "best_csv_id")
        .filter(pl.col("edition_csv_id") != pl.col("best_csv_id"))
    )

    mo.md(f"""
    ### Edition to Best Book lookup (CSV space)
    - Goodreads-space mappings: {len(_edition_to_best_gr):,}
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
    _output_path = os.path.join(data_dir, "goodreads_reviews_dedup_clean.parquet")

    _lf = pl.scan_parquet(_input_path)
    _n_in = _lf.select(pl.len()).collect().item()

    # 1. Map book_id (Goodreads) to book_id_csv
    # 2. Replace with best_csv_id if exists
    # 3. Map user_id (Goodreads) to user_id_csv

    _cleaned = (
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

    _cleaned.sink_parquet(_output_path)
    _n_out = pl.scan_parquet(_output_path).select(pl.len()).collect().item()

    # Count nulls (failed mappings)
    _result = pl.read_parquet(_output_path)
    _null_books = _result.filter(pl.col("book_id").is_null()).height
    _null_users = _result.filter(pl.col("user_id").is_null()).height

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
    _output_path = os.path.join(data_dir, "goodreads_reviews_spoiler_clean.parquet")

    _lf = pl.scan_parquet(_input_path)
    _n_in = _lf.select(pl.len()).collect().item()

    _cleaned = (
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

    _cleaned.sink_parquet(_output_path)
    _n_out = pl.scan_parquet(_output_path).select(pl.len()).collect().item()

    # Count nulls (failed mappings)
    _result = pl.read_parquet(_output_path)
    _null_books = _result.filter(pl.col("book_id").is_null()).height
    _null_users = _result.filter(pl.col("user_id").is_null()).height

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
    _dedup_out = pl.read_parquet(os.path.join(data_dir, "goodreads_reviews_dedup_clean.parquet"), n_rows=1)

    mo.md(f"""
    ### Schema comparison (reviews_dedup)
    **Input columns:** {list(_dedup_in.columns)}
    **Output columns:** {list(_dedup_out.columns)}
    """)
    return

if __name__ == "__main__":
    app.run()
