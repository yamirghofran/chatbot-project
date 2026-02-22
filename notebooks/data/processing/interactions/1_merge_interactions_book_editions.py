import marimo

__generated_with = "0.19.8"
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
    # Merge Book Editions into Best Book ID

    Replace `book_id` in the interactions with the canonical `best_book_id`.

    The CSV interactions use anonymized `book_id_csv` IDs, not Goodreads IDs.
    `best_book_id_map.json` is in Goodreads space, so we translate everything
    to CSV space via `raw_book_id_map.parquet`.
    """)
    return


@app.cell
def _(json, mo, os, pl):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    data_dir = os.path.join(project_root, "data")

    with open(os.path.join(data_dir, "best_book_id_map.json")) as f:
        best_book_id_map = json.load(f)

    # invert: edition -> best (goodreads IDs)
    edition_to_best_gr = {
        int(eid): int(best_id)
        for best_id, edition_ids in best_book_id_map.items()
        for eid in edition_ids
    }

    gr_lookup = pl.DataFrame({
        "edition_book_id": list(edition_to_best_gr.keys()),
        "best_book_id": list(edition_to_best_gr.values()),
    })

    book_id_map = pl.read_parquet(os.path.join(data_dir, "raw_book_id_map.parquet"))
    user_id_map = pl.read_parquet(os.path.join(data_dir, "raw_user_id_map.parquet"))

    # translate both sides to CSV ID space
    csv_lookup = (
        gr_lookup
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
    )

    lookup_df = csv_lookup.filter(pl.col("edition_csv_id") != pl.col("best_csv_id"))

    mo.vstack([
        mo.md(f"""
    ### Edition mapping (CSV space)
    - Goodreads-space mappings: **{gr_lookup.shape[0]:,}**
    - Translated to CSV space: **{csv_lookup.shape[0]:,}**
    - After removing self-maps: **{lookup_df.shape[0]:,}**
    - Lost (no CSV ID found): **{gr_lookup.shape[0] - csv_lookup.shape[0]:,}**
    """),
        lookup_df.head(10),
    ])
    return book_id_map, data_dir, lookup_df, user_id_map


@app.cell
def _(data_dir, lookup_df, mo, os, pl):
    _input = os.path.join(data_dir, "raw_goodreads_interactions.parquet")
    _output = os.path.join(data_dir, "1_goodreads_interactions_merged.parquet")

    _lf = pl.scan_parquet(_input)
    _n_in = _lf.select(pl.len()).collect().item()

    _merged = (
        _lf.join(
            lookup_df.lazy().rename({"edition_csv_id": "book_id"}),
            on="book_id",
            how="left",
        )
        .with_columns(
            pl.coalesce("best_csv_id", "book_id").alias("book_id")
        )
        .drop("best_csv_id")
    )

    _merged.sink_parquet(_output, engine="streaming")
    _n_out = pl.scan_parquet(_output).select(pl.len()).collect().item()

    mo.md(f"""
    ### raw_goodreads_interactions
    {_n_in:,} rows in -> {_n_out:,} rows out | `{os.path.basename(_output)}`
    """)
    return


@app.cell
def _(book_id_map, data_dir, lookup_df, mo, os, pl, user_id_map):
    _input = os.path.join(data_dir, "raw_goodreads_interactions_dedup.parquet")
    _output = os.path.join(data_dir, "1_goodreads_interactions_dedup_merged.parquet")

    _lf = pl.scan_parquet(_input)
    _n_in = _lf.select(pl.len()).collect().item()

    # Step 1: Translate both user_id and book_id to CSV IDs
    _translated = (
        _lf
        # Cast book_id to Int64 for joining (user_id is already a string hash)
        .with_columns(
            pl.col("book_id").cast(pl.Int64),
        )
        # Translate book_id: Goodreads -> CSV
        .join(
            book_id_map.lazy().select(
                pl.col("book_id").alias("book_id_gr"),
                pl.col("book_id_csv"),
            ),
            left_on="book_id",
            right_on="book_id_gr",
            how="inner",
        )
        .drop("book_id")
        .rename({"book_id_csv": "book_id"})
        # Translate user_id: hashed string -> CSV integer
        .join(
            user_id_map.lazy().select(
                pl.col("user_id").alias("user_id_hash"),
                pl.col("user_id_csv"),
            ),
            left_on="user_id",
            right_on="user_id_hash",
            how="inner",
        )
        .drop("user_id")
        .rename({"user_id_csv": "user_id"})
    )

    # Step 2: Apply edition merging (edition_csv_id -> best_csv_id)
    _merged = (
        _translated
        .join(
            lookup_df.lazy().rename({"edition_csv_id": "book_id"}),
            on="book_id",
            how="left",
        )
        .with_columns(
            pl.coalesce("best_csv_id", "book_id").alias("book_id")
        )
        .drop("best_csv_id")
    )

    _merged.sink_parquet(_output, engine="streaming")
    _n_out = pl.scan_parquet(_output).select(pl.len()).collect().item()

    mo.md(f"""
    ### raw_goodreads_interactions_dedup
    {_n_in:,} rows in -> {_n_out:,} rows out | `{os.path.basename(_output)}`
    """)
    return

if __name__ == "__main__":
    app.run()
