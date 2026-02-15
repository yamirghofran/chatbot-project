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
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_dir = os.path.join(project_root, "data")

    with open(os.path.join(data_dir, "best_book_id_map.json")) as f:
        best_book_id_map = json.load(f)

    # invert: edition -> best (goodreads IDs)
    edition_to_best_gr = {}
    for best_id, edition_ids in best_book_id_map.items():
        best_id_int = int(best_id)
        for eid in edition_ids:
            edition_to_best_gr[int(eid)] = best_id_int

    gr_lookup = pl.DataFrame({
        "edition_book_id": list(edition_to_best_gr.keys()),
        "best_book_id": list(edition_to_best_gr.values()),
    })

    id_map = pl.read_parquet(os.path.join(data_dir, "raw_book_id_map.parquet"))

    # translate both sides to CSV ID space
    csv_lookup = (
        gr_lookup
        .join(
            id_map.select(
                pl.col("book_id").alias("edition_book_id"),
                pl.col("book_id_csv").alias("edition_csv_id"),
            ),
            on="edition_book_id",
            how="inner",
        )
        .join(
            id_map.select(
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
    return data_dir, lookup_df


@app.cell
def _(data_dir, lookup_df, mo, os, pl):
    _input = os.path.join(data_dir, "raw_goodreads_interactions.parquet")
    _output = os.path.join(data_dir, "goodreads_interactions_merged.parquet")

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

    _merged.collect(engine="streaming").write_parquet(_output)
    _n_out = pl.scan_parquet(_output).select(pl.len()).collect().item()

    mo.md(f"""
    ### raw_goodreads_interactions
    {_n_in:,} rows in -> {_n_out:,} rows out | `{os.path.basename(_output)}`
    """)
    return


@app.cell
def _(data_dir, lookup_df, mo, os, pl):
    _input = os.path.join(data_dir, "raw_goodreads_interactions_dedup.parquet")
    _output = os.path.join(data_dir, "goodreads_interactions_dedup_merged.parquet")

    _lf = pl.scan_parquet(_input)
    _n_in = _lf.select(pl.len()).collect().item()

    _lookup_str = lookup_df.with_columns(
        pl.col("edition_csv_id").cast(pl.String),
        pl.col("best_csv_id").cast(pl.String),
    )

    _merged = (
        _lf.join(
            _lookup_str.lazy().rename({"edition_csv_id": "book_id"}),
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
