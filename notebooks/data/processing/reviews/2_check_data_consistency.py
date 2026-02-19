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
    # Data Consistency Check — Reviews

    Verify that all `book_id` values in the reviews dataset
    have a corresponding entry in `goodreads_books_with_metrics.parquet`,
    then drop unmatched rows.
    """)
    return


@app.cell
def _(os, pl):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    data_dir = os.path.join(project_root, "data")

    books_df = pl.read_parquet(
        os.path.join(data_dir, "3_goodreads_books_with_metrics.parquet"),
        columns=["book_id"],
    )
    return books_df, data_dir


@app.cell
def _(books_df, mo):
    mo.md(f"""
    ## Books with Metrics
    - **Total books**: {books_df.shape[0]:,}
    - **book_id dtype**: `{books_df.schema['book_id']}`
    """)
    return


@app.cell
def _(books_df, data_dir, mo, os, pl):
    # Check reviews_dedup_merged
    _path = os.path.join(data_dir, "1_goodreads_reviews_dedup_merged.parquet")
    _lf = pl.scan_parquet(_path)

    _total = _lf.select(pl.len()).collect().item()
    _unique_books = _lf.select("book_id").unique().collect()

    _missing = _unique_books.join(books_df, on="book_id", how="anti")
    _missing_count = (
        _lf.join(_missing.lazy(), on="book_id", how="inner")
        .select(pl.len()).collect().item()
    )

    mo.md(f"""
    ## Reviews Dedup Merged
    - **Total reviews**: {_total:,}
    - **Unique book_ids**: {_unique_books.shape[0]:,}
    - **Book IDs NOT in books_with_metrics**: {_missing.shape[0]:,}
    - **Reviews with missing book_id**: {_missing_count:,} ({_missing_count / _total * 100:.2f}%)

    {"All book_ids have a corresponding book!" if _missing.shape[0] == 0 else "Some book_ids are missing from books_with_metrics!"}
    """)
    return


@app.cell
def _(books_df, data_dir, mo, os, pl):
    # Check reviews_spoiler_merged
    _path = os.path.join(data_dir, "1_goodreads_reviews_spoiler_merged.parquet")
    _lf = pl.scan_parquet(_path)

    _total = _lf.select(pl.len()).collect().item()
    _unique_books = _lf.select("book_id").unique().collect()

    _missing = _unique_books.join(books_df, on="book_id", how="anti")
    _missing_count = (
        _lf.join(_missing.lazy(), on="book_id", how="inner")
        .select(pl.len()).collect().item()
    )

    mo.md(f"""
    ## Reviews Spoiler Merged
    - **Total reviews**: {_total:,}
    - **Unique book_ids**: {_unique_books.shape[0]:,}
    - **Book IDs NOT in books_with_metrics**: {_missing.shape[0]:,}
    - **Reviews with missing book_id**: {_missing_count:,} ({_missing_count / _total * 100:.2f}%)

    {"All book_ids have a corresponding book!" if _missing.shape[0] == 0 else "Some book_ids are missing from books_with_metrics!"}
    """)
    return


@app.cell
def _(books_df, data_dir, mo, os, pl):
    # Clean dedup reviews — drop rows with no matching book_id
    _in_path = os.path.join(data_dir, "1_goodreads_reviews_dedup_merged.parquet")
    _out_path = os.path.join(data_dir, "2_goodreads_reviews_dedup_reduced.parquet")
    _lf = pl.scan_parquet(_in_path)
    _n_before = _lf.select(pl.len()).collect().item()

    _lf.join(books_df.lazy(), on="book_id", how="semi").sink_parquet(_out_path, engine="streaming")

    _n_after = pl.scan_parquet(_out_path).select(pl.len()).collect().item()
    _dropped = _n_before - _n_after

    mo.md(f"""
    ## Cleaned Reviews Dedup
    - **Before**: {_n_before:,}
    - **After**: {_n_after:,}
    - **Dropped**: {_dropped:,} ({_dropped / _n_before * 100:.2f}%)
    """)
    return


@app.cell
def _(books_df, data_dir, mo, os, pl):
    # Clean spoiler reviews — drop rows with no matching book_id
    _in_path = os.path.join(data_dir, "1_goodreads_reviews_spoiler_merged.parquet")
    _out_path = os.path.join(data_dir, "2_goodreads_reviews_spoiler_reduced.parquet")
    _lf = pl.scan_parquet(_in_path)
    _n_before = _lf.select(pl.len()).collect().item()

    _lf.join(books_df.lazy(), on="book_id", how="semi").sink_parquet(_out_path, engine="streaming")

    _n_after = pl.scan_parquet(_out_path).select(pl.len()).collect().item()
    _dropped = _n_before - _n_after

    mo.md(f"""
    ## Cleaned Reviews Spoiler
    - **Before**: {_n_before:,}
    - **After**: {_n_after:,}
    - **Dropped**: {_dropped:,} ({_dropped / _n_before * 100:.2f}%)
    """)
    return


if __name__ == "__main__":
    app.run()