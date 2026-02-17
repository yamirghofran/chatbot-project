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
    # Data Consistency Check

    Verify that all `book_id` values in interactions and reviews datasets
    have a corresponding entry in `goodreads_books_with_metrics.parquet`.
    """)
    return


@app.cell
def _(os, pl):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    data_dir = os.path.join(project_root, "data")

    books_df = pl.read_parquet(
        os.path.join(data_dir, "goodreads_books_with_metrics.parquet"),
        columns=["book_id"],
    )
    book_ids = books_df.select("book_id").to_series().to_list()
    book_ids_set = set(book_ids)
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
def _(data_dir, mo, os, pl):
    # Check interactions_merged_timestamps
    interactions_path = os.path.join(data_dir, "goodreads_interactions_merged_timestamps.parquet")

    if not os.path.exists(interactions_path):
        _output = mo.md("`goodreads_interactions_merged_timestamps.parquet` does not exist yet. Skipping.")
    else:
        interactions_lf = pl.scan_parquet(interactions_path)

        total_interactions = interactions_lf.select(pl.len()).collect().item()
        unique_book_ids = (
            interactions_lf
            .select("book_id")
            .unique()
            .collect()
        )

        # Check which book_ids are NOT in books_with_metrics
        books_ref = pl.read_parquet(
            os.path.join(data_dir, "goodreads_books_with_metrics.parquet"),
            columns=["book_id"],
        )

        missing = unique_book_ids.join(
            books_ref,
            on="book_id",
            how="anti",  # Keep only rows NOT in books_ref
        )

        # Count interactions with missing book_ids
        missing_interaction_count = (
            interactions_lf
            .join(
                missing.lazy(),
                on="book_id",
                how="inner",
            )
            .select(pl.len())
            .collect()
            .item()
        )

        _output = mo.md(f"""
    ## Interactions Merged Timestamps
    - **Total interactions**: {total_interactions:,}
    - **Unique book_ids**: {unique_book_ids.shape[0]:,}
    - **Book IDs NOT in books_with_metrics**: {missing.shape[0]:,}
    - **Interactions with missing book_id**: {missing_interaction_count:,} ({missing_interaction_count / total_interactions * 100:.2f}%)

    {"All book_ids have a corresponding book!" if missing.shape[0] == 0 else "Some book_ids are missing from books_with_metrics!"}
        """)
    _output
    return


@app.cell
def _(data_dir, mo, os, pl):
    # Check reviews_dedup_merged
    reviews_path = os.path.join(data_dir, "goodreads_reviews_dedup_merged.parquet")
    reviews_lf = pl.scan_parquet(reviews_path)

    total_reviews = reviews_lf.select(pl.len()).collect().item()
    unique_review_book_ids = (
        reviews_lf
        .select("book_id")
        .unique()
        .collect()
    )

    # Check which book_ids are NOT in books_with_metrics
    books_ref2 = pl.read_parquet(
        os.path.join(data_dir, "goodreads_books_with_metrics.parquet"),
        columns=["book_id"],
    )

    missing_reviews = unique_review_book_ids.join(
        books_ref2,
        on="book_id",
        how="anti",
    )

    # Count reviews with missing book_ids
    missing_review_count = (
        reviews_lf
        .join(
            missing_reviews.lazy(),
            on="book_id",
            how="inner",
        )
        .select(pl.len())
        .collect()
        .item()
    )

    mo.md(f"""
    ## Reviews Dedup Merged
    - **Total reviews**: {total_reviews:,}
    - **Unique book_ids**: {unique_review_book_ids.shape[0]:,}
    - **Book IDs NOT in books_with_metrics**: {missing_reviews.shape[0]:,}
    - **Reviews with missing book_id**: {missing_review_count:,} ({missing_review_count / total_reviews * 100:.2f}%)
    """)
    return


@app.cell
def _(books_df, data_dir, mo, os, pl):
    # Drop interactions with no matching book_id
    _int_path = os.path.join(data_dir, "goodreads_interactions_merged_timestamps.parquet")
    _int_out = os.path.join(data_dir, "goodreads_interactions_reduced.parquet")
    _int_lf = pl.scan_parquet(_int_path)
    _n_before = _int_lf.select(pl.len()).collect().item()

    # Keep only interactions with a matching book_id
    _cleaned = _int_lf.join(
        books_df.lazy(),
        on="book_id",
        how="semi",  # Keep rows from left that have a match in right
    )

    _cleaned.sink_parquet(_int_out, engine="streaming")

    _n_after = pl.scan_parquet(_int_out).select(pl.len()).collect().item()
    _dropped = _n_before - _n_after

    mo.md(f"""
    ## Cleaned Interactions
    - **Before**: {_n_before:,}
    - **After**: {_n_after:,}
    - **Dropped**: {_dropped:,} ({_dropped / _n_before * 100:.2f}%)
    """)
    return


@app.cell
def _(books_df, data_dir, mo, os, pl):
    # Drop reviews with no matching book_id
    reviews_path = os.path.join(data_dir, "goodreads_reviews_dedup_merged.parquet")
    _reviews_out = os.path.join(data_dir, "goodreads_reviews_reduced.parquet")
    reviews_lf = pl.scan_parquet(reviews_path)
    n_reviews_before = reviews_lf.select(pl.len()).collect().item()

    # Keep only reviews with a matching book_id
    cleaned_reviews = reviews_lf.join(
        books_df.lazy(),
        on="book_id",
        how="semi",
    )

    cleaned_reviews.sink_parquet(_reviews_out, engine="streaming")

    n_reviews_after = pl.scan_parquet(_reviews_out).select(pl.len()).collect().item()
    reviews_dropped = n_reviews_before - n_reviews_after

    mo.md(f"""
    ## Cleaned Reviews
    - **Before**: {n_reviews_before:,}
    - **After**: {n_reviews_after:,}
    - **Dropped**: {reviews_dropped:,} ({reviews_dropped / n_reviews_before * 100:.2f}%)
    """)
    return


if __name__ == "__main__":
    app.run()
