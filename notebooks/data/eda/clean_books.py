import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from scipy import stats
    import json
    import os
    return json, mo, os, pl


@app.cell
def _(mo, os, pl):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_path = os.path.join(project_root, "data", "raw_goodreads_books.parquet")
    df = pl.read_parquet(data_path)

    df = df.with_columns(
        [
            pl.when(pl.col(c).str.len_chars() == 0)
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in [
                "publication_month",
                "publication_year",
                "publication_day",
            ]
        ]
    )

    df = df.with_columns(
        [
            pl.col("text_reviews_count").cast(pl.Int64, strict=False),
            pl.col("ratings_count").cast(pl.Int64, strict=False),
            pl.col("average_rating").cast(pl.Float64, strict=False),
            pl.col("num_pages").cast(pl.Int64, strict=False),
            pl.col("publication_day").cast(pl.Int64, strict=False),
            pl.col("publication_month").cast(pl.Int64, strict=False),
            pl.col("publication_year").cast(pl.Int64, strict=False),
        ]
    )

    mo.vstack(
        [
            df.head(),
            mo.md(f"Dataset contains {df.shape[0]} books and {df.shape[1]} columns."),
        ]
    )
    return df, project_root


@app.cell
def _(df, json, mo, os, pl, project_root):
    best_book_map_path = os.path.join(project_root, "data", "best_book_id_map.json")

    # Load the best book ID map
    with open(best_book_map_path, "r") as f:
        best_book_map = json.load(f)

    # Extract non-best book IDs (values from the JSON to drop)
    # Convert to strings to match the book_id column type in the dataframe
    non_best_book_ids = set()
    for book_id_list in best_book_map.values():
        non_best_book_ids.update(str(book_id) for book_id in book_id_list)

    # Filter to EXCLUDE non-best book IDs
    # This keeps: (1) best book IDs and (2) books with only one version
    df_filtered = df.filter(~pl.col("book_id").is_in(non_best_book_ids))

    mo.vstack(
        [
            mo.md(f"**Before filtering:** {df.shape[0]} books"),
            mo.md(f"**After filtering:** {df_filtered.shape[0]} books"),
            mo.md(
                f"**Removed:** {df.shape[0] - df_filtered.shape[0]} books ({((df.shape[0] - df_filtered.shape[0]) / df.shape[0] * 100):.1f}%)"
            ),
            mo.md("### Filtered Dataset Sample"),
            df_filtered.head(),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
