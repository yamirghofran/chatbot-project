import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import duckdb
    import os

    return duckdb, mo, os, pl


@app.cell
def _(mo, os, pl):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    books_path = os.path.join(project_root, "data", "raw_goodreads_books.parquet")
    books_df = pl.read_parquet(books_path)

    interactions_path = os.path.join(project_root, "data", "goodreads_interactions_merged.parquet")
    interactions_df = pl.read_parquet(interactions_path)

    mo.vstack([
        mo.md("## Books Dataset"),
        books_df.head(),
        mo.md("## Interactions Dataset"),
        interactions_df.head()
    ])
    return books_df, project_root


@app.cell
def _(mo):
    mo.md("""
    ## Aggregate Interaction Metrics

    Using DuckDB to aggregate metrics for each book from the interactions dataset:

    - **num_interactions**: Total count of interactions for each book
    - **num_read**: Count of interactions where `is_read = 1`
    - **num_ratings**: Count of interactions where `rating` is between 1 and 5 (inclusive)
    - **num_reviews**: Count of interactions where `is_reviewed = 1`
    """)
    return


@app.cell
def _(duckdb):
    book_metrics_df = duckdb.sql("""
        SELECT
            book_id,
            COUNT(*) AS num_interactions,
            SUM(CASE WHEN is_read = 1 THEN 1 ELSE 0 END) AS num_read,
            SUM(CASE WHEN rating >= 1 AND rating <= 5 THEN 1 ELSE 0 END) AS num_ratings,
            SUM(CASE WHEN is_reviewed = 1 THEN 1 ELSE 0 END) AS num_reviews
        FROM interactions_df
        GROUP BY book_id
    """).pl()

    book_metrics_df.head()
    return (book_metrics_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Join Metrics to Books Dataset

    Left join the aggregated metrics to the books dataframe so that books without any interactions
    will have `null` values for the metric columns, which we'll then fill with 0.
    """)
    return


@app.cell
def _(book_metrics_df, books_df, pl):
    books_with_metrics_df = books_df.join(
        book_metrics_df,
        on="book_id",
        how="left"
    ).with_columns(
        pl.col("num_interactions").fill_null(0).cast(pl.Int64),
        pl.col("num_read").fill_null(0).cast(pl.Int64),
        pl.col("num_ratings").fill_null(0).cast(pl.Int64),
        pl.col("num_reviews").fill_null(0).cast(pl.Int64),
    )

    books_with_metrics_df.select(
        "book_id", "title", "num_interactions", "num_read", "num_ratings", "num_reviews"
    ).head(10)
    return (books_with_metrics_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Summary Statistics

    Let's check the distribution of the aggregated metrics to ensure they look reasonable.
    """)
    return


@app.cell
def _(books_with_metrics_df):
    books_with_metrics_df.select(
        "num_interactions", "num_read", "num_ratings", "num_reviews"
    ).describe()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Save to Parquet

    Save the enriched books dataset with the new aggregated metric columns to a parquet file.
    """)
    return


@app.cell
def _(books_with_metrics_df, mo, os, project_root):
    output_path = os.path.join(project_root, "data", "books_with_metrics.parquet")
    books_with_metrics_df.write_parquet(output_path)

    mo.md(f"✅ Saved enriched books dataset to: `{output_path}`")
    return


if __name__ == "__main__":
    app.run()
