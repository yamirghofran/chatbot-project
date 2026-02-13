import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    return mo, os, pl


@app.cell
def _(mo):
    mo.md("""
    # Preparing Data for SAR Algorithm Training

    In this notebook, we take the dedup interactions dataset from goodreads (after aggregated book interactions), and prepare it in the format that the SAR algorithm expects. For that we need to:

    1. Devise a way to calculate weight/rating based on both implicit and explicit ratings.
    2. Remove any columns not necessary (only keeping `user_id`, `book_id`, `weight`, `timestamp`)
    3. Convert the timestamp to unix timestamp
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Loading the Interactions Dedup Dataset
    """)
    return


@app.cell
def _(os, pl):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_path = os.path.join(
        project_root, "data", "raw_goodreads_interactions_dedup.parquet"
    )

    _df_full = pl.read_parquet(data_path)
    _full_rows = _df_full.shape[0]
    df = _df_full.sample(fraction=0.05, seed=42)
    del _df_full
    return (df,)


@app.cell
def _(df):
    df.head(3)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Weight Calculation Strategy

    We calculate a combined weight based on multiple engagement signals:

    - **Explicit rating (1-5)**: Primary signal, normalized to 0-1 range (rating / 5)
    - **is_read**: Binary signal indicating if the user read the book
    - **has_review**: Binary signal indicating if the user left a review

    The weight formula combines these with configurable weights:
    `weight = w1 * rating_normalized + w2 * is_read + w3 * has_review`
    """)
    return


@app.cell
def _(pl):
    def calculate_weight(df, rating_weight=0.5, is_read_weight=0.3, review_weight=0.2):
        """
        Calculate a combined engagement weight for each interaction.

        Parameters:
        - df: Polars DataFrame with rating, is_read, and review_text_incomplete columns
        - rating_weight: Weight for explicit rating component (default 0.5)
        - is_read_weight: Weight for is_read component (default 0.3)
        - review_weight: Weight for review component (default 0.2)

        Returns:
        - Polars DataFrame with added 'weight' column
        """
        return df.with_columns(
            [
                (pl.col("rating") / 5).alias("rating_normalized"),
                (pl.col("review_text_incomplete").str.len_chars() > 0).alias("has_review"),
            ]
        ).with_columns(
            (
                rating_weight * pl.col("rating_normalized")
                + is_read_weight * pl.col("is_read").cast(pl.Float64)
                + review_weight * pl.col("has_review").cast(pl.Float64)
            ).alias("weight")
        )

    return (calculate_weight,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Convert Timestamp to Unix Timestamp
    """)
    return


@app.cell
def _(pl):
    def convert_to_unix_timestamp(df, timestamp_col="date_updated"):
        """
        Convert a date string column to unix timestamp.

        Parameters:
        - df: Polars DataFrame
        - timestamp_col: Name of the column containing date strings

        Returns:
        - Polars DataFrame with 'timestamp' column containing unix timestamps
        """
        return df.with_columns(
            pl.col(timestamp_col)
            .str.strptime(pl.Datetime, format="%a %b %d %H:%M:%S %z %Y")
            .dt.epoch("s")
            .alias("timestamp")
        )

    return (convert_to_unix_timestamp,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Process and Transform the Data
    """)
    return


@app.cell
def _(calculate_weight, convert_to_unix_timestamp, df):
    df_with_weight = calculate_weight(df)
    df_with_timestamp = convert_to_unix_timestamp(df_with_weight)
    df_with_timestamp.head(5)
    return (df_with_timestamp,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Select Final Columns and Save to Parquet
    """)
    return


@app.cell
def _(df_with_timestamp):
    sar_df = df_with_timestamp.select(["user_id", "book_id", "weight", "timestamp"])

    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    )
    output_path = _os.path.join(_project_root, "data", "sar_interactions.parquet")

    sar_df.write_parquet(output_path)

    f"Saved {sar_df.shape[0]:,} rows to {output_path}"
    return (sar_df,)


@app.cell
def _(mo, sar_df):
    mo.md(f"""
    ### Final Dataset Summary

    - **Total interactions**: {sar_df.shape[0]:,}
    - **Unique users**: {sar_df["user_id"].n_unique():,}
    - **Unique books**: {sar_df["book_id"].n_unique():,}
    - **Weight statistics**:
      - Min: {sar_df["weight"].min():.3f}
      - Max: {sar_df["weight"].max():.3f}
      - Mean: {sar_df["weight"].mean():.3f}
    """)
    return


if __name__ == "__main__":
    app.run()
