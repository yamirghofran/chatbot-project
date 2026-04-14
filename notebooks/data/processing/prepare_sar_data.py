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
    from bookdb.utils.paths import find_project_root
    project_root = find_project_root()
    data_path = os.path.join(
        project_root, "data", "raw_goodreads_interactions_dedup_sample.parquet"
    )

    df = pl.read_parquet(data_path)
    return df, project_root


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
def _():
    from bookdb.processing.interactions import calculate_weight_sar as calculate_weight
    return (calculate_weight,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Convert Timestamp to Unix Timestamp
    """)
    return


@app.cell
def _():
    from bookdb.processing.interactions import convert_to_unix_timestamp
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
def _(df_with_timestamp, os, project_root):
    sar_df = df_with_timestamp.select(["user_id", "book_id", "weight", "timestamp"])

    output_path = os.path.join(project_root, "data", "sar_interactions_sample.parquet")

    sar_df.write_parquet(output_path)

    f"Saved {sar_df.shape[0]:,} rows to {output_path}"
    return (sar_df,)


@app.cell
def _(sar_df):
    sar_df.head(3)
    return


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
