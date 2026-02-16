import marimo

__generated_with = "0.19.8"
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
    # Preparing Data for BPR Algorithm Training

    In this notebook, we take the interactions dataset from goodreads (after merged and standardized book interactions), and prepare it in the format that the BPR algorithm expects. For that we need to:

    1. Devise a way to calculate weight/rating based on both implicit and explicit ratings.
    2. Remove any columns not necesbpry (only keeping `user_id`, `book_id`, `weight`)
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
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_path = os.path.join(
        project_root, "data", "goodreads_interactions_merged.parquet"
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

    We calculate a combined weight based on multiple engagement signals using a tiered approach:

    | Interaction Type | Weight |
    |-----------------|--------|
    | Base interaction (viewed/clicked) | +1.0 |
    | Actually read the book | +2.0 |
    | Rating (1-5 scale) | +(rating - 1), i.e., 0 to 4 |
    | Wrote a review | +3.0 |

    **Example weights:**
    - Interaction only: 1.0
    - Interaction + is_read: 3.0
    - Interaction + is_read + rating=5: 7.0
    - Interaction + is_read + rating=5 + review: 10.0
    """)
    return


@app.cell
def _(pl):
    def calculate_weight(df):
        """
        Calculate confidence weight for user-book interactions.

        Higher weight = stronger signal the user liked this book.

        Formula:
            weight = 1.0 (base interaction)
                   + 2.0 * is_read
                   + (rating - 1) if rating exists, else 0
                   + 3.0 * is_reviewed
        """
        return df.with_columns(
            [
                # rating contribution: (rating - 1) for 1-5 scale, or 0 if no rating
                pl.when(pl.col("rating") > 0)
                .then(pl.col("rating") - 1)
                .otherwise(0.0)
                .alias("rating_contrib"),
            ]
        ).with_columns(
            (
                # Base interaction (viewed/clicked) - weak signal
                1.0
                # Actually read it - strong engagement signal
                + 2.0 * pl.col("is_read").cast(pl.Float32)
                # Rating - explicit preference signal (0-4 contribution)
                + pl.col("rating_contrib")
                # Wrote a review - highest engagement
                + 3.0 * pl.col("is_reviewed").cast(pl.Float32)
            )
            .cast(pl.Float32)  # Downcast to 32-bit to save space
            .alias("weight")
        ).drop(["rating_contrib"])  # Clean up intermediate cols

    return (calculate_weight,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Process and Transform the Data
    """)
    return


@app.cell
def _(calculate_weight, df):
    df_with_weight = calculate_weight(df)
    df_with_weight.head(5)
    return (df_with_weight,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Select Final Columns and Save to Parquet
    """)
    return


@app.cell
def _(df_with_weight, os, project_root):
    bpr_df = df_with_weight.select(["user_id", "book_id", "weight"])

    output_path = os.path.join(project_root, "data", "bpr_interactions_merged.parquet")

    bpr_df.write_parquet(output_path)

    f"Saved {bpr_df.shape[0]:,} rows to {output_path}"
    return (bpr_df,)


@app.cell
def _(bpr_df):
    bpr_df.head(3)
    return


@app.cell
def _(bpr_df, mo):
    mo.md(f"""
    ### Final Dataset Summary

    - **Total interactions**: {bpr_df.shape[0]:,}
    - **Unique users**: {bpr_df["user_id"].n_unique():,}
    - **Unique books**: {bpr_df["book_id"].n_unique():,}
    - **Weight statistics**:
      - Min: {bpr_df["weight"].min():.3f}
      - Max: {bpr_df["weight"].max():.3f}
      - Mean: {bpr_df["weight"].mean():.3f}
    """)
    return


if __name__ == "__main__":
    app.run()
