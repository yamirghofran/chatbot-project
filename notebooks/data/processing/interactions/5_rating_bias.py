import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import os

    return mo, os, pl


@app.cell
def _(os, pl):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    data_dir = os.path.join(project_root, "data")

    interactions_path = os.path.join(data_dir, "4_goodreads_reviews_reduced.parquet")
    interactions_lf = pl.scan_parquet(interactions_path)

    return data_dir, interactions_lf


@app.cell
def _(interactions_lf):
    interactions_lf.head().collect()
    return


@app.cell
def _(interactions_lf):
    interactions_lf.collect().sample(n=1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Method 1: Mean-shift
    """)
    return


@app.cell
def _(interactions_lf, pl):
    #Get global mean
    global_mean = interactions_lf.select(
        pl.col("rating").mean()
    ).collect().item()

    #Get user's mean
    user_stats_lf = interactions_lf.group_by("user_id").agg(
        pl.col("rating").mean().alias("user_mean"),    
        pl.col("rating").std().alias("user_std"),
    )

    #Apply Mean-shift
    mean_shifted_lf = (
        interactions_lf
        .join(user_stats_lf, on="user_id", how="left")
        .with_columns(
            (
                pl.col("rating") - pl.col("user_mean") + pl.lit(global_mean)
            )
            .clip(0.0, 5.0)
            .round(1)
            .alias("rating")
        )
    )

    original_columns = list(interactions_lf.collect_schema().keys())
    mean_shifted_lf = mean_shifted_lf.select(original_columns)
    return mean_shifted_lf, original_columns, user_stats_lf


@app.cell
def _(mean_shifted_lf):
    mean_shifted_lf.collect().sample(n=1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Method 2: Z-Score
    """)
    return


@app.cell
def _(interactions_lf, pl, user_stats_lf):
    normalized_lf = (
        interactions_lf
        .join(user_stats_lf, on="user_id", how="left")
        .with_columns(
            pl.when(pl.col("user_std").is_null() | (pl.col("user_std") == 0))
            .then(0.0)
            .otherwise(
                (pl.col("rating") - pl.col("user_mean")) / pl.col("user_std")
            )
            .alias("rating")
        )
    )
    return


@app.cell
def _(interactions_lf, original_columns, pl, user_stats_lf):
    rescaled_lf_1 = (
        interactions_lf
        .join(user_stats_lf, on="user_id", how="left")
        .with_columns(
            pl.when(pl.col("user_std").is_null() | (pl.col("user_std") == 0))
            .then(2.5)
            .otherwise(
                (
                    5 / (
                        1
                        + (-((pl.col("rating") - pl.col("user_mean")) / pl.col("user_std"))).exp()
                    )
                )
            )
                .round(1)
            .alias("rating")
        )
        .select(original_columns)
    )
    rescaled_lf_1 = rescaled_lf_1.select(original_columns)
    rescaled_lf_1.collect().sample(n=1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Genre-Adjusted
    """)
    return


@app.cell
def _(data_dir, mean_shifted_lf, os):
    output_path = os.path.join(data_dir, "5_goodreads_reviews_reduced.parquet")
    mean_shifted_lf.sink_parquet(output_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
