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
    interactions_lf.collect().sample(n=1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Method 1: Mean-shift

    This method adjusts each user's ratings based on their personal average rating.
    For each user, we first compute the user's average rating (`user_mean`) and the global average rating across all users (`global_mean`) then transform each rating using `adjusted_rating = rating - user_mean + global_mean`

    The aim is to remove rating bias by centering each user's ratings around the overall average.
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
    # Method 2: Sigmoid-transformed Gaussian normalization

    This method first standardizes each user's ratings using their own mean and standard deviation, then maps the result back into the 0–5 rating range using a sigmoid function.

    We compute a z-sore `z = (rating - user_mean) / user_std` based on the user's average rating (`user_mean`) and the user's rating standard deviation (`user_std`)

    Since Gaussian normalization produces values centered around 0, often including negative values and values above 1. The z-score is then passed through a sigmoid function and rescaled to the 0–5 as such : `adjusted_rating = 5 / (1 + exp(-z))`
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
    rescaled_lf = (
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
    rescaled_lf = rescaled_lf.select(original_columns)
    rescaled_lf.collect().sample(n=1000)
    return (rescaled_lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Method 3:  Decoupling Normalization
    The approach converts the rating of an item by a user into a probability for the item to be favored by the user, based on the user's own rating distribution.

    For each user and rating value `r`, it computes:
    - `p(Rating <= r)` = the proportion of the user's ratings that are less than or equal to `r`
    - `p(Rating = r)` = the proportion of the user's ratings that are exactly equal to `r`

    The transformed score is then computed as: `decoupled_score = p(Rating <= r) - p(Rating = r) / 2`.
    It converts a raw rating into a measure of how favorable that rating is for that specific user.

    This method was tested as the ratings are discrete values rather than fully continuous values. Decoupling normalization works directly with the empirical distribution of each user's rating categories, so it can be better suited to discrete rating data.

    A high transformed score means that, according to that user's own rating habits, the item is relatively strongly favored. A lower score means the rating is not especially meaningful for that user, even if the raw rating value looks high.
    """)
    return


@app.cell
def _(interactions_lf, original_columns, pl):
    # one row per (user, rating) with count
    rating_counts_lf = (
        interactions_lf
        .group_by(["user_id", "rating"])
        .agg(pl.len().alias("rating_count"))
    )

    # total ratings per user
    user_totals_lf = (
        interactions_lf
        .group_by("user_id")
        .agg(pl.len().alias("user_total"))
    )

    # combine and compute p_eq
    rating_dist_lf = (
        rating_counts_lf
        .join(user_totals_lf, on="user_id", how="left")
        .with_columns(
            (pl.col("rating_count") / pl.col("user_total")).alias("p_eq")
        )
    )

    # compute cumulative count <= r within each user
    rating_dist_df = (
        rating_dist_lf
        .collect()
        .sort(["user_id", "rating"])
        .with_columns(
            pl.col("rating_count").cum_sum().over("user_id").alias("cum_count")
        )
        .with_columns(
            (pl.col("cum_count") / pl.col("user_total")).alias("p_le")
        )
        .with_columns(
            (pl.col("p_le") - pl.col("p_eq") / 2).alias("decoupled_score")
        )
    )

    # keep only mapping columns
    decoupled_map_df = rating_dist_df.select(["user_id", "rating", "decoupled_score"])

    # join back to original rows
    decoupled_lf = (
        interactions_lf
        .join(decoupled_map_df.lazy(), on=["user_id", "rating"], how="left")
        .with_columns(
            (pl.col("decoupled_score") * 5).round(1).clip(0.0, 5.0).alias("rating")
        )
        .select(original_columns)
    )

    decoupled_lf.collect().sample(n=1000)
    return (decoupled_lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparing the methods
    """)
    return


@app.cell
def _(pl):
    def user_mean_spread(lf, method_name):
        return (
            lf.group_by("user_id")
            .agg(pl.col("rating").mean().alias("user_mean"))
            .select(
                pl.lit(method_name).alias("method"),
                pl.col("user_mean").mean().alias("avg_user_mean"),
                pl.col("user_mean").std().alias("spread_user_means"),
            )
        )

    def overall_stats(lf, method_name):
        return (
            lf.select(
                pl.lit(method_name).alias("method"),
                pl.col("rating").cast(pl.Float64).min().alias("min_rating"),
                pl.col("rating").cast(pl.Float64).quantile(0.25).alias("p25"),
                pl.col("rating").cast(pl.Float64).median().alias("p50"),
                pl.col("rating").cast(pl.Float64).quantile(0.75).alias("p75"),
                pl.col("rating").cast(pl.Float64).max().alias("max_rating"),
                pl.col("rating").cast(pl.Float64).mean().alias("mean_rating"),
                pl.col("rating").cast(pl.Float64).std().alias("std_rating"),
            )
        )

    return overall_stats, user_mean_spread


@app.cell
def _(
    decoupled_lf,
    interactions_lf,
    mean_shifted_lf,
    pl,
    rescaled_lf,
    user_mean_spread,
):
    comparison_user_means = pl.concat([
        user_mean_spread(interactions_lf, "raw"),
        user_mean_spread(mean_shifted_lf, "mean_shift"),
        user_mean_spread(rescaled_lf, "gaussian"),
        user_mean_spread(decoupled_lf, "decoupling"),
    ]).collect()

    comparison_user_means
    return


@app.cell
def _(
    decoupled_lf,
    interactions_lf,
    mean_shifted_lf,
    overall_stats,
    pl,
    rescaled_lf,
):
    comparison_distribution = pl.concat([
        overall_stats(interactions_lf, "raw"),
        overall_stats(mean_shifted_lf, "mean_shift"),
        overall_stats(rescaled_lf, "gaussian"),
        overall_stats(decoupled_lf, "decoupling"),
    ]).collect()

    comparison_distribution
    return


@app.cell
def _(decoupled_lf, interactions_lf, mean_shifted_lf, pl, rescaled_lf):
    raw_df = interactions_lf.select(["user_id", "book_id", "rating"]).collect().rename({"rating": "raw_rating"})
    mean_df = mean_shifted_lf.select(["user_id", "book_id", "rating"]).collect().rename({"rating": "mean_rating"})
    gaussian_df = rescaled_lf.select(["user_id", "book_id", "rating"]).collect().rename({"rating": "gaussian_rating"})
    decoupled_df = decoupled_lf.select(["user_id", "book_id", "rating"]).collect().rename({"rating": "decouple_rating"})


    joined_df = (
        raw_df
        .join(mean_df, on=["user_id", "book_id"])
        .join(gaussian_df, on=["user_id", "book_id"])
        .join(decoupled_df, on=["user_id", "book_id"])
    )


    correlation_stats = pl.DataFrame({
        "method": ["mean_shift", "guassian", "decoupling"],
        "correlation_with_raw": [
            joined_df.select(pl.corr("raw_rating", "mean_rating")).item(),
            joined_df.select(pl.corr("raw_rating", "gaussian_rating")).item(),        
            joined_df.select(pl.corr("raw_rating", "decouple_rating")).item()
        ]
    })

    correlation_stats
    return (joined_df,)


@app.cell
def _(interactions_lf, joined_df, pl):
    # Getting a sample of users visually show the change in user rating after each method

    sample_users = (
        interactions_lf
        .select("user_id")
        .unique()
        .collect()
        .sample(n=1000)["user_id"]
        .to_list()
    )

    sample_df = (
        joined_df
        .filter(pl.col("user_id").is_in(sample_users))
        .head(20)
    )

    sample_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Justification for Choosing Decoupling Normalization

    - Decoupling has the lowest spread of user means: 0.0132 showing that it reduced the user bias most.
    - Decoupling kept the spread closest to the original at 1.243

    Decoupling normalization is the most suitable method as it achieves the strongest reduction in user bias while maintaining a realistic rating distribution.
    """)
    return


@app.cell
def _(data_dir, decoupled_lf, os):
    output_path = os.path.join(data_dir, "5_goodreads_reviews_reduced.parquet")
    decoupled_lf.sink_parquet(output_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
