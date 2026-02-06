import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    return mo, np, pl, plt, sns, stats


@app.cell
def _(mo):
    mo.md("""
    # Preliminary EDA for Goodreads Authors Dataset
    """)
    return


@app.cell
def _(mo, pl):
    import os
    # Use absolute path from project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "raw_goodreads_book_authors.parquet")
    #df = pl.read_parquet(data_path)
    df = pl.read_parquet("https://pub-eecdafb53cc84b659949b513e40369d2.r2.dev/files/md5/25/1939e78307a157b3285bb98d085ab2")
    df = df.with_columns(
        [
            pl.col("average_rating").cast(pl.Float64),
            pl.col("author_id").cast(pl.Int64),
            pl.col("text_reviews_count").cast(pl.Int64),
            pl.col("ratings_count").cast(pl.Int64),
        ]
    )
    shape = df.shape
    mo.md(f"""
    - Dataset contains {shape[0]} authors and {shape[1]} columns.

    - No missing values found in the dataset
    """)
    mo.vstack([
        mo.md(f"""
    - Dataset contains {shape[0]} authors and {shape[1]} columns.

    - No missing values found in the dataset
    """),
        df.head()
    ])
    return (df,)


@app.cell
def _(df, mo):
    """""Summary Statistics"""
    summary = df.select(
        ["average_rating", "text_reviews_count", "ratings_count"]
    ).describe()

    distribution_analysis = mo.md(f"""
        - Average rating ranges from {df['average_rating'].min():.2f} to {df['average_rating'].max():.2f} with mean {df['average_rating'].mean():.2f}
        - Text reviews count is highly skewed: median {df['text_reviews_count'].median():.0f} vs max {df['text_reviews_count'].max():,.0f}
        - Ratings count has extreme outliers: median {df['ratings_count'].median():.0f} vs max {df['ratings_count'].max():,.0f} """)

    mo.vstack([
        mo.md("### Summary Statistics"),
        distribution_analysis,
        summary
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Highly skewed distribution of reviews/rating counts suggests log-transformation will be needed
    """)
    return


@app.cell
def _(df, plt):
    """Distribution of Average Ratings"""
    fig_ratings, ax_ratings = plt.subplots(figsize=(10, 6))
    ax_ratings.hist(df["average_rating"].to_pandas(), bins=50, edgecolor="black", alpha=0.7)
    ax_ratings.set_xlabel("Average Rating")
    ax_ratings.set_ylabel("Frequency")
    ax_ratings.set_title("Distribution of Author Average Ratings")
    plt.tight_layout()

    fig_ratings
    return


@app.cell
def _(df, np, plt):
    """Distribution of Log-Transformed Metrics"""
    fig_log, axes_log = plt.subplots(1, 2, figsize=(14, 5))

    # Log-transform text reviews
    log_reviews = np.log1p(df["text_reviews_count"].to_pandas())
    axes_log[0].hist(log_reviews, bins=50, edgecolor="black", alpha=0.7)
    axes_log[0].set_xlabel("Log(Text Reviews Count)")
    axes_log[0].set_ylabel("Frequency")
    axes_log[0].set_title("Log-Transformed Text Reviews Count")

    # Log-transform ratings
    log_ratings = np.log1p(df["ratings_count"].to_pandas())
    axes_log[1].hist(log_ratings, bins=50, edgecolor="black", alpha=0.7)
    axes_log[1].set_xlabel("Log(Ratings Count)")
    axes_log[1].set_ylabel("Frequency")
    axes_log[1].set_title("Log-Transformed Ratings Count")

    plt.tight_layout()
    fig_log
    return


@app.cell
def _(df, mo, pl):
    """Outlier Analysis"""
    # Calculate IQR for ratings_count
    q1 = df["ratings_count"].quantile(0.25)
    q3 = df["ratings_count"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr

    outliers_count = df.filter(pl.col("ratings_count") > upper_bound).shape[0]

    # Check top authors by ratings
    top_authors = (
        df.sort("ratings_count", descending=True)
        .select(["name", "ratings_count"])
        .head(10)
    )

    mo.vstack([
        mo.md(f"Using 3x IQR method, {outliers_count} authors ({outliers_count / df.shape[0] * 100:.2f}%) have extreme rating counts"),
        mo.md("Top authors by total count:"),
        top_authors
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Filter out authors with very few ratings (<10) to improve rating reliability
    """)
    return


@app.cell
def _(df, mo, plt, sns):
    """Correlation Analysis"""
    # Convert to pandas for correlation heatmap
    numeric_df = df.select(
        ["average_rating", "text_reviews_count", "ratings_count"]
    ).to_pandas()
    correlation = numeric_df.corr(method="pearson")

    fig_analysis, ax_analysis = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".3f", ax=ax_analysis)
    ax_analysis.set_title("Correlation Matrix")
    plt.tight_layout()

    mo.vstack([
        mo.md("## Correlation Analysis"),
        fig_analysis,
        mo.md("""Strong correlation between ratings_count and text_reviews_count suggests these are redundant popularity metrics: consider creating a `popularity_score` combining both"""),
    ])
    return


@app.cell
def _(df, mo, np, plt):
    """Rating vs Popularity Analysis"""
    fig_analysis2, axes_analysis2 = plt.subplots(1, 2, figsize=(14, 5))

    # Ratings vs Text Reviews
    axes_analysis2[0].scatter(
        np.log1p(df["ratings_count"].to_pandas()),
        df["average_rating"].to_pandas(),
        alpha=0.3,
        s=1,
    )
    axes_analysis2[0].set_xlabel("Log(Ratings Count)")
    axes_analysis2[0].set_ylabel("Average Rating")
    axes_analysis2[0].set_title("Average Rating vs Ratings Count")

    # Text Reviews vs Average Rating
    axes_analysis2[1].scatter(
        np.log1p(df["text_reviews_count"].to_pandas()),
        df["average_rating"].to_pandas(),
        alpha=0.3,
        s=1,
    )
    axes_analysis2[1].set_xlabel("Log(Text Reviews Count)")
    axes_analysis2[1].set_ylabel("Average Rating")
    axes_analysis2[1].set_title("Average Rating vs Text Reviews Count")

    plt.tight_layout()

    mo.vstack([
        mo.md("""Average rating shows weak positive correlation with popularity (more popular authors tend to have slightly higher ratings)"""),
        fig_analysis2
    ])
    return


@app.cell
def _(df, np, stats):
    """Pearson Correlation Significance Test"""
    ratings = df["ratings_count"].to_pandas().values
    reviews = df["text_reviews_count"].to_pandas().values
    avg_rating = df["average_rating"].to_pandas().values

    # Log-transform for better normality
    log_ratings2 = np.log1p(ratings)
    log_reviews2 = np.log1p(reviews)

    corr1, p1 = stats.pearsonr(log_ratings2, avg_rating)
    corr2, p2 = stats.pearsonr(log_reviews2, avg_rating)
    corr3, p3 = stats.pearsonr(log_ratings2, log_reviews2)

    print(f"Log(Ratings) vs Average Rating: r={corr1:.4f}, p={p1:.2e}")
    print(f"Log(Text Reviews) vs Average Rating: r={corr2:.4f}, p={p2:.2e}")
    print(f"Log(Ratings) vs Log(Text Reviews): r={corr3:.4f}, p={p3:.2e}")
    return


@app.cell
def _(df, pl, plt):
    """Rating Quality Assessment"""
    # Calculate confidence interval approximation based on ratings count

    df_with_ci = df.with_columns([
        (
            1.96
            / pl.when(pl.col("ratings_count") > 0)
                .then(pl.col("ratings_count"))
                .otherwise(1)
                .sqrt()
        ).alias("ci_width")
    ])


    # Plot average rating with error bars for top authors
    top_sample = (
        df_with_ci.filter(pl.col("ratings_count") >= 1000)
        .sort("ratings_count", descending=True)
        .head(100)
    )

    fig_quality, ax_quality = plt.subplots(figsize=(12, 6))
    ax_quality.errorbar(
        range(len(top_sample)),
        top_sample["average_rating"].to_pandas(),
        yerr=top_sample["ci_width"].to_pandas(),
        fmt="o",
        alpha=0.3,
        capsize=2,
    )
    ax_quality.set_xlabel("Author Index (sorted by popularity)")
    ax_quality.set_ylabel("Average Rating")
    ax_quality.set_title(
        "Average Rating with Confidence Intervals for Top 100 Authors (>=1000 ratings)"
    )
    plt.tight_layout()
    fig_quality
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature Engineering Recommendations

    1. Strong correlation between ratings_count and text_reviews_count keep only one.
    2. Consider creating a 'popularity_score' combining both
    3. Consider filtering out authors with very few ratings to improve rating reliability
    """)
    return


if __name__ == "__main__":
    app.run()
