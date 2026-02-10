import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from scipy import stats
    return mo, np, pl, plt


@app.cell
def _(mo):
    mo.md("""
    # Preliminary EDA for Goodreads Books Dataset
    """)
    return


@app.cell
def _(mo, pl):
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    return (df,)


@app.cell
def _(df, mo, pl):
    """Data Quality Assessment"""
    neg_ratings = df.filter(pl.col("ratings_count") < 0).shape[0]
    neg_reviews = df.filter(pl.col("text_reviews_count") < 0).shape[0]
    invalid_year = df.filter(
        (pl.col("publication_year") < 1000) | (pl.col("publication_year") > 2026)
    ).shape[0]
    invalid_pages = df.filter(
        (pl.col("num_pages") < 0) | (pl.col("num_pages") > 10000)
    ).shape[0]

    low_ratings = df.filter(pl.col("ratings_count") < 10).shape[0]

    null_counts = df.null_count()

    mo.vstack(
        [
            mo.md("### Missing values per column:"),
            null_counts,
            mo.md("### Reviews Quality"),
            mo.md(f"- **Negative ratings_count**: {neg_ratings} rows"),
            mo.md(f"- **Negative text_reviews_count**: {neg_reviews} rows"),
            mo.md(
                f"- **Authors with < 10 ratings**: {low_ratings} ({low_ratings / df.shape[0] * 100:.1f}%)"
            ),
            mo.md("### Books Quality"),
            mo.md(
                f"- **Books with invalid publication year < 1000 or >2026**: {invalid_year} books"
            ),
            mo.md(f"- **Books with invalid pages > 10000**: {invalid_pages} books"),
            mo.md("### Findings: "),
            mo.md(r"""
        - Rating features missing values, however no negative values arise.
        - Original publication dates have a lot of missing values, as well as impossible dates. Will need imputation or filtering strategies.
        - Number of pages also missing. Consider relevance.
        """),
        ]
    )
    return


@app.cell
def _(df, mo, pl, plt):
    """Rating Distribution"""
    ratings = df.filter(pl.col("average_rating").is_not_null())["average_rating"]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _ax1.hist(ratings, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    _ax1.axvline(
        ratings.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {ratings.mean():.2f}",
    )
    _ax1.axvline(
        ratings.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {ratings.median():.2f}",
    )
    _ax1.set_xlabel("Average Rating")
    _ax1.set_ylabel("Frequency")
    _ax1.set_title("Distribution of Average Ratings")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    _bins = [0, 1, 2, 3, 4, 5]
    _labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]

    rating_binned = (
        pl.DataFrame({"rating": ratings.cast(pl.Float64)})
        .with_columns(
            rating=pl.col("rating").clip(0.0, 5.0),
            bin=(
                pl.when((pl.col("rating") >= 0) & (pl.col("rating") < 1)).then(pl.lit("0-1"))
                .when((pl.col("rating") >= 1) & (pl.col("rating") < 2)).then(pl.lit("1-2"))
                .when((pl.col("rating") >= 2) & (pl.col("rating") < 3)).then(pl.lit("2-3"))
                .when((pl.col("rating") >= 3) & (pl.col("rating") < 4)).then(pl.lit("3-4"))
                .when((pl.col("rating") >= 4) & (pl.col("rating") <= 5)).then(pl.lit("4-5"))
                .otherwise(None)
            ),
        )
        .filter(pl.col("bin").is_not_null())
        .group_by("bin")
        .agg(pl.len().alias("count"))
        # keep the bin order stable:
        .with_columns(
            bin_order=pl.col("bin").replace(
                {"0-1": 0, "1-2": 1, "2-3": 2, "3-4": 3, "4-5": 4}
            )
        )
        .sort("bin_order")
        .drop("bin_order")
    )



    _ax2.bar(
        rating_binned["bin"],
        rating_binned["count"],
        color="coral",
        edgecolor="black",
        alpha=0.7,
    )
    _ax2.set_xlabel("Rating Range")
    _ax2.set_ylabel("Count")
    _ax2.set_title("Books by Rating Range")
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Rating Statistics"),
            _fig,
            mo.md(f"- **Mean**: {ratings.mean():.3f}"),
            mo.md(f"- **Median**: {ratings.median():.3f}"),
            mo.md(f"- **Std Dev**: {ratings.std():.3f}"),
            mo.md(f"- **Min**: {ratings.min():.3f} | **Max**: {ratings.max():.3f}"),
            mo.md(
                f"- **Skewness**: {ratings.skew():.3f} | **Kurtosis**: {ratings.kurtosis():.3f}"
            ),
        ]
    )
    return


@app.cell
def _(df, mo, pl, plt):
    """Review Counts Distribution"""
    _review_data = (
        df.filter(
            (pl.col("ratings_count").is_not_null()) & (pl.col("ratings_count") > 0)
        )
        .select("ratings_count", "text_reviews_count")
        .filter(pl.col("text_reviews_count").is_not_null())
    )

    ratings_count = _review_data["ratings_count"]
    text_reviews = _review_data["text_reviews_count"]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _ax1.hist(ratings_count, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    _ax1.set_xlabel("Ratings Count (log scale)")
    _ax1.set_ylabel("Frequency")
    _ax1.set_title("Distribution of Ratings Count")
    _ax1.set_xscale("log")
    _ax1.set_yscale("log")
    _ax1.grid(True, alpha=0.3)

    _ax2.hist(text_reviews, bins=50, color="coral", edgecolor="black", alpha=0.7)
    _ax2.set_xlabel("Text Reviews Count (log scale)")
    _ax2.set_ylabel("Frequency")
    _ax2.set_title("Distribution of Text Reviews Count")
    _ax2.set_xscale("log")
    _ax2.set_yscale("log")
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Review Counts Distribution"),
            _fig,
            mo.md("### Findings:"),
            mo.md("""- Popularity is highly unequal
            - The dataset is dominated by low-engagement books
            - Ratings > text reviews --> Star ratings are cheap vs written reviews are costly
            """)
        ])
    return


@app.cell
def _(df, mo, pl, plt):
    """Ratings Count vs Text Reviews Count"""
    review_data = df.filter(
        (pl.col("ratings_count").is_not_null())
        & (pl.col("ratings_count") > 0)
        & (pl.col("text_reviews_count").is_not_null())
        & (pl.col("ratings_count") < 500000)
        & (pl.col("text_reviews_count") < 50000)
    ).select("ratings_count", "text_reviews_count")

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _ax1.scatter(
        review_data["ratings_count"],
        review_data["text_reviews_count"],
        alpha=0.3,
        s=5,
        color="steelblue",
    )
    _ax1.set_xlabel("Ratings Count")
    _ax1.set_ylabel("Text Reviews Count")
    _ax1.set_title("Ratings Count vs Text Reviews Count")
    _ax1.grid(True, alpha=0.3)

    _ax2.loglog(
        review_data["ratings_count"],
        review_data["text_reviews_count"],
        "o",
        alpha=0.2,
        markersize=2,
        color="coral",
    )
    _ax2.set_xlabel("Ratings Count (log)")
    _ax2.set_ylabel("Text Reviews Count (log)")
    _ax2.set_title("Log-Log Scatter: Ratings vs Reviews")
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Ratings Count vs Text Reviews Count"),
            _fig,
            mo.md("### Findings:"),
            mo.md("""- Popularity drives engagement but unevenly
            - Ratings scale faster than reviews makes sense because writing is effort. Same findings as previous plot
            - Books with more ratings tend to have more text reviews.
            - Each text review is also a rating but not every rating is a text review""")
        ])
    return


@app.cell
def _(df, mo, pl, plt):
    """Page Count Distribution"""
    pages = df.filter(
        (pl.col("num_pages").is_not_null())
        & (pl.col("num_pages") > 0)
        & (pl.col("num_pages") <= 10000)
    )["num_pages"]


    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _ax1.hist(pages, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    _ax1.axvline(
        pages.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {pages.mean():.0f}",
    )
    _ax1.axvline(
        pages.median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {pages.median():.0f}",
    )
    _ax1.set_xlabel("Number of Pages")
    _ax1.set_ylabel("Frequency")
    _ax1.set_title("Distribution of Page Counts")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    _ax2.boxplot(
        pages, vert=True, patch_artist=True, boxprops=dict(facecolor="lightcoral")
    )
    _ax2.set_ylabel("Number of Pages")
    _ax2.set_title("Box Plot of Page Counts")
    _ax2.grid(True, alpha=0.3)


    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Page Count Statistics"),
            _fig,
            mo.md(f"- **Mean**: {pages.mean():.0f} pages"),
            mo.md(f"- **Median**: {pages.median():.0f} pages"),
            mo.md(f"- **Std Dev**: {pages.std():.0f} pages"),
            mo.md(f"- **Min**: {pages.min():.0f} | **Max**: {pages.max():.0f}"),
            mo.md("### Findings:"),
            mo.md("""- Page counts follow a highly right-skewed distribution
            - Most books are around 250 pages long while a small number of extremely long books create a long tail and inflate the mean""")
    ])
    
    return


@app.cell
def _(df, mo, pl, plt):
    """Publication Year Trends"""
    year_data = df.filter(
        (pl.col("publication_year").is_not_null())
        & (pl.col("publication_year") >= 1000)
        & (pl.col("publication_year") <= 2026)
    )

    yearly_stats = (
        year_data.group_by("publication_year")
        .agg(
            pl.len().alias("count"), pl.col("average_rating").mean().alias("avg_rating")
        )
        .sort("publication_year")
    )

    year_data_with_centuries = year_data.with_columns(
        century=(pl.col("publication_year") // 100) * 100
    )
    century_stats = (
        year_data_with_centuries.group_by("century")
        .agg(pl.len().alias("count"))
        .sort("century")
    )

    _fig, (_ax2, _ax3) = plt.subplots(1, 2, figsize=(16, 5))


    _ax2.bar(
        century_stats["century"].cast(str),
        century_stats["count"],
        color="coral",
        alpha=0.7,
    )
    _ax2.set_xlabel("Century")
    _ax2.set_ylabel("Number of Books")
    _ax2.set_title("Books Published per Century")
    _ax2.tick_params(axis="x", rotation=45)
    _ax2.grid(True, alpha=0.3)

    _ax3.plot(
        yearly_stats["publication_year"],
        yearly_stats["avg_rating"],
        color="green",
        linewidth=1.5,
    )
    _ax3.set_xlabel("Publication Year")
    _ax3.set_ylabel("Average Rating")
    _ax3.set_title("Average Rating by Publication Year")
    _ax3.set_ylim(3, 5)
    _ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Publication Year Statistics"),
            _fig,
            mo.md(f"- **Earliest Year**: {yearly_stats['publication_year'][0]}"),
            mo.md(f"- **Latest Year**: {yearly_stats['publication_year'][-1]}"),
            mo.md(
                f"- **Peak Year**: {yearly_stats.sort('count', descending=True)['publication_year'][0]} ({yearly_stats.sort('count', descending=True)['count'][0]:,} books)"
            ),
            mo.md("""- Book publishing increases dramatically over time
            - Vast majority of books come from the 20th and 21st centuries. 
            - Early publication years are sparsely represented 
            """)
        ]
    )
    return


@app.cell
def _(df, mo, np, pl, plt):
    """Correlations Heatmap"""
    corr_data = df.filter(
        (pl.col("average_rating").is_not_null())
        & (pl.col("ratings_count").is_not_null())
        & (pl.col("ratings_count") > 0)
        & (pl.col("text_reviews_count").is_not_null())
        & (pl.col("num_pages").is_not_null())
        & (pl.col("num_pages") > 0)
        & (pl.col("publication_year").is_not_null())
        & (pl.col("publication_year") >= 1900)
        & (pl.col("publication_year") <= 2026)
    ).select(
        "average_rating",
        "ratings_count",
        "text_reviews_count",
        "num_pages",
        "publication_year",
    )

    corr_matrix = corr_data.corr()

    _fig, _ax = plt.subplots(figsize=(10, 8))
    _im = _ax.imshow(
        corr_matrix.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-1, vmax=1
    )

    _ax.set_xticks(np.arange(len(corr_matrix.columns)))
    _ax.set_yticks(np.arange(len(corr_matrix.columns)))
    _ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    _ax.set_yticklabels(corr_matrix.columns)

    for _i in range(len(corr_matrix.columns)):
        for _j in range(len(corr_matrix.columns)):
            _text = _ax.text(
                _j,
                _i,
                f"{corr_matrix.to_numpy()[_i, _j]:.2f}",
                ha="center",
                va="center",
                color="black" if abs(corr_matrix.to_numpy()[_i, _j]) < 0.5 else "white",
            )

    _ax.set_title("Correlation Heatmap")
    plt.colorbar(_im, ax=_ax)
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Correlations Heatmap"),
            _fig,
            mo.md("Only `text_reviews_count` and `ratings_count` appear to be correlated, consider dropping one.")
        ])
    return


@app.cell
def _(df, mo, pl, plt):
    """Categorical Analysis"""
    top_languages = (
        df.filter(pl.col("language_code").is_not_null())
        .group_by("language_code")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
    )

    top_formats = (
        df.filter(pl.col("format").is_not_null())
        .group_by("format")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
    )

    series_counts = (
        df.with_columns(is_series=pl.col("series").list.len() > 0)
        .group_by("is_series")
        .agg(pl.len().alias("count"))
        .sort("is_series", descending=False)
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))

    _axes[0].barh(
        top_languages["language_code"], top_languages["count"], color="steelblue"
    )
    _axes[0].set_xlabel("Count")
    _axes[0].set_title("Top 10 Languages")
    _axes[0].invert_yaxis()
    _axes[0].grid(True, alpha=0.3, axis="x")

    _axes[1].barh(top_formats["format"], top_formats["count"], color="coral")
    _axes[1].set_xlabel("Count")
    _axes[1].set_title("Top 10 Formats")
    _axes[1].invert_yaxis()
    _axes[1].grid(True, alpha=0.3, axis="x")

    _labels = ["No Series", "Has Series"]
    _axes[2].bar(_labels, series_counts["count"], color=["gray", "orange"], alpha=0.7)
    _axes[2].set_ylabel("Count")
    _axes[2].set_title("Series vs Standalone")
    _axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Categorical Statistics"),
            _fig,
            mo.md(f"- **Languages**: {df['language_code'].n_unique()} unique values"),
            mo.md(f"- **Formats**: {df['format'].n_unique()} unique values"),
            mo.md(
                f"- **Books with Series**: {df.filter(pl.col('series').list.len() > 0).shape[0]:,} ({df.filter(pl.col('series').list.len() > 0).shape[0] / df.shape[0] * 100:.1f}%)"
            ),
            mo.md("Only book related formats are relevant, consider dropping works of other types.")
        ]
    )
    return


@app.cell
def _(df, mo, pl, plt):
    """Key Relationships"""
    relation_data = df.filter(
        (pl.col("num_pages").is_not_null())
        & (pl.col("num_pages") > 0)
        & (pl.col("num_pages") < 2000)
        & (pl.col("average_rating").is_not_null())
        & (pl.col("ratings_count").is_not_null())
        & (pl.col("ratings_count") > 0)
        & (pl.col("ratings_count") < 50000)
    ).select("num_pages", "average_rating", "ratings_count")

    series_rating = (
        df.with_columns(is_series=pl.col("series").list.len() > 0)
        .group_by("is_series")
        .agg(
            pl.col("average_rating").mean().alias("avg_rating"), pl.len().alias("count")
        )
        .sort("is_series", descending=False)
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))

    _axes[0].scatter(
        relation_data["num_pages"],
        relation_data["average_rating"],
        alpha=0.3,
        s=5,
        color="steelblue",
    )
    _axes[0].set_xlabel("Number of Pages")
    _axes[0].set_ylabel("Average Rating")
    _axes[0].set_title("Page Count vs Average Rating")
    _axes[0].grid(True, alpha=0.3)

    _axes[1].scatter(
        relation_data["ratings_count"],
        relation_data["average_rating"],
        alpha=0.3,
        s=5,
        color="coral",
    )
    _axes[1].set_xlabel("Ratings Count")
    _axes[1].set_ylabel("Average Rating")
    _axes[1].set_title("Ratings Count vs Average Rating")
    _axes[1].grid(True, alpha=0.3)

    _labels = ["Standalone", "Series"]
    _axes[2].bar(
        _labels, series_rating["avg_rating"], color=["gray", "orange"], alpha=0.7
    )
    _axes[2].set_ylabel("Average Rating")
    _axes[2].set_title("Average Rating: Series vs Standalone")
    _axes[2].set_ylim(3.5, 4.2)
    _axes[2].grid(True, alpha=0.3, axis="y")

    for _i, (_label, _row) in enumerate(
        zip(_labels, series_rating.iter_rows(named=True))
    ):
        _axes[2].text(
            _i, _row["avg_rating"] + 0.02, f"{_row['avg_rating']:.3f}", ha="center"
        )


    plt.tight_layout()

    mo.vstack([
        mo.md("### Key Relationships"),
        _fig,
        mo.md("### Findings:"),
        mo.md("""- Length alone does not predict rating quality
        - Popular books aren’t necessarily better rated but their ratings are more reliable
        - Series benefit from reader investment and familiarity
        """)

    ])

    return


if __name__ == "__main__":
    app.run()
