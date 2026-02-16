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
    import json
    import os
    return json, mo, os, pl, plt, sns


@app.cell
def _(mo):
    mo.md("""
    # Preliminary EDA for Goodreads Books Dataset
    """)
    return


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
def _(df, json, mo, os, pl, plt):
    """Empty String Analysis for Kept Features"""
    # Apply the same filtering and dropping logic as clean_books.py

    # Load best_book_id_map.json
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    best_book_map_path = os.path.join(_project_root, "data", "best_book_id_map.json")

    with open(best_book_map_path, "r") as f:
        best_book_map = json.load(f)

    # Extract non-best book IDs (values from the JSON to drop)
    non_best_book_ids = set()
    for book_id_list in best_book_map.values():
        non_best_book_ids.update(str(book_id) for book_id in book_id_list)

    # Filter to EXCLUDE non-best book IDs
    df_filtered = df.filter(~pl.col("book_id").is_in(non_best_book_ids))

    # Define columns to drop (same as clean_books.py)
    columns_to_drop = [
        "text_reviews_count",
        "series",
        "country_code",
        "asin",
        "is_ebook",
        "average_rating",
        "kindle_asin",
        "publication_day",
        "publication_month",
        "edition_information",
        "ratings_count",
        "title_without_series",
        "isbn",
    ]

    # Drop the specified columns
    df_cleaned = df_filtered.drop(columns_to_drop)

    # Define the 16 kept columns
    kept_columns = [
        "isbn13",
        "popular_shelves",
        "similar_books",
        "description",
        "link",
        "authors",
        "publisher",
        "num_pages",
        "publication_year",
        "url",
        "image_url",
        "book_id",
        "work_id",
        "title",
        "format",
        "language_code",
    ]

    # Define list columns that need special handling
    list_columns = ["popular_shelves", "similar_books", "authors"]

    # Check for empty strings in each column
    empty_string_results = []
    for col in kept_columns:
        col_type = df_cleaned.schema[col]

        if col in list_columns:
            # For list columns: check for empty lists
            if col_type == pl.List:
                empty_count = df_cleaned.filter(pl.col(col).list.len() == 0).shape[0]
            else:
                empty_count = 0
        else:
            # For non-list columns: check for empty strings
            if col_type == pl.String:
                empty_count = df_cleaned.filter(pl.col(col) == "").shape[0]
            else:
                empty_count = 0

        empty_percentage = (empty_count / df_cleaned.shape[0]) * 100
        empty_string_results.append(
            {
                "column": col,
                "type": str(col_type),
                "empty_count": empty_count,
                "empty_percentage": empty_percentage,
            }
        )

    # Convert to DataFrame for easier display
    results_df = pl.DataFrame(empty_string_results).sort("empty_count", descending=True)

    # Create visualization
    _fig, _ax = plt.subplots(figsize=(12, 6))
    colors = [
        "coral" if x > 10 else "steelblue" for x in results_df["empty_percentage"]
    ]
    bars = _ax.barh(
        results_df["column"],
        results_df["empty_percentage"],
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    _ax.set_xlabel("Empty String Percentage (%)", fontsize=12)
    _ax.set_ylabel("Column Name", fontsize=12)
    _ax.set_title(
        "Empty String Percentage in Kept Features", fontsize=14, fontweight="bold"
    )
    _ax.set_xlim(0, max(results_df["empty_percentage"]) * 1.1)
    _ax.grid(True, alpha=0.3, axis="x")
    _ax.invert_yaxis()

    # Add value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, results_df["empty_percentage"])):
        _ax.text(
            pct + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )

    # Highlight columns with high empty string rates (>10%)
    high_empty = results_df.filter(pl.col("empty_percentage") > 10)
    high_empty_cols = high_empty["column"].to_list()

    mo.vstack(
        [
            mo.md("### Empty String Analysis for Kept Features"),
            mo.md(f"**Total books after filtering:** {df_cleaned.shape[0]:,}"),
            mo.md(f"**Total kept columns:** {len(kept_columns)}"),
            mo.md("#### Empty String Summary Table"),
            results_df,
            mo.md("#### Empty String Visualization"),
            _fig,
            mo.md("#### Key Findings"),
            mo.md(
                f"- **Columns with >10% empty strings:** {len(high_empty)} "
                f"({', '.join([f'`{col}`' for col in high_empty_cols])})"
            ),
            mo.md(
                f"- **Highest empty string rate:** `{results_df[0, 'column']}` "
                f"with {results_df[0, 'empty_percentage']:.1f}% empty strings"
            ),
            mo.md(
                f"- **Lowest empty string rate:** `{results_df[-1, 'column']}` "
                f"with {results_df[-1, 'empty_percentage']:.1f}% empty strings"
            ),
            mo.md("#### Recommendations"),
            mo.md(r"""
        - Consider dropping columns with very high empty string rates (>50%)
        - For columns with moderate empty string rates (10-50%), consider:
          * Imputation strategies (if appropriate)
          * Creating a separate "missing" category
          * Dropping if not critical for analysis
        - For columns with low empty string rates (<10%), consider:
          * Dropping rows with missing values (if few)
          * Simple imputation (mean, median, mode)
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

    _labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]

    # Create rating bins using qcut for quantile-based binning
    rating_binned = (
        pl.DataFrame({"rating": ratings.cast(pl.Float64)})
        .with_columns(rating=pl.col("rating").clip(0.0, 5.0))
        .with_columns(
            bin=pl.col("rating").cut(breaks=[0, 1, 2, 3, 4, 5], include_breaks=True)
        )
        .with_columns(
            bin=pl.when(pl.col("rating").is_between(0, 1, closed="left"))
            .then(pl.lit("0-1"))
            .when(pl.col("rating").is_between(1, 2, closed="left"))
            .then(pl.lit("1-2"))
            .when(pl.col("rating").is_between(2, 3, closed="left"))
            .then(pl.lit("2-3"))
            .when(pl.col("rating").is_between(3, 4, closed="left"))
            .then(pl.lit("3-4"))
            .when(pl.col("rating").is_between(4, 5, closed="both"))
            .then(pl.lit("4-5"))
            .otherwise(None)
        )
        .filter(pl.col("bin").is_not_null())
        .group_by("bin")
        .agg(pl.len().alias("count"))
        .sort("bin")
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
            """),
        ]
    )
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
            - Each text review is also a rating but not every rating is a text review"""),
        ]
    )
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
            - Most books are around 250 pages long while a small number of extremely long books create a long tail and inflate the mean"""),
        ]
    )
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
            """),
        ]
    )
    return


@app.cell
def _(df, mo, pl, plt, sns):
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
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=_ax)
    _ax.set_title("Correlation Heatmap")
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Correlations Heatmap"),
            _fig,
            mo.md(
                "Only `text_reviews_count` and `ratings_count` appear to be correlated, consider dropping one."
            ),
        ]
    )
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
            mo.md(
                "Only book related formats are relevant, consider dropping works of other types."
            ),
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

    mo.vstack(
        [
            mo.md("### Key Relationships"),
            _fig,
            mo.md("### Findings:"),
            mo.md("""- Length alone does not predict rating quality
        - Popular books aren’t necessarily better rated but their ratings are more reliable
        - Series benefit from reader investment and familiarity
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleaning
    1. filter format for paperback, Hardcover, Mass market paperback then drop column
    2. drop text_reviews_count and ratings_counts and series asin, is_ebook, language, average_rating, kindle_asin, format, publication_day, publication_month, edition_information, ratings_count, title_witout_series
    3. Drop non best books id
    """)
    return


if __name__ == "__main__":
    app.run()
