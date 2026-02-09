import marimo

__generated_with = "0.19.7"
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
def _(pl):
    import os
    # Use absolute path from project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "raw_goodreads_book_works.parquet")
    df = pl.read_parquet(data_path)

    # Replace empty strings with null for date fields
    df = df.with_columns(
        [
            pl.when(pl.col(c).str.len_chars() == 0)
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in [
                "original_publication_month",
                "original_publication_year",
                "original_publication_day",
            ]
        ]
    )

    # Cast to appropriate types
    df = df.with_columns(
        [
            pl.col("books_count").cast(pl.Int64),
            pl.col("reviews_count").cast(pl.Int64),
            pl.col("text_reviews_count").cast(pl.Int64),
            pl.col("best_book_id").cast(pl.Int64),
            pl.col("ratings_count").cast(pl.Int64),
            pl.col("ratings_sum").cast(pl.Int64),
            pl.col("work_id").cast(pl.Int64),
            pl.col("original_publication_month").cast(pl.Int64),
            pl.col("original_publication_year").cast(pl.Int64),
            pl.col("original_publication_day").cast(pl.Int64),
        ]
    )

    df.head()
    return (df,)


@app.cell
def _(df, mo):
    """Dataset Overview"""
    shape = df.shape
    mo.md(f"Dataset contains {shape[0]:,.0f} works and {shape[1]} columns")
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Large dataset with 1.5M works, all numeric fields are highly skewed with extreme outliers
    """)
    return


@app.cell
def _(df, mo):
    """Missing Values Analysis"""
    null_counts = df.null_count()

    missing_year_pct = (
        df["original_publication_year"].is_null().sum() / df.shape[0] * 100
    )

    missing_month_pct = (
        df["original_publication_month"].is_null().sum() / df.shape[0] * 100
    )

    missing_day_pct = (
        df["original_publication_day"].is_null().sum() / df.shape[0] * 100
    )

    mo.vstack([
        mo.md(f"{missing_year_pct:.1f}% of works missing publication year"),
        mo.md(f"{missing_month_pct:.1f}% of works missing publication month"),
        mo.md(f"{missing_day_pct:.1f}% of works missing publication day"),
        mo.md("### Missing values per column:"),
        null_counts
    ])
    return


@app.cell
def _(df):

    df["media_type"].unique_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Original publication dates have a lot of missing values, will need imputation or filtering strategies
    """)
    return


@app.cell
def _(df):
    """Summary Statistics"""
    summary = df.select(
        [
            "books_count",
            "reviews_count",
            "text_reviews_count",
            "ratings_count",
            "ratings_sum",
            "original_publication_year",
        ]
    ).describe()
    summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    Extreme outliers present: negative review counts (-6069), unrealistic publication years (-2600, 32767), and highly skewed rating counts
    """)
    return


@app.cell
def _(df, mo, pl):
    """Data Quality Checks"""
    # Check for negative values
    neg_reviews = df.filter(pl.col("reviews_count") < 0).shape[0]
    neg_year = df.filter(pl.col("original_publication_year") < 1000).shape[0]
    future_year = df.filter(pl.col("original_publication_year") > 2026).shape[0]


    # Check rating_sum consistency
    rating_issues = df.filter(
        pl.col("ratings_sum") > pl.col("ratings_count") * 5
    ).shape[0]

    mo.vstack([mo.md(f"Works with negative reviews: {neg_reviews}"),
    mo.md(f"Works with publication year < 1000: {neg_year}"),
    mo.md(f"Works with publication year > 2026: {future_year}"),
    mo.md(
        f"\nWorks where rating_sum exceeds max possible (ratings_count * 5): {rating_issues}"
    )])

    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    Data contains negative review counts and impossible publication years, requiring cleaning before modeling
    """)
    return


@app.cell
def _(df, mo):
    """Distribution Analysis"""
    mo.vstack([mo.md(
        f"Books per work: median {df['books_count'].median():.0f}, max {df['books_count'].max():.0f}"
    ),
    mo.md(
        f"Ratings per work: median {df['ratings_count'].median():.0f}, max {df['ratings_count'].max():,.0f}"
    ),
    mo.md(
        f"Text reviews per work: median {df['text_reviews_count'].median():.0f}, max {df['text_reviews_count'].max():,.0f}"
    ),
    mo.md(
        f"Publication year: {df['original_publication_year'].min():.0f} to {df['original_publication_year'].max():.0f}"
    )])

    return


@app.cell
def _(mo):
    mo.md(r"""
    Metrics are extremely right-skewed with long tails, requiring log-transformation for analysis
    """)
    return


@app.cell
def _(df, plt):
    """Distribution of Ratings Count"""
    fig_ratings, ax_ratings = plt.subplots(figsize=(10, 6))
    ax_ratings.hist(
        df["ratings_count"].to_pandas(), bins=100, edgecolor="black", alpha=0.7
    )
    ax_ratings.set_xlabel("Ratings Count")
    ax_ratings.set_ylabel("Frequency")
    ax_ratings.set_title("Distribution of Ratings Count (Raw)")
    ax_ratings.set_xlim(0, 1000) 
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, np, plt):
    """Distribution of Log-Transformed Metrics"""
    fig_log, axes_log = plt.subplots(2, 2, figsize=(14, 10))

    # Log-transform metrics
    log_ratings = np.log1p(df["ratings_count"].to_pandas())
    log_reviews = np.log1p(df["text_reviews_count"].to_pandas())
    log_sum = np.log1p(df["ratings_sum"].to_pandas())
    log_books = np.log1p(df["books_count"].to_pandas())

    axes_log[0, 0].hist(log_ratings, bins=50, edgecolor="black", alpha=0.7)
    axes_log[0, 0].set_xlabel("Log(Ratings Count)")
    axes_log[0, 0].set_ylabel("Frequency")
    axes_log[0, 0].set_title("Log-Transformed Ratings Count")

    axes_log[0, 1].hist(log_reviews, bins=50, edgecolor="black", alpha=0.7)
    axes_log[0, 1].set_xlabel("Log(Text Reviews Count)")
    axes_log[0, 1].set_ylabel("Frequency")
    axes_log[0, 1].set_title("Log-Transformed Text Reviews Count")

    axes_log[1, 0].hist(log_sum, bins=50, edgecolor="black", alpha=0.7)
    axes_log[1, 0].set_xlabel("Log(Ratings Sum)")
    axes_log[1, 0].set_ylabel("Frequency")
    axes_log[1, 0].set_title("Log-Transformed Ratings Sum")

    axes_log[1, 1].hist(log_books, bins=50, edgecolor="black", alpha=0.7)
    axes_log[1, 1].set_xlabel("Log(Books Count)")
    axes_log[1, 1].set_ylabel("Frequency")
    axes_log[1, 1].set_title("Log-Transformed Books Count")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Log-transformation reveals approximately normal distributions, suggesting it's appropriate for modeling
    """)
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

    # Check top works by ratings
    top_works = (
        df.sort("ratings_count", descending=True)
        .select(["original_title", "ratings_count", "ratings_sum"])
        .head(10)
    )
    mo.vstack([mo.md(
        f"Using 3x IQR method, {outliers_count:,} works ({outliers_count / df.shape[0] * 100:.2f}%) are outliers"
    ),
    mo.md("\nTop 10 works by total ratings count:"),
    top_works])

    return


@app.cell
def _(mo):
    mo.md(r"""
    Small percentage of extreme outliers dominate the metrics, consider cap or separate treatment
    """)
    return


@app.cell
def _(df, pl, plt):
    """Publication Year Distribution"""
    valid_years = df.filter(pl.col("original_publication_year").is_not_null())

    fig_year, ax_year = plt.subplots(figsize=(12, 6))
    ax_year.hist(
        valid_years["original_publication_year"].to_pandas(),
        bins=100,
        edgecolor="black",
        alpha=0.7,
    )
    ax_year.set_xlabel("Publication Year")
    ax_year.set_ylabel("Frequency")
    ax_year.set_title("Distribution of Publication Years")
    ax_year.set_xlim(1800, 2026)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Most works are from recent decades (2000-2020), with fewer older classics available
    """)
    return


@app.cell
def _(df, mo, pl):
    """Media Type Analysis"""
    media_counts = df.group_by("media_type").len().sort("len", descending=True)
    # Count works with non-empty language
    has_language = df.filter(pl.col("original_language_id") != "").shape[0]
    mo.vstack([
        mo.md("Distribution of media types:"),
        media_counts,
        mo.md(
        f"\nWorks with language specified: {has_language:,} ({has_language / df.shape[0] * 100:.1f}%)"
    )
    ])

    return


@app.cell
def _(mo):
    mo.md(r"""
    Most entries are 'book' type
    Consider getting rid of other media types
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Both `original_language_id` and `default_description_language_code` have empty values (same for all).
    Consider droppig later
    """)
    return


@app.cell
def _(df, plt, sns):
    """Correlation Analysis"""
    # Convert to pandas for correlation heatmap
    numeric_df = df.select(
        [
            "books_count",
            "reviews_count",
            "text_reviews_count",
            "ratings_count",
            "ratings_sum",
        ]
    ).to_pandas()

    correlation = numeric_df.corr(method="pearson")

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation, annot=True, cmap="coolwarm", center=0, fmt=".3f", ax=ax_corr
    )
    ax_corr.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Very high correlation between ratings_count and ratings_sum (r=0.99) suggests redundancy, consider removing one
    Same with review_count and text_reviews_count.

    Consider dropping _count/_sum variables for later, all redundant.
    """)
    return


@app.cell
def _(df, pl, plt):
    """Rating Quality Analysis"""
    # Calculate average rating from ratings_sum and ratings_count
    df_rating = df.with_columns(
        [(pl.col("ratings_sum") / pl.col("ratings_count")).alias("avg_rating")]
    )

    fig_rating_hist, ax_rating_hist = plt.subplots(figsize=(10, 6))
    ax_rating_hist.hist(
        df_rating["avg_rating"].to_pandas(),
        bins=50,
        edgecolor="black",
        alpha=0.7,
    )
    ax_rating_hist.set_xlabel("Average Rating")
    ax_rating_hist.set_ylabel("Frequency")
    ax_rating_hist.set_title("Distribution of Calculated Average Ratings")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Ratings are normally distributed around 3.5-4.0 with slight left skew, typical for user-generated content
    """)
    return


@app.cell
def _(df, np, pl, plt):
    """Popularity vs Rating Analysis"""
    df_rating2 = df.with_columns(
        [(pl.col("ratings_sum") / pl.col("ratings_count")).alias("avg_rating")]
    )

    fig_pop, ax_pop = plt.subplots(figsize=(10, 6))
    ax_pop.scatter(
        np.log1p(df_rating2["ratings_count"].to_pandas()),
        df_rating2["avg_rating"].to_pandas(),
        alpha=0.3,
        s=1,
    )
    ax_pop.set_xlabel("Log(Ratings Count)")
    ax_pop.set_ylabel("Average Rating")
    ax_pop.set_title("Average Rating vs Popularity")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Weak relationship between popularity and rating quality, popular works don't consistently have higher ratings
    """)
    return


@app.cell
def _(df):
    """Rating Distribution Parsing Analysis"""
    # Parse a sample of rating_dist to understand the format
    sample_dists = df["rating_dist"].head(10).to_list()

    print("Sample rating distributions (format: 5:X|4:Y|3:Z|2:W|1:V|total:N):")
    for dist in sample_dists:
        print(f"  {dist}")

    print(
        "\nThis format contains detailed breakdown by star rating, can be used for weighted averaging or sentiment analysis"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Rating_dist field provides granular rating breakdown that could enable weighted scoring or preference modeling

    can be used for weighted averaging or sentiment analysis
    """)
    return


@app.cell
def _(df, mo, pl):
    """Books per Work Analysis"""
    multi_book = df.filter(pl.col("books_count") > 1).shape[0]

    books_distribution = df.group_by("books_count").len().sort("books_count")
    mo.vstack([
        (f"Works with multiple book editions: {multi_book:,} ({multi_book / df.shape[0] * 100:.1f}%)"),
        mo.md("\nDistribution of books count:"),
        books_distribution.head(20)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Most works have a single book edition, but 4.4% have multiple editions requiring deduplication or ranking.

    Best solution is to keep the best version based on book id
    """)
    return


@app.cell
def _(df, mo, np, stats):
    """Pearson Correlation Significance Tests"""
    ratings_count = df["ratings_count"].to_pandas().values
    text_reviews = df["text_reviews_count"].to_pandas().values
    reviews_count = df["reviews_count"].to_pandas().values

    # Log-transform for better normality
    log_ratings2 = np.log1p(ratings_count)
    log_reviews2 = np.log1p(text_reviews)
    log_reviews_count = np.log1p(np.maximum(0, reviews_count))  # Clip negative values at 0

    corr1, p1 = stats.pearsonr(log_ratings2, log_reviews2)
    corr2, p2 = stats.pearsonr(log_ratings2, log_reviews_count)
    corr3, p3 = stats.pearsonr(log_reviews2, log_reviews_count)

    mo.vstack([
        mo.md(f"Log(Ratings) vs Log(Text Reviews): r={corr1:.4f}, p={p1:.2e}"),
        mo.md(f"Log(Ratings) vs Log(Reviews): r={corr2:.4f}, p={p2:.2e}"),
        mo.md(f"Log(Text Reviews) vs Log(Reviews): r={corr3:.4f}, p={p3:.2e}")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    All three popularity metrics are highly correlated, suggesting they measure similar underlying construct
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Feature Engineering Recommendations

    1. ratings_sum and ratings_count are highly correlated (r=0.99), reviews_count and text_reviews_count too so consider using only one. (all popularity metrics are somewhat similar)
    2. Missing publication years/month/day need imputation strategy (median, KNN, or exclude)
    3. Negative review counts and impossible publication years require data cleaning
    4. All popularity metrics are highly skewed, always use log-transformation
    5. Parse rating_dist field for weighted scoring instead of simple average
    6. Consider filtering to works with minimum rating threshold (e.g., >= 10 ratings)
    7. Media type is mostly 'book', may not be useful as a feature
    8. Language data (id and description) are empty and unique
    9. Works with multiple editions might need to do a join with the books db to select best one.
    11. Outlier treatment needed for extreme rating counts
    12. Consider creating an average_rating column (ratings_sum/ratings_count) to measure popularity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Cleaning
       1. Filter to keep only books
       2. Fix invalid publication years
       3. Drop redundant and unused columns
    """)
    return


@app.cell
def _(df, mo, pl):
    """Filter to keep only book entries"""
    # Check current media type distribution
    media_dist = df.group_by("media_type").len().sort("len", descending=True)
    
    # Filter to keep only books
    df_books = df.filter(pl.col("media_type") == "book")
    
    mo.vstack([
        mo.md("### Filter by Media Type"),
        mo.md(f"Original dataset: {df.shape[0]:,} works"),
        media_dist,
        mo.md(f"\nAfter filtering to 'book' only: {df_books.shape[0]:,} works"),
        mo.md(f"Rows removed: {df.shape[0] - df_books.shape[0]:,} ({(1 - df_books.shape[0]/df.shape[0])*100:.1f}%)")
        ])

    return (df_books,)


@app.cell
def _(df_books, mo, pl):
    """Fix invalid publication years"""
    # Count invalid years before fixing
    year_too_old = df_books.filter(pl.col("original_publication_year") < 1000).shape[0]
    year_too_new = df_books.filter(pl.col("original_publication_year") > 2026).shape[0]
    year_null_before = df_books["original_publication_year"].is_null().sum()
    
    # Fix invalid years by setting them to null
    df_cleaned = df_books.with_columns(
        pl.when(
            (pl.col("original_publication_year") < 1000) | 
            (pl.col("original_publication_year") > 2026)
        )
        .then(None)
        .otherwise(pl.col("original_publication_year"))
        .alias("original_publication_year")
    )
    
    year_null_after = df_cleaned["original_publication_year"].is_null().sum()
    
    mo.vstack([
        mo.md("### Fix Invalid Publication Years"),
        mo.md(f"Years < 1000: {year_too_old:,} rows → set to null"),
        mo.md(f"Years > 2026: {year_too_new:,} rows → set to null"),
        mo.md(f"Null years before: {year_null_before:,} ({year_null_before/df_books.shape[0]*100:.1f}%)"),
        mo.md(f"Null years after: {year_null_after:,} ({year_null_after/df_cleaned.shape[0]*100:.1f}%)"),
        mo.md(f"Valid years (1000-2026): {df_cleaned.shape[0] - year_null_after:,} rows")
    ])
    return (df_cleaned,)


@app.cell
def _(df_cleaned, mo):
    """Drop redundant and unused columns"""
    # Define columns to drop
    columns_to_drop = [
        "original_publication_month", 
        "original_publication_day",     
        "original_language_id",         
        "default_description_language_code",  
        "reviews_count",                
        "ratings_sum",                  
        "media_type",                   
        "default_chaptering_book_id",  
    ]
    
    # Drop the columns
    df_final = df_cleaned.drop(columns_to_drop)
    
    mo.vstack([
        mo.md("### Drop Redundant Columns"),
        mo.md("Columns dropped:"),
        mo.md("- `original_publication_month` - Too many missing values"),
        mo.md("- `original_publication_day` - Too many missing values"),
        mo.md("- `original_language_id` - All empty values"),
        mo.md("- `default_description_language_code` - All empty values"),
        mo.md("- `reviews_count` - Redundant with text_reviews_count"),
        mo.md("- `ratings_sum` - Redundant with ratings_count"),
        mo.md("- `media_type` - All values are 'book' after filtering"),
        mo.md("- `default_chaptering_book_id` - Mostly empty"),
        mo.md(f"\nColumns before: {df_cleaned.shape[1]}"),
        mo.md(f"Columns after: {df_final.shape[1]}"),
        mo.md(f"Final dataset: {df_final.shape[0]:,} works × {df_final.shape[1]} columns")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Impute Missing Publication Years
    14.8% of works have null original_publication_year after cleaning.
    - Drop rows: Simple but lose 14.8% of data
    - Impute with median: Quick fix but ignores other signals
    2. Parse rating_dist
    The format is: 5:X|4:Y|3:Z|2:W|1:V|total:N
    - Extract star counts as separate columns
    - Create weighted average rating
    - Calculate rating variance/sentiment metrics
    2. Create Derived Features
    - average_rating: ratings_sum / ratings_count
    - log_ratings_count: log1p(ratings_count) for normality
    - popularity_score: Combined metric (log(ratings) × avg_rating)
    3. Handle Outliers
    The EDA showed extreme outliers:
    - Winsorization: Cap values at 95th/99th percentile
    4.Join with Books Table
    best_book_id could be used to:
    - Get language_code from books table (more informative than the empty language fields)
    - Get format information (hardcover, paperback, ebook)
    - Get publisher for publisher-level analysis
    5. Join with Authors Table
    - Add author information for author-level features
    - Author popularity, average ratings, ...
    """)
    return


if __name__ == "__main__":
    app.run()
