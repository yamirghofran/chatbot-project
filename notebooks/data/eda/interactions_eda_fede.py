import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    return mo, pl, plt, sns


@app.cell
def _(pl):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "raw_goodreads_interactions.parquet")
    df = pl.read_parquet(data_path)
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    **Columns:**
    - `user_id`: Unique user identifier
    - `book_id`: Unique book identifier
    - `rating`: User's rating 0-5 
    - `is_read`: Boolean indicating if user marked book as read
    - `is_reviewed`: User wrote a text review
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **INITIAL DATA OVERVIEW**
    """)
    return


@app.cell
def _(df):
    # Stats overview
    df.describe()
    return


@app.cell
def _(df):
    # Total unique users and books
    n_users = df["user_id"].n_unique()
    n_books = df["book_id"].n_unique()

    print(f"Total unique users: {n_users:,}")
    print(f"Total unique books: {n_books:,}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Key Observations:**
    - `Median rating = 0`: Over 50% of interactions are either rated as 0, or most probably unrated
    - `Mean rating = 1.80`: Heavily skewed by low rated / unrated (0) values
    - `Mean reviews = 0.07`: Only 7% have reviews
    - `around 50% marked as read`
    """)
    return


@app.cell
def _(df):
    # Shape and nulls at a glance
    print(f"Shape: {df.shape}")
    print(f"Nulls: {df.null_count().to_dicts()[0]}")
    return


@app.cell
def _(df):
    # Check data types are as expected
    print("Column types:")
    print(df.schema)

    # Check for unexpected values in boolean columns
    print("\nis_read unique values:", df["is_read"].unique().to_list())
    print("is_reviewed unique values:", df["is_reviewed"].unique().to_list())

    # Check rating range 
    print(f"\nRating range: {df['rating'].min()} to {df['rating'].max()}")
    print("Unique ratings:", sorted(df["rating"].unique().to_list()))
    return


@app.cell
def _(df, pl):
    # Rating = 0 investigation (likely means "no rating given")
    _total = df.height
    _zero_ratings = df.filter(pl.col("rating") == 0).height
    _nonzero_ratings = df.filter(pl.col("rating") > 0).height
    _total_read = df.filter(pl.col("is_read") == True).height

    print(f"Total interactions: {_total:,}")
    print(f"Rating = 0 (unrated): {_zero_ratings:,} ({_zero_ratings/_total*100:.1f}%)")
    print(f"Rating > 0 (rated): {_nonzero_ratings:,} ({_nonzero_ratings/_total*100:.1f}%)")

    # Check unrated interactions with is_read = True
    _zero_but_read = df.filter((pl.col("rating") == 0) & (pl.col("is_read") == True)).height
    print(f"\nRead interactions: {_total_read:,} ({_total_read/_total*100:.1f}%)")
    print(f"\nUnrated but marked as read: {_zero_but_read:,} ({_zero_but_read/_zero_ratings*100:.1f}% of unrated)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Of the 124M of likely unrated (0) interactions:
    6.1% are marked as read, suggesting many users engage with books possibly without providing a rating.
    93.9% of 0 rated interactions are not marked as read, reinforcing the idea that many 0 ratings likely represent "no rating given" rather than an actual rating of zero. This supports the decision not to treat them as explicit negative feedback.
    This also suggests strong rating engagement: about 93% of read books receive explicit ratings (104.5M rated / 112M total read), indicating active user participation.
    """)
    return


@app.cell
def _(df):
    # Sparsity of user-item matrix
    _n_users = df["user_id"].n_unique()
    _n_books = df["book_id"].n_unique()
    _n_interactions = df.height

    # assuming every user could interact with every book
    _possible_interactions = _n_users * _n_books 
    _sparsity = 1 - (_n_interactions / _possible_interactions)

    print(f"Unique users: {_n_users:,}")
    print(f"Unique books: {_n_books:,}")
    print(f"Total interactions: {_n_interactions:,}")
    print(f"Possible interactions: {_possible_interactions:,}")
    print(f"Matrix sparsity: {_sparsity:.4%}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The user–item interaction matrix is extremely sparse (99.9889% sparse), meaning almost all
    user–book combinations have no interaction data.

    On average, each user interacted with only 261 books out of 2.36M available books (228M / 876K).
    When predicting preferences, the model has very little data for most user–book pairs and must generalize
    from a small sample to a massive catalog, making personalization difficult.

    Each book is interacted with by only 97 users on average (228M / 2.36M). Most books have very weak signals,
    while popular books dominate the data (see distribution analysis below).
    Popular books get more exposure.

    Collaborative filtering relies on shared interactions to find similar users or books.
    With 99.99% sparsity and each user interacting with only around 0.011% of books, shared interactions between
    users are rare. 

    New users have no ratings, so the system falls back to recommending popular books. New books have no ratings, so they rarely get
    recommended.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ** USER ACTIVITY PATTERNS**
    """)
    return


@app.cell
def _(df, pl):
    # Outliers in user activity (books per user)
    _books_per_user = df.group_by("user_id").len().rename({"len": "book_count"})

    _q1 = _books_per_user["book_count"].quantile(0.25)
    _q3 = _books_per_user["book_count"].quantile(0.75)
    _iqr = _q3 - _q1
    _lower_bound = _q1 - 1.5 * _iqr
    _upper_bound = _q3 + 1.5 * _iqr

    _upper_outliers = _books_per_user.filter(pl.col("book_count") > _upper_bound)
    _lower_outliers = _books_per_user.filter(pl.col("book_count") < _lower_bound)

    _total_users = _books_per_user.height
    _upper_pct = (_upper_outliers.height / _total_users) * 100
    _lower_pct = (_lower_outliers.height / _total_users) * 100

    print(f"Upper outliers (>{_upper_bound:.0f} books): {_upper_outliers.height:,} users ({_upper_pct:.1f}%)")
    print(f"Lower outliers (<{_lower_bound:.0f} books): {_lower_outliers.height:,} users ({_lower_pct:.1f}%)")
    if _lower_bound < 0:
        print("  Note: Lower bound is negative (impossible) - no lower outliers for this skewed distribution")
    print(f"Total outliers: {_upper_outliers.height + _lower_outliers.height:,} users ({(_upper_pct + _lower_pct):.1f}%)")

    _upper_outliers.sort("book_count", descending=True).head(10)
    return


@app.cell
def _(df, plt):
    # Books per user distribution
    _books_per_user = df.group_by("user_id").len().rename({"len": "book_count"})

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram 
    _axes[0].hist(_books_per_user["book_count"].to_numpy(), bins=50, edgecolor='black')
    _axes[0].set_xlabel("Books per User")
    _axes[0].set_ylabel("Frequency")
    _axes[0].set_title("User Activity Distribution")
    _axes[0].set_yscale('log')

    # Boxplot
    _axes[1].boxplot(_books_per_user["book_count"].to_numpy())
    _axes[1].set_ylabel("Books per User")
    _axes[1].set_title("User Activity (Boxplot)")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(df, plt):
    # Visualization - distribution of user activity
    _books_per_user_viz = df.group_by("user_id").len().rename({"len": "book_count"})
    _counts = _books_per_user_viz["book_count"].to_numpy()

    # Calculate outlier threshold
    _q3_viz = _books_per_user_viz["book_count"].quantile(0.75)
    _iqr_viz = _q3_viz - _books_per_user_viz["book_count"].quantile(0.25)
    _outlier_threshold = _q3_viz + 1.5 * _iqr_viz

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram with log-log scale 
    _axes[0].hist(_counts, bins=100, edgecolor='black', alpha=0.7)
    _axes[0].axvline(_outlier_threshold, color='red', linestyle='--', linewidth=2,
                     label=f'Outlier threshold ({_outlier_threshold:.0f} books)')
    _axes[0].set_xlabel("Books per User")
    _axes[0].set_ylabel("Number of Users")
    _axes[0].set_yscale('log')
    _axes[0].set_xscale('log')
    _axes[0].set_title("User Activity Distribution (Log-Log Scale)")
    _axes[0].legend()
    _axes[0].grid(True, alpha=0.3)

    # Right: Cumulative distribution
    _sorted_counts = sorted(_counts)
    _cumulative = [i/len(_sorted_counts)*100 for i in range(len(_sorted_counts))]
    _axes[1].plot(_sorted_counts, _cumulative, linewidth=2)
    _axes[1].axvline(_outlier_threshold, color='red', linestyle='--', linewidth=2,
                     label=f'{100 - 9.7:.1f}% of users')
    _axes[1].set_xlabel("Books per User")
    _axes[1].set_ylabel("Cumulative % of Users")
    _axes[1].set_xscale('log')
    _axes[1].set_title("Cumulative Distribution of User Activity")
    _axes[1].grid(True, alpha=0.3)
    _axes[1].legend()

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Heavily skewed user activity distribution.
    Approximately 85k users exceed the upper bound of 614 books, with extreme cases showing over 100,000 interactions.

    More active users contribute disproportionately to the training data and may bias learned patterns.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **BOOK POPULARITY PATTERNS**
    """)
    return


@app.cell
def _(df, pl):
    # Outliers in book popularity (users per book)
    _users_per_book = df.group_by("book_id").len().rename({"len": "user_count"})

    _q1 = _users_per_book["user_count"].quantile(0.25)
    _q3 = _users_per_book["user_count"].quantile(0.75)
    _iqr = _q3 - _q1
    _lower_bound = _q1 - 1.5 * _iqr
    _upper_bound = _q3 + 1.5 * _iqr

    _upper_outliers_books = _users_per_book.filter(pl.col("user_count") > _upper_bound)
    _lower_outliers_books = _users_per_book.filter(pl.col("user_count") < _lower_bound)

    _total_books = _users_per_book.height
    _upper_pct = (_upper_outliers_books.height / _total_books) * 100
    _lower_pct = (_lower_outliers_books.height / _total_books) * 100

    print(f"IQR bounds: Q1={_q1:.0f}, Q3={_q3:.0f}, IQR={_iqr:.0f}")
    print(f"Upper bound: {_upper_bound:.0f} users")
    print(f"\nUpper outliers (>{_upper_bound:.0f} users): {_upper_outliers_books.height:,} books ({_upper_pct:.1f}%)")
    print(f"Lower outliers (<{_lower_bound:.0f} users): {_lower_outliers_books.height:,} books ({_lower_pct:.1f}%)")
    if _lower_bound < 0:
        print("  Note: Lower bound is negative (impossible) - no lower outliers for this skewed distribution")
    print(f"Total outliers: {_upper_outliers_books.height + _lower_outliers_books.height:,} books ({(_upper_pct + _lower_pct):.1f}%)")

    _upper_outliers_books.sort("user_count", descending=True).head(10)
    return


@app.cell
def _(df, plt):
    # Users per book distribution
    _users_per_book = df.group_by("book_id").len().rename({"len": "user_count"})

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    _axes[0].hist(_users_per_book["user_count"].to_numpy(), bins=50, edgecolor='black')
    _axes[0].set_xlabel("Users per Book")
    _axes[0].set_ylabel("Frequency")
    _axes[0].set_title("Book Popularity Distribution")
    _axes[0].set_yscale('log')

    # Boxplot
    _axes[1].boxplot(_users_per_book["user_count"].to_numpy())
    _axes[1].set_ylabel("Users per Book")
    _axes[1].set_title("Book Popularity (Boxplot)")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    25% of books have 3 or fewer interactions, while 75% have 34 or fewer.
    
    13.8% of books are outliers (> 80 users) vs 9.7% of users (>614 books)
    
    Data is more skewed toward popular books than toward active users.
    This reinforces book popularity bias problem (more exposure for popular books).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ** RATINGS ANALYSIS **
    """)
    return


@app.cell
def _(df, pl):
    # Rating summary statistics
    _rating_stats = df.select([
        pl.col("rating").mean().alias("mean"),
        pl.col("rating").median().alias("median"),
        pl.col("rating").std().alias("std"),
        pl.col("rating").min().alias("min"),
        pl.col("rating").max().alias("max"),
        pl.col("rating").skew().alias("skewness"),
    ])
    _rating_stats
    return


@app.cell
def _(df, plt):
    # Rating distribution - histogram + boxplot
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _ratings = df["rating"].to_numpy()

    # Histogram
    _axes[0].hist(_ratings, bins=range(0, 7), edgecolor='black', align='left')
    _axes[0].set_xlabel("Rating")
    _axes[0].set_ylabel("Frequency")
    _axes[0].set_title("Rating Distribution (Histogram)")
    _axes[0].set_xticks(range(0, 6))

    # Boxplot
    _axes[1].boxplot(_ratings, vert=True)
    _axes[1].set_ylabel("Rating")
    _axes[1].set_title("Rating Distribution (Boxplot)")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(df, pl):
    # Rating frequency table
    _rating_freq = df.group_by("rating").len().sort("rating")
    _rating_freq = _rating_freq.with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("percentage")
    )
    _rating_freq
    return


@app.cell
def _(mo):
    mo.md(r"""
    As already seen 54.3% of interactions are "unrated" (rating = 0), and 45.7% have explicit ratings (1-5).
    From the rated interactions, the distribution is heavily skewed toward positive ratings:
    4 and 5 stars account for 70% of all rated interactions (16.4% + 15.5%), ratings 1-2 are rare (0.9% + 2.7%).
    Indicating users tend to rate books they liked.

    For modeling, rating = 0 should be treated as "unrated" rather than a zero score.
    """)
    return


@app.cell
def _(df, pl):
    # is_read frequency table
    _is_read_freq = df.group_by("is_read").len().sort("is_read")
    _is_read_freq = _is_read_freq.with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("percentage")
    )
    print("is_read distribution:")
    _is_read_freq
    return


@app.cell
def _(mo):
    mo.md(r"""
    Balanced distribution (51% / 49%). Strong correlation with ratings. is_read is an implicit signal.
    """)
    return


@app.cell
def _(df, pl):
    # is_reviewed frequency table
    _is_reviewed_freq = df.group_by("is_reviewed").len().sort("is_reviewed")
    _is_reviewed_freq = _is_reviewed_freq.with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("percentage")
    )
    print("is_reviewed distribution:")
    _is_reviewed_freq
    return


@app.cell
def _(mo):
    mo.md(r"""
    Reviews are sparse (7.09%).
    Reviews indicate strong engagement, but the low frequency makes them unsuitable as a primary feature.
    They can be used as an additional engagement signal or to weight ratings (rating that also have a review may have more weight)
    """)
    return


@app.cell
def _(df, pl, plt, sns):
    # Correlation matrix (numeric columns only)
    _numeric_df = df.select([
        pl.col("rating"),
        pl.col("is_read").cast(pl.Int8).alias("is_read"),
        pl.col("is_reviewed").cast(pl.Int8).alias("is_reviewed"),
    ])

    _corr_matrix = _numeric_df.to_pandas().corr()

    _fig, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(_corr_matrix, annot=True, cmap="coolwarm", center=0,
                fmt=".3f", square=True, ax=_ax)
    _ax.set_title("Correlation Heatmap")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Strong correlation between rating and is_read (0.886).
    The correlation is inflated because unread books almost never receive ratings. It does not really explain variation in rating scores.

    Weak correlations with is_reviewed (0.262 and 0.272).
    While reviews indicate engagement, their low frequency limits their predictive value as a feature.
    """)
    return

@app.cell
def _(df, pl):
    # Cross-tabulation: is_read x rating
    _cross_tab = df.group_by(["is_read", "rating"]).agg([
        pl.len().alias("count")
    ]).sort(["is_read", "rating"])

    _cross_tab = _cross_tab.with_columns(
        (pl.col("count") / pl.col("count").sum() * 100).round(2).alias("percentage")
    )

    print("Cross-tabulation: is_read x rating")
    _cross_tab
    return

@app.cell
def _(mo):
    mo.md(r"""
    - is_read = 0: Almost all are rating = 0 (unread = unrated)
    - is_read = 1: Rating distribution skews positive (4-5 dominate)
    Users rate books they have read, and rate them positively.
    """)
    return


@app.cell
def _(df, pl):
    # Cross-tabulation with counts and percentages
    _cross_tab = df.group_by(["is_read", "is_reviewed"]).agg([
        pl.len().alias("count"),
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").filter(pl.col("rating") > 0).mean().alias("avg_rating_nonzero"),
    ]).sort(["is_read", "is_reviewed"])

    # Add percentage column
    _total = df.height
    _cross_tab = _cross_tab.with_columns(
        (pl.col("count") / _total * 100).round(2).alias("percentage")
    )
    _cross_tab
    return


@app.cell
def _(df, pl, plt, sns):
    # Heatmap of cross-tabulation
    _cross_counts = df.group_by(["is_read", "is_reviewed"]).len()

    _pivot_data = _cross_counts.pivot(
        on="is_reviewed",
        index="is_read",
        values="len"
    ).sort("is_read")

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _heatmap_data = _pivot_data.select(pl.exclude("is_read")).to_numpy()
    sns.heatmap(
        _heatmap_data,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["Not Reviewed", "Reviewed"],
        yticklabels=["Not Read", "Read"],
        ax=_ax
    )
    _ax.set_xlabel("is_reviewed")
    _ax.set_ylabel("is_read")
    _ax.set_title("Cross-Tabulation: is_read vs is_reviewed")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    - **(0, 0)**: No engagement beyond interaction record
    - **(1, 0)**: Read but no review - "silent readers"
    - **(1, 1)**: Full engagement - read + reviewed - most active users
    - **(0, 1)**: Data inconsistency? Reviewed without marking as read


    283,912 interactions where books are reviewed but not marked as read.
    This likely represents data inconsistency.
    For modeling, consider setting is_read = True for all reviewed books during data cleaning.
    """)
    return


@app.cell
def _(df, pl):
    # Breakdown of minimal engagement interactions (is_read = False, is_reviewed = False)
    _minimal = df.filter((pl.col("is_read") == False) & (pl.col("is_reviewed") == False))
    _minimal_by_rating = _minimal.group_by("rating").len().sort("rating")
    _minimal_by_rating = _minimal_by_rating.with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("percentage")
    )
    print("Minimal engagement (not read, not reviewed) by rating:")
    _minimal_by_rating
    return


@app.cell
def _(mo):
    mo.md(r"""
    Minimal engagement - likely "want to read".
    """)
    return


@app.cell
def _(df, pl, plt):
    # Does user activity level affect their average rating?
    _user_stats = df.group_by("user_id").agg([
        pl.len().alias("book_count"),
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").filter(pl.col("rating") > 0).mean().alias("avg_rating_nonzero"),
    ])

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.scatter(
        _user_stats["book_count"].to_numpy(),
        _user_stats["avg_rating_nonzero"].to_numpy(),
        alpha=0.3,
        s=5
    )
    _ax.set_xlabel("Books per User")
    _ax.set_ylabel("Average Rating (non-zero)")
    _ax.set_title("User Activity vs Average Rating")
    _ax.set_xscale('log')
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Low-activity users have averages at integer values because of limited ratings.
    High-activity users (right) converge to 4.0-4.5.
    Almost no users with 1000+ interactions have averages below 3.0 - active users select books they expect to like.
    For modeling: Use activity level as a confidence signal. High-activity users are more predictable, low-activity users have high variance.
    """)
    return


@app.cell
def _(df, pl, plt):
    # Does book popularity affect its average rating?
    _book_stats = df.group_by("book_id").agg([
        pl.len().alias("user_count"),
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").filter(pl.col("rating") > 0).mean().alias("avg_rating_nonzero"),
    ])

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.scatter(
        _book_stats["user_count"].to_numpy(),
        _book_stats["avg_rating_nonzero"].to_numpy(),
        alpha=0.3,
        s=5
    )
    _ax.set_xlabel("Users per Book (Popularity)")
    _ax.set_ylabel("Average Rating (non-zero)")
    _ax.set_title("Book Popularity vs Average Rating")
    _ax.set_xscale('log')
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Similar pattern: discrete clusters at low popularity, converge at high popularity.
    Popular books (> 1000 users) cluster at 4.5 - 5.0.
    Very few popular books average below 3.5 - Books with low ratings don't become popular.
    Popular books have less rating variance than popular users - more ratings = less disagreement on book quality.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Feature Engineering Recommendations**
    1. Treat rating = 0 as missing explicit feedback (unrated) rather than an explicit zero rating.
    2. Use is_read = True as implicit positive signal.
    3. Use weighting with is_reviewed: reviewed ratings may deserve higher weights due to higher engagement. 
    4. For prediction confidence weight predictions by interaction count - high-activity users (>1000) have stable averages, low-activity users have high variance.
    5. Handle interactions with is_reviewed=True but is_read = False - set is_read = True for all reviewed interactions.
    6. Consider data sparsity when selecting models.
    7. Use side information (book metadata, user demographics, etc) for cold-start.
    """)
    return


if __name__ == "__main__":
    app.run()
