import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, pl, plt


@app.cell
def _(mo):
    mo.md("""
    #EDA for Interactions Dataset

    Dataset: `raw_goodreads_interactions.parquet`
    """)
    return


@app.cell
def _(mo, pl):
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_path = os.path.join(
        project_root, "data", "raw_goodreads_interactions.parquet"
    )
    df = pl.read_parquet(data_path)
    shape = df.shape

    mo.vstack([
        mo.md(f"""
    ### Dataset Overview
    - **Rows:** {shape[0]:,} interactions
    - **Columns:** {shape[1]}
    - **Columns list:** {', '.join(df.columns)}
    """),
        df.head(10),
    ])
    return (df,)


@app.cell
def _(df, mo):
    """Null and Data Type Analysis"""
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))

    mo.vstack([
        mo.md(f"""
    ### Data Quality
    - **Total null values:** {total_nulls}
    - All columns are `Int64` type
    """),
        mo.md("**Null counts per column:**"),
        null_counts,
    ])
    return


@app.cell
def _(df, mo):
    """Summary Statistics"""
    summary = df.describe()

    mo.vstack([
        mo.md("### Summary Statistics"),
        summary,
    ])
    return


@app.cell
def _(df, mo, pl):
    """Rating Distribution"""
    rating_counts = (
        df.group_by("rating")
        .len()
        .sort("rating")
        .with_columns(
            (pl.col("len") / df.shape[0] * 100).round(2).alias("percentage")
        )
    )

    unrated = df.filter(pl.col("rating") == 0).shape[0]
    rated = df.filter(pl.col("rating") > 0).shape[0]
    rated_mean = df.filter(pl.col("rating") > 0)["rating"].mean()

    mo.vstack([
        mo.md(f"""
    ### Rating Distribution
    - **Unrated interactions (rating=0):** {unrated:,} ({unrated / df.shape[0] * 100:.1f}%)
    - **Rated interactions (rating>0):** {rated:,} ({rated / df.shape[0] * 100:.1f}%)
    - **Mean rating (excluding 0s):** {rated_mean:.2f}
    """),
        rating_counts,
    ])
    return


@app.cell
def _(df, plt):
    """Rating Distribution Plot"""
    rating_counts_plot = (
        df.group_by("rating").len().sort("rating")
    )

    fig_rating, ax_rating = plt.subplots(figsize=(10, 6))
    ax_rating.bar(
        rating_counts_plot["rating"].to_list(),
        rating_counts_plot["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
    )
    ax_rating.set_xlabel("Rating")
    ax_rating.set_ylabel("Count")
    ax_rating.set_title("Distribution of Ratings (0 = unrated)")
    ax_rating.set_xticks([0, 1, 2, 3, 4, 5])

    for _, (_r, _c) in enumerate(
        zip(
            rating_counts_plot["rating"].to_list(),
            rating_counts_plot["len"].to_list(),
        )
    ):
        ax_rating.text(_r, _c, f"{_c / 1e6:.1f}M", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_rating
    return


@app.cell
def _(df, pl, plt):
    """Rating Distribution (Excluding Unrated)"""
    rated_only = df.filter(pl.col("rating") > 0)
    rating_counts_rated = rated_only.group_by("rating").len().sort("rating")

    fig_rated, ax_rated = plt.subplots(figsize=(10, 6))
    ax_rated.bar(
        rating_counts_rated["rating"].to_list(),
        rating_counts_rated["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    ax_rated.set_xlabel("Rating")
    ax_rated.set_ylabel("Count")
    ax_rated.set_title("Distribution of Ratings (Excluding Unrated)")
    ax_rated.set_xticks([1, 2, 3, 4, 5])

    for _r, _c in zip(
        rating_counts_rated["rating"].to_list(),
        rating_counts_rated["len"].to_list(),
    ):
        ax_rated.text(_r, _c, f"{_c / 1e6:.1f}M", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_rated
    return


@app.cell
def _(mo):
    mo.md("""
    Rating distribution is heavily skewed toward higher ratings (4-5 stars), typical J-shaped pattern in user-generated ratings. Large portion of interactions are unrated (rating=0).
    """)
    return



@app.cell
def _(df, mo):
    """is_read and is_reviewed Distribution"""
    read_counts = df["is_read"].value_counts().sort("is_read")
    reviewed_counts = df["is_reviewed"].value_counts().sort("is_reviewed")

    read_pct = df["is_read"].mean() * 100
    reviewed_pct = df["is_reviewed"].mean() * 100

    mo.vstack([
        mo.md(f"""
    ### Interaction Type Distribution
    - **Read:** {read_pct:.1f}% of interactions
    - **Reviewed:** {reviewed_pct:.1f}% of interactions
    """),
        mo.hstack([
            mo.vstack([mo.md("**is_read**"), read_counts]),
            mo.vstack([mo.md("**is_reviewed**"), reviewed_counts]),
        ]),
    ])
    return


@app.cell
def _(df, mo, pl):
    """is_read vs rating"""
    crosstab = (
        df.group_by(["is_read", "rating"])
        .len()
        .sort(["is_read", "rating"])
        .with_columns(
            (pl.col("len") / df.shape[0] * 100).round(2).alias("pct")
        )
    )

    mo.vstack([
        mo.md("### is_read x rating"),
        crosstab,
    ])
    return


@app.cell
def _(df, mo, pl):
    """is_reviewed vs rating"""
    crosstab_rev = (
        df.group_by(["is_reviewed", "rating"])
        .len()
        .sort(["is_reviewed", "rating"])
        .with_columns(
            (pl.col("len") / df.shape[0] * 100).round(2).alias("pct")
        )
    )

    # Users who review vs who don't - rating comparison
    reviewed_ratings = df.filter(
        (pl.col("is_reviewed") == 1) & (pl.col("rating") > 0)
    )["rating"]
    not_reviewed_ratings = df.filter(
        (pl.col("is_reviewed") == 0) & (pl.col("rating") > 0)
    )["rating"]

    mo.vstack([
        mo.md(f"""
    ### is_reviewed x rating
    - Mean rating when reviewed: **{reviewed_ratings.mean():.2f}**
    - Mean rating when not reviewed: **{not_reviewed_ratings.mean():.2f}**
    """),
        crosstab_rev,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    Users who write reviews tend to give slightly different ratings than those who don't, suggesting review presence could be a useful signal.
    """)
    return


@app.cell
def _(df, mo, np, plt):
    """User Activity Distribution"""
    user_activity = df.group_by("user_id").len().rename({"len": "interaction_count"})

    mo.vstack([
        mo.md(f"""
    ### User Activity Distribution
    - **Total unique users:** {user_activity.shape[0]:,}
    - **Mean interactions/user:** {user_activity["interaction_count"].mean():.1f}
    - **Median interactions/user:** {user_activity["interaction_count"].median():.0f}
    - **Max interactions/user:** {user_activity["interaction_count"].max():,}
    - **Min interactions/user:** {user_activity["interaction_count"].min()}
    """),
    ])

    fig_user, axes_user = plt.subplots(1, 2, figsize=(14, 5))

    # Raw distribution (capped for visibility)
    _user_counts = user_activity["interaction_count"].to_pandas()
    axes_user[0].hist(_user_counts.clip(upper=500), bins=100, edgecolor="black", alpha=0.7)
    axes_user[0].set_xlabel("Interactions per User (capped at 500)")
    axes_user[0].set_ylabel("Number of Users")
    axes_user[0].set_title("User Activity Distribution (Raw)")

    # Log-transformed
    _log_user_counts = np.log1p(_user_counts)
    axes_user[1].hist(_log_user_counts, bins=100, edgecolor="black", alpha=0.7)
    axes_user[1].set_xlabel("Log(Interactions per User)")
    axes_user[1].set_ylabel("Number of Users")
    axes_user[1].set_title("User Activity Distribution (Log)")

    plt.tight_layout()
    fig_user
    return (user_activity,)


@app.cell
def _(mo):
    mo.md("""
    User activity is heavily right-skewed with most users having few interactions and a long tail of power users.
    """)
    return


@app.cell
def _(df, mo, np, plt):
    """Book Popularity Distribution"""
    book_popularity = df.group_by("book_id").len().rename({"len": "interaction_count"})

    mo.vstack([
        mo.md(f"""
    ### Book Popularity Distribution
    - **Total unique books:** {book_popularity.shape[0]:,}
    - **Mean interactions/book:** {book_popularity["interaction_count"].mean():.1f}
    - **Median interactions/book:** {book_popularity["interaction_count"].median():.0f}
    - **Max interactions/book:** {book_popularity["interaction_count"].max():,}
    - **Min interactions/book:** {book_popularity["interaction_count"].min()}
    """),
    ])

    fig_book, axes_book = plt.subplots(1, 2, figsize=(14, 5))

    _book_counts = book_popularity["interaction_count"].to_pandas()
    axes_book[0].hist(_book_counts.clip(upper=2000), bins=100, edgecolor="black", alpha=0.7)
    axes_book[0].set_xlabel("Interactions per Book (capped at 2000)")
    axes_book[0].set_ylabel("Number of Books")
    axes_book[0].set_title("Book Popularity Distribution (Raw)")

    _log_book_counts = np.log1p(_book_counts)
    axes_book[1].hist(_log_book_counts, bins=100, edgecolor="black", alpha=0.7)
    axes_book[1].set_xlabel("Log(Interactions per Book)")
    axes_book[1].set_ylabel("Number of Books")
    axes_book[1].set_title("Book Popularity Distribution (Log)")

    plt.tight_layout()
    fig_book
    return (book_popularity,)


@app.cell
def _(mo):
    mo.md("""
    Book popularity also follows a power-law distribution. A small number of books receive the majority of interactions (long-tail effect).
    """)
    return


@app.cell
def _(book_popularity, df, mo, user_activity):
    """Sparsity Analysis"""
    n_users = user_activity.shape[0]
    n_books = book_popularity.shape[0]
    n_interactions = df.shape[0]
    total_possible = n_users * n_books
    sparsity = 1 - (n_interactions / total_possible)

    mo.md(f"""
    ### Sparsity Analysis (User-Book Matrix)
    - **Users:** {n_users:,}
    - **Books:** {n_books:,}
    - **Interactions:** {n_interactions:,}
    - **Possible interactions:** {total_possible:,.0f}
    - **Sparsity:** {sparsity * 100:.4f}%
    - **Density:** {(1 - sparsity) * 100:.6f}%

    The interaction matrix is extremely sparse. Collaborative filtering methods need to handle this sparsity effectively.
    """)
    return


@app.cell
def _(df, mo, pl, plt):
    """Rating Conditional on Read Status"""
    read_ratings = df.filter(pl.col("is_read") == 1).group_by("rating").len().sort("rating")
    unread_ratings = df.filter(pl.col("is_read") == 0).group_by("rating").len().sort("rating")

    fig_cond, axes_cond = plt.subplots(1, 2, figsize=(14, 5))

    axes_cond[0].bar(
        read_ratings["rating"].to_list(),
        read_ratings["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
        color="green",
    )
    axes_cond[0].set_xlabel("Rating")
    axes_cond[0].set_ylabel("Count")
    axes_cond[0].set_title("Ratings for Read Books (is_read=1)")
    axes_cond[0].set_xticks([0, 1, 2, 3, 4, 5])

    axes_cond[1].bar(
        unread_ratings["rating"].to_list(),
        unread_ratings["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
        color="orange",
    )
    axes_cond[1].set_xlabel("Rating")
    axes_cond[1].set_ylabel("Count")
    axes_cond[1].set_title("Ratings for Unread Books (is_read=0)")
    axes_cond[1].set_xticks([0, 1, 2, 3, 4, 5])

    plt.tight_layout()

    mo.vstack([
        mo.md("### Rating Distribution Conditional on Read Status"),
        fig_cond,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    Unread books are almost entirely unrated (rating=0), which makes sense. Read books show the classic J-shaped rating distribution skewed toward 4-5 stars.
    """)
    return


@app.cell
def _(df, mo, pl):
    """Cold Start Analysis"""
    user_counts = df.group_by("user_id").len()
    book_counts = df.group_by("book_id").len()

    cold_users_5 = user_counts.filter(pl.col("len") <= 5).shape[0]
    cold_users_10 = user_counts.filter(pl.col("len") <= 10).shape[0]
    cold_books_5 = book_counts.filter(pl.col("len") <= 5).shape[0]
    cold_books_10 = book_counts.filter(pl.col("len") <= 10).shape[0]

    _n_users = user_counts.shape[0]
    _n_books = book_counts.shape[0]

    mo.md(f"""
    ### Cold Start Analysis
    | Threshold | Cold Users | % | Cold Books | % |
    |-----------|-----------|---|-----------|---|
    | <= 5 interactions | {cold_users_5:,} | {cold_users_5 / _n_users * 100:.1f}% | {cold_books_5:,} | {cold_books_5 / _n_books * 100:.1f}% |
    | <= 10 interactions | {cold_users_10:,} | {cold_users_10 / _n_users * 100:.1f}% | {cold_books_10:,} | {cold_books_10 / _n_books * 100:.1f}% |

    Cold start is a significant challenge - many users and books have very few interactions.
    """)
    return


@app.cell
def _(df, mo, pl):
    """Rating Behavior Analysis"""
    user_rating_stats = (
        df.filter(pl.col("rating") > 0)
        .group_by("user_id")
        .agg([
            pl.col("rating").mean().alias("avg_rating"),
            pl.col("rating").std().alias("std_rating"),
            pl.col("rating").len().alias("n_ratings"),
        ])
    )

    generous = user_rating_stats.filter(pl.col("avg_rating") >= 4.5).shape[0]
    harsh = user_rating_stats.filter(pl.col("avg_rating") <= 2.5).shape[0]
    low_variance = user_rating_stats.filter(
        (pl.col("std_rating") < 0.5) & (pl.col("n_ratings") >= 5)
    ).shape[0]

    mo.md(f"""
    ### User Rating Behavior
    - **Users who rated at least 1 book:** {user_rating_stats.shape[0]:,}
    - **"Generous" raters (avg >= 4.5):** {generous:,} ({generous / user_rating_stats.shape[0] * 100:.1f}%)
    - **"Harsh" raters (avg <= 2.5):** {harsh:,} ({harsh / user_rating_stats.shape[0] * 100:.1f}%)
    - **Low-variance raters (std < 0.5, n >= 5):** {low_variance:,} ({low_variance / user_rating_stats.shape[0] * 100:.1f}%)

    Many users are "generous raters" giving mostly high scores. Low-variance raters provide less discriminating signal for recommendations.
    """)
    return (user_rating_stats,)


@app.cell
def _(pl, plt, user_rating_stats):
    """User Average Rating Distribution"""
    fig_user_avg, axes_ua = plt.subplots(1, 2, figsize=(14, 5))

    avg_ratings = user_rating_stats.filter(
        pl.col("n_ratings") >= 5
    )["avg_rating"].to_pandas()

    axes_ua[0].hist(avg_ratings, bins=50, edgecolor="black", alpha=0.7)
    axes_ua[0].set_xlabel("Average Rating per User")
    axes_ua[0].set_ylabel("Number of Users")
    axes_ua[0].set_title("Distribution of User Average Ratings (n>=5)")

    std_ratings = user_rating_stats.filter(
        pl.col("n_ratings") >= 5
    )["std_rating"].to_pandas()

    axes_ua[1].hist(std_ratings.dropna(), bins=50, edgecolor="black", alpha=0.7, color="coral")
    axes_ua[1].set_xlabel("Rating Std Dev per User")
    axes_ua[1].set_ylabel("Number of Users")
    axes_ua[1].set_title("Distribution of User Rating Variability (n>=5)")

    plt.tight_layout()
    fig_user_avg
    return


@app.cell
def _(mo):
    mo.md("""
    Most users cluster around 3.5-4.0 average rating. Rating variability shows many users have moderate standard deviations (~0.8-1.2), but a notable group has very low variance.
    """)
    return

@app.cell
def _(df, mo, pl):
    """Interaction Patterns Summary"""
    # How many users both read and rate?
    read_and_rated = df.filter(
        (pl.col("is_read") == 1) & (pl.col("rating") > 0)
    ).shape[0]

    # Read but not rated
    read_not_rated = df.filter(
        (pl.col("is_read") == 1) & (pl.col("rating") == 0)
    ).shape[0]

    # Not read but rated (unusual)
    not_read_but_rated = df.filter(
        (pl.col("is_read") == 0) & (pl.col("rating") > 0)
    ).shape[0]

    # Not read and not rated (just added/shelved)
    shelved_only = df.filter(
        (pl.col("is_read") == 0) & (pl.col("rating") == 0)
    ).shape[0]

    total = df.shape[0]

    mo.md(f"""
    ### Interaction Pattern Breakdown
    | Pattern | Count | % |
    |---------|-------|---|
    | Read + Rated | {read_and_rated:,} | {read_and_rated / total * 100:.1f}% |
    | Read + Not Rated | {read_not_rated:,} | {read_not_rated / total * 100:.1f}% |
    | Not Read + Rated | {not_read_but_rated:,} | {not_read_but_rated / total * 100:.1f}% |
    | Shelved Only (no read, no rating) | {shelved_only:,} | {shelved_only / total * 100:.1f}% |

    Interesting: some users rate books without marking them as read. This could indicate data quality issues or users rating from memory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature Engineering Recommendations

    1. **Rating=0 means unrated**, not a zero-star rating. Filter these out for rating-based models.
    2. **Implicit vs explicit signals:** `is_read` and `is_reviewed` are implicit signals; `rating` is explicit. Consider separate models or combine them.
    3. **User/book activity thresholds:** Filter cold-start users/books (e.g., min 5-10 interactions) for collaborative filtering.
    4. **Generous rater bias:** Many users rate everything 4-5 stars. Consider user-mean normalization for ratings.
    5. **Sparsity:** collaborative filtering is better suited
    """)
    return


if __name__ == "__main__":
    app.run()
