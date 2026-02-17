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
    from collections import Counter
    import pandas as pd

    return mo, np, pl, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    # EDA for Reviews Dataset
    **Dataset:** `raw_goodreads_reviews_dedup.parquet`

    **Columns:**
    - `user_id`: Unique user identifier
    - `book_id`: Unique book identifier
    - `review_id`: Unique review identifier
    - `rating`: User's rating (0-5)
    - `review_text`: Full text of the review
    - `date_added`: When the review was added
    - `date_updated`: When the review was last updated
    - `read_at`: When the user finished reading
    - `started_at`: When the user started reading
    - `n_votes`: Number of helpful votes on the review
    - `n_comments`: Number of comments on the review
    """)
    return


@app.cell
def _(pl):
    import os
    project_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    data_path = os.path.join(project_root, "data", "raw_goodreads_reviews_dedup.parquet")
    df = pl.read_parquet(data_path)
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    **INITIAL DATA OVERVIEW**
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    n_users = df["user_id"].n_unique()
    n_books = df["book_id"].n_unique()
    n_reviews = df["review_id"].n_unique()

    print(f"Total rows: {df.height:,}")
    print(f"Total unique users: {n_users:,}")
    print(f"Total unique books: {n_books:,}")
    print(f"Total unique reviews: {n_reviews:,}")
    return


@app.cell
def _(df):
    print("Column types:")
    print(df.schema)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **DATA QUALITY CHECKS**
    """)
    return


@app.cell
def _(df, mo, pl):
    _empty_counts = {}
    for _col in df.columns:
        if df[_col].dtype == pl.Utf8:
            _empty_counts[_col] = df.filter(pl.col(_col) == "").height
    pl.from_dict(_empty_counts)

    mo.vstack([
        mo.md(f"Shape: {df.shape}"),
        mo.md("Nulls per column:"),
        df.null_count(),
        mo.md("Empty strings per column:"),
        pl.from_dict(_empty_counts)
    ])
    return


@app.cell
def _(df, mo, pl):
    # Rating range validation
    _min_rating = df['rating'].min()
    _max_rating = df['rating'].max()
    _unique_ratings = sorted(df["rating"].unique().to_list())

    _invalid_ratings = df.filter((pl.col("rating") < 0) | (pl.col("rating") > 5)).height

    mo.vstack([
        mo.md("Rating Validation:"),
        mo.md(f"""
        - Range: {_min_rating} to {_max_rating}
        - Unique values: {_unique_ratings}
        - Invalid ratings (outside 0-5): {_invalid_ratings}""")
    ])
    return


@app.cell
def _(df, mo, pl):
    # Votes and comments validation - check for negative values
    _negative_votes = df.filter(pl.col("n_votes") < 0).height
    _negative_comments = df.filter(pl.col("n_comments") < 0).height

    mo.vstack([
        mo.md("Numeric Columns Validation:"),
        mo.md(f"""
        - n_votes range: {df['n_votes'].min()} to {df['n_votes'].max()}
        - Negative n_votes: {_negative_votes}"""),
        mo.md(f"""
        - n_comments range: {df['n_comments'].min()} to {df['n_comments'].max()}
        - Negative n_comments: {_negative_comments}""")
    ])
    return


@app.cell
def _(df, mo, pl, plt):
    # Focus on lower range to see negative values
    _votes_low = df.filter(pl.col("n_votes") <= 20)["n_votes"].to_numpy()
    _comments_low = df.filter(pl.col("n_comments") <= 20)["n_comments"].to_numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # n_votes distribution (focused on -5 to 20)
    _axes[0].hist(_votes_low, bins=range(-5, 22), edgecolor="black", alpha=0.7, align='left')
    _axes[0].set_xlabel("Number of Votes")
    _axes[0].set_ylabel("Number of Reviews")
    _axes[0].set_title("n_votes Distribution (values -5 to 20)")
    _axes[0].set_yscale("log")
    _axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)

    # n_comments distribution (focused on -20 to 20)
    _axes[1].hist(_comments_low, bins=range(-20, 22), edgecolor="black", alpha=0.7, color="coral", align='left')
    _axes[1].set_xlabel("Number of Comments")
    _axes[1].set_ylabel("Number of Reviews")
    _axes[1].set_title("n_comments Distribution (values -20 to 20)")
    _axes[1].set_yscale("log")
    _axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()

    mo.vstack([
        mo.md("""
        - n_votes and n_comments contain negative values, which are invalid.
        - These negative values should be cleaned (clip to 0) or excluded during analysis."""),
        _fig
    ])
    return


@app.cell
def _(df, pl):
    date_cols = ["date_added", "date_updated", "read_at", "started_at"]

    for col in date_cols:
        valid_dates = df.filter(pl.col(col) != "")

        # Extract years and days using Polars expressions
        years_df = valid_dates.with_columns(
            pl.col(col).str.slice(-4).cast(pl.Int32, strict=False).alias("year"),
            pl.col(col).str.slice(8, 2).cast(pl.Int32, strict=False).alias("day")
        )

        years_list = years_df["year"].drop_nulls().to_list()
        days_list = years_df["day"].drop_nulls().to_list()

        if years_list:
            print(f"{col}: year range {min(years_list)} to {max(years_list)}")
            # Print dates with year < 2000
            old_dates = years_df.filter(pl.col("year") < 2000).select(col).to_series().to_list()
            if old_dates:
                print(f"  Dates with year < 2000: {old_dates[:5]}{' ...' if len(old_dates) > 5 else ''} (total: {len(old_dates)})")
            # Print dates with year > 2026
            future_dates = years_df.filter(pl.col("year") > 2026).select(col).to_series().to_list()
            if future_dates:
                print(f"  Dates with year > 2026: {future_dates[:5]}{' ...' if len(future_dates) > 5 else ''} (total: {len(future_dates)})")

        if days_list:
            print(f"{col}: day range {min(days_list)} to {max(days_list)}")
            if min(days_list) < 1 or max(days_list) > 31:
                print(f"  Warning: Found days outside 1-31 in {col}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Many dates appear invalid:
    some date_added values are very early (year 1092 and 1943)
    read_at and started_at contain thousands of dates with years after 2026 (e.g. 2558) and very early years (like 0).
    They should be cleaned.
    All day values are within a valid range.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Summary of Data Quality Issues:
    - The dataset is complete with no null values, but some columns contain empty strings.
    - n_votes and n_comments have a small number of negative values, which are likely data errors and should be cleaned.
    - For NLP modeling, empty reviews should be excluded.
    - Many dates must be cleaned.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **RATING ANALYSIS**
    """)
    return


@app.cell
def _(df, mo, pl):
    _total = df.height
    _zero_ratings = df.filter(pl.col("rating") == 0).height
    _nonzero_ratings = df.filter(pl.col("rating") > 0).height

    output = [
        mo.md(f"**Total reviews:** {_total:,}"),
        mo.md(f"**Rating = 0 (unrated):** {_zero_ratings:,} ({_zero_ratings/_total*100:.1f}%)"),
        mo.md(f"**Rating > 0 (rated):** {_nonzero_ratings:,} ({_nonzero_ratings/_total*100:.1f}%)")
    ]

    if _nonzero_ratings > 0:
        _mean_rated = df.filter(pl.col("rating") > 0)["rating"].mean()
        output.append(mo.md(f"**Mean rating (excluding 0):** {_mean_rated:.2f}"))

    mo.vstack(output)
    return


@app.cell
def _(df, pl):
    _rating_freq = df.group_by("rating").len().sort("rating")
    _rating_freq = _rating_freq.with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("percentage")
    )
    _rating_freq
    return


@app.cell
def _(df, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _rating_counts = df.group_by("rating").len().sort("rating")

    # All ratings
    _axes[0].bar(
        _rating_counts["rating"].to_list(),
        _rating_counts["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
    )
    _axes[0].set_xlabel("Rating")
    _axes[0].set_ylabel("Count")
    _axes[0].set_title("Rating Distribution (0 = unrated)")
    _axes[0].set_xticks([0, 1, 2, 3, 4, 5])

    for _r, _c in zip(_rating_counts["rating"].to_list(), _rating_counts["len"].to_list()):
        _axes[0].text(_r, _c, f"{_c/1e6:.2f}M", ha="center", va="bottom", fontsize=9)

    # Excluding 0
    _rated_only = _rating_counts.filter(_rating_counts["rating"] > 0)
    _axes[1].bar(
        _rated_only["rating"].to_list(),
        _rated_only["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    _axes[1].set_xlabel("Rating")
    _axes[1].set_ylabel("Count")
    _axes[1].set_title("Rating Distribution (Excluding Unrated)")
    _axes[1].set_xticks([1, 2, 3, 4, 5])

    for _r, _c in zip(_rated_only["rating"].to_list(), _rated_only["len"].to_list()):
        _axes[1].text(_r, _c, f"{_c/1e6:.2f}M", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(df, mo, pl):
    # Analyze relationship between rating, votes, and comments
    _engagement_by_rating = (
        df.filter(pl.col("rating") > 0)
        .group_by("rating")
        .agg([
            pl.col("n_votes").mean().alias("avg_votes"),
            pl.col("n_comments").mean().alias("avg_comments"),
            pl.len().alias("n_reviews"),
        ])
        .sort("rating")
    )

    mo.vstack([
        mo.md("Engagement by Rating:"),
        _engagement_by_rating
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Reviews with extreme ratings (1 and 5) receive the highest average votes and comments, suggesting that both highly positive and highly negative opinions drive more engagement.
    More neutral ratings (2-4) attract less interaction, with the lowest engagement observed for 3-star reviews.
    This pattern indicates that users are more likely to react to reviews expressing strong sentiment.
    Higher weights can be assigned to reviews with more votes and comments, especially those with extreme ratings, so that user queries and recommendations are influenced more by the most impactful opinions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **REVIEW TEXT ANALYSIS**
    """)
    return


@app.cell
def _(df, mo, pl):
    _df_text = df.with_columns([
        pl.col("review_text").str.len_chars().alias("char_count"),
        pl.col("review_text").str.split(" ").list.len().alias("word_count"),
    ])

    _text_stats = _df_text.select([
        pl.col("char_count").mean().alias("avg_chars"),
        pl.col("char_count").median().alias("median_chars"),
        pl.col("char_count").min().alias("min_chars"),
        pl.col("char_count").max().alias("max_chars"),
        pl.col("word_count").mean().alias("avg_words"),
        pl.col("word_count").median().alias("median_words"),
        pl.col("word_count").max().alias("max_words"),
    ])

    stats_display = mo.vstack([
        mo.md("**Review Text Statistics:**"),
        mo.md(f"  Avg characters: {_text_stats['avg_chars'][0]:,.0f}"),
        mo.md(f"  Median characters: {_text_stats['median_chars'][0]:,.0f}"),
        mo.md(f"  Max characters: {_text_stats['max_chars'][0]:,}"),
        mo.md(f"  Avg words: {_text_stats['avg_words'][0]:,.0f}"),
        mo.md(f"  Median words: {_text_stats['median_words'][0]:,.0f}"),
        mo.md(f"  Max words: {_text_stats['max_words'][0]:,}")
    ])

    _empty = _df_text.filter(pl.col("char_count") == 0).height
    _short = _df_text.filter(pl.col("word_count") < 10).height
    _long = _df_text.filter(pl.col("word_count") > 500).height

    counts_display = mo.vstack([
        mo.md(f"\nEmpty reviews: {_empty:,} ({_empty/df.height*100:.2f}%)"),
        mo.md(f"Short reviews (<10 words): {_short:,} ({_empty/df.height*100:.1f}%)"),
        mo.md(f"Long reviews (>500 words): {_long:,} ({_long/df.height*100:.1f}%)")
    ])

    mo.vstack([stats_display, counts_display])
    return


@app.cell
def _(df, pl):
    # Analyze empty review_text entries - categorize by activity
    _empty_reviews = df.filter(pl.col("review_text").str.strip_chars() == "")

    # Create categories
    _empty_reviews_cat = _empty_reviews.with_columns([
        (pl.col("rating") == 0).alias("zero_rating"),
        (pl.col("n_votes") == 0).alias("zero_votes"),
        (pl.col("n_comments") == 0).alias("zero_comments"),
    ])

    # Count combinations
    _all_zeros = _empty_reviews_cat.filter(
        pl.col("zero_rating") & pl.col("zero_votes") & pl.col("zero_comments")
    ).height

    _only_rating = _empty_reviews_cat.filter(
        ~pl.col("zero_rating") & pl.col("zero_votes") & pl.col("zero_comments")
    ).height

    _has_votes_or_comments = _empty_reviews_cat.filter(
        ~pl.col("zero_votes") | ~pl.col("zero_comments")
    ).height

    _total_empty = _empty_reviews.height

    # Summary table
    _summary = pl.DataFrame({
        "Category": [
            "All zeros (rating=0, votes=0, comments=0)",
            "Only has rating (votes=0, comments=0)",
            "Has votes or comments",
            "TOTAL empty reviews"
        ],
        "Count": [_all_zeros, _only_rating, _has_votes_or_comments, _total_empty],
        "Percentage": [
            f"{_all_zeros/_total_empty*100:.1f}%",
            f"{_only_rating/_total_empty*100:.1f}%",
            f"{_has_votes_or_comments/_total_empty*100:.1f}%",
            "100%"
        ]
    })

    print("Empty review_text breakdown:")
    _summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    A small percentage of reviews (0.044%) contain only whitespace.
    However, these empty reviews still show different levels of engagement with nearly half of them containing votes or comments.
    """)
    return


@app.cell
def _(df, np, pl, plt):
    _df_text = df.with_columns([
        pl.col("review_text").str.split(" ").list.len().alias("word_count"),
    ])

    _word_counts = _df_text["word_count"].to_numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _axes[0].hist(np.clip(_word_counts, 0, 500), bins=100, edgecolor="black", alpha=0.7)
    _axes[0].set_xlabel("Word Count (capped at 500)")
    _axes[0].set_ylabel("Number of Reviews")
    _axes[0].set_title("Review Length Distribution")

    _axes[1].hist(np.log1p(_word_counts), bins=100, edgecolor="black", alpha=0.7, color="coral")
    _axes[1].set_xlabel("Log(Word Count + 1)")
    _axes[1].set_ylabel("Number of Reviews")
    _axes[1].set_title("Review Length Distribution (Log Scale)")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(df, pl):
    # Sample extreme cases - shortest reviews
    _non_empty = df.filter(pl.col("review_text").str.strip_chars() != "")

    _shortest = _non_empty.with_columns(
        pl.col("review_text").str.split(" ").list.len().alias("word_count")
    ).sort("word_count").head(10)

    print("Sample: Shortest reviews (by word count):")
    for _row in _shortest.iter_rows(named=True):
        _text = _row["review_text"][:80] + "..." if len(_row["review_text"]) > 80 else _row["review_text"]
        print(f"  [{_row['word_count']} words, rating={_row['rating']}] '{_text}'")
    return


@app.cell
def _(df, pl):
    # Sample extreme cases - longest reviews
    _non_empty = df.filter(pl.col("review_text").str.strip_chars() != "")

    _longest = _non_empty.with_columns(
        pl.col("review_text").str.split(" ").list.len().alias("word_count")
    ).sort("word_count", descending=True).head(5)

    print("Sample: Longest reviews (by word count):")
    for _row in _longest.iter_rows(named=True):
        _preview = _row["review_text"][:150] + "..."
        print(f"  [{_row['word_count']:,} words, rating={_row['rating']}]")
        print(f"    '{_preview}'")
        print()
    return


@app.cell
def _(df, pl):
    _df_text = df.with_columns([
        pl.col("review_text").str.split(" ").list.len().alias("word_count"),
    ])

    _length_by_rating = (
        _df_text.filter(pl.col("rating") > 0)
        .group_by("rating")
        .agg([
            pl.col("word_count").mean().alias("avg_words"),
            pl.col("word_count").median().alias("median_words"),
            pl.col("word_count").std().alias("std_words"),
        ])
        .sort("rating")
    )

    print("Review Length by Rating:")
    _length_by_rating
    return


@app.cell
def _(df, pl):
    # Compute word count for non-empty reviews
    _df_text = df.filter(pl.col("review_text").str.strip_chars() != "").with_columns([
        pl.col("review_text").str.split(" ").list.len().alias("word_count"),
    ])

    # Define breaks (internal division points) and labels
    # 5 breaks create 6 bins: [-inf, 10), [10, 50), [50, 100), [100, 250), [250, 500), [500, inf)
    breaks = [10, 50, 100, 250, 500]
    labels = ["0-10", "10-50", "50-100", "100-250", "250-500", "500+"]

    # Bin word counts using pl.col().cut()
    _binned_df = _df_text.with_columns(
        pl.col("word_count").cut(breaks, labels=labels).alias("length_bin")
    )

    # Group and summarize
    summary = _binned_df.group_by("length_bin", maintain_order=True).agg([
        pl.len().alias("n_reviews"),
        pl.mean("word_count"),
        pl.mean("n_votes"),
        pl.mean("n_comments"),
    ])

    print("Review Length Bin Summary (using polars):")
    print(summary)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Review length does not strongly vary by rating, but longer reviews consistently attract more votes and comments, this is:
    Review length is not a good predictor of rating.
    Review length is a good predictor of engagement.
    Weighting reviews by both engagement and length yields more relevant and impactful suggestions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **VOTES ANALYSIS**
    """)
    return


@app.cell
def _(df, mo, pl):
    _vote_stats = df.select([
        pl.col("n_votes").mean().alias("avg_votes"),
        pl.col("n_votes").median().alias("median_votes"),
        pl.col("n_votes").max().alias("max_votes"),
        pl.col("n_votes").sum().alias("total_votes"),
    ])

    _no_votes = df.filter(pl.col("n_votes") == 0).height
    _has_votes = df.filter(pl.col("n_votes") > 0).height
    _high_votes = df.filter(pl.col("n_votes") >= 10).height

    mo.vstack([
        mo.md("**Vote Statistics:**"),
        mo.md(f"  Avg votes/review: {_vote_stats['avg_votes'][0]:.2f}"),
        mo.md(f"  Median votes: {_vote_stats['median_votes'][0]:.0f}"),
        mo.md(f"  Max votes: {_vote_stats['max_votes'][0]:,}"),
        mo.md(f"  Total votes: {_vote_stats['total_votes'][0]:,}"),
        mo.md(f"\nReviews with 0 votes: {_no_votes:,} ({_no_votes/df.height*100:.1f}%)"),
        mo.md(f"Reviews with 1+ votes: {_has_votes:,} ({_has_votes/df.height*100:.1f}%)"),
        mo.md(f"Reviews with 10+ votes: {_high_votes:,} ({_high_votes/df.height*100:.2f}%)")
    ])
    return


@app.cell
def _(df, np, plt):
    _votes = df["n_votes"].to_numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _axes[0].hist(np.clip(_votes, 0, 50), bins=50, edgecolor="black", alpha=0.7)
    _axes[0].set_xlabel("Number of Votes (capped at 50)")
    _axes[0].set_ylabel("Number of Reviews")
    _axes[0].set_title("Vote Distribution")
    _axes[0].set_yscale("log")

    _votes_nonzero = _votes[_votes > 0]
    if len(_votes_nonzero) > 0:
        _axes[1].hist(np.log1p(_votes_nonzero), bins=50, edgecolor="black", alpha=0.7, color="green")
        _axes[1].set_xlabel("Log(Votes + 1)")
        _axes[1].set_ylabel("Number of Reviews")
        _axes[1].set_title("Vote Distribution (Reviews with votes, Log)")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(df, mo, pl):
    _votes_by_rating = (
        df.filter(pl.col("rating") > 0)
        .group_by("rating")
        .agg([
            pl.col("n_votes").mean().alias("avg_votes"),
            pl.col("n_votes").median().alias("median_votes"),
            pl.col("n_votes").sum().alias("total_votes"),
            pl.len().alias("n_reviews"),
        ])
        .sort("rating")
    )

    mo.vstack([
        mo.md("Votes by Rating:"),
        _votes_by_rating
    ])
    return


@app.cell
def _(df, mo, pl):
    # Filter reviews with 10+ votes
    high_vote_reviews = df.filter(pl.col("n_votes") >= 10)

    # Count unique books among these reviews
    n_high_vote_reviews = high_vote_reviews.height
    n_unique_books = high_vote_reviews["book_id"].n_unique()

    # What percentage of all books do these represent?
    total_books = df["book_id"].n_unique()
    pct_books = n_unique_books / total_books * 100

    mo.vstack([
        mo.md(f"Reviews with 10+ votes: {n_high_vote_reviews:,}"),
        mo.md(f"Unique books among these reviews: {n_unique_books:,} ({pct_books:.2f}% of all books)")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Most reviews do not have votes.
    As previously seen in rating analysis section, reviews with extreme ratings (1 and 5) attract more votes and comments.
    Reviews with 10+ votes are concentrated on a small subset of books (7.2% of all books), engagement is highly focused on a few popular books.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **USER ACTIVITY PATTERNS**
    """)
    return


@app.cell
def _(df, mo, pl):
    _reviews_per_user = df.group_by("user_id").len().rename({"len": "review_count"})

    _q1 = _reviews_per_user["review_count"].quantile(0.25)
    _q3 = _reviews_per_user["review_count"].quantile(0.75)
    _iqr = _q3 - _q1
    _upper_bound = _q3 + 1.5 * _iqr

    _upper_outliers = _reviews_per_user.filter(pl.col("review_count") > _upper_bound)

    mo.vstack([
        mo.md(f"**Unique users:** {_reviews_per_user.height:,}"),
        mo.md(f"**Mean reviews/user:** {_reviews_per_user['review_count'].mean():.1f}"),
        mo.md(f"**Median reviews/user:** {_reviews_per_user['review_count'].median():.0f}"),
        mo.md(f"**Max reviews/user:** {_reviews_per_user['review_count'].max():,}"),
        mo.md(f"\n**IQR:** Q1={_q1:.0f}, Q3={_q3:.0f}"),
        mo.md(f"**Upper outlier threshold:** >{_upper_bound:.0f} reviews"),
        mo.md(f"**Upper outliers:** {_upper_outliers.height:,} users ({_upper_outliers.height/_reviews_per_user.height*100:.1f}%)")
    ])
    return


@app.cell
def _(df, mo, np, plt):
    _reviews_per_user = df.group_by("user_id").len().rename({"len": "review_count"})
    _counts = _reviews_per_user["review_count"].to_pandas()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _axes[0].hist(_counts.clip(upper=100), bins=100, edgecolor="black", alpha=0.7)
    _axes[0].set_xlabel("Reviews per User (capped at 100)")
    _axes[0].set_ylabel("Number of Users")
    _axes[0].set_title("User Activity Distribution")

    _axes[1].hist(np.log1p(_counts), bins=100, edgecolor="black", alpha=0.7)
    _axes[1].set_xlabel("Log(Reviews per User + 1)")
    _axes[1].set_ylabel("Number of Users")
    _axes[1].set_title("User Activity Distribution (Log Scale)")

    plt.tight_layout()
    mo.vstack([
        mo.md("""Most users write only a few reviews.
    13.4% of users are highly active, with more than 52 reviews each.
    Care should be taken to avoid biasing suggestions toward their preferences."""),
        _fig
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **BOOK POPULARITY PATTERNS**
    """)
    return


@app.cell
def _(df, pl):
    _reviews_per_book = df.group_by("book_id").len().rename({"len": "review_count"})

    _q1 = _reviews_per_book["review_count"].quantile(0.25)
    _q3 = _reviews_per_book["review_count"].quantile(0.75)
    _iqr = _q3 - _q1
    _upper_bound = _q3 + 1.5 * _iqr

    _upper_outliers = _reviews_per_book.filter(pl.col("review_count") > _upper_bound)

    print(f"Unique books: {_reviews_per_book.height:,}")
    print(f"Mean reviews/book: {_reviews_per_book['review_count'].mean():.1f}")
    print(f"Median reviews/book: {_reviews_per_book['review_count'].median():.0f}")
    print(f"Max reviews/book: {_reviews_per_book['review_count'].max():,}")
    print(f"\nIQR: Q1={_q1:.0f}, Q3={_q3:.0f}")
    print(f"Upper outlier threshold: >{_upper_bound:.0f} reviews")
    print(f"Upper outliers: {_upper_outliers.height:,} books ({_upper_outliers.height/_reviews_per_book.height*100:.1f}%)")
    return


@app.cell
def _(df, mo, np, plt):
    _reviews_per_book = df.group_by("book_id").len().rename({"len": "review_count"})
    _counts = _reviews_per_book["review_count"].to_pandas()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _axes[0].hist(_counts.clip(upper=500), bins=100, edgecolor="black", alpha=0.7)
    _axes[0].set_xlabel("Reviews per Book (capped at 500)")
    _axes[0].set_ylabel("Number of Books")
    _axes[0].set_title("Book Popularity Distribution")

    _axes[1].hist(np.log1p(_counts), bins=100, edgecolor="black", alpha=0.7)
    _axes[1].set_xlabel("Log(Reviews per Book + 1)")
    _axes[1].set_ylabel("Number of Books")
    _axes[1].set_title("Book Popularity Distribution (Log Scale)")

    plt.tight_layout()
    mo.vstack([
        mo.md("""Most books receive only a little amount of reviews, but a small subset (12.2%) are highly popular with more than 8 reviews each.
    A higher review count provides a more stable estimate of overall reader sentiment.
    However, review volume does not imply positive reception, and care should be taken to avoid overlooking books with fewer reviews."""),
        _fig
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **COLD START ANALYSIS**
    """)
    return


@app.cell
def _(df, mo, pl):
    _user_counts = df.group_by("user_id").len()
    _book_counts = df.group_by("book_id").len()

    _cold_users_1 = _user_counts.filter(pl.col("len") == 1).height
    _cold_users_5 = _user_counts.filter(pl.col("len") <= 5).height
    _cold_books_1 = _book_counts.filter(pl.col("len") == 1).height
    _cold_books_5 = _book_counts.filter(pl.col("len") <= 5).height

    _n_users = _user_counts.height
    _n_books = _book_counts.height

    mo.vstack([
        mo.md("**Cold Start Analysis:**"),
        mo.md(f"\n**Users with only 1 review:** {_cold_users_1:,} ({_cold_users_1/_n_users*100:.1f}%)"),
        mo.md(f"**Users with ≤5 reviews:** {_cold_users_5:,} ({_cold_users_5/_n_users*100:.1f}%)"),
        mo.md(f"\n**Books with only 1 review:** {_cold_books_1:,} ({_cold_books_1/_n_books*100:.1f}%)"),
        mo.md(f"**Books with ≤5 reviews:** {_cold_books_5:,} ({_cold_books_5/_n_books*100:.1f}%)")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Almost half of users and the majority of books have five or fewer reviews - cold start problem
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **CORRELATION**
    """)
    return


@app.cell
def _(df, mo, mod, pl, plt, sns):
    _numeric_df = df.select([
        pl.col("rating"),
        pl.col("n_votes"),
        pl.col("review_text").str.split(" ").list.len().alias("word_count"),
    ])

    _corr_matrix = _numeric_df.to_pandas().corr()

    _fig, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(_corr_matrix, annot=True, cmap="coolwarm", center=0,
                fmt=".3f", square=True, ax=_ax)
    _ax.set_title("Correlation Heatmap: Rating, Votes, Review Length")

    mo.vstack([
        _fig,
        mod.md("Weak correlations.")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Recommendations
    - Remove or impute invalid dates (years outside a reasonable range).
    - Clip or exclude negative values in `n_votes` and `n_comments`.
    - Exclude empty reviews for NLP and modeling tasks.
    - Ensure recommendations are not overly biased toward popular books or highly active users.
    - Prioritize reviews with higher engagement (votes/comments), especially those with extreme ratings
    """)
    return


if __name__ == "__main__":
    app.run()
