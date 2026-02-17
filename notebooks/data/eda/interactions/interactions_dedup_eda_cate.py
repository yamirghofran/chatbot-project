import marimo

__generated_with = "0.19.8"
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
    # Preliminary EDA for Goodreads Interactions (Dedup) Dataset

    This dataset contains user-book interactions with richer metadata including
    review text, timestamps, and original string IDs.
    """)
    return


@app.cell
def _(mo, pl):
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_path = os.path.join(
        project_root, "data", "raw_goodreads_interactions_dedup.parquet"
    )
    _df_full = pl.read_parquet(data_path)
    _full_rows = _df_full.shape[0]
    df = _df_full.sample(fraction=0.05, seed=42)
    del _df_full
    shape = df.shape

    mo.vstack([
        mo.md(f"""
    ### Dataset Overview
    - **Full dataset:** {_full_rows:,} interactions
    - **Sampled (5%):** {shape[0]:,} interactions
    - **Columns ({shape[1]}):** {', '.join(df.columns)}
    """),
        mo.md("**Schema:**"),
        mo.as_html(
            pl.DataFrame({
                "column": df.columns,
                "dtype": [str(dt) for dt in df.dtypes],
            })
        ),
        mo.md("**Sample rows:**"),
        df.head(10),
    ])
    return (df,)


@app.cell
def _(df, mo, pl):
    """Null and Empty String Analysis"""
    null_counts = df.null_count()

    # For string columns, also count empty strings
    string_cols = [
        col for col, dt in zip(df.columns, df.dtypes) if dt == pl.String
    ]
    empty_counts = {}
    for col in string_cols:
        empty_counts[col] = df.filter(pl.col(col) == "").shape[0]

    empty_df = pl.DataFrame({
        "column": list(empty_counts.keys()),
        "empty_string_count": list(empty_counts.values()),
        "empty_pct": [
            f"{v / df.shape[0] * 100:.1f}%" for v in empty_counts.values()
        ],
    })

    mo.vstack([
        mo.md("### Data Quality: Nulls and Empty Strings"),
        mo.md("**Null counts:**"),
        null_counts,
        mo.md("**Empty string counts (string columns only):**"),
        empty_df,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    No null values in the dataset, but `read_at` and `started_at` have many empty strings representing missing data. `review_text_incomplete` is also mostly empty.
    """)
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
    - **Unrated (rating=0):** {unrated:,} ({unrated / df.shape[0] * 100:.1f}%)
    - **Rated (rating>0):** {rated:,} ({rated / df.shape[0] * 100:.1f}%)
    - **Mean rating (excluding 0):** {rated_mean:.2f}
    """),
        rating_counts,
    ])
    return


@app.cell
def _(df, mo, pl, plt):
    """Rating Distribution Plots"""
    rating_all = df.group_by("rating").len().sort("rating")
    rating_nonzero = df.filter(pl.col("rating") > 0).group_by("rating").len().sort("rating")

    fig_rating, axes_r = plt.subplots(1, 2, figsize=(14, 5))

    axes_r[0].bar(
        rating_all["rating"].to_list(),
        rating_all["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
    )
    axes_r[0].set_xlabel("Rating")
    axes_r[0].set_ylabel("Count")
    axes_r[0].set_title("All Ratings (0 = unrated)")
    axes_r[0].set_xticks([0, 1, 2, 3, 4, 5])
    for _r, _c in zip(rating_all["rating"].to_list(), rating_all["len"].to_list()):
        axes_r[0].text(_r, _c, f"{_c / 1e6:.1f}M", ha="center", va="bottom", fontsize=8)

    axes_r[1].bar(
        rating_nonzero["rating"].to_list(),
        rating_nonzero["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    axes_r[1].set_xlabel("Rating")
    axes_r[1].set_ylabel("Count")
    axes_r[1].set_title("Rated Only (1-5)")
    axes_r[1].set_xticks([1, 2, 3, 4, 5])
    for _r, _c in zip(rating_nonzero["rating"].to_list(), rating_nonzero["len"].to_list()):
        axes_r[1].text(_r, _c, f"{_c / 1e6:.1f}M", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    mo.vstack([
        fig_rating,
        mo.md("Same J-shaped rating distribution as the integer-encoded dataset. Majority of interactions are unrated (shelved/added).")
    ])
    return


@app.cell
def _(df, mo, pl):
    """is_read Distribution"""
    read_counts = df["is_read"].value_counts().sort("is_read")
    read_pct = df.filter(pl.col("is_read") == True).shape[0] / df.shape[0] * 100

    mo.vstack([
        mo.md(f"""
    ### Read Status Distribution
    - **Read:** {read_pct:.1f}%
    - **Not Read:** {100 - read_pct:.1f}%
    """),
        read_counts,
    ])
    return


@app.cell
def _(df, mo, pl):
    """Review Text Analysis"""
    has_review = df.filter(pl.col("review_text_incomplete") != "").shape[0]
    no_review = df.shape[0] - has_review

    # Length distribution of reviews
    review_lengths = (
        df.filter(pl.col("review_text_incomplete") != "")
        .with_columns(
            pl.col("review_text_incomplete").str.len_chars().alias("review_length")
        )
    )

    mo.vstack([
        mo.md(f"""
    ### Review Text Analysis
    - **Has review text:** {has_review:,} ({has_review / df.shape[0] * 100:.1f}%)
    - **No review text:** {no_review:,} ({no_review / df.shape[0] * 100:.1f}%)
    - **Mean review length:** {review_lengths["review_length"].mean():.0f} chars
    - **Median review length:** {review_lengths["review_length"].median():.0f} chars
    - **Max review length:** {review_lengths["review_length"].max():,} chars
    """),
    ])
    return (review_lengths,)


@app.cell
def _(mo, np, plt, review_lengths):
    """Review Length Distribution"""
    fig_rev, axes_rev = plt.subplots(1, 2, figsize=(14, 5))

    lengths = review_lengths["review_length"].to_pandas()

    axes_rev[0].hist(lengths.clip(upper=2000), bins=100, edgecolor="black", alpha=0.7)
    axes_rev[0].set_xlabel("Review Length (chars, capped at 2000)")
    axes_rev[0].set_ylabel("Frequency")
    axes_rev[0].set_title("Review Length Distribution (Raw)")

    axes_rev[1].hist(np.log1p(lengths), bins=100, edgecolor="black", alpha=0.7, color="coral")
    axes_rev[1].set_xlabel("Log(Review Length)")
    axes_rev[1].set_ylabel("Frequency")
    axes_rev[1].set_title("Review Length Distribution (Log)")

    plt.tight_layout()

    mo.vstack([
        fig_rev,
        mo.md("Review text is highly variable in length with a right-skewed distribution. The field is called `review_text_incomplete`, suggesting reviews may be truncated.")
    ])

    return


@app.cell
def _(df, mo, pl):
    """Review Text vs Rating"""
    # Average review length by rating
    avg_len_by_rating = (
        df.filter(pl.col("review_text_incomplete") != "")
        .with_columns(
            pl.col("review_text_incomplete").str.len_chars().alias("review_length")
        )
        .group_by("rating")
        .agg(
            pl.col("review_length").mean().alias("avg_length"),
            pl.col("review_length").median().alias("median_length"),
            pl.len().alias("count"),
        )
        .sort("rating")
    )

    mo.vstack([
        mo.md("### Review Presence and Length by Rating"),
        mo.md("**Average review length by rating:**"),
        avg_len_by_rating,
    ])
    return


@app.cell
def _(df, mo, pl):
    """Date Parsing and Temporal Coverage"""
    # Parse date_added (format: "Tue Oct 17 09:40:11 -0700 2017")
    # Polars strptime format
    date_format = "%a %b %d %H:%M:%S %z %Y"

    df_dates = df.with_columns([
        pl.col("date_added").str.to_datetime(date_format, strict=False).alias("date_added_parsed"),
        pl.col("date_updated").str.to_datetime(date_format, strict=False).alias("date_updated_parsed"),
        pl.when(pl.col("read_at") != "")
        .then(pl.col("read_at").str.to_datetime(date_format, strict=False))
        .otherwise(None)
        .alias("read_at_parsed"),
        pl.when(pl.col("started_at") != "")
        .then(pl.col("started_at").str.to_datetime(date_format, strict=False))
        .otherwise(None)
        .alias("started_at_parsed"),
    ])

    parsed_success = df_dates.filter(pl.col("date_added_parsed").is_not_null()).shape[0]
    parsed_fail = df_dates.filter(pl.col("date_added_parsed").is_null()).shape[0]

    mo.vstack([
        mo.md(f"""
    ### Temporal Coverage
    - **date_added parsed successfully:** {parsed_success:,} ({parsed_success / df.shape[0] * 100:.1f}%)
    - **date_added parse failures:** {parsed_fail:,}
    - **date_added range:** {df_dates["date_added_parsed"].min()} to {df_dates["date_added_parsed"].max()}
    - **date_updated range:** {df_dates["date_updated_parsed"].min()} to {df_dates["date_updated_parsed"].max()}
    - **read_at non-empty:** {df_dates.filter(pl.col("read_at_parsed").is_not_null()).shape[0]:,}
    - **started_at non-empty:** {df_dates.filter(pl.col("started_at_parsed").is_not_null()).shape[0]:,}
    """),
    ])
    return (df_dates,)


@app.cell
def _(df_dates, mo, pl, plt):
    """Temporal Distribution: date_added"""
    added_by_month = (
        df_dates.filter(pl.col("date_added_parsed").is_not_null())
        .with_columns(
            pl.col("date_added_parsed").dt.truncate("1mo").alias("month")
        )
        .group_by("month")
        .len()
        .sort("month")
    )

    fig_temporal, ax_temporal = plt.subplots(figsize=(14, 6))
    ax_temporal.plot(
        added_by_month["month"].to_pandas(),
        added_by_month["len"].to_pandas(),
        linewidth=0.8,
    )
    ax_temporal.set_xlabel("Month")
    ax_temporal.set_ylabel("Interactions Added")
    ax_temporal.set_title("Interactions Added Over Time (Monthly)")
    ax_temporal.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    mo.vstack([
        mo.md("### Temporal Distribution: date_added"),
        fig_temporal,
    ])
    return


@app.cell
def _(df_dates, mo, pl, plt):
    """Temporal Distribution: date_updated"""
    updated_by_month = (
        df_dates.filter(pl.col("date_updated_parsed").is_not_null())
        .with_columns(
            pl.col("date_updated_parsed").dt.truncate("1mo").alias("month")
        )
        .group_by("month")
        .len()
        .sort("month")
    )

    fig_updated, ax_updated = plt.subplots(figsize=(14, 6))
    ax_updated.plot(
        updated_by_month["month"].to_pandas(),
        updated_by_month["len"].to_pandas(),
        linewidth=0.8,
        color="orange",
    )
    ax_updated.set_xlabel("Month")
    ax_updated.set_ylabel("Interactions Updated")
    ax_updated.set_title("Interactions Updated Over Time (Monthly)")
    ax_updated.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    mo.vstack([
        mo.md("### Temporal Distribution: date_updated"),
        fig_updated,
    ])
    return


@app.cell
def _(df_dates, mo, pl, plt):
    """Day of Week Patterns"""
    dow_added = (
        df_dates.filter(pl.col("date_added_parsed").is_not_null())
        .with_columns(
            pl.col("date_added_parsed").dt.weekday().alias("day_of_week")
        )
        .group_by("day_of_week")
        .len()
        .sort("day_of_week")
    )

    day_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

    fig_dow, ax_dow = plt.subplots(figsize=(10, 6))
    ax_dow.bar(
        [day_names[d] for d in dow_added["day_of_week"].to_list()],
        dow_added["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
    )
    ax_dow.set_xlabel("Day of Week")
    ax_dow.set_ylabel("Interactions Added")
    ax_dow.set_title("Interactions by Day of Week")
    plt.tight_layout()

    mo.vstack([
        mo.md("### Day of Week Patterns"),
        fig_dow,
    ])
    return


@app.cell
def _(df_dates, mo, pl, plt):
    """Hour of Day Patterns"""
    hour_added = (
        df_dates.filter(pl.col("date_added_parsed").is_not_null())
        .with_columns(
            pl.col("date_added_parsed").dt.hour().alias("hour")
        )
        .group_by("hour")
        .len()
        .sort("hour")
    )

    fig_hour, ax_hour = plt.subplots(figsize=(12, 6))
    ax_hour.bar(
        hour_added["hour"].to_list(),
        hour_added["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
        color="teal",
    )
    ax_hour.set_xlabel("Hour of Day (UTC offset adjusted)")
    ax_hour.set_ylabel("Interactions")
    ax_hour.set_title("Interactions by Hour of Day")
    ax_hour.set_xticks(range(24))
    plt.tight_layout()

    mo.vstack([
        mo.md("### Hour of Day Patterns"),
        fig_hour,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    Temporal patterns reveal user behavior rhythms - useful for time-aware recommendation models and understanding engagement windows.
    """)
    return


@app.cell
def _(df_dates, mo, np, pl, plt):
    """Reading Duration Analysis"""
    # For interactions with both started_at and read_at
    reading_duration = (
        df_dates.filter(
            pl.col("started_at_parsed").is_not_null()
            & pl.col("read_at_parsed").is_not_null()
        )
        .with_columns(
            (pl.col("read_at_parsed") - pl.col("started_at_parsed"))
            .dt.total_days()
            .alias("reading_days")
        )
        .filter(
            (pl.col("reading_days") >= 0) & (pl.col("reading_days") <= 365)
        )
    )

    fig_dur, axes_dur = plt.subplots(1, 2, figsize=(14, 5))

    days = reading_duration["reading_days"].to_pandas()
    axes_dur[0].hist(days.clip(upper=60), bins=60, edgecolor="black", alpha=0.7)
    axes_dur[0].set_xlabel("Reading Duration (days, capped at 60)")
    axes_dur[0].set_ylabel("Frequency")
    axes_dur[0].set_title("Reading Duration Distribution")

    axes_dur[1].hist(
        np.log1p(days), bins=50, edgecolor="black", alpha=0.7, color="green"
    )
    axes_dur[1].set_xlabel("Log(Reading Duration in days)")
    axes_dur[1].set_ylabel("Frequency")
    axes_dur[1].set_title("Reading Duration Distribution (Log)")

    plt.tight_layout()

    mo.vstack([
        mo.md(f"""
    ### Reading Duration Analysis
    - **Interactions with both start & finish dates:** {reading_duration.shape[0]:,}
    - **Mean reading duration:** {reading_duration["reading_days"].mean():.1f} days
    - **Median reading duration:** {reading_duration["reading_days"].median():.0f} days
    - **Max reading duration (capped at 365):** {reading_duration["reading_days"].max():.0f} days
    """),
        fig_dur,
    ])
    return (reading_duration,)


@app.cell
def _(mo, pl, plt, reading_duration):
    """Reading Duration vs Rating"""
    dur_by_rating = (
        reading_duration.filter(pl.col("rating") > 0)
        .group_by("rating")
        .agg([
            pl.col("reading_days").mean().alias("mean_days"),
            pl.col("reading_days").median().alias("median_days"),
            pl.len().alias("count"),
        ])
        .sort("rating")
    )

    fig_dur_rat, ax_dur_rat = plt.subplots(figsize=(10, 6))
    x = dur_by_rating["rating"].to_list()
    ax_dur_rat.bar(
        x,
        dur_by_rating["median_days"].to_list(),
        edgecolor="black",
        alpha=0.7,
        label="Median",
    )
    ax_dur_rat.set_xlabel("Rating")
    ax_dur_rat.set_ylabel("Median Reading Duration (days)")
    ax_dur_rat.set_title("Median Reading Duration by Rating")
    ax_dur_rat.set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()

    mo.vstack([
        mo.md("### Reading Duration vs Rating"),
        mo.md("Reading duration patterns may reveal engagement signals. Books rated lower might be abandoned faster or dragged out longer."),
        fig_dur_rat,
        dur_by_rating,
    ])
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
    """),
    ])

    fig_user, axes_user = plt.subplots(1, 2, figsize=(14, 5))

    _user_counts = user_activity["interaction_count"].to_pandas()
    axes_user[0].hist(_user_counts.clip(upper=500), bins=100, edgecolor="black", alpha=0.7)
    axes_user[0].set_xlabel("Interactions per User (capped at 500)")
    axes_user[0].set_ylabel("Number of Users")
    axes_user[0].set_title("User Activity Distribution (Raw)")

    axes_user[1].hist(
        np.log1p(_user_counts), bins=100, edgecolor="black", alpha=0.7
    )
    axes_user[1].set_xlabel("Log(Interactions per User)")
    axes_user[1].set_ylabel("Number of Users")
    axes_user[1].set_title("User Activity Distribution (Log)")

    plt.tight_layout()
    fig_user
    return (user_activity,)


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
    """),
    ])

    fig_book, axes_book = plt.subplots(1, 2, figsize=(14, 5))

    _book_counts = book_popularity["interaction_count"].to_pandas()
    axes_book[0].hist(_book_counts.clip(upper=2000), bins=100, edgecolor="black", alpha=0.7)
    axes_book[0].set_xlabel("Interactions per Book (capped at 2000)")
    axes_book[0].set_ylabel("Number of Books")
    axes_book[0].set_title("Book Popularity Distribution (Raw)")

    axes_book[1].hist(
        np.log1p(_book_counts), bins=100, edgecolor="black", alpha=0.7
    )
    axes_book[1].set_xlabel("Log(Interactions per Book)")
    axes_book[1].set_ylabel("Number of Books")
    axes_book[1].set_title("Book Popularity Distribution (Log)")

    plt.tight_layout()
    fig_book
    return (book_popularity,)


@app.cell
def _(book_popularity, df, mo, user_activity):
    """Sparsity Analysis"""
    n_users = user_activity.shape[0]
    n_books = book_popularity.shape[0]
    n_interactions = df.shape[0]
    total_possible = n_users * n_books
    sparsity = 1 - (n_interactions / total_possible)

    mo.md(f"""
    ### Sparsity Analysis
    - **Users:** {n_users:,}
    - **Books:** {n_books:,}
    - **Interactions:** {n_interactions:,}
    - **Possible interactions:** {total_possible:,.0f}
    - **Sparsity:** {sparsity * 100:.4f}%
    - **Density:** {(1 - sparsity) * 100:.6f}%
    """)
    return


@app.cell
def _(df_dates, mo, pl, plt):
    """Time Between Add and Update"""
    add_update_gap = (
        df_dates.filter(
            pl.col("date_added_parsed").is_not_null()
            & pl.col("date_updated_parsed").is_not_null()
        )
        .with_columns(
            (pl.col("date_updated_parsed") - pl.col("date_added_parsed"))
            .dt.total_days()
            .alias("days_to_update")
        )
        .filter(
            (pl.col("days_to_update") >= 0) & (pl.col("days_to_update") <= 3650)
        )
    )

    mo.vstack([
        mo.md(f"""
    ### Time Between Adding and Updating
    - **Mean:** {add_update_gap["days_to_update"].mean():.1f} days
    - **Median:** {add_update_gap["days_to_update"].median():.0f} days
    - **Same day updates (0 days):** {add_update_gap.filter(pl.col("days_to_update") == 0).shape[0]:,} ({add_update_gap.filter(pl.col("days_to_update") == 0).shape[0] / add_update_gap.shape[0] * 100:.1f}%)
    """),
    ])

    fig_gap, ax_gap = plt.subplots(figsize=(12, 6))
    gap_days = add_update_gap["days_to_update"].to_pandas()
    ax_gap.hist(gap_days.clip(upper=365), bins=100, edgecolor="black", alpha=0.7)
    ax_gap.set_xlabel("Days Between Add and Update (capped at 365)")
    ax_gap.set_ylabel("Frequency")
    ax_gap.set_title("Gap Between date_added and date_updated")
    plt.tight_layout()

    mo.vstack([
        fig_gap,
        mo.md("Many interactions are updated on the same day they are added. The gap distribution could indicate engagement patterns.")
    ])
    return


@app.cell
def _(df_dates, mo, pl, plt):
    """Seasonal Reading Patterns"""
    monthly_reads = (
        df_dates.filter(pl.col("read_at_parsed").is_not_null())
        .with_columns(
            pl.col("read_at_parsed").dt.month().alias("month")
        )
        .group_by("month")
        .len()
        .sort("month")
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig_season, ax_season = plt.subplots(figsize=(12, 6))
    ax_season.bar(
        [month_names[m - 1] for m in monthly_reads["month"].to_list()],
        monthly_reads["len"].to_list(),
        edgecolor="black",
        alpha=0.7,
        color="mediumpurple",
    )
    ax_season.set_xlabel("Month")
    ax_season.set_ylabel("Books Read")
    ax_season.set_title("Seasonal Reading Patterns")
    plt.tight_layout()

    mo.vstack([
        mo.md("### Seasonal Reading Patterns (from read_at)"),
        fig_season,
    ])
    return


@app.cell
def _(df, mo, pl):
    """Review ID Analysis"""
    has_review_id = df.filter(pl.col("review_id") != "").shape[0]
    unique_review_ids = df.filter(pl.col("review_id") != "")["review_id"].n_unique()

    mo.md(f"""
    ### Review ID Analysis
    - **Interactions with review_id:** {has_review_id:,} ({has_review_id / df.shape[0] * 100:.1f}%)
    - **Unique review IDs:** {unique_review_ids:,}
    - **Duplicate review IDs:** {has_review_id - unique_review_ids:,}

    Review IDs could be used to link this dataset with the reviews dataset for richer text features.
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
    """)
    return


@app.cell
def _(df, mo):
    """Comparison: Integer vs String Dataset"""
    # Check if the user/book IDs look like hashed strings
    sample_users = df["user_id"].head(5).to_list()
    sample_books = df["book_id"].head(5).to_list()

    n_unique_users = df["user_id"].n_unique()
    n_unique_books = df["book_id"].n_unique()

    mo.vstack([
        mo.md(f"""
    ### ID Format Analysis
    - **User ID format (samples):** {sample_users[:3]}
    - **Book ID format (samples):** {sample_books[:3]}
    - **Unique users:** {n_unique_users:,}
    - **Unique books:** {n_unique_books:,}

    User IDs are hashed hex strings (anonymized). Book IDs are string-encoded integers.
    This dataset uses original Goodreads identifiers vs the integer-mapped version.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature Engineering Recommendations

    1. **Parse dates:** `date_added`, `date_updated`, `read_at`, `started_at` should be parsed into datetime for temporal features.
    2. **Reading duration:** `read_at - started_at` provides reading speed/engagement signal. Filter for positive, reasonable values.
    3. **Temporal features:** Extract year, month, day-of-week, hour from dates for time-aware models.
    4. **Review text:** Despite being "incomplete", can be used for:
       - Sentiment analysis features
       - Review length as engagement proxy
       - NLP embeddings for content-based filtering
    5. **Recency features:** Time since last interaction, time between add and update.
    6. **Seasonal patterns:** Month of reading could capture seasonal preferences.
    7. **ID mapping:** Link `review_id` to full reviews dataset for richer text features.
    8. **Empty string handling:** Convert empty strings in `read_at`, `started_at`, `review_text_incomplete` to null for proper handling.
    9. **User engagement tiers:** Segment users by activity level (casual, moderate, power users) using interaction count quantiles.
    10. **Same-day updates:** High proportion of same-day add+update suggests batch imports or automated shelving.
    11. First date being in 1072?
    """)
    return


if __name__ == "__main__":
    app.run()
