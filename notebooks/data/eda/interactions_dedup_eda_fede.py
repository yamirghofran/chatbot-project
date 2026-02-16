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
    return mo, pl, plt


@app.cell
def _(pl):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path = os.path.join(project_root, "data", "raw_goodreads_interactions_dedup.parquet")

    # Random sampling for computational reasons 
    print("Loading random 50% sample")
    df = pl.scan_parquet(data_path).filter(
        pl.col("user_id").hash(seed=42) % 100 < 50
    ).collect()
    print(f"Loaded {df.height:,} rows (~50% random sample)")
    df.head()
    return (df,)

@app.cell
def _(mo):
    mo.md(r"""
    **Columns:**
    - `user_id`: Unique user identifier
    - `book_id`: Unique book identifier
    - `review_id`: Unique identifier for each review
    - `is_read`: Boolean indicating if user marked book as read
    - `rating`: User's rating (0-5)
    - `review_text_incomplete`: Text content of the review
    - `date_added`: Date when book/review was added to user's shelf
    - `date_updated`: Date when interaction was last modified
    - `read_at`: Date when user finished reading the book
    - `started_at`: Date when user started reading the book
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
def _(mo):
    mo.md(r"""
    **Key Observations:**
    - `Median rating = 0`: Over 50% of interactions are unrated (rating = 0)
    - `Mean rating = 1.79`: Heavily skewed by unrated (0) values, similar to Interactions dataset (1.80)
    - `Mean is_read = 0.489`: About 49% marked as read, consistent with Interactions dataset (51%)
    - `Temporal fields available`: date_added, date_updated, read_at, started_at
    - `Review text available`: review_text_incomplete present, likely sparse based on Interactions dataset (7.09%)
    - Temporal fields allow for the analysis of reading process and engagement patterns.
    """)
    return


@app.cell
def _(df):
    # Shape and nulls
    print(f"Shape: {df.shape}")
    print(f"\nNull counts:")
    print(df.null_count().to_dicts()[0])
    return


@app.cell
def _(mo):
    mo.md(r"""
    No nulls detected.
    Missing data uses empty strings instead of nulls.
    """)
    return


@app.cell
def _(df, pl):
    # Empty strings vs null for review_text
    _total = df.height
    _null_review = df.filter(pl.col("review_text_incomplete").is_null()).height
    _empty_review = df.filter(pl.col("review_text_incomplete") == "").height
    _has_review = df.filter(pl.col("review_text_incomplete").str.len_chars() > 0).height

    print("review_text_incomplete breakdown (null vs empty string):")
    print(f"Null values:    {_null_review:,} ({_null_review/_total*100:.2f}%)")
    print(f"Empty strings:  {_empty_review:,} ({_empty_review/_total*100:.2f}%)")
    print(f"Has review:     {_has_review:,} ({_has_review/_total*100:.2f}%)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Around 93% empty strings, only 7.12% have actual review text.
    Consistent with Interactions dataset (7.09% is_reviewed = True).
    """)
    return


@app.cell
def _(df):
    # Check data types
    print("Column types:")
    print(df.schema)

    # Check is_read unique values
    print("\nis_read unique values:", df["is_read"].unique().to_list())
    print("is_read value counts:", df.group_by("is_read").len().sort("is_read"))
    return


@app.cell
def _(df, pl):
    # Rating distribution: 0 vs 1-5
    _total = df.height
    _zero_ratings = df.filter(pl.col("rating") == 0).height
    _nonzero_ratings = df.filter(pl.col("rating") > 0).height
    _total_read = df.filter(pl.col("is_read") == True).height

    print(f"Total interactions: {_total:,}")
    print(f"\nRating breakdown:")
    print(f"Rating = 0 (unrated): {_zero_ratings:,} ({_zero_ratings/_total*100:.1f}%)")
    print(f"Rating > 0 (rated): {_nonzero_ratings:,} ({_nonzero_ratings/_total*100:.1f}%)")
    print(f"\nRead interactions: {_total_read:,} ({_total_read/_total*100:.1f}%)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    54.4% unrated (rating = 0), 45.6% rated - matches Interactions dataset.
    Interactions Dedup dataset preserves same distributional characteristics as Interactions dataset.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **TEMPORAL DATA**
    """)
    return


@app.cell
def _(df, pl):
    # Empty strings and null for temporal fields
    _total = df.height

    print("Temporal fields: null and empty string breakdown")
    for col in ["date_added", "date_updated", "read_at", "started_at"]:
        _null = df.filter(pl.col(col).is_null()).height
        _empty = df.filter(pl.col(col) == "").height
        _valid = df.filter(pl.col(col).str.len_chars() > 0).height

        print(f"\n{col}:")
        print(f"  Null:   {_null:,} ({_null/_total*100:.2f}%)")
        print(f"  Empty:  {_empty:,} ({_empty/_total*100:.2f}%)")
        print(f"  Valid:  {_valid:,} ({_valid/_total*100:.2f}%)")
    return


@app.cell
def _(df, pl):
    # Check temporal field completeness
    _total = df.height
    _has_date_added = df.filter(pl.col("date_added").str.len_chars() > 0).height
    _has_date_updated = df.filter(pl.col("date_updated").str.len_chars() > 0).height
    _has_read_at = df.filter(pl.col("read_at").str.len_chars() > 0).height
    _has_started_at = df.filter(pl.col("started_at").str.len_chars() > 0).height


    print("Temporal field completeness:")
    print(f"date_added:    {_has_date_added:,} ({_has_date_added/_total*100:.1f}%)")
    print(f"date_updated:  {_has_date_updated:,} ({_has_date_updated/_total*100:.1f}%)")
    print(f"read_at:       {_has_read_at:,} ({_has_read_at/_total*100:.1f}%)")
    print(f"started_at:    {_has_started_at:,} ({_has_started_at/_total*100:.1f}%)")

    # Complete lifecycle interactions
    _complete_lifecycle = df.filter(
    (pl.col("started_at").str.len_chars() > 0) & (pl.col("read_at").str.len_chars() > 0)
    ).height
    print(f"\nComplete lifecycle (started_at + read_at): {_complete_lifecycle:,} ({_complete_lifecycle/_total*100:.1f}%)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    date_added and date_updated always present (100%).
    read_at (20.5%) and started_at (15.7%) are sparse: most users don't track reading progress.
    Only 13.8% have complete lifecycle (both started_at and read_at).
    This small subset represents users with a lot of engagement that track their reading.

    Given sparsity of data, consider using presence of read_at/started_at as binary engagement signals rather than relying on actual dates for all interactions.
    Derived duration features (like reading time (read_at - started_at)) can be explored, but they may not be usable for the majority of interactions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **REVIEW CONTENT ANALYSIS**
    """)
    return


@app.cell
def _(df, pl):
    # Review text presence
    _total = df.height
    _has_review = df.filter(pl.col("review_text_incomplete").str.len_chars() > 0).height
    _no_review = df.filter(pl.col("review_text_incomplete") == "").height

    print(f"Review text presence:")
    print(f"Has review text: {_has_review:,} ({_has_review/_total*100:.2f}%)")
    print(f"No review text:  {_no_review:,} ({_no_review/_total*100:.2f}%)")
    print(f"\nNote: Interactions dataset showed 7.09% with is_reviewed=True")
    return


@app.cell
def _(df, pl):
    # Review text length distribution
    _df_with_review = df.filter(pl.col("review_text_incomplete").str.len_chars() > 0)
    _df_with_review = _df_with_review.with_columns(
        pl.col("review_text_incomplete").str.len_chars().alias("review_length")
    )

    _stats = _df_with_review.select([
        pl.col("review_length").mean().alias("mean"),
        pl.col("review_length").median().alias("median"),
        pl.col("review_length").min().alias("min"),
        pl.col("review_length").max().alias("max"),
        pl.col("review_length").quantile(0.25).alias("q25"),
        pl.col("review_length").quantile(0.75).alias("q75"),
    ])

    print("Review text length statistics (characters):")
    _stats
    return


@app.cell
def _(df, pl, plt):
    # Review text length distribution plot
    _df_with_review = df.filter(pl.col("review_text_incomplete").str.len_chars() > 0)
    _df_with_review = _df_with_review.with_columns(
        pl.col("review_text_incomplete").str.len_chars().alias("review_length")
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    _axes[0].hist(_df_with_review["review_length"].to_numpy(), bins=50, edgecolor='black')
    _axes[0].set_xlabel("Review Length (characters)")
    _axes[0].set_ylabel("Frequency")
    _axes[0].set_title("Review Text Length Distribution")
    _axes[0].set_yscale('log')

    # Boxplot
    _axes[1].boxplot(_df_with_review["review_length"].to_numpy())
    _axes[1].set_ylabel("Review Length (characters)")
    _axes[1].set_title("Review Length (Boxplot)")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Sparse review text presence (7.12%), but for those with reviews:
    - Median length: 299 chars
    - Mean lenght: 233 chars
    - Right-skewed: most reviews are short (100-500 chars)
    - Some outliers reach more than 12,000 characters

    For the recommendation system consider using `has_review` as binary engagement signal.
    For the chatbot review text might be valuable and can be used for:
    - Content-based recommendations (sentiment, topics)
    - Generating responses about books ("users said...")
    - Understanding user preferences from their writing
    Consider keeping review_text_incomplete for NLP pipeline.
    """)
    return


@app.cell
def _(df, pl):
    # Relationship between review presence and rating
    _review_by_rating = df.with_columns(
        (pl.col("review_text_incomplete").str.len_chars() > 0).alias("has_review")
    ).group_by("rating").agg([
        pl.len().alias("total"),
        pl.col("has_review").sum().alias("with_review"),
    ]).sort("rating")

    _review_by_rating = _review_by_rating.with_columns(
        (pl.col("with_review") / pl.col("total") * 100).round(2).alias("review_pct")
    )

    print("Review presence by rating:")
    _review_by_rating
    return


@app.cell
def _(mo):
    mo.md(r"""
    Strong relationship between reviews and ratings:
    - Unrated (0): only 0.46% have reviews - no rating = no review
    - Rated (1-5): 13-23% have reviews
    - Rating 1 has highest review rate (23%) 
    - Ratings 3-5 similar review rates (14-16%)

    Reviews almost always come with explicit ratings.
    Reviews are more common at rating 1 - users may be more motivated to explain negative experiences.
    Note: check reviews sentiment matches ratings score.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Note
    For rating distribution and user engagement analysis, see `interactions_eda.py`.
    The distributions in this dataset match those findings.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **TEMPORAL ENGAGEMENT PATTERNS**
    """)
    return


@app.cell
def _(df):
    # Parse date columns for temporal analysis
    _sample = df.select([
        "date_added", "date_updated", "read_at", "started_at"
    ]).head(10)

    print("Sample of temporal columns:")
    _sample
    return


@app.cell
def _(df, pl):
    # Analyze temporal segments
    _total = df.height

    # both started_at and read_at
    _complete = df.filter((pl.col("started_at").str.len_chars() > 0) & (pl.col("read_at").str.len_chars() > 0)).height

    # Read but no start date
    _read_no_start = df.filter((pl.col("read_at").str.len_chars() > 0) & (pl.col("started_at") == "")).height

    # Not read
    _not_read = df.filter((pl.col("read_at") == "") & (pl.col("is_read") == False)).height

    # Marked as read but no read_at date
    _read_no_date = df.filter((pl.col("is_read") == True) & (pl.col("read_at") == "")).height

    print("Temporal engagement segments:")
    print(f"Complete lifecycle (started + read_at): {_complete:,} ({_complete/_total*100:.1f}%)")
    print(f"Read but no start date:                 {_read_no_start:,} ({_read_no_start/_total*100:.1f}%)")
    print(f"Not read:                     {_not_read:,} ({_not_read/_total*100:.1f}%)")
    print(f"Marked as read but no read_at:          {_read_no_date:,} ({_read_no_date/_total*100:.1f}%)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Date format is human-readable string ("Tue Jan 26 17:38:09 -0800 2016") - will need parsing for temporal features.
    -0800 corresponds to (UTC offset)
    
    Reveal user behavior patterns:
    - 51.1% "Not read": wishlist/want to read - weak interest signal
    - 28.3% "Marked as read but no read_at": users don't log completion dates - is_read is more reliable than read_at
    - 13.8% "Complete lifecycle": highly engaged users who track reading progress - high quality subset
    - 6.7% "Read but no start date": users who log completion but not start

    Use is_read as primary read signal (more complete than read_at)
    Treat "not read" as a weak implicit interest signal
    Complete lifecycle subset (13.8%) can be used for reading duration analysis if needed
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **FEATURE ENGINEERING RECOMMENDATIONS**
    1. Use presence of read_at/started_at as binary engagement signals.
    2. Use has_review as binary engagement signal for the recommendation system.
    3. Keep review_text_incomplete for NLP: sentiment analysis, topic extraction, chatbot responses.
    4. Consider review length as engagement intensity signal (longer = more invested).
    """)
    return


if __name__ == "__main__":
    app.run()
