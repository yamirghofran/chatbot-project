import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import os
    import re

    return mo, os, pl, re


@app.cell
def _(mo):
    mo.md("""
    # Review Reduction and Spoiler Flagging

    - Drop date_updated from dedup dataset
    - Check empty/whitespace-only reviews, duplicate reviews, non-informative reviews (single char, punctuation, numbers, repeated chars, short low-variety)
    - Drop non-informative reviews from dedup dataset
    - Add has_spoiler flag (1 if in spoiler dataset, 0 otherwise)

    Output Files
    - 5_goodreads_reviews_final_clean.parquet: Cleaned reviews with spoiler flag
    - 5_dropped_review_ids.json: List of dropped review IDs
    """)
    return


@app.cell
def _(os, pl):
    from bookdb.utils.paths import find_project_root
    data_dir = os.path.join(find_project_root(), "data")

    # Load datasets
    dedup_path = os.path.join(data_dir, "4_goodreads_reviews_reduced.parquet")
    spoiler_path = os.path.join(data_dir, "2_goodreads_reviews_spoiler_reduced.parquet")

    df_dedup = pl.read_parquet(dedup_path)
    df_spoiler = pl.read_parquet(spoiler_path)
    return data_dir, df_dedup, df_spoiler


@app.cell
def _(df_dedup, mo):
    mo.md(f"""
    ## 1. Drop date_updated from dedup dataset
    Columns before: {df_dedup.columns}
    """)
    return


@app.cell
def _(df_dedup):
    # Drop date_updated if it exists
    if "date_updated" in df_dedup.columns:
        df_dedup_clean = df_dedup.drop("date_updated")
    else:
        df_dedup_clean = df_dedup

    df_dedup_clean.columns
    return (df_dedup_clean,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Check for empty reviews
    (Empty = length 0 or whitespace-only)
    """)
    return


@app.cell
def _(df_dedup_clean, df_spoiler, mo, pl):
    # Check empty reviews in dedup
    _empty_dedup = df_dedup_clean.filter(
        (pl.col("review_text").str.len_chars() == 0) |
        (pl.col("review_text").str.strip_chars().str.len_chars() == 0)
    )

    # Check empty reviews in spoiler
    _empty_spoiler = df_spoiler.filter(
        (pl.col("review_text").str.len_chars() == 0) |
        (pl.col("review_text").str.strip_chars().str.len_chars() == 0)
    )

    mo.md(f"""
    ### Empty Reviews Summary

    **Dedup dataset:**
    Total reviews: {df_dedup_clean.shape[0]:,}
    Empty reviews: {_empty_dedup.shape[0]:,} ({_empty_dedup.shape[0] / df_dedup_clean.shape[0] * 100:.2f}%)

    **Spoiler dataset:**
    Total reviews: {df_spoiler.shape[0]:,}
    Empty reviews: {_empty_spoiler.shape[0]:,} ({_empty_spoiler.shape[0] / df_spoiler.shape[0] * 100:.2f}%)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Check for non-informative reviews
    This includes:
    1. Single character
    2. Punctuation only
    3. Repeated symbols/characters
    4. Numbers only
    5. Short reviews with low variety (INSPECT FIRST)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Inspect: Short reviews (<=5 chars) with low variety (<=2 unique chars)
    These might include words like "sad", "good" - (reviews that are short but could be informative).
    We will inspect these manually before deciding whether to drop them.
    """)
    return


@app.cell
def _(df_dedup_clean, is_short_low_variety, mo, pl):
    # Find short reviews with low variety in dedup
    _short_dedup = df_dedup_clean.filter(
        pl.col("review_text").map_elements(is_short_low_variety, return_dtype=pl.Boolean)
    ).select(["review_id", "review_text", "rating"])

    mo.md(f"""
    DEDUP: Short + low variety reviews: {_short_dedup.shape[0]:,}
    """)
    return


@app.cell
def _(df_dedup_clean, is_short_low_variety, pl):
    # Show ALL short low variety reviews from dedup
    _short_reviews = df_dedup_clean.filter(
        pl.col("review_text").map_elements(is_short_low_variety, return_dtype=pl.Boolean)
    ).select(["review_id", "review_text", "rating"])

    # Show ALL unique review_text values
    _unique_texts = _short_reviews.select("review_text").unique().sort("review_text")["review_text"].to_list()
    print(f"ALL {len(_unique_texts)} unique short+low variety texts in DEDUP")
    for _t in _unique_texts:
        print(repr(_t))
    return


@app.cell
def _(df_spoiler, is_short_low_variety, mo, pl):
    # Find short reviews with low variety in spoiler
    _short_spoiler = df_spoiler.filter(
        pl.col("review_text").map_elements(is_short_low_variety, return_dtype=pl.Boolean)
    ).select(["review_id", "review_text", "rating"])

    mo.md(f"""
    SPOILER: Short + low variety reviews: {_short_spoiler.shape[0]:,}
    """)
    return


@app.cell
def _(df_spoiler, is_short_low_variety, pl):
    # Show ALL short low variety reviews from spoiler
    _short_reviews = df_spoiler.filter(
        pl.col("review_text").map_elements(is_short_low_variety, return_dtype=pl.Boolean)
    ).select(["review_id", "review_text", "rating"])

    # Show ALL unique review_text values
    _unique_texts = _short_reviews.select("review_text").unique().sort("review_text")["review_text"].to_list()
    print(f"ALL {len(_unique_texts)} unique short+low variety texts in SPOILER")
    for _t in _unique_texts:
        print(repr(_t))
    return


@app.cell
def _(mo):
    mo.md("""
    ### Non-informative check (without short+low variety)
    """)
    return


@app.cell
def _():
    from bookdb.processing.text import is_non_informative, is_short_low_variety
    return is_non_informative, is_short_low_variety


@app.cell
def _(df_dedup_clean, df_spoiler, is_non_informative, pl):
    # Apply to dedup
    _non_info_dedup = df_dedup_clean.filter(
        pl.col("review_text").map_elements(is_non_informative, return_dtype=pl.Boolean)
    )

    # Apply to spoiler
    _non_info_spoiler = df_spoiler.filter(
        pl.col("review_text").map_elements(is_non_informative, return_dtype=pl.Boolean)
    )

    print(f"Non-informative reviews in dedup: {_non_info_dedup.shape[0]:,}")
    print(f"Non-informative reviews in spoiler: {_non_info_spoiler.shape[0]:,}")
    return



@app.cell
def _(df_dedup_clean, is_non_informative, pl):
    # Display samples
    df_dedup_clean.filter(
        pl.col("review_text").map_elements(is_non_informative, return_dtype=pl.Boolean)
    ).select(["review_id", "review_text", "rating"]).head(50)
    return



@app.cell
def _(df_spoiler, is_non_informative, pl):
    # Display samples
    df_spoiler.filter(
        pl.col("review_text").map_elements(is_non_informative, return_dtype=pl.Boolean)
    ).select(["review_id", "review_text", "rating"]).head(50)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Check for duplicate reviews
    """)
    return


@app.cell
def _(df_dedup_clean, mo, pl):
    # Check duplicates in dedup by review_id
    _dup_ids_dedup = df_dedup_clean.group_by("review_id").agg(pl.len().alias("count")).filter(pl.col("count") > 1)
    _n_dup_dedup = _dup_ids_dedup.shape[0]

    if _n_dup_dedup > 0:
        _dup_id_list = _dup_ids_dedup["review_id"].to_list()
        _dup_reviews_dedup = df_dedup_clean.filter(
            pl.col("review_id").is_in(_dup_id_list)
        ).sort("review_id")

        # Check if all columns match for duplicates
        _cols_to_check = ['review_id', 'rating', 'review_text', 'n_votes', 'n_comments', 'book_id', 'user_id']
        _cols_to_check = [c for c in _cols_to_check if c in df_dedup_clean.columns]

        _dup_check = _dup_reviews_dedup.select(_cols_to_check).unique()
        _all_match = _dup_check.shape[0] == _dup_ids_dedup.shape[0]
    else:
        _all_match = True

    mo.md(f"""
    ### Duplicate Reviews in DEDUP dataset

    Duplicate review_ids: {_n_dup_dedup:,}
    All duplicates have identical values: {_all_match}
    """)
    return


@app.cell
def _(df_dedup_clean, pl):
    # Show duplicate reviews in dedup
    _dup_ids_dedup = df_dedup_clean.group_by("review_id").agg(pl.len().alias("count")).filter(pl.col("count") > 1)

    if _dup_ids_dedup.shape[0] > 0:
        _dup_id_list = _dup_ids_dedup["review_id"].to_list()
        df_dedup_clean.filter(
            pl.col("review_id").is_in(_dup_id_list)
        ).sort("review_id").head(20)
    else:
        print("No duplicates found")
    return


@app.cell
def _(df_spoiler, mo, pl):
    # Check duplicates in spoiler by review_id
    _dup_ids_spoiler = df_spoiler.group_by("review_id").agg(pl.len().alias("count")).filter(pl.col("count") > 1)
    _n_dup_spoiler = _dup_ids_spoiler.shape[0]

    if _n_dup_spoiler > 0:
        _dup_id_list = _dup_ids_spoiler["review_id"].to_list()
        _dup_reviews_spoiler = df_spoiler.filter(
            pl.col("review_id").is_in(_dup_id_list)
        ).sort("review_id")

        # Check if all columns match for duplicates
        _cols_to_check = ['review_id', 'rating', 'review_text', 'n_votes', 'n_comments', 'book_id', 'user_id']
        _cols_to_check = [c for c in _cols_to_check if c in df_spoiler.columns]

        _dup_check = _dup_reviews_spoiler.select(_cols_to_check).unique()
        _all_match = _dup_check.shape[0] == _dup_ids_spoiler.shape[0]
    else:
        _all_match = True

    mo.md(f"""
    ### Duplicate Reviews in SPOILER dataset

    Duplicate review_ids: {_n_dup_spoiler:,}
    All duplicates have identical values: {_all_match}
    """)
    return


@app.cell
def _(df_spoiler, pl):
    # Show duplicate reviews in spoiler
    _dup_ids_spoiler = df_spoiler.group_by("review_id").agg(pl.len().alias("count")).filter(pl.col("count") > 1)

    if _dup_ids_spoiler.shape[0] > 0:
        _dup_id_list = _dup_ids_spoiler["review_id"].to_list()
        df_spoiler.filter(
            pl.col("review_id").is_in(_dup_id_list)
        ).sort("review_id").head(20)
    else:
        print("No duplicates found")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Drop non-informative reviews from DEDUP

    Dropping reviews that are:
    1. Non-informative (single char, punctuation only, numbers only, repeated char)
    2. Short + low variety (<=5 chars with <=2 unique chars)
    """)
    return


@app.cell
def _(df_dedup_clean, is_non_informative, is_short_low_variety, pl):
    # Combined filter: non-informative OR short+low variety
    def _should_drop(text: str) -> bool:
        return is_non_informative(text) or is_short_low_variety(text)

    # Get reviews to drop
    df_to_drop = df_dedup_clean.filter(
        pl.col("review_text").map_elements(_should_drop, return_dtype=pl.Boolean)
    )

    # Get clean reviews (keep)
    df_dedup_final = df_dedup_clean.filter(
        ~pl.col("review_text").map_elements(_should_drop, return_dtype=pl.Boolean)
    )

    # Stats
    n_before = df_dedup_clean.shape[0]
    n_dropped = df_to_drop.shape[0]
    n_after = df_dedup_final.shape[0]

    print(f"DEDUP Reviews Cleanup")
    print(f"Before: {n_before:,}")
    print(f"After: {n_after:,}")
    return df_dedup_final, df_to_drop, n_after, n_before, n_dropped


@app.cell
def _(data_dir, df_to_drop, os):
    import json

    # Save dropped review IDs to JSON
    dropped_ids = df_to_drop["review_id"].to_list()
    dropped_ids_path = os.path.join(data_dir, "5_dropped_review_ids.json")

    with open(dropped_ids_path, "w") as f:
        json.dump(dropped_ids, f)

    print(f"Saved {len(dropped_ids):,} dropped review IDs to {dropped_ids_path}")
    return (dropped_ids,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Add spoiler flag
    Flag reviews that exist in the spoiler dataset (has_spoiler = 1 or 0)
    """)
    return


@app.cell
def _(df_dedup_final, df_spoiler, n_after, pl):
    # Get review_ids from spoiler dataset
    spoiler_ids = set(df_spoiler["review_id"].to_list())

    # Add has_spoiler flag: 1 if in spoiler dataset, 0 otherwise
    df_with_spoiler = df_dedup_final.with_columns(
        pl.col("review_id").map_elements(
            lambda x: 1 if x in spoiler_ids else 0,
            return_dtype=pl.Int8
        ).alias("has_spoiler")
    )

    # Stats
    n_with_spoiler = df_with_spoiler.filter(pl.col("has_spoiler") == 1).shape[0]
    n_without_spoiler = df_with_spoiler.filter(pl.col("has_spoiler") == 0).shape[0]

    print(f"Spoiler Flag")
    print(f"Reviews with spoiler: {n_with_spoiler:,} ({n_with_spoiler / n_after * 100:.2f}%)")
    print(f"Reviews without spoiler: {n_without_spoiler:,} ({n_without_spoiler / n_after * 100:.2f}%)")
    return (df_with_spoiler,)


@app.cell
def _(
    data_dir,
    df_with_spoiler,
    dropped_ids,
    mo,
    n_after,
    n_before,
    n_dropped,
    os,
    pl,
):
    # Save final dataset
    output_path = os.path.join(data_dir, "5_goodreads_reviews_final_clean.parquet")
    df_with_spoiler.write_parquet(output_path)

    _n_with_spoiler = df_with_spoiler.filter(pl.col("has_spoiler") == 1).shape[0]

    mo.md(f"""
    ## Summary

    ### Reviews Dropped from DEDUP
    - Before: {n_before:,}
    - Dropped: {n_dropped:,} ({n_dropped / n_before * 100:.2f}%)
    - After: {n_after:,}

    ### Spoiler Flag
    - With spoiler: {_n_with_spoiler:,} ({_n_with_spoiler / n_after * 100:.2f}%)
    - Without spoiler: {n_after - _n_with_spoiler:,} ({(n_after - _n_with_spoiler) / n_after * 100:.2f}%)

    ### Output Files
    - Cleaned reviews: `5_goodreads_reviews_final_clean.parquet` (with `has_spoiler` column)
    - Dropped review IDs: `5_dropped_review_ids.json` ({len(dropped_ids):,} IDs)
    """)
    return


@app.cell
def _(df_with_spoiler, n_after, n_before, n_dropped, pl):
    # Validation checks
    assert n_before - n_dropped == n_after, "Row count mismatch"
    assert set(df_with_spoiler["has_spoiler"].unique().to_list()) <= {0, 1}, "Invalid has_spoiler values"

    # review_id checks (null and empty)
    assert df_with_spoiler["review_id"].null_count() == 0, "Null review_ids found"
    assert df_with_spoiler.filter(pl.col("review_id").str.len_chars() == 0).shape[0] == 0, "Empty review_ids found"

    # review_text checks (null and empty)
    assert df_with_spoiler["review_text"].null_count() == 0, "Null review_text found"
    assert df_with_spoiler.filter(pl.col("review_text").str.len_chars() == 0).shape[0] == 0, "Empty review_text found"

    print("All validation checks passed")
    print(f"Final columns: {df_with_spoiler.columns}")
    return


if __name__ == "__main__":
    app.run()
