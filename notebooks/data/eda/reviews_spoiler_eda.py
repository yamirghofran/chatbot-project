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
    import re 

    return mo, np, pl, plt, re


@app.cell
def _(mo):
    mo.md(r"""
    # Comparative EDA: Spoiler vs Non-Spoiler Reviews
    **Datasets:**
    - `raw_goodreads_reviews_dedup.parquet` - All reviews
    - `raw_goodreads_reviews_spoiler.parquet` - Spoiler reviews
    """)
    return


@app.cell
def _(mo, pl):
    import os
    project_root = __import__("pathlib").Path(__file__).resolve().parents[3]

    df_all = pl.read_parquet(os.path.join(project_root, "data", "raw_goodreads_reviews_dedup.parquet"))
    df_spoiler = pl.read_parquet(os.path.join(project_root, "data", "raw_goodreads_reviews_spoiler.parquet"))

    mo.vstack([
        mo.md(f"Reviews dedup: {df_all.height:,}"),
        mo.md(f"Reviews spoiler: {df_spoiler.height:,}")
    ])
    return df_all, df_spoiler


@app.cell
def _(df_spoiler):
    df_spoiler.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
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
def _(mo):
    mo.md(r"""
    # Part 1: ASSUMPTION VERIFICATION: Is spoiler a subset of dedup?
    """)
    return


@app.cell
def _(df_all, df_spoiler, mo):
    # Verify schemas match
    _schemas_match = df_all.schema == df_spoiler.schema
    output = [mo.md(f"**Schemas match:** {_schemas_match}")]
    if not _schemas_match:
        output.append(mo.md(f"  Dedup columns: {list(df_all.schema.keys())}"))
        output.append(mo.md(f"  Spoiler columns: {list(df_spoiler.schema.keys())}"))
    mo.vstack(output)
    return


@app.cell
def _(df_all, df_spoiler, pl):
    # Check if spoiler review_ids are a subset of dedup review_ids
    _only_in_spoiler = df_spoiler.filter(~pl.col("review_id").is_in(df_all["review_id"]))
    _n_only_in_spoiler = _only_in_spoiler.height
    _n_spoiler = df_spoiler.height
    _n_in_both = _n_spoiler - _n_only_in_spoiler

    print(f"Spoiler review_ids in dedup: {_n_in_both:,} ({_n_in_both/_n_spoiler*100:.1f}%)")
    print(f"Spoiler review_ids NOT in dedup: {_n_only_in_spoiler:,} ({_n_only_in_spoiler/_n_spoiler*100:.1f}%)")

    _is_subset = _n_only_in_spoiler == 0
    print(f"\nSpoiler is strict subset of dedup: {_is_subset}")

    if not _is_subset:
        print(f"  WARNING: {_n_only_in_spoiler:,} spoiler reviews are not in dedup dataset")
    return


@app.cell
def _(df_all, df_spoiler, pl):
    # Join - compare ALL columns
    joined_comparison = df_spoiler.join(
        df_all,
        on="review_id",
        how="inner",
        suffix="_dedup"
    )

    print(f"Matched rows for comparison: {joined_comparison.height:,}")

    # Compare each column (excluding review_id - joined on)
    _cols_to_compare = [c for c in df_spoiler.columns if c != "review_id"]
    _mismatches = {}

    for _col in _cols_to_compare:
        _col_dedup = f"{_col}_dedup"
        if _col_dedup in joined_comparison.columns:
            _diff_count = joined_comparison.filter(
                pl.col(_col).ne_missing(pl.col(_col_dedup))
            ).height
            if _diff_count > 0:
                _mismatches[_col] = _diff_count

    if _mismatches:
        print("\nColumn mismatches found:")
        for _col, _count in _mismatches.items():
            print(f"  {_col}: {_count:,} rows differ ({_count/joined_comparison.height*100:.2f}%)")
    else:
        print("\nAll columns match - spoiler rows are identical to their dedup counterparts")
    return (joined_comparison,)


@app.cell
def _(mo):
    mo.md(r"""
    Only review_text differs between datasets (around 6.5% of rows)
    All other columns (user_id, book_id, rating, dates, n_votes, n_comments) are identical
    """)
    return


@app.cell
def _(joined_comparison, pl):
    # Investigate review_text differences 
    _diff_rows = joined_comparison.filter(
        pl.col("review_text").ne_missing(pl.col("review_text_dedup"))
    )

    print(f"Rows with different review_text: {_diff_rows.height:,}")

    if _diff_rows.height > 0:
        _analysis = _diff_rows.select([
            pl.col("review_text").str.len_chars().mean().alias("avg_len_spoiler"),
            pl.col("review_text_dedup").str.len_chars().mean().alias("avg_len_dedup"),
            (pl.col("review_text").str.len_chars() > pl.col("review_text_dedup").str.len_chars()).sum().alias("spoiler_longer"),
            (pl.col("review_text_dedup").str.len_chars() > pl.col("review_text").str.len_chars()).sum().alias("dedup_longer"),
            (pl.col("review_text").str.len_chars() == pl.col("review_text_dedup").str.len_chars()).sum().alias("same_length"),
        ])

        print("\nLength comparison of differing texts:")
        print(f"  Avg spoiler text length: {_analysis['avg_len_spoiler'][0]:,.0f}")
        print(f"  Avg dedup text length: {_analysis['avg_len_dedup'][0]:,.0f}")
        print(f"  Avg length difference: {_analysis['avg_len_spoiler'][0] - _analysis['avg_len_dedup'][0]:+,.0f}")
        print(f"\nLength comparison:")
        print(f"  Spoiler text longer: {_analysis['spoiler_longer'][0]:,} ({_analysis['spoiler_longer'][0]/_diff_rows.height*100:.1f}%)")
        print(f"  Dedup text longer: {_analysis['dedup_longer'][0]:,} ({_analysis['dedup_longer'][0]/_diff_rows.height*100:.1f}%)")
        print(f"  Same length: {_analysis['same_length'][0]:,} ({_analysis['same_length'][0]/_diff_rows.height*100:.1f}%)")
    return


@app.cell
def _(joined_comparison, pl):
    # Sample - show first 200 chars 
    _diff_rows = joined_comparison.filter(
        pl.col("review_text").ne_missing(pl.col("review_text_dedup"))
    )

    print("Sample of differing review texts (first 200 chars):\n")
    for _row in _diff_rows.head(3).iter_rows(named=True):
        print(f"Review ID: {_row['review_id']}")
        print(f"  SPOILER:  {str(_row['review_text'])[:200]}...")
        print(f"  DEDUP:    {str(_row['review_text_dedup'])[:200]}...")
        print()
    return


@app.cell
def _(joined_comparison, pl):
    # Show the ACTUAL DIFFERENCE between texts
    _diff_rows = joined_comparison.filter(
        pl.col("review_text").ne_missing(pl.col("review_text_dedup"))
    )

    print("Analyzing the actual difference between texts:\n")

    _prefix_match = 0
    _not_prefix = 0

    for _row in _diff_rows.head(10).iter_rows(named=True):
        _spoiler_text = str(_row['review_text'])
        _dedup_text = str(_row['review_text_dedup'])

        # Check if dedup is a prefix of spoiler (i.e., spoiler = dedup + extra)
        if _spoiler_text.startswith(_dedup_text):
            _extra = _spoiler_text[len(_dedup_text):]
            _prefix_match += 1
            print(f"Review ID: {_row['review_id']}")
            print(f"  Dedup IS a prefix of spoiler")
            print(f"  EXTRA CONTENT ({len(_extra)} chars): '{_extra}'")
            print()
        else:
            _not_prefix += 1
            # Find where they diverge
            _diverge_at = 0
            for _i, (_c1, _c2) in enumerate(zip(_spoiler_text, _dedup_text)):
                if _c1 != _c2:
                    _diverge_at = _i
                    break
            print(f"Review ID: {_row['review_id']}")
            print(f"  Texts diverge at position {_diverge_at}")
            print(f"  Around divergence point:")
            print(f"    SPOILER: ...{_spoiler_text[max(0,_diverge_at-20):_diverge_at+50]}...")
            print(f"    DEDUP:   ...{_dedup_text[max(0,_diverge_at-20):_diverge_at+50]}...")
            print()

    print(f"\nSummary (first 10): {_prefix_match} prefix matches, {_not_prefix} divergent")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Spoilers seem to be either content within/flagged with '(view spoiler)[...]'
    """)
    return


@app.cell
def _(joined_comparison, pl, re):

    # Extract and print FULL spoiler content within (view spoiler)[...] tags
    _diff_rows = joined_comparison.filter(
        pl.col("review_text").ne_missing(pl.col("review_text_dedup"))
    )

    print("Extracting FULL spoiler blocks including tags:\n")

    # Capture the ENTIRE pattern including tags: (view spoiler)[...(hide spoiler)]
    _spoiler_pattern = re.compile(r'(\(view spoiler\)\[.*?\(hide spoiler\)\])', re.DOTALL)

    for _row in _diff_rows.head(5).iter_rows(named=True):
        _spoiler_text = str(_row['review_text'])
        _matches = _spoiler_pattern.findall(_spoiler_text)

        print(f"Review ID: {_row['review_id']}")
        if _matches:
            for _i, _match in enumerate(_matches, 1):
                print(f"  Spoiler block {_i} ({len(_match)} chars):")
                print(f"    '''{_match}'''")
        else:
            print("  No (view spoiler)[...(hide spoiler)] pattern found")
        print()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Spoilers seem to be either content within/flagged with '(view spoiler)[... (hide spoiler)]'
    """)
    return


@app.cell
def _(df_spoiler, pl):
    # Validate: does every (view spoiler)[ have a matching (hide spoiler)]?
    # Vectorized approach using Polars expressions
    validation_df = df_spoiler.with_columns(
        view_count=pl.col("review_text").str.count_matches(r'\(view spoiler\)\[', literal=False).fill_null(0),
        hide_count=pl.col("review_text").str.count_matches(r'\(hide spoiler\)\]', literal=False).fill_null(0),
    )

    _total_view = validation_df["view_count"].sum()
    _total_hide = validation_df["hide_count"].sum()

    mismatched_reviews_df = validation_df.filter(pl.col("view_count") != pl.col("hide_count"))
    mismatched_reviews = mismatched_reviews_df.select("review_id", "view_count", "hide_count").to_dicts()

    print("Validating spoiler tag pairs:")
    print(f"  Total (view spoiler)[ tags: {_total_view:,}")
    print(f"  Total (hide spoiler)] tags: {_total_hide:,}")
    print(f"  Tags match globally: {_total_view == _total_hide}")
    print(f"\n  Reviews with mismatched tags: {len(mismatched_reviews):,}")

    if mismatched_reviews:
        print("\n  Sample mismatched reviews:")
        for _r in mismatched_reviews[:5]:
            print(f"    {_r['review_id']}: {_r['view_count']} view, {_r['hide_count']} hide")
    return (mismatched_reviews,)


@app.cell
def _(df_spoiler, mismatched_reviews, pl):
    # Categorize mismatched reviews by tag type
    _only_view = [r for r in mismatched_reviews if r['view_count'] > 0 and r['hide_count'] == 0]
    _only_hide = [r for r in mismatched_reviews if r['view_count'] == 0 and r['hide_count'] > 0]
    _partial = [r for r in mismatched_reviews if r['view_count'] > 0 and r['hide_count'] > 0]
    _neither = [r for r in mismatched_reviews if not (r['view_count'] > 0 or r['hide_count'] > 0)]

    _id_to_info = {r['review_id']: r for r in mismatched_reviews}

    def _print_reviews(_category_name, _id_list):
        if not _id_list:
            print(f"\n{_category_name}: 0 reviews\n")
            return
        print(f"\n{'='*80}")
        print(f"{_category_name}: {len(_id_list)} reviews")
        print("="*80 + "\n")

        # Efficiently filter the DataFrame for the relevant IDs
        _ids = {r['review_id'] for r in _id_list}
        reviews_to_print = df_spoiler.filter(pl.col('review_id').is_in(_ids))

        for _row in reviews_to_print.iter_rows(named=True):
            _text = str(_row['review_text']) if _row['review_text'] else ""
            _info = _id_to_info[_row['review_id']]
            print(f"Review ID: {_row['review_id']}")
            print(f"  Tags: {_info['view_count']} view, {_info['hide_count']} hide")
            print(f"  Length: {len(_text):,} chars")
            print(f"\n  FULL TEXT:\n  '''{_text}'''")
            print("\n" + "-"*80 + "\n")

    print(f"Categorizing {len(mismatched_reviews)} mismatched reviews:")
    print(f"  Only view (truncated): {len(_only_view)}")
    print(f"  Only hide (orphan): {len(_only_hide)}")
    print(f"  Partial (both but unequal): {len(_partial)}")
    print(f"  Neither: {len(_neither)}")

    _print_reviews("ONLY VIEW - missing hide (truncated)", _only_view)
    _print_reviews("ONLY HIDE - missing view (orphan)", _only_hide)
    _print_reviews("PARTIAL - both tags but unequal counts", _partial)
    _print_reviews("NEITHER - no spoiler markup", _neither)
    return


@app.cell
def _(mo):
    mo.md(r"""
    22 reviews have mismatched view/hide spoiler tags (0.0016% - negligible)
      - 15 truncated: have `(view spoiler)[` but missing `(hide spoiler)]`
      - 5 orphan: have `(hide spoiler)]` but missing opener
      - 2 partial: have both tags but in unequal numbers
    These edge cases are data quality issues from scraping, not meaningful for analysis
    """)
    return


@app.cell
def _(df_spoiler, pl):
    # Analyze how many spoiler tags per review
    from collections import Counter
    spoiler_counts_df = df_spoiler.with_columns(
        spoiler_tags=pl.col("review_text").str.count_matches(r'\(view spoiler\)\[', literal=False).fill_null(0)
    )
    _spoiler_counts = spoiler_counts_df["spoiler_tags"].to_list()

    # Distribution of spoiler tag counts
    _count_dist = Counter(_spoiler_counts)

    print("Distribution of (view spoiler) tags per review:")
    print(f"  Total reviews in spoiler dataset: {len(_spoiler_counts):,}")
    print(f"\nSpoiler tag count distribution:")
    for _n_tags in sorted(_count_dist.keys()):
        _freq = _count_dist[_n_tags]
        print(f"  {_n_tags} tags: {_freq:,} reviews ({_freq/len(_spoiler_counts)*100:.2f}%)")

    _reviews_with_tags = spoiler_counts_df.filter(pl.col("spoiler_tags") > 0).height
    _reviews_multi_tags = spoiler_counts_df.filter(pl.col("spoiler_tags") > 1).height
    print(f"\nSummary:")
    print(f"  Reviews with at least 1 spoiler tag: {_reviews_with_tags:,} ({_reviews_with_tags/len(_spoiler_counts)*100:.1f}%)")
    print(f"  Reviews with multiple spoiler tags: {_reviews_multi_tags:,} ({_reviews_multi_tags/len(_spoiler_counts)*100:.1f}%)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    93.49% of spoiler-flagged reviews do not have inline spoiler markup
    The spoiler dataset = reviews flagged as spoilers 
    Only 6.51% of reviews have actual '(view spoiler)[...(hide spoiler)]' tags in text
    """)
    return


@app.cell
def _(df_all, pl):
    # Check if dedup dataset contains ANY (view spoiler) tags
    _has_spoiler_tag = df_all.filter(
        pl.col("review_text").str.contains(r"\(view spoiler\)\[", literal=False)
    )

    print("Checking dedup dataset for (view spoiler) tags:")
    print(f"  Total reviews in dedup: {df_all.height:,}")
    print(f"  Reviews with spoiler tags: {_has_spoiler_tag.height:,} ({_has_spoiler_tag.height/df_all.height*100:.2f}%)")
    print(f"  Reviews without spoiler tags: {df_all.height - _has_spoiler_tag.height:,}")
    return


@app.cell
def _(joined_comparison, pl):
    # Check if content is the same after stripping spoiler markup
    _diff_rows = joined_comparison.filter(
        pl.col("review_text").ne_missing(pl.col("review_text_dedup"))
    )

    # Vectorized string stripping
    stripped_df = _diff_rows.with_columns(
        spoiler_stripped=pl.col("review_text")
            .str.replace_all(r'\(view spoiler\)\[', '', literal=True)
            .str.replace_all(r'\(hide spoiler\)\]', '', literal=True)
            .str.replace_all(r'\(hide spoiler\)', '', literal=True)
            .str.replace_all(r'\s+', ' ', literal=False)
            .str.strip_chars(),
        dedup_stripped=pl.col("review_text_dedup")
            .str.replace_all(r'\(view spoiler\)\[', '', literal=True)
            .str.replace_all(r'\(hide spoiler\)\]', '', literal=True)
            .str.replace_all(r'\(hide spoiler\)', '', literal=True)
            .str.replace_all(r'\s+', ' ', literal=False)
            .str.strip_chars(),
    )

    same_count = stripped_df.filter(pl.col("spoiler_stripped") == pl.col("dedup_stripped")).height
    diff_count = stripped_df.height - same_count

    # Get examples of differences
    _diff_examples_df = stripped_df.filter(pl.col("spoiler_stripped") != pl.col("dedup_stripped")).head(5)

    _diff_examples = []
    for _row in _diff_examples_df.iter_rows(named=True):
        _spoiler_stripped = str(_row['spoiler_stripped'])
        _dedup_stripped = str(_row['dedup_stripped'])

        # Find where they differ
        _diverge_at = 0
        for _i, (_c1, _c2) in enumerate(zip(_spoiler_stripped, _dedup_stripped)):
            if _c1 != _c2:
                _diverge_at = _i
                break
        else:
            _diverge_at = min(len(_spoiler_stripped), len(_dedup_stripped))

        _diff_examples.append({
            'review_id': _row['review_id'],
            'diverge_at': _diverge_at,
            'spoiler_around': _spoiler_stripped[max(0,_diverge_at-30):_diverge_at+50],
            'dedup_around': _dedup_stripped[max(0,_diverge_at-30):_diverge_at+50],
            'len_spoiler': len(_spoiler_stripped),
            'len_dedup': len(_dedup_stripped),
        })

    print("After stripping spoiler markup ((view spoiler)[, (hide spoiler)]):")
    print(f"  Content identical: {same_count:,} ({same_count/_diff_rows.height*100:.2f}%)")
    print(f"  Content still different: {diff_count:,} ({diff_count/_diff_rows.height*100:.2f}%)")

    if _diff_examples:
        print("\nExamples still different after stripping:")
        for _ex in _diff_examples:
            print(f"\nReview ID: {_ex['review_id']}")
            print(f"  Lengths: spoiler={_ex['len_spoiler']}, dedup={_ex['len_dedup']}")
            print(f"  Diverge at position {_ex['diverge_at']}:")
            print(f"    SPOILER: ...{_ex['spoiler_around']}...")
            print(f"    DEDUP:   ...{_ex['dedup_around']}...")
    return


@app.cell
def _(df_all, df_spoiler, pl):
    # Create non-spoiler subset by excluding spoiler review_ids
    df_nonspoiler = df_all.join(df_spoiler, on="review_id", how="anti")

    print(f"All reviews (dedup): {df_all.height:,}")
    print(f"Spoiler reviews: {df_spoiler.height:,} ({df_spoiler.height/df_all.height*100:.1f}%)")
    print(f"Non-spoiler reviews: {df_nonspoiler.height:,} ({df_nonspoiler.height/df_all.height*100:.1f}%)")

    # Sanity check
    _expected_nonspoiler = df_all.height - df_all.join(df_spoiler, on="review_id", how="semi").height
    print(f"\nSanity check - expected non-spoiler count: {_expected_nonspoiler:,}")
    print(f"Actual non-spoiler count: {df_nonspoiler.height:,}")
    print(f"Match: {_expected_nonspoiler == df_nonspoiler.height}")
    return (df_nonspoiler,)


@app.cell
def _(mo):
    mo.md(r"""
    100% of text differences are only spoiler tags + whitespace
    After stripping '(view spoiler)[' and '(hide spoiler)]' and normalizing whitespace:
      - All previously different texts become identical
    **Conclusion:**
    1. Spoiler dataset = subset of dedup 
    2. Spoiler flag does not imply spoiler markup: - 93% of flagged reviews have no inline tags
    3. All other columns identical - ratings, dates, votes are the same
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Part 2: COMPARATIVE ANALYSIS: behavior of spoiler vs non-spoiler reviews
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **RATING DISTRIBUTION COMPARISON**
    """)
    return


@app.cell
def _(df_nonspoiler, df_spoiler, pl):
    _spoiler_ratings = df_spoiler.group_by("rating").len().sort("rating")
    _spoiler_ratings = _spoiler_ratings.with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("spoiler_pct")
    ).rename({"len": "spoiler_count"})

    _nonspoiler_ratings = df_nonspoiler.group_by("rating").len().sort("rating")
    _nonspoiler_ratings = _nonspoiler_ratings.with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("nonspoiler_pct")
    ).rename({"len": "nonspoiler_count"})

    _comparison = _spoiler_ratings.join(_nonspoiler_ratings, on="rating", how="full").sort("rating")
    _comparison = _comparison.with_columns(
        (pl.col("spoiler_pct") - pl.col("nonspoiler_pct")).round(2).alias("diff_pct")
    )

    print("Rating Distribution Comparison:")
    _comparison
    return


@app.cell
def _(df_nonspoiler, df_spoiler, pl):
    _spoiler_mean = df_spoiler.filter(pl.col("rating") > 0)["rating"].mean()
    _nonspoiler_mean = df_nonspoiler.filter(pl.col("rating") > 0)["rating"].mean()
    _spoiler_median = df_spoiler.filter(pl.col("rating") > 0)["rating"].median()
    _nonspoiler_median = df_nonspoiler.filter(pl.col("rating") > 0)["rating"].median()

    print("Rating Statistics (excluding 0):")
    print(f"  Spoiler mean: {_spoiler_mean:.3f}, median: {_spoiler_median:.1f}")
    print(f"  Non-spoiler mean: {_nonspoiler_mean:.3f}, median: {_nonspoiler_median:.1f}")
    print(f"  Difference in mean: {_spoiler_mean - _nonspoiler_mean:+.3f}")
    return


@app.cell
def _(df_nonspoiler, df_spoiler, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _spoiler_counts = df_spoiler.group_by("rating").len().sort("rating")
    _nonspoiler_counts = df_nonspoiler.group_by("rating").len().sort("rating")

    # Normalize for comparison
    _spoiler_pct = [c / df_spoiler.height * 100 for c in _spoiler_counts["len"].to_list()]
    _nonspoiler_pct = [c / df_nonspoiler.height * 100 for c in _nonspoiler_counts["len"].to_list()]

    _x = range(6)
    _width = 0.35

    _axes[0].bar([i - _width/2 for i in _x], _spoiler_pct, _width, label="Spoiler", alpha=0.7)
    _axes[0].bar([i + _width/2 for i in _x], _nonspoiler_pct, _width, label="Non-Spoiler", alpha=0.7)
    _axes[0].set_xlabel("Rating")
    _axes[0].set_ylabel("Percentage")
    _axes[0].set_title("Rating Distribution: Spoiler vs Non-Spoiler")
    _axes[0].set_xticks(_x)
    _axes[0].legend()

    # Difference plot
    _diff = [s - n for s, n in zip(_spoiler_pct, _nonspoiler_pct)]
    _colors = ["red" if d < 0 else "green" for d in _diff]
    _axes[1].bar(_x, _diff, color=_colors, alpha=0.7, edgecolor="black")
    _axes[1].axhline(0, color="black", linewidth=0.5)
    _axes[1].set_xlabel("Rating")
    _axes[1].set_ylabel("Difference (Spoiler - Non-Spoiler) %")
    _axes[1].set_title("Rating Distribution Difference")
    _axes[1].set_xticks(_x)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Spoiler reviews rate 0.1 lower on average (3.82 vs 3.92) - negligible
    Same median rating (4.0) for both groups
    Spoiler reviews have slightly fewer 5-stars (-4.6%) and more mid-range ratings
    Reviewers who write spoilers may be slightly more critical, but the difference is too small (<3%) to be actionable
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **REVIEW LENGTH COMPARISON**
    """)
    return


@app.cell
def _(df_nonspoiler, df_spoiler, pl):
    _spoiler_text = df_spoiler.with_columns([
        pl.col("review_text").str.len_chars().alias("char_count"),
        pl.col("review_text").str.split(" ").list.len().alias("word_count"),
    ])

    _nonspoiler_text = df_nonspoiler.with_columns([
        pl.col("review_text").str.len_chars().alias("char_count"),
        pl.col("review_text").str.split(" ").list.len().alias("word_count"),
    ])

    print("Review Length Statistics:")
    print("\nSpoiler reviews:")
    print(f"  Avg words: {_spoiler_text['word_count'].mean():,.0f}")
    print(f"  Median words: {_spoiler_text['word_count'].median():,.0f}")
    print(f"  Avg chars: {_spoiler_text['char_count'].mean():,.0f}")

    print("\nNon-spoiler reviews:")
    print(f"  Avg words: {_nonspoiler_text['word_count'].mean():,.0f}")
    print(f"  Median words: {_nonspoiler_text['word_count'].median():,.0f}")
    print(f"  Avg chars: {_nonspoiler_text['char_count'].mean():,.0f}")

    _word_diff = _spoiler_text['word_count'].mean() - _nonspoiler_text['word_count'].mean()
    _word_ratio = _spoiler_text['word_count'].mean() / _nonspoiler_text['word_count'].mean()
    print(f"\nSpoiler reviews are {_word_ratio:.2f}x longer on average ({_word_diff:+,.0f} words)")
    return


@app.cell
def _(df_nonspoiler, df_spoiler, np, pl, plt):
    _spoiler_words = df_spoiler.with_columns(
        pl.col("review_text").str.split(" ").list.len().alias("word_count")
    )["word_count"].to_numpy()

    _nonspoiler_words = df_nonspoiler.with_columns(
        pl.col("review_text").str.split(" ").list.len().alias("word_count")
    )["word_count"].to_numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograms (capped)
    _axes[0].hist(np.clip(_nonspoiler_words, 0, 500), bins=100, alpha=0.5, label="Non-Spoiler", density=True)
    _axes[0].hist(np.clip(_spoiler_words, 0, 500), bins=100, alpha=0.5, label="Spoiler", density=True)
    _axes[0].set_xlabel("Word Count (capped at 500)")
    _axes[0].set_ylabel("Density")
    _axes[0].set_title("Review Length Distribution")
    _axes[0].legend()

    # Log scale
    _axes[1].hist(np.log1p(_nonspoiler_words), bins=100, alpha=0.5, label="Non-Spoiler", density=True)
    _axes[1].hist(np.log1p(_spoiler_words), bins=100, alpha=0.5, label="Spoiler", density=True)
    _axes[1].set_xlabel("Log(Word Count + 1)")
    _axes[1].set_ylabel("Density")
    _axes[1].set_title("Review Length Distribution (Log Scale)")
    _axes[1].legend()

    plt.tight_layout()
    _fig
    return


@app.cell
def _(df_nonspoiler, df_spoiler, pl):
    _spoiler_by_rating = (
        df_spoiler.filter(pl.col("rating") > 0)
        .with_columns(pl.col("review_text").str.split(" ").list.len().alias("word_count"))
        .group_by("rating")
        .agg(pl.col("word_count").mean().alias("spoiler_avg_words"))
        .sort("rating")
    )

    _nonspoiler_by_rating = (
        df_nonspoiler.filter(pl.col("rating") > 0)
        .with_columns(pl.col("review_text").str.split(" ").list.len().alias("word_count"))
        .group_by("rating")
        .agg(pl.col("word_count").mean().alias("nonspoiler_avg_words"))
        .sort("rating")
    )

    _comparison = _spoiler_by_rating.join(_nonspoiler_by_rating, on="rating")
    _comparison = _comparison.with_columns(
        (pl.col("spoiler_avg_words") / pl.col("nonspoiler_avg_words")).round(2).alias("ratio")
    )

    print("Average Word Count by Rating:")
    _comparison
    return


@app.cell
def _(mo):
    mo.md(r"""
    Spoiler reviews are 1.63x longer on average (197 vs 121 words)
    Median difference is even larger: 112 vs 59 words
    This ratio is consistent across all rating levels (1.5-1.8x) - length difference is not driven by positive/negative rating (no interaction effect)
    Reviews flagged as spoilers tend to be longer.
    For the chatbot: Could use review length as a proxy for review quality/depth, but note it tends to include spoiler content - A user asking "tell me about this book" could get plot details revealed unintentionally if we prioritize longer reviews without checking the spoiler flag.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **VOTES COMPARISON**
    """)
    return


@app.cell
def _(df_nonspoiler, df_spoiler, pl):
    _spoiler_votes = df_spoiler.select([
        pl.col("n_votes").mean().alias("avg"),
        pl.col("n_votes").median().alias("median"),
        pl.col("n_votes").max().alias("max"),
    ])

    _nonspoiler_votes = df_nonspoiler.select([
        pl.col("n_votes").mean().alias("avg"),
        pl.col("n_votes").median().alias("median"),
        pl.col("n_votes").max().alias("max"),
    ])

    print("Vote Statistics:")
    print(f"\nSpoiler reviews:")
    print(f"  Avg votes: {_spoiler_votes['avg'][0]:.2f}")
    print(f"  Median votes: {_spoiler_votes['median'][0]:.0f}")

    print(f"\nNon-spoiler reviews:")
    print(f"  Avg votes: {_nonspoiler_votes['avg'][0]:.2f}")
    print(f"  Median votes: {_nonspoiler_votes['median'][0]:.0f}")

    _ratio = _spoiler_votes['avg'][0] / _nonspoiler_votes['avg'][0]
    print(f"\nSpoiler reviews get {_ratio:.2f}x more votes on average")
    return


@app.cell
def _(df_nonspoiler, df_spoiler, pl):
    _spoiler_with_votes = df_spoiler.filter(pl.col("n_votes") > 0).height
    _nonspoiler_with_votes = df_nonspoiler.filter(pl.col("n_votes") > 0).height

    _spoiler_pct = _spoiler_with_votes / df_spoiler.height * 100
    _nonspoiler_pct = _nonspoiler_with_votes / df_nonspoiler.height * 100

    print("Reviews with at least 1 vote:")
    print(f"  Spoiler: {_spoiler_with_votes:,} ({_spoiler_pct:.1f}%)")
    print(f"  Non-spoiler: {_nonspoiler_with_votes:,} ({_nonspoiler_pct:.1f}%)")
    return


@app.cell
def _(df_nonspoiler, df_spoiler, np, plt):
    _spoiler_votes = df_spoiler["n_votes"].to_numpy()
    _nonspoiler_votes = df_nonspoiler["n_votes"].to_numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Capped histogram
    _axes[0].hist(np.clip(_nonspoiler_votes, 0, 50), bins=50, alpha=0.5, label="Non-Spoiler", density=True)
    _axes[0].hist(np.clip(_spoiler_votes, 0, 50), bins=50, alpha=0.5, label="Spoiler", density=True)
    _axes[0].set_xlabel("Number of Votes (capped at 50)")
    _axes[0].set_ylabel("Density")
    _axes[0].set_title("Vote Distribution")
    _axes[0].set_yscale("log")
    _axes[0].legend()

    # Non-zero votes only, log scale
    _spoiler_nonzero = _spoiler_votes[_spoiler_votes > 0]
    _nonspoiler_nonzero = _nonspoiler_votes[_nonspoiler_votes > 0]

    if len(_spoiler_nonzero) > 0 and len(_nonspoiler_nonzero) > 0:
        _axes[1].hist(np.log1p(_nonspoiler_nonzero), bins=50, alpha=0.5, label="Non-Spoiler", density=True)
        _axes[1].hist(np.log1p(_spoiler_nonzero), bins=50, alpha=0.5, label="Spoiler", density=True)
        _axes[1].set_xlabel("Log(Votes + 1)")
        _axes[1].set_ylabel("Density")
        _axes[1].set_title("Vote Distribution (Reviews with votes, Log)")
        _axes[1].legend()

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Spoiler reviews get 3.17x more votes on average (3.10 vs 0.98)
    Both have median of 0 - most reviews receive no votes regardless of spoiler status
    40.8% of spoiler reviews have at least 1 vote vs 26.4% of non-spoiler
    For the chatbot: When choosing which reviews to show users, n_votes can be a good signal that a review was helpful to other readers. However, highly upvoted reviews are also more likely to contain spoilers, so always check the spoiler flag before displaying them.
    For the recommendation system: Reviews could be weighted by n_votes when aggregating sentiment, but this may overemphasize highly engaged and potentially more extreme reviews. Consider monitoring how this affects the overall book score.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **BOOK ANALYSIS**
    """)
    return


@app.cell
def _(df_nonspoiler, df_spoiler):
    _spoiler_books = set(df_spoiler["book_id"].unique().to_list())
    _nonspoiler_books = set(df_nonspoiler["book_id"].unique().to_list())
    _all_books = _spoiler_books | _nonspoiler_books

    _only_spoiler = _spoiler_books - _nonspoiler_books
    _only_nonspoiler = _nonspoiler_books - _spoiler_books
    _both = _spoiler_books & _nonspoiler_books

    print("Book Overlap Analysis:")
    print(f"  Total unique books: {len(_all_books):,}")
    print(f"  Books with spoiler reviews: {len(_spoiler_books):,}")
    print(f"  Books with non-spoiler reviews: {len(_nonspoiler_books):,}")
    print(f"\n  Books with ONLY spoiler reviews: {len(_only_spoiler):,} ({len(_only_spoiler)/len(_all_books)*100:.2f}%)")
    print(f"  Books with ONLY non-spoiler reviews: {len(_only_nonspoiler):,} ({len(_only_nonspoiler)/len(_all_books)*100:.1f}%)")
    print(f"  Books with BOTH: {len(_both):,} ({len(_both)/len(_all_books)*100:.1f}%)")
    return


@app.cell
def _(df_all, df_spoiler, pl, plt):
    # For books that have spoiler reviews, what % of reviews are spoilers?
    _book_spoiler_counts = df_spoiler.group_by("book_id").len().rename({"len": "spoiler_count"})
    _book_total_counts = df_all.group_by("book_id").len().rename({"len": "total_count"})

    _book_stats = _book_spoiler_counts.join(_book_total_counts, on="book_id")
    _book_stats = _book_stats.with_columns(
        (pl.col("spoiler_count") / pl.col("total_count") * 100).alias("spoiler_pct")
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _axes[0].hist(_book_stats["spoiler_pct"].to_numpy(), bins=50, edgecolor="black", alpha=0.7)
    _axes[0].set_xlabel("% of Reviews that are Spoilers")
    _axes[0].set_ylabel("Number of Books")
    _axes[0].set_title("Spoiler Rate per Book")

    _axes[1].scatter(
        _book_stats["total_count"].to_numpy(),
        _book_stats["spoiler_pct"].to_numpy(),
        alpha=0.3,
        s=5,
    )
    _axes[1].set_xlabel("Total Reviews (Book Popularity)")
    _axes[1].set_ylabel("% Spoiler Reviews")
    _axes[1].set_title("Book Popularity vs Spoiler Rate")
    _axes[1].set_xscale("log")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(df_all, df_spoiler, pl):
    # Top books by spoiler count
    _book_spoiler_counts = df_spoiler.group_by("book_id").len().rename({"len": "spoiler_count"})
    _book_total_counts = df_all.group_by("book_id").len().rename({"len": "total_count"})

    _top_spoiler_books = (
        _book_spoiler_counts.join(_book_total_counts, on="book_id")
        .with_columns(
            (pl.col("spoiler_count") / pl.col("total_count") * 100).round(1).alias("spoiler_pct")
        )
        .sort("spoiler_count", descending=True)
        .head(20)
    )

    print("Top 20 Books by Spoiler Review Count:")
    _top_spoiler_books
    return


@app.cell
def _(mo):
    mo.md(r"""
    98.8% of books have no spoiler reviews - spoilers are concentrated in popular/discussed books
    Then for most books, spoiler filtering is irrelevant, no spoiler reviews exist
    For books with more reviews (almost 25K), expect around 1 in 3 reviews to be spoilers (25-35% spoiler rate) - spoiler filtering is important for these
    For chatbot: spoiler filtering needs to be carefully addressed and matters most for well-known books
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **SUMMARY STATISTICS TABLE**
    """)
    return


@app.cell
def _(df_nonspoiler, df_spoiler, pl):
    _spoiler_words = df_spoiler.with_columns(
        pl.col("review_text").str.split(" ").list.len().alias("word_count")
    )["word_count"]

    _nonspoiler_words = df_nonspoiler.with_columns(
        pl.col("review_text").str.split(" ").list.len().alias("word_count")
    )["word_count"]

    _summary = pl.DataFrame({
        "Metric": [
            "Total Reviews",
            "Unique Users",
            "Unique Books",
            "Avg Rating (excl 0)",
            "Median Rating (excl 0)",
            "Avg Word Count",
            "Median Word Count",
            "Avg Votes",
            "% with Votes",
        ],
        "Spoiler": [
            f"{df_spoiler.height:,}",
            f"{df_spoiler['user_id'].n_unique():,}",
            f"{df_spoiler['book_id'].n_unique():,}",
            f"{df_spoiler.filter(pl.col('rating') > 0)['rating'].mean():.2f}",
            f"{df_spoiler.filter(pl.col('rating') > 0)['rating'].median():.0f}",
            f"{_spoiler_words.mean():,.0f}",
            f"{_spoiler_words.median():,.0f}",
            f"{df_spoiler['n_votes'].mean():.2f}",
            f"{df_spoiler.filter(pl.col('n_votes') > 0).height / df_spoiler.height * 100:.1f}%",
        ],
        "Non-Spoiler": [
            f"{df_nonspoiler.height:,}",
            f"{df_nonspoiler['user_id'].n_unique():,}",
            f"{df_nonspoiler['book_id'].n_unique():,}",
            f"{df_nonspoiler.filter(pl.col('rating') > 0)['rating'].mean():.2f}",
            f"{df_nonspoiler.filter(pl.col('rating') > 0)['rating'].median():.0f}",
            f"{_nonspoiler_words.mean():,.0f}",
            f"{_nonspoiler_words.median():,.0f}",
            f"{df_nonspoiler['n_votes'].mean():.2f}",
            f"{df_nonspoiler.filter(pl.col('n_votes') > 0).height / df_nonspoiler.height * 100:.1f}%",
        ],
    })

    print("Summary Comparison:")
    _summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Recommendations
    - Use dedup dataset as the primary source (clean text)
    - Add `is_spoiler` boolean column by checking if `review_id` exists in spoiler dataset
    - Strip any remaining spoiler markup for NLP tasks

    For the chatbot:
    1. Use the `is_spoiler` flag (from dataset membership) to avoid revealing plot details unless the user explicitly asks. Don't rely on `(view spoiler)[...]` tags - most spoiler reviews don't have them.
    2. Review selection strategy:
       - Use `n_votes` as a quality signal (more votes = more helpful to other readers)
       - Spoiler reviews are 1.63x longer - if using review length for quality, note it correlates with spoiler content
       - Rating distributions are nearly identical - no need to weight spoiler/non-spoiler differently
    3. Text preprocessing:
       - Strip `(view spoiler)[...(hide spoiler)]` markup before any NLP/embedding
       - The 22 malformed tags are negligible (<0.002%) and can be ignored or cleaned with a simple regex fix if desired
    """)
    return


if __name__ == "__main__":
    app.run()
