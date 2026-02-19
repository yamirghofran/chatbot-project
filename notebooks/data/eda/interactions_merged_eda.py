import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import os

    return mo, os, pl


@app.cell
def _(mo):
    mo.md("""
    # EDA: Interactions – Book Popularity

    Compare `goodreads_interactions_merged.parquet` (editions merged)
    with `raw_goodreads_interactions.parquet` (original).

    Titles are resolved via `book_id_map -> books -> works`.
    """)
    return


@app.cell
def _(mo, os, pl):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    data_dir = os.path.join(project_root, "data")

    merged_lf = pl.scan_parquet(os.path.join(data_dir, "goodreads_interactions_merged.parquet"))
    raw_lf = pl.scan_parquet(os.path.join(data_dir, "raw_goodreads_interactions.parquet"))
    books_lf = pl.scan_parquet(os.path.join(data_dir, "raw_goodreads_books.parquet"))
    works_lf = pl.scan_parquet(os.path.join(data_dir, "raw_goodreads_book_works.parquet"))
    id_map = pl.read_parquet(os.path.join(data_dir, "raw_book_id_map.parquet"))

    mo.md(f"""
    ### Datasets
    | Dataset | Rows |
    |---------|-----:|
    | Merged interactions | {merged_lf.select(pl.len()).collect().item():,} |
    | Raw interactions | {raw_lf.select(pl.len()).collect().item():,} |
    | Book ID map | {id_map.shape[0]:,} |
    | Books metadata | {books_lf.select(pl.len()).collect().item():,} |
    | Works metadata | {works_lf.select(pl.len()).collect().item():,} |
    """)
    return books_lf, id_map, merged_lf, raw_lf, works_lf


@app.cell
def _(books_lf, id_map, mo, pl, works_lf):
    _books = (
        books_lf.select(
            pl.col("book_id").cast(pl.Int64, strict=False),
            pl.col("work_id").cast(pl.Int64, strict=False),
            "title",
        ).collect()
    )
    _works = (
        works_lf.select(
            pl.col("work_id").cast(pl.Int64, strict=False),
            "original_title",
        ).collect()
    )

    title_lookup = (
        id_map
        .join(_books, on="book_id", how="left")
        .join(_works, on="work_id", how="left")
        .with_columns(
            pl.coalesce("title", "original_title").alias("resolved_title")
        )
        .select("book_id_csv", "book_id", "resolved_title")
    )

    _total = title_lookup.shape[0]
    _matched = title_lookup.filter(
        pl.col("resolved_title").is_not_null()
        & (pl.col("resolved_title").str.len_chars() > 0)
    ).shape[0]

    mo.md(f"""
    ### Title lookup
    `book_id_csv -> id_map -> book_id -> books.title / works.original_title`

    {_matched:,} / {_total:,} with title ({_matched / _total * 100:.1f}%)
    """)
    return (title_lookup,)


@app.cell
def _(merged_lf, mo, pl):
    merged_stats = (
        merged_lf.group_by("book_id")
        .agg(
            pl.len().alias("total_interactions"),
            pl.col("is_read").sum().alias("total_reads"),
            pl.col("is_reviewed").sum().alias("total_reviews"),
            pl.col("rating").filter(pl.col("rating") > 0).mean().alias("avg_rating"),
            pl.col("rating").filter(pl.col("rating") > 0).count().alias("num_ratings"),
            pl.col("user_id").n_unique().alias("unique_users"),
        )
        .sort("total_interactions", descending=True)
        .collect(engine="streaming")
    )

    mo.vstack([
        mo.md(f"### Merged interactions — {merged_stats.shape[0]:,} unique books"),
        merged_stats.head(10),
    ])
    return (merged_stats,)


@app.cell
def _(mo, pl, raw_lf):
    raw_stats = (
        raw_lf.group_by("book_id")
        .agg(
            pl.len().alias("total_interactions"),
            pl.col("is_read").sum().alias("total_reads"),
            pl.col("is_reviewed").sum().alias("total_reviews"),
            pl.col("rating").filter(pl.col("rating") > 0).mean().alias("avg_rating"),
            pl.col("rating").filter(pl.col("rating") > 0).count().alias("num_ratings"),
            pl.col("user_id").n_unique().alias("unique_users"),
        )
        .sort("total_interactions", descending=True)
        .collect(engine="streaming")
    )

    mo.vstack([
        mo.md(f"### Raw interactions — {raw_stats.shape[0]:,} unique books"),
        raw_stats.head(10),
    ])
    return (raw_stats,)


@app.cell
def _(merged_stats, mo, pl, title_lookup):
    _valid = title_lookup.filter(
        pl.col("resolved_title").is_not_null()
        & (pl.col("resolved_title").str.len_chars() > 0)
    )
    _missing = merged_stats.filter(
        ~pl.col("book_id").is_in(_valid.get_column("book_id_csv"))
    )
    _n = _missing.shape[0]
    _interactions = _missing.select(pl.col("total_interactions").sum()).item()

    _all_csv = title_lookup.get_column("book_id_csv")
    _not_in_map = _missing.filter(~pl.col("book_id").is_in(_all_csv))
    _in_map = _missing.filter(pl.col("book_id").is_in(_all_csv))

    mo.md(f"""
    ### Missing titles breakdown ({_n:,} books, {_interactions:,} interactions)

    | Where it breaks | Books | Interactions |
    |-----------------|------:|------------:|
    | Not in book_id_map | {_not_in_map.shape[0]:,} | {_not_in_map.select(pl.col("total_interactions").sum()).item():,} |
    | In map but no title | {_in_map.shape[0]:,} | {_in_map.select(pl.col("total_interactions").sum()).item():,} |
    """)
    return


@app.cell
def _(merged_stats, mo, pl, raw_stats, title_lookup):
    _valid_ids = (
        title_lookup.filter(
            pl.col("resolved_title").is_not_null()
            & (pl.col("resolved_title").str.len_chars() > 0)
        )
        .get_column("book_id_csv")
    )

    _mt = merged_stats.shape[0]
    _mm = merged_stats.filter(pl.col("book_id").is_in(_valid_ids)).shape[0]
    _mi = merged_stats.filter(pl.col("book_id").is_in(_valid_ids)).select(pl.col("total_interactions").sum()).item()
    _mi_total = merged_stats.select(pl.col("total_interactions").sum()).item()

    _rt = raw_stats.shape[0]
    _rm = raw_stats.filter(pl.col("book_id").is_in(_valid_ids)).shape[0]
    _ri = raw_stats.filter(pl.col("book_id").is_in(_valid_ids)).select(pl.col("total_interactions").sum()).item()
    _ri_total = raw_stats.select(pl.col("total_interactions").sum()).item()

    mo.md(f"""
    ### Title coverage

    | | Merged | Raw |
    |--|--------|-----|
    | Unique books | {_mt:,} | {_rt:,} |
    | With title | {_mm:,} ({_mm/_mt*100:.1f}%) | {_rm:,} ({_rm/_rt*100:.1f}%) |
    | Missing | {_mt-_mm:,} ({(_mt-_mm)/_mt*100:.1f}%) | {_rt-_rm:,} ({(_rt-_rm)/_rt*100:.1f}%) |
    | Interactions covered | {_mi:,}/{_mi_total:,} ({_mi/_mi_total*100:.1f}%) | {_ri:,}/{_ri_total:,} ({_ri/_ri_total*100:.1f}%) |
    """)
    return


@app.cell
def _(merged_stats, mo, pl, title_lookup):
    _lookup = title_lookup.filter(
        pl.col("resolved_title").is_not_null()
        & (pl.col("resolved_title").str.len_chars() > 0)
    ).select(
        pl.col("book_id_csv").alias("book_id"),
        pl.col("resolved_title").alias("title"),
    )

    top_merged = (
        merged_stats.filter(pl.col("book_id").is_in(_lookup.get_column("book_id")))
        .head(50)
        .join(_lookup, on="book_id", how="left")
        .select("book_id", "title", "total_interactions", "unique_users",
                "num_ratings", "avg_rating", "total_reads", "total_reviews")
    )

    mo.vstack([
        mo.md("### Top 50 — Merged"),
        top_merged,
    ])
    return


@app.cell
def _(mo, pl, raw_stats, title_lookup):
    _lookup = title_lookup.filter(
        pl.col("resolved_title").is_not_null()
        & (pl.col("resolved_title").str.len_chars() > 0)
    ).select(
        pl.col("book_id_csv").alias("book_id"),
        pl.col("resolved_title").alias("title"),
    )

    top_raw = (
        raw_stats.filter(pl.col("book_id").is_in(_lookup.get_column("book_id")))
        .head(50)
        .join(_lookup, on="book_id", how="left")
        .select("book_id", "title", "total_interactions", "unique_users",
                "num_ratings", "avg_rating", "total_reads", "total_reviews")
    )

    mo.vstack([
        mo.md("### Top 50 — Raw"),
        top_raw,
    ])
    return


@app.cell
def _(merged_stats, mo, pl, title_lookup):
    _lookup = title_lookup.filter(
        pl.col("resolved_title").is_not_null()
        & (pl.col("resolved_title").str.len_chars() > 0)
    ).select(
        pl.col("book_id_csv").alias("book_id"),
        pl.col("resolved_title").alias("title"),
    )

    top_rated = (
        merged_stats.filter(
            (pl.col("num_ratings") >= 1000)
            & pl.col("book_id").is_in(_lookup.get_column("book_id"))
        )
        .sort("avg_rating", descending=True)
        .head(50)
        .join(_lookup, on="book_id", how="left")
        .select("book_id", "title", "avg_rating", "num_ratings",
                "total_interactions", "unique_users")
    )

    mo.vstack([
        mo.md("### Top 50 by rating (min 1k ratings) — Merged"),
        top_rated,
    ])
    return


@app.cell
def _(merged_stats, mo, pl):
    _q = merged_stats.select(
        pl.col("total_interactions").quantile(0.25).alias("p25"),
        pl.col("total_interactions").quantile(0.50).alias("p50"),
        pl.col("total_interactions").quantile(0.75).alias("p75"),
        pl.col("total_interactions").quantile(0.90).alias("p90"),
        pl.col("total_interactions").quantile(0.95).alias("p95"),
        pl.col("total_interactions").quantile(0.99).alias("p99"),
        pl.col("total_interactions").mean().alias("mean"),
        pl.col("total_interactions").max().alias("max"),
    ).row(0, named=True)

    mo.md(f"""
    ### Interaction distribution (merged)
    | Metric | Value |
    |--------|------:|
    | P25 | {_q['p25']:,.0f} |
    | Median | {_q['p50']:,.0f} |
    | P75 | {_q['p75']:,.0f} |
    | P90 | {_q['p90']:,.0f} |
    | P95 | {_q['p95']:,.0f} |
    | P99 | {_q['p99']:,.0f} |
    | Mean | {_q['mean']:,.1f} |
    | Max | {_q['max']:,.0f} |
    """)
    return


@app.cell
def _(merged_stats, mo, pl):
    _thresholds = [1, 5, 10, 50, 100, 500, 1000, 5000]
    _total = merged_stats.shape[0]
    _rows = []

    for t in _thresholds:
        n = merged_stats.filter(pl.col("total_interactions") <= t).shape[0]
        _rows.append({
            "max_interactions": t,
            "num_books": n,
            "pct": f"{n / _total * 100:.1f}%",
        })

    mo.vstack([
        mo.md("### Long tail"),
        pl.DataFrame(_rows),
    ])
    return


if __name__ == "__main__":
    app.run()
