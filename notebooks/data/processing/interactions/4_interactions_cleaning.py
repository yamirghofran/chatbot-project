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
    # Interactions Reduced - Quality Checks

    1. Timestamp range
    2. is_read vs rating consistency
    3. is_reviewed implies is_read
    """)
    return


@app.cell
def _(os, pl):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    data_dir = os.path.join(project_root, "data")

    interactions_path = os.path.join(data_dir, "3_goodreads_interactions_reduced.parquet")
    interactions_lf = pl.scan_parquet(interactions_path)

    total_rows = interactions_lf.select(pl.len()).collect().item()
    return data_dir, interactions_lf, total_rows


@app.cell
def _(interactions_lf, mo, pl, total_rows):
    # 1. Timestamp range
    ts_stats = (
        interactions_lf
        .select(
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
            pl.col("timestamp").is_null().sum().alias("null_count"),
        )
        .collect()
    )

    _min_ts = ts_stats["min_ts"][0]
    _max_ts = ts_stats["max_ts"][0]
    _null_count = ts_stats["null_count"][0]

    from datetime import datetime, timezone

    _min_dt = datetime.fromtimestamp(_min_ts, tz=timezone.utc).strftime("%Y-%m-%d") if _min_ts is not None else "N/A"
    _max_dt = datetime.fromtimestamp(_max_ts, tz=timezone.utc).strftime("%Y-%m-%d") if _max_ts is not None else "N/A"

    mo.md(f"""
    ## 1. Timestamp Range
    - **Min timestamp**: {_min_ts} ({_min_dt})
    - **Max timestamp**: {_max_ts} ({_max_dt})
    - **Null timestamps**: {_null_count:,} ({_null_count / total_rows * 100:.2f}%)
    """)
    return


@app.cell
def _(interactions_lf, mo, pl, total_rows):
    # 2. is_read vs rating consistency
    # Check: is_read = 0 but rating != 0 
    _not_read_but_rated = (
        interactions_lf
        .filter((pl.col("is_read") == 0) & (pl.col("rating") != 0))
        .select(pl.len())
        .collect()
        .item()
    )

    # Check: is_read = 1 but rating = 0 
    _read_but_not_rated = (
        interactions_lf
        .filter((pl.col("is_read") == 1) & (pl.col("rating") == 0))
        .select(pl.len())
        .collect()
        .item()
    )

    # Check: rating != 0 but is_read = 0 
    _rated_not_read = (
        interactions_lf
        .filter((pl.col("rating") != 0) & (pl.col("is_read") == 0))
        .select(pl.len())
        .collect()
        .item()
    )

    mo.md(f"""
    ## 2. is_read vs rating Consistency
    - **is_read=0 AND rating!=0** (not read but rated): {_not_read_but_rated:,} ({_not_read_but_rated / total_rows * 100:.2f}%) {"- suspicious!" if _not_read_but_rated > 0 else ""}
    - **is_read=1 AND rating=0** (read but not rated): {_read_but_not_rated:,} ({_read_but_not_rated / total_rows * 100:.2f}%) - expected
    - **rating!=0 AND is_read=0** (rated but not read): {_rated_not_read:,} ({_rated_not_read / total_rows * 100:.2f}%) {"- suspicious!" if _rated_not_read > 0 else ""}

    {"All consistent!" if _not_read_but_rated == 0 else "Some inconsistencies found - users rated books without marking as read."}
    """)
    return


@app.cell
def _(interactions_lf, mo, pl, total_rows):
    # 3. is_reviewed = 1 implies is_read = 1
    _reviewed_not_read = (
        interactions_lf
        .filter((pl.col("is_reviewed") == 1) & (pl.col("is_read") == 0))
        .select(pl.len())
        .collect()
        .item()
    )

    _total_reviewed = (
        interactions_lf
        .filter(pl.col("is_reviewed") == 1)
        .select(pl.len())
        .collect()
        .item()
    )

    mo.md(f"""
    ## 3. is_reviewed implies is_read
    - **Total reviewed**: {_total_reviewed:,} ({_total_reviewed / total_rows * 100:.2f}%)
    - **is_reviewed=1 AND is_read=0** (reviewed but not read): {_reviewed_not_read:,} {"- suspicious!" if _reviewed_not_read > 0 else ""}

    {"All consistent! Every reviewed book is marked as read." if _reviewed_not_read == 0 else f"Found {_reviewed_not_read:,} inconsistencie."}
    """)
    return


@app.cell
def _(interactions_lf, mo, pl, total_rows):
    # 4. Null values per column
    _schema = interactions_lf.collect_schema()
    _cols = list(_schema.keys())

    _null_counts = (
        interactions_lf
        .select([pl.col(c).is_null().sum().alias(c) for c in _cols])
        .collect()
    )

    _rows = []
    for c in _cols:
        _n = _null_counts[c][0]
        _pct = _n / total_rows * 100
        _rows.append(f"| `{c}` | `{_schema[c]}` | {_n:,} | {_pct:.2f}% |")

    _table = "\n".join(_rows)

    mo.md(f"""
    ## 4. Null Values

    | Column | Type | Null Count | % |
    |--------|------|-----------|---|
    {_table}
    """)
    return


@app.cell
def _(interactions_lf, mo, pl, total_rows):
    # 5. Future timestamps (later than today)
    import time

    _now = int(time.time())

    _future_count = (
        interactions_lf
        .filter(pl.col("timestamp") > _now)
        .select(pl.len())
        .collect()
        .item()
    )

    _future_sample = (
        interactions_lf
        .filter(pl.col("timestamp") > _now)
        .head(10)
        .collect()
    )

    import datetime as _dt

    _now_dt = _dt.datetime.fromtimestamp(_now, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M")

    mo.vstack([
        mo.md(f"""
    ## 5. Future Timestamps
    - **Current time**: {_now} ({_now_dt})
    - **Timestamps after today**: {_future_count:,} ({_future_count / total_rows * 100:.2f}%)

    {"All timestamps are in the past!" if _future_count == 0 else "Some timestamps are in the future - these are suspicious."}
        """),
        mo.ui.table(_future_sample) if _future_count > 0 else mo.md(""),
    ])
    return


@app.cell
def _(data_dir, interactions_lf, mo, os, pl):
    # 6. Clip future timestamps to now
    import time as _time

    _now_ts = int(_time.time())
    _out_path = os.path.join(data_dir, "3_goodreads_interactions_reduced.parquet")
    _tmp_path = os.path.join(data_dir, "4_goodreads_interactions_reduced_tmp.parquet")

    _clipped = interactions_lf.with_columns(
        pl.when(pl.col("timestamp") > _now_ts)
        .then(_now_ts)
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )

    _clipped.sink_parquet(_tmp_path, engine="streaming")
    os.replace(_tmp_path, _out_path)

    _clipped_count = (
        pl.scan_parquet(_out_path)
        .filter(pl.col("timestamp") > _now_ts)
        .select(pl.len())
        .collect()
        .item()
    )

    mo.md(f"""
    ## 6. Clipped Future Timestamps
    - **Clipped to**: {_now_ts}
    - **Remaining future timestamps**: {_clipped_count:,}
    """)
    return


if __name__ == "__main__":
    app.run()
