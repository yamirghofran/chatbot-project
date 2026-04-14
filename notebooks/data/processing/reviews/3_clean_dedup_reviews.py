import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")

@app.cell
def _():
    import os
    import polars as pl
    return (os, pl)

@app.cell
def _(os, pl):
    from bookdb.utils.paths import find_project_root
    project_root = find_project_root()
    data_path = os.path.join(project_root, "data", "2_goodreads_reviews_dedup_reduced.parquet")
    df = pl.scan_parquet(data_path)
    return (df, project_root)
    

@app.cell
def _(df, pl):
    from bookdb.processing.interactions import GOODREADS_DATE_FORMAT as _GDFMT
    from bookdb.validation import clip_negative_values

    # Clean the data
    # Drop read_at, started_at, date_added
    # Clip n_votes and n_comments to 0
    # Transform date_updated to unix timestamp
    df_clean = (
        df.select(
            pl.exclude("read_at", "started_at", "date_added"),
        )
        .with_columns(
            # Parse date to unix timestamp
            pl.col("date_updated").str.strptime(pl.Datetime, _GDFMT, strict=False).dt.epoch("s").alias("ts_updated"),
        )
        .collect()
    )
    
    # Clip negative values
    df_clean = clip_negative_values(df_clean, ["n_votes", "n_comments"])
    
    return (df_clean,)


@app.cell
def _(df_clean):
    # Preview cleaned data
    df_clean.head(10).collect()


@app.cell
def _(df_clean, os, project_root):
    # Save cleaned data
    output_path = os.path.join(project_root, "data", "3_goodreads_reviews_dedup_clean.parquet")
    df_clean.sink_parquet(output_path)
    return ()

# Later: add spoiler flag, clean empty reviews

if __name__ == "__main__":
    app.run()
