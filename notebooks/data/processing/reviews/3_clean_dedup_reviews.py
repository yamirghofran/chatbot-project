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
    project_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    data_path = os.path.join(project_root, "data", "2_goodreads_reviews_dedup_reduced.parquet")
    df = pl.scan_parquet(data_path)
    return (df, project_root)
    

@app.cell
def _(df, pl):
    # Clean the data
    # Drop read_at, started_at, date_added
    # Clip n_votes and n_comments to 0
    # Transform date_updated to unix timestamp 
    df_clean = df.select(
        pl.exclude("read_at", "started_at", "date_added"),
    ).with_columns(
        # Clip n_votes and n_comments to 0
        pl.when(pl.col("n_votes") < 0).then(0).otherwise(pl.col("n_votes")).alias("n_votes"),
        pl.when(pl.col("n_comments") < 0).then(0).otherwise(pl.col("n_comments")).alias("n_comments"),

        # Parse dates to unix timestamps 
        pl.col("date_updated").str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False).dt.epoch("s").alias("ts_updated"),
    )
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
