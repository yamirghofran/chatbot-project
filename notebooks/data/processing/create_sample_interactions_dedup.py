import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import polars as pl

    return os, pl


@app.cell
def _(os, pl):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_path = os.path.join(
        project_root, "data", "raw_goodreads_interactions_dedup.parquet"
    )

    _df_full = pl.read_parquet(data_path)
    _full_rows = _df_full.shape[0]
    df = _df_full.sample(fraction=0.05, seed=42)
    del _df_full

    # Save the sampled dataframe with "sample" suffix
    output_path = data_path.replace(".parquet", "_sample.parquet")
    df.write_parquet(output_path)

    output_path, df


if __name__ == "__main__":
    app.run()
