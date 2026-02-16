import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os

    import duckdb
    import marimo as mo
    import polars as pl

    return mo, os, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The purpose of this notebook is to add a timestamp field from interactions-dedup to interactions.
    """)
    return


@app.cell
def _(mo, os, pl):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    interactions_dedup_path = os.path.join(project_root, "data", "goodreads_interactions_dedup_merged.parquet") # replace this with merged remapped interactions dedup path
    interactions_dedup_df = pl.read_parquet(interactions_dedup_path)

    interactions_path = os.path.join(project_root, "data", "goodreads_interactions_merged.parquet") # replace this with merged remapped interactions path
    interactions_df = pl.read_parquet(interactions_path)

    mo.vstack(
        [
            mo.md("## Interactions Dedup Dataset"),
            interactions_dedup_df.head(),
            mo.md("## Interactions Dataset"),
            interactions_df.head(),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
