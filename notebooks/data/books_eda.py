import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from scipy import stats
    return mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Preliminary EDA for Goodreads Books Dataset
    """)
    return


@app.cell
def _(mo, pl):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "raw_goodreads_books.parquet")
    df = pl.read_parquet(data_path)

    df = df.with_columns(
            [
                pl.when(pl.col(c).str.len_chars() == 0)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in [
                    "publication_month",
                    "publication_year",
                    "publication_day",
                ]
            ]
        )

    df = df.with_columns([
        pl.col("text_reviews_count").cast(pl.Int64, strict=False),
        pl.col("ratings_count").cast(pl.Int64, strict=False),
        pl.col("average_rating").cast(pl.Float64, strict=False),
        pl.col("num_pages").cast(pl.Int64, strict=False),
        pl.col("publication_day").cast(pl.Int64, strict=False),
        pl.col("publication_month").cast(pl.Int64, strict=False),
        pl.col("publication_year").cast(pl.Int64, strict=False),
    ])

    mo.vstack([
        df.head(),
        mo.md(f"Dataset contains {df.shape[0]} books and {df.shape[1]} columns.")
    ])

    return (df,)


@app.cell
def _(df, mo, pl):
    """Data Quality Assestment"""
    neg_ratings = df.filter(pl.col("ratings_count") < 0).shape[0]
    neg_reviews = df.filter(pl.col("text_reviews_count") < 0).shape[0]
    invalid_year = df.filter((pl.col("publication_year") < 1000) | (pl.col("publication_year") > 2026)).shape[0]
    invalid_pages = df.filter((pl.col("num_pages") < 0) | (pl.col("num_pages") > 10000)).shape[0]

    low_ratings = df.filter(pl.col("ratings_count") < 10).shape[0]

    null_counts = df.null_count()


    mo.vstack([
        mo.md("### Missing values per column:"),
        null_counts,   
        mo.md("### Reviews Quality"),
        mo.md(f"- **Negative ratings_count**: {neg_ratings} rows"),
        mo.md(f"- **Negative text_reviews_count**: {neg_reviews} rows"),
        mo.md(f"- **Authors with < 10 ratings**: {low_ratings} ({low_ratings/df.shape[0]*100:.1f}%)"),
        mo.md("### Books Quality"),
        mo.md(f"- **Books with invalid publication year < 1000 or >2026**: {invalid_year} books"),
        mo.md(f"- **Books with invalid pages**: {invalid_pages} books"),
        mo.md("### Findings: "),
        mo.md(r"""
        - Rating features missing values, however no negative values arise.
        - Original publication dates have a lot of missing values, as well as impossible dates. Will need imputation or filtering strategies.
        - Number of pages also missing. Consider relevance.
        """)
    ])
    return


app._unparsable_cell(
    r"""
    mo.hstack([
        df["language_code"].
    ])
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
