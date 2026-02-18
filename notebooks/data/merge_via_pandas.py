import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    from pathlib import Path

    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "data"


    books_df = pd.read_parquet(DATA_DIR / "3_goodreads_books_with_metrics.parquet")
    # interactions_df = pd.read_parquet("3_goodreads_interactions_reduced.parquet")
    # reviews_df = pd.read_parquet("3_goodreads_reviews_dedup_clean.parquet")
    authors_df = pd.read_parquet(DATA_DIR / "raw_goodreads_book_authors.parquet")

    return authors_df, books_df


@app.cell
def _(authors_df, books_df):
    def summarize(name, df):
        print(f"\n{name.upper()}")
        print("-" * 60)
        print(f"Rows: {len(df):,}")
        print("\nColumns & dtypes:")
        print(df.dtypes)

    summarize("books", books_df)
    # summarize("interactions", interactions_df)
    # summarize("reviews", reviews_df)
    summarize("authors", authors_df)
    return


@app.cell
def _(authors_df, books_df):
    _authors_clean = authors_df.copy()
    _authors_clean["author_id"] = _authors_clean["author_id"].astype("string")

    _books_exploded = books_df.explode("authors")

    _books_exploded["author_id"] = _books_exploded["authors"].apply(
        lambda x: x.get("author_id") if isinstance(x, dict) else None
    ).astype("string")

    books_authors_joined = _books_exploded.merge(
        _authors_clean[["author_id", "name"]],
        on="author_id",
        how="left"
    )

    print("\nBOOK → AUTHOR SAMPLE")
    print("-" * 60)
    cols = [c for c in ["title", "author_id", "name"] if c in books_authors_joined.columns]
    print(books_authors_joined[cols].head(10))
    return (books_authors_joined,)


@app.cell
def _(books_authors_joined):
    key_col = next(
        (c for c in ["book_id", "id", "goodreads_book_id", "title"]
         if c in books_authors_joined.columns),
        None
    )

    books_with_author_names = (
        books_authors_joined
        .groupby(key_col)["name"]
        .apply(lambda s: [x for x in s.dropna().unique()])
        .reset_index(name="author_names")
    )

    print("\nAGGREGATED SAMPLE")
    print("-" * 60)
    print(books_with_author_names.head(10))
    return


if __name__ == "__main__":
    app.run()
