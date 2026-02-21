import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import math
    import polars as pl

    return math, os, pl


@app.cell
def _(os, pl):
    project_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    reviews_path = os.path.join(project_root, "data", "3_goodreads_reviews_dedup_clean.parquet")
    books_path = os.path.join(project_root, "data", "3_goodreads_books_with_metrics.parquet")

    df_reviews = pl.scan_parquet(reviews_path)
    df_books = pl.scan_parquet(books_path).select("book_id")
    return df_books, df_reviews, project_root


@app.cell
def _(df_books, df_reviews, pl):
    total_books = df_books.select(pl.len()).collect().item()
    books_with_reviews = df_reviews.select(pl.col("book_id").n_unique()).collect().item()
    books_without_reviews = total_books - books_with_reviews

    print(f"total books: {total_books:,}")
    print(f"with reviews: {books_with_reviews:,}")
    print(f"without reviews: {books_without_reviews:,}")
    return (books_with_reviews,)


@app.cell
def _(books_with_reviews, math):
    TARGET = 1_500_000
    k = math.ceil(TARGET / books_with_reviews)
    print(f"k = {TARGET:,} / {books_with_reviews:,} = {k} reviews per book")
    return (k,)


@app.cell
def _(k, os, pl, project_root):
    import pyarrow.parquet as pq
    import gc
    import time

    t0 = time.time()

    pf = pq.ParquetFile(os.path.join(project_root, "data", "3_goodreads_reviews_dedup_clean.parquet"))
    chunks = []
    for i, arrow_batch in enumerate(pf.iter_batches(batch_size=500_000)):
        tc = time.time()
        df_batch = pl.from_arrow(arrow_batch)
        df_batch = df_batch.filter(
            pl.col("review_text").is_not_null()
            & (pl.col("review_text").str.len_chars() > 0)
        )
        if len(df_batch) == 0:
            continue
        print(f"batch {i+1}: {len(df_batch):,} reviews")

        df_topk = (
            df_batch
            .with_columns(
                (
                    pl.col("n_votes") * 3
                    + pl.col("review_text").str.len_chars()
                    + (pl.col("rating") - 3).abs() * 10
                ).alias("score")
            )
            .sort("score", descending=True)
            .group_by("book_id")
            .head(k)
        )
        chunks.append(df_topk)
        del df_batch, df_topk
        print(f"batch {i+1}: done ({time.time()-tc:.1f}s)")

    df_reduced = (
        pl.concat(chunks)
        .sort("score", descending=True)
        .group_by("book_id")
        .head(k)
        .drop("score")
    )
    del chunks
    gc.collect()
    print(f"reduced to {len(df_reduced):,} reviews")

    output_path = os.path.join(project_root, "data", "4_goodreads_reviews_reduced.parquet")
    df_reduced.write_parquet(output_path)
    print(f"saved ({time.time()-t0:.1f}s total)")
    return (df_reduced,)


@app.cell
def _(df_reduced):
    df_reduced.head(100)
    return


if __name__ == "__main__":
    app.run()
